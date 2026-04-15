import torch
import torch.nn as nn
import random
from typing import Callable, Optional
import time

from learning_algorithms.MixedObservationEncoder import infer_encoder_out_dim
from learning_algorithms.RandomDistilation import RNDModel, RunningMeanStd
from learning_algorithms.RainbowNetworks import EV_Q_Network, IQN_Network
from learning_algorithms.agent import Agent


class RainbowBase(Agent):
    """
    Base class for Rainbow DQN agents, housing shared hyperparameters,
    running statistics, RND intrinsic reward models, and common utilities.
    """
    def __init__(
        self,
        input_dim,
        n_action_dims,
        n_action_bins,
        hidden_layer_sizes=[128, 128],
        lr: float = 1e-3,
        gamma: float = 0.99,
        alpha: float = 0.9,
        tau: float = 0.03,
        polyak_tau: float = 0.03,
        l_clip: float = -1.0,
        soft: bool = False,
        munchausen: bool = False,
        Thompson: bool = False,
        dueling: bool = False,

        Beta: float = 0.0,
        delayed: bool = True,
        ent_reg_coef: float = 0.0,
        rnd_output_dim: int = 128,
        rnd_lr: float = 1e-3,
        intrinsic_lr: float = 1e-3,
        int_r_clip: float = 5.0,
        ext_r_clip: float = 5.0,
        beta_half_life_steps: Optional[int] = None,
        norm_obs: bool = True,
        burn_in_updates: int = 0,
        encoder_factory: Optional[Callable[[], nn.Module]] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_action_dims = n_action_dims
        self.n_action_bins = n_action_bins
        self.n_actions = n_action_dims * n_action_bins
        self.hidden_layer_sizes = hidden_layer_sizes
        
        self.lr = lr
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.polyak_tau = polyak_tau
        self.l_clip = l_clip
        self.soft = soft
        self.munchausen = munchausen
        self.Thompson = Thompson
        self.dueling = dueling
                
        self.Beta = Beta
        self.start_Beta = Beta
        self.delayed_target = delayed
        self.ent_reg_coef = ent_reg_coef
        self.rnd_output_dim = rnd_output_dim
        self.rnd_lr = rnd_lr
        self.intrinsic_lr = intrinsic_lr
        
        # Support kwargs naming flexibly for unified interface
        self.int_r_clip = int_r_clip
        self.ext_r_clip = ext_r_clip
        
        self.beta_half_life_steps = beta_half_life_steps
        self.norm_obs = norm_obs
        self.burn_in_updates = burn_in_updates
        self.encoder_factory = encoder_factory
        
        self.step = 0
        self.last_eps = 1.0
        self.update_timings = None

        # RND and running stats setup
        rnd_target_encoder = encoder_factory() if encoder_factory is not None else None
        rnd_predictor_encoder = encoder_factory() if encoder_factory is not None else None
        
        self.rnd = RNDModel(
            input_dim,
            rnd_output_dim,
            encoder_target=rnd_target_encoder,
            encoder_predictor=rnd_predictor_encoder,
        ).float()
        
        self.rnd_optim = torch.optim.Adam(self.rnd.predictor.parameters(), lr=rnd_lr)
        self.obs_rms = RunningMeanStd(shape=(input_dim,))
        self.int_rms = RunningMeanStd(shape=())

    def to(self, device):
        """Move the agent and all its subcomponents to a specific device."""
        main_lr = self.optim.param_groups[0]["lr"] if hasattr(self, "optim") else self.lr
        int_lr = self.int_optim.param_groups[0]["lr"] if hasattr(self, "int_optim") else self.intrinsic_lr
        rnd_lr = self.rnd_optim.param_groups[0]["lr"] if hasattr(self, "rnd_optim") else self.rnd_lr

        if hasattr(self, "ext_online"): self.ext_online.to(device)
        if hasattr(self, "ext_target"): self.ext_target.to(device)
        if hasattr(self, "int_online"): self.int_online.to(device)
        if hasattr(self, "int_target"): self.int_target.to(device)
        if hasattr(self, "rnd") and hasattr(self.rnd, "to"): self.rnd.to(device)
        if hasattr(self, "obs_rms") and hasattr(self.obs_rms, "to"): self.obs_rms.to(device)
        if hasattr(self, "int_rms") and hasattr(self.int_rms, "to"): self.int_rms.to(device)

        if hasattr(self, "ext_online"):
            self.optim = torch.optim.Adam(self.ext_online.parameters(), lr=main_lr)
        if hasattr(self, "int_online"):
            self.int_optim = torch.optim.Adam(self.int_online.parameters(), lr=int_lr)
        if hasattr(self, "rnd"):
            self.rnd_optim = torch.optim.Adam(self.rnd.predictor.parameters(), lr=rnd_lr)
        return self

    def update_target(self):
        """Polyak averaging: target = (1 - tau) * target + tau * online."""
        if not self.delayed_target:
            return
        with torch.no_grad():
            for tp, op in zip(self.ext_target.parameters(), self.ext_online.parameters()):
                tp.data.mul_(1.0 - self.polyak_tau).add_(op.data, alpha=self.polyak_tau)
            for tp, op in zip(self.int_target.parameters(), self.int_online.parameters()):
                tp.data.mul_(1.0 - self.polyak_tau).add_(op.data, alpha=self.polyak_tau)

    @torch.no_grad()
    def update_running_stats(self, next_obs: torch.Tensor, r: torch.Tensor):
        x64 = next_obs.to(dtype=torch.float64, device=self.obs_rms.mean.device)
        self.obs_rms.update(x64)
        with torch.no_grad():
            norm_x64 = self.obs_rms.normalize(x64)
            if norm_x64.ndim == 1:
                norm_x64 = norm_x64.unsqueeze(0)
            rnd_err = self.rnd(norm_x64.to(dtype=torch.float32)).squeeze().detach()
            self.int_rms.update(rnd_err.to(dtype=torch.float64))

    def _update_RND(self, next_obs: torch.Tensor, batch_norm=False):
        with torch.no_grad():        
            if batch_norm:
                norm_next_obs_f32 = (next_obs - next_obs.mean(dim=0, keepdim=True)) / (
                    next_obs.std(dim=0, keepdim=True, unbiased=False) + 1e-6
                )
            else:
                norm_next_obs = self.obs_rms.normalize(next_obs.to(dtype=torch.float64))
                norm_next_obs_f32 = norm_next_obs.to(dtype=torch.float32)

        rnd_errors = self.rnd(norm_next_obs_f32)
        rnd_loss = rnd_errors.mean()
        self.rnd_optim.zero_grad()
        rnd_loss.backward()
        self.rnd_optim.step()
        return rnd_errors, rnd_loss


class EVRainbowDQN(RainbowBase):
    """Non-distributional counterpart to RainbowDQN with optional dueling and all five pillars."""

    def __init__(
        self,
        input_dim,
        n_action_dims,
        n_action_bins,
        hidden_layer_sizes=[128, 128],
        lr: float = 1e-3,
        gamma: float = 0.99,
        alpha: float = 0.9,
        tau: float = 0.03,
        polyak_tau: float = 0.005,
        l_clip: float = -1.0,
        soft: bool = False,
        munchausen: bool = False,
        Thompson: bool = False,
        dueling: bool = False,

        Beta: float = 0.0,
        delayed: bool = True,
        ent_reg_coef: float = 0.0,
        beta_half_life_steps: Optional[int] = None,
        rnd_output_dim: int = 128,
        rnd_lr: float = 1e-3,
        intrinsic_lr: float = 1e-3,
        norm_obs: bool = True,
        burn_in_updates: int = 0,
        int_r_clip=5,
        ext_r_clip=5,
        encoder_factory: Optional[Callable[[], nn.Module]] = None,
    ):
        super().__init__(
            input_dim=input_dim, n_action_dims=n_action_dims, n_action_bins=n_action_bins,
            hidden_layer_sizes=hidden_layer_sizes, lr=lr, gamma=gamma, alpha=alpha, tau=tau,
            polyak_tau=polyak_tau, l_clip=l_clip, soft=soft, munchausen=munchausen, Thompson=Thompson,
            dueling=dueling, Beta=Beta, delayed=delayed, ent_reg_coef=ent_reg_coef,
            rnd_output_dim=rnd_output_dim, rnd_lr=rnd_lr, intrinsic_lr=intrinsic_lr,
            int_r_clip=int_r_clip, ext_r_clip=ext_r_clip, beta_half_life_steps=beta_half_life_steps,
            norm_obs=norm_obs, burn_in_updates=burn_in_updates, encoder_factory=encoder_factory
        )

        def _encoder_kwargs():
            if encoder_factory is None: return {}
            encoder = encoder_factory()
            return {"encoder": encoder, "encoder_out_dim": infer_encoder_out_dim(encoder, int(input_dim))}

        ext_online_kwargs = _encoder_kwargs()
        ext_target_kwargs = _encoder_kwargs()
        int_online_kwargs = _encoder_kwargs()
        int_target_kwargs = _encoder_kwargs()

        self.ext_online = EV_Q_Network(
            input_dim, n_action_dims, n_action_bins, hidden_layer_sizes=hidden_layer_sizes,
            dueling=dueling, popart=True, **ext_online_kwargs,
        ).float()
        self.ext_target = EV_Q_Network(
            input_dim, n_action_dims, n_action_bins, hidden_layer_sizes=hidden_layer_sizes,
            dueling=dueling, popart=True, **ext_target_kwargs,
        ).float()
        self.int_online = EV_Q_Network(
            input_dim, n_action_dims, n_action_bins, hidden_layer_sizes=hidden_layer_sizes,
            dueling=dueling, popart=True, **int_online_kwargs,
        ).float()
        self.int_target = EV_Q_Network(
            input_dim, n_action_dims, n_action_bins, hidden_layer_sizes=hidden_layer_sizes,
            dueling=dueling, popart=True, **int_target_kwargs,
        ).float()

        self.ext_target.requires_grad_(False)
        self.ext_target.load_state_dict(self.ext_online.state_dict())
        self.int_target.requires_grad_(False)
        self.int_target.load_state_dict(self.int_online.state_dict())

        self.optim = torch.optim.Adam(self.ext_online.parameters(), lr=lr)
        self.int_optim = torch.optim.Adam(self.int_online.parameters(), lr=intrinsic_lr)

    @torch.no_grad()
    def _soft_policy(self, q_values: torch.Tensor):
        logpi = torch.clamp(torch.log_softmax(q_values / self.tau, dim=-1), min=-1e8)
        pi = torch.exp(logpi)
        ent = -(pi * logpi).sum(dim=-1)  # [B]
        return pi, logpi, ent

    def update(
        self, obs, a, r, next_obs, term, batch_size=None, step=0, extrinsic_only=False
    ):
        # Set up wall clock time for optimization later
        if self.update_timings is None:
            self.update_timings = {
                "update_rnd": 0.0, "extrinsic_loss": 0.0,
                "intrinsic_loss": 0.0, "logging": 0.0, "total": 0.0,
            }
        update_timings = self.update_timings
        t__ = time.time()

        self.step += 1
        if batch_size is None:
            idx = torch.arange(0, len(r))
            batch_size = len(r)
        else:
            idx = torch.randint(low=0, high=step, size=(batch_size,))
        b_next_obs = next_obs[idx]
        t_ = time.time()
        b_obs = obs[idx]
        b_r_ext = r[idx]
        b_term = term[idx]

        t_ = time.time()
        if self.step < self.burn_in_updates:
            if self.Beta > 0.0 or getattr(self, 'always_update_rnd', False):
                rnd_errors, rnd_loss = self._update_RND(b_next_obs, batch_norm=True)
                return 0.0
            return 0.0
        update_timings["update_rnd"] += time.time() - t_
        
        if self.Beta > 0.0 or getattr(self, 'always_update_rnd', False):
            rnd_errors, rnd_loss = self._update_RND(b_next_obs, batch_norm=False)
            if self.beta_half_life_steps is not None and self.beta_half_life_steps > 0:
                self.Beta = self.start_Beta * (
                    0.5 ** (self.step / self.beta_half_life_steps)
                )
            with torch.no_grad():
                norm_int = self.int_rms.scale(
                    rnd_errors.detach().to(dtype=torch.float64)
                )
                b_r_int = norm_int.to(dtype=torch.float32)
        else:
            rnd_errors, rnd_loss, b_r_int = torch.zeros_like(b_r_ext), 0, 0

        logpi_now = None
        pi_now = None
        entropy_loss = 0
        current_sigma = self.ext_online.output_layer.sigma.detach()

        # Get target
        with torch.no_grad():
            q_ext_norm = self.ext_online(b_obs, normalized=True)  # [B,D,Bins]
            # if self.Beta>0.0:
            #     q_int_norm = self.int_online(b_obs, normalized=True)  # [B,D,Bins]
            #     q_mixed_now = self.Beta * q_int_norm + (1-self.Beta)*q_ext_norm
            # else:
            #     q_mixed_now = q_ext_norm
            
            b_actions_idx = a[idx]
            if b_actions_idx.ndim == 1:
                b_act_view = b_actions_idx.view(-1, 1)
            else:
                b_act_view = b_actions_idx  # [B,D]
            b_actions_idx = b_act_view.unsqueeze(-1)
            
            q_next_online_norm = self.ext_online(b_next_obs, normalized=True)
            # if self.Beta > 0.0:
            #     q_next_int_online_norm = self.int_online(b_next_obs, normalized=True)
            #     if self.delayed_target:
            #         q_next_int_target_norm = self.int_target(b_next_obs, normalized=True)       
            #     else: 
            #         q_next_int_target_norm = q_next_int_online_norm 
            #     #q_next_online_mixed = (1.0 - self.Beta) * q_next_online_norm + self.Beta * q_next_int_online_norm
            # else:
            #     #q_next_online_mixed = q_next_online_norm

            q_next_target_raw = self.ext_target(b_next_obs, normalized=False) if self.delayed_target else self.ext_online(b_next_obs, normalized=False)   
            if self.munchausen:
                if logpi_now is None:
                    logpi_now = torch.clamp(
                        torch.log_softmax(q_ext_norm / self.tau, dim=-1), min=-1e8
                    )

                selected_logpi = torch.gather(logpi_now, -1, b_actions_idx).squeeze(-1)  # [B,D]
                r_kl = torch.clamp(selected_logpi, min=self.l_clip)
                if r_kl.ndim > 1:
                    r_kl = r_kl.sum(-1)
                b_r_ext += current_sigma * self.alpha * self.tau * r_kl 

            if self.munchausen or self.soft:
                logpi_next = torch.clamp(
                    torch.log_softmax(q_next_online_norm / self.tau, dim=-1), min=-1e8
                )
                pi_next = torch.exp(logpi_next)
                next_head_vals = (pi_next * (current_sigma * self.alpha * logpi_next + q_next_target_raw)).sum(-1)
            else:
                target_actions_next = q_next_online_norm.argmax(dim=-1, keepdim=True).detach()
                next_head_vals = torch.gather(q_next_target_raw, -1, target_actions_next).squeeze(-1)

            assert isinstance(next_head_vals, torch.Tensor)
            online_ext_target = b_r_ext + self.gamma * (1 - b_term).view(-1, 1) * next_head_vals

        target_for_stats_ext = online_ext_target.detach() # maintain [B, D]
        self.ext_online.output_layer.update_stats(target_for_stats_ext)
        if self.delayed_target:
            self.ext_target.output_layer.sigma.copy_(self.ext_online.output_layer.sigma)
            self.ext_target.output_layer.mu.copy_(self.ext_online.output_layer.mu)
        td_target_norm = self.ext_online.output_layer.normalize(target_for_stats_ext)
        q_ext_now_norm = self.ext_online(b_obs, normalized=True)
        q_selected_norm = torch.gather(q_ext_now_norm, -1, b_actions_idx).squeeze(-1) # [B, D]
        extrinsic_loss = torch.nn.functional.mse_loss(q_selected_norm, td_target_norm)
        
        # if self.Beta>0.0:
        #     q_mixed_now = self.Beta * q_int_norm.detach() + (1-self.Beta)*q_ext_now_norm
        # else:
        #     q_mixed_now = q_ext_now_norm
        if self.ent_reg_coef > 0.0:
            logpi_now = torch.clamp(
                torch.log_softmax(q_ext_now_norm / self.tau, dim=-1), min=-1e8
            )
            if torch.isnan(logpi_now).any():
                print("NaN detected in logpi_now!")
            pi_now = torch.exp(logpi_now)
            entropy_loss = (pi_now * logpi_now).sum(dim=-1).mean()
        extrinsic_loss = extrinsic_loss + self.ent_reg_coef * entropy_loss
        
        self.optim.zero_grad()
        extrinsic_loss.backward()
        if hasattr(self, "ext_online"):
            torch.nn.utils.clip_grad_norm_(self.ext_online.parameters(), max_norm=10.0)
        self.optim.step()
        update_timings["extrinsic_loss"] += time.time() - t_

        if extrinsic_only:
            update_timings["total"] += time.time() - t__
            return float(extrinsic_loss.item())

        t_ = time.time()

        # Intrinsic Q update
        with torch.no_grad():
            next_int_actions = self.int_online(b_next_obs, normalized=True).argmax(-1,keepdim=True).detach()
            int_q_next = (self.int_target if self.delayed_target else self.int_online)(
                b_next_obs, normalized=False
            )
            int_q_next_target = torch.gather(int_q_next, -1, next_int_actions).squeeze(-1)
            r_int_only = (
                b_r_int if isinstance(b_r_int, torch.Tensor) else torch.zeros_like(b_r_ext)
            )
            int_td_target = r_int_only + self.gamma * int_q_next_target

        target_for_stats_int = int_td_target.detach()
        self.int_online.output_layer.update_stats(target_for_stats_int)
        if self.delayed_target:
            self.int_target.output_layer.sigma.copy_(self.int_online.output_layer.sigma)
            self.int_target.output_layer.mu.copy_(self.int_online.output_layer.mu)
        
        int_td_target_norm = self.int_online.output_layer.normalize(target_for_stats_int)
        int_q_now_norm = self.int_online(b_obs, normalized=True)
        int_q_selected_norm = torch.gather(int_q_now_norm, -1, b_actions_idx).squeeze(-1)
            
        intrinsic_loss = torch.nn.functional.mse_loss(
            int_q_selected_norm, int_td_target_norm
        )

        self.int_optim.zero_grad()
        intrinsic_loss.backward()
        if hasattr(self, "int_online"):
            torch.nn.utils.clip_grad_norm_(self.int_online.parameters(), max_norm=10.0)
        self.int_optim.step()
        update_timings["intrinsic_loss"] += time.time() - t_

        t_ = time.time()

        if isinstance(b_r_int, torch.Tensor):
            r_int_log = float(b_r_int.mean().item())
        else:
            r_int_log = 0.0

        if isinstance(entropy_loss, torch.Tensor):
            entropy_val = float(entropy_loss.item())
        else:
            entropy_val = entropy_loss
        if isinstance(rnd_loss, torch.Tensor):
            rnd_loss = rnd_loss.item()
        self.last_losses = {
            "extrinsic": float(extrinsic_loss.item()),
            "intrinsic": float(intrinsic_loss.item()),
            "rnd": float(rnd_loss),
            "avg_r_int": r_int_log,
            "entropy_reg": entropy_val,
            "batch_nonzero_r_frac": float((b_r_ext != 0).float().mean().item()),
            "target_mean": float(target_for_stats_ext.mean().item()),
            "entropy_loss": entropy_val,
            "Beta": float(self.Beta),
            "Q_ext_mean": float(q_ext_now_norm.mean().item()),
            "Q_int_mean": float(int_q_now_norm.mean().item()) if 'int_q_now_norm' in locals() else 0.0,
            "last_eps": float(self.last_eps),
        }
        if self.tb_writer is not None:
            try:
                for k, v in self.last_losses.items():
                    if isinstance(v, (int, float)):
                        self.tb_writer.add_scalar(
                            f"{self.tb_prefix}/{k}", float(v), step
                        )
            except Exception:
                pass

        update_timings["logging"] += time.time() - t_
        update_timings["total"] += time.time() - t__

        if self.delayed_target:
            self.update_target()
        return float(extrinsic_loss.item())

    def sample_action(
        self,
        obs: torch.Tensor,
        eps: float,
        step: int,
        n_steps: int = 100000,
        min_ent=0.01,
        verbose: bool = False,
    ):
        self.last_eps = eps
        is_batched = obs.ndim > 1
        obs_b = obs if is_batched else obs.unsqueeze(0)
        batch_size = obs_b.size(0)

        with torch.no_grad():
            q_ext = self.ext_online(obs_b, normalized=True)  # [B,n_actions]
            rand_vals = torch.rand(batch_size, device=obs_b.device)
            explore_mask = (rand_vals < min_ent) | (rand_vals < eps)
            
            if self.Beta > 0.0:
                int_q = self.int_online(obs_b, normalized=True)
                q_ext = (1.0 - self.Beta) * q_ext + self.Beta * int_q

            if self.soft or self.munchausen:
                q_ext = q_ext / self.tau
                actions = torch.distributions.Categorical(logits=q_ext).sample()
                if verbose:
                    print(f"Logits: {logits.cpu().numpy()}")
            else:
                actions = torch.argmax(q_ext, dim=-1)
                
            if explore_mask.any():
                explore_actions = torch.randint(
                    0,
                    self.n_action_bins,
                    (batch_size, self.n_action_dims),
                    device=obs_b.device,
                )
                actions = torch.where(
                    explore_mask.unsqueeze(1) if actions.ndim > 1 else explore_mask, explore_actions, actions
                )
            if verbose:
                print(f"Q-values ext: {q_ext.cpu().numpy()}")

            if is_batched:
                return actions.tolist()
            else:
                return actions.squeeze(0).tolist()


class IQNRainbowDQN(RainbowBase):
    """Maintains online (current) and target Q_Networks and training logic for IQN."""

    def __init__(
        self,
        input_dim,
        n_action_dims,
        n_action_bins,
        hidden_layer_sizes=[128, 128],
        lr: float = 1e-3,
        gamma: float = 0.99,
        alpha: float = 0.9,
        tau: float = 0.03,
        polyak_tau: float = 0.03,
        l_clip: float = -1.0,
        soft: bool = False,
        munchausen: bool = False,
        Thompson: bool = False,
        dueling: bool = False,

        Beta: float = 0.0,
        delayed: bool = True,
        ent_reg_coef: float = 0.0,
        rnd_output_dim: int = 128,
        rnd_lr: float = 1e-3,
        intrinsic_lr: float = 1e-3,
        int_r_clip=5.0,
        ext_r_clip=5.0,
        beta_half_life_steps: Optional[int] = None,
        norm_obs: bool = True,
        burn_in_updates: int = 0,
        encoder_factory: Optional[Callable[[], nn.Module]] = None,
    ):
        super().__init__(
            input_dim=input_dim, n_action_dims=n_action_dims, n_action_bins=n_action_bins,
            hidden_layer_sizes=hidden_layer_sizes, lr=lr, gamma=gamma, alpha=alpha, tau=tau,
            polyak_tau=polyak_tau, l_clip=l_clip, soft=soft, munchausen=munchausen, Thompson=Thompson,
            dueling=dueling, Beta=Beta, delayed=delayed, ent_reg_coef=ent_reg_coef,
            rnd_output_dim=rnd_output_dim, rnd_lr=rnd_lr, intrinsic_lr=intrinsic_lr,
            int_r_clip=int_r_clip, ext_r_clip=ext_r_clip, beta_half_life_steps=beta_half_life_steps,
            norm_obs=norm_obs, burn_in_updates=burn_in_updates, encoder_factory=encoder_factory
        )

        def _encoder_kwargs():
            if encoder_factory is None: return {}
            encoder = encoder_factory()
            return {"encoder": encoder, "encoder_out_dim": infer_encoder_out_dim(encoder, int(input_dim))}

        ext_online_kwargs = _encoder_kwargs()
        ext_target_kwargs = _encoder_kwargs()
        int_online_kwargs = _encoder_kwargs()
        int_target_kwargs = _encoder_kwargs()

        self.ext_online = IQN_Network(
            input_dim, n_action_dims, n_action_bins, hidden_layer_sizes=hidden_layer_sizes,
            dueling=dueling,  popart=True,**ext_online_kwargs,
        ).float()
        self.ext_target = IQN_Network(
            input_dim, n_action_dims, n_action_bins, hidden_layer_sizes=hidden_layer_sizes,
            dueling=dueling,  popart=True,**ext_target_kwargs,
        ).float()
        self.int_online = IQN_Network(
            input_dim, n_action_dims, n_action_bins, hidden_layer_sizes=hidden_layer_sizes,
            dueling=dueling, popart=True, **int_online_kwargs,
        ).float()
        self.int_target = IQN_Network(
            input_dim, n_action_dims, n_action_bins, hidden_layer_sizes=hidden_layer_sizes,
            dueling=dueling, popart=True, **int_target_kwargs,
        ).float()

        self.ext_target.requires_grad_(False)
        self.ext_target.load_state_dict(self.ext_online.state_dict())
        self.int_target.requires_grad_(False)
        self.int_target.load_state_dict(self.int_online.state_dict())

        self.optim = torch.optim.Adam(self.ext_online.parameters(), lr=lr)
        self.int_optim = torch.optim.Adam(self.int_online.parameters(), lr=intrinsic_lr)

        self.n_quantiles = 32
        self.n_target_quantiles = 32
        self.n_cosines = 64

    def _sample_taus(self, batch_size: int, n: int, device: torch.device) -> torch.Tensor:
        return torch.rand(batch_size, n, device=device)

    def _quantile_huber_loss(self, pred: torch.Tensor, target: torch.Tensor, taus: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
        B, N = pred.shape
        td = target.unsqueeze(1) - pred.unsqueeze(2)  # [B, N, N]
        abs_td = torch.abs(td)
        huber = torch.where(abs_td <= kappa, 0.5 * td.pow(2), kappa * (abs_td - 0.5 * kappa))
        I_ = (td < 0).float()
        taus_expanded = taus.unsqueeze(2)  # [B,N,1]
        loss = (torch.abs(taus_expanded - I_) * huber).mean()
        return loss

    def update(
        self, obs, a, r, next_obs, term, batch_size=None, step=0, extrinsic_only=False
    ):
        if self.update_timings is None:
            self.update_timings = {
                "update_rnd": 0.0, "extrinsic_loss": 0.0,
                "intrinsic_loss": 0.0, "logging": 0.0, "total": 0.0,
            }
        update_timings = self.update_timings
        t__ = time.time()

        if batch_size is None:
            idx = torch.arange(0, len(r))
            batch_size = len(r)
        else:
            idx = torch.randint(low=0, high=step, size=(batch_size,))
        b_next_obs = next_obs[idx]

        self.step += 1
        t_ = time.time()
        if self.step < self.burn_in_updates:
            if self.Beta > 0.0 or getattr(self, 'always_update_rnd', False):
                rnd_errors, rnd_loss = self._update_RND(b_next_obs, batch_norm=True)
            return 0.0
        update_timings["update_rnd"] += time.time() - t_
        
        if self.Beta > 0.0 or getattr(self, 'always_update_rnd', False):
            rnd_errors, rnd_loss = self._update_RND(b_next_obs, batch_norm=False)
            if self.beta_half_life_steps is not None and self.beta_half_life_steps > 0:
                self.Beta = self.start_Beta * (0.5 ** (self.step / self.beta_half_life_steps))
            
            with torch.no_grad():
                norm_int = self.int_rms.scale(rnd_errors.detach().to(dtype=torch.float64))
                if self.int_r_clip is not None and self.int_r_clip > 0.0:
                    norm_int = torch.clamp(norm_int, min=-self.int_r_clip, max=self.int_r_clip)
                b_r_int = norm_int.to(dtype=torch.float32)
        else:
            rnd_errors, rnd_loss, b_r_int = torch.zeros_like(r[idx]), 0, 0

        b_obs = obs[idx]
        b_r_ext = r[idx]
        b_term = term[idx]
        b_actions_idx = a[idx]

        t_ = time.time()
        loss_ext, m_r, e_loss = self._rl_update_extrinsic(b_obs, b_actions_idx, b_r_ext, b_next_obs, b_term, batch_size)
        update_timings["extrinsic_loss"] += time.time() - t_

        if extrinsic_only:
            update_timings["total"] += time.time() - t__
            return float(loss_ext.item())

        t_ = time.time()
        loss_int = torch.tensor(0.0)
        if self.Beta > 0.0:
            loss_int = self._rl_update_intrinsic(b_obs, b_actions_idx, b_r_int, b_next_obs, batch_size, b_term)
        update_timings["intrinsic_loss"] += time.time() - t_

        t_ = time.time()
        with torch.no_grad():
            taus = self._sample_taus(b_obs.shape[0], self.n_quantiles, b_obs.device)
            Q_ext = self.ext_online(b_obs, taus, normalized=True).mean(dim=1)
            Q_int = self.int_online(b_obs, taus, normalized=True).mean(dim=1)

        self.last_losses = {
            "extrinsic": float(loss_ext.item()),
            "intrinsic": float(loss_int.item()),
            "rnd": float(rnd_loss.item()) if isinstance(rnd_loss, torch.Tensor) else float(rnd_loss),
            "entropy_reg": float(e_loss) if not isinstance(e_loss, torch.Tensor) else float(e_loss.item()),
            "abs_r_ext": float(b_r_ext.abs().mean().item()),
            "avg_r_int": float(b_r_int.mean().item()) if isinstance(b_r_int, torch.Tensor) else 0.0,
            "munchausen_r": float(m_r),
            "Beta": float(self.Beta),
            "Q_ext_mean": float(Q_ext.mean().item()),
            "Q_int_mean": float(Q_int.mean().item()),
            "last_eps": float(self.last_eps),
        }
        
        if self.tb_writer is not None:
            try:
                for k, v in self.last_losses.items():
                    if isinstance(v, (int, float)):
                        self.tb_writer.add_scalar(f"{self.tb_prefix}/{k}", float(v), self.step)
            except Exception:
                pass
                
        update_timings["logging"] += time.time() - t_
        update_timings["total"] += time.time() - t__
        if self.delayed_target:
            self.update_target()
        return loss_ext.item()

    def _rl_update_extrinsic(self, b_obs, b_actions_idx, b_r_ext, b_next_obs, b_term, batch_size):
        device = b_obs.device
        current_sigma = self.ext_online.output_layer.sigma.detach()

        if b_actions_idx.ndim == 1:
            b_act_view = b_actions_idx.view(-1, 1)
        else:
            b_act_view = b_actions_idx

        with torch.no_grad():
            taus = self._sample_taus(batch_size, self.n_quantiles, device)
            target_taus = self._sample_taus(batch_size, self.n_target_quantiles, device)

            online_next_q_norm = self.ext_online.forward(b_next_obs, taus, normalized=True).mean(dim=1)

            t_net = self.ext_target if self.delayed_target else self.ext_online
            target_quantiles_all = t_net.forward(b_next_obs, target_taus, normalized=False)

            q_ext_norm_now = self.ext_online.forward(b_obs, taus, normalized=True).mean(dim=1)

            logpi_now = torch.clamp(torch.log_softmax(q_ext_norm_now / self.tau, dim=-1), min=-1e8)

            m_r = 0.0
            if self.munchausen or self.soft:
                logpi_next = torch.clamp(torch.log_softmax(online_next_q_norm / self.tau, dim=-1), min=-1e8)
                pi_next = torch.exp(logpi_next)
                ent_bonus_per_dim = -(pi_next * logpi_next).sum(dim=-1)
                ent_bonus = ent_bonus_per_dim.sum(dim=-1).unsqueeze(1) # [B, 1]
                
                mixed_target = (pi_next.unsqueeze(1) * target_quantiles_all).sum(dim=-1).sum(dim=-1) # [B, Nt]
            else:
                target_actions = online_next_q_norm.argmax(dim=-1)
                action_idx = target_actions.unsqueeze(1).unsqueeze(-1).expand(-1, self.n_target_quantiles, -1, 1)
                mixed_target = torch.gather(target_quantiles_all, 3, action_idx).squeeze(-1).sum(dim=-1)
                ent_bonus = 0.0
            
            if self.munchausen:
                selected_logpi = torch.gather(logpi_now, -1, b_act_view.unsqueeze(-1)).squeeze(-1)
                r_kl = torch.clamp(selected_logpi.mean(dim=-1), min=self.l_clip)
                m_r = current_sigma * self.alpha * self.tau * r_kl

            b_r_final = b_r_ext + m_r
            target_values = b_r_final.unsqueeze(1) + (1 - b_term).unsqueeze(1) * self.gamma * (mixed_target + current_sigma * ent_bonus)

        # Apply PopArt stats precisely
        self.ext_online.output_layer.update_stats(target_values.detach())
        if self.delayed_target:
            self.ext_target.output_layer.sigma.copy_(self.ext_online.output_layer.sigma)
            self.ext_target.output_layer.mu.copy_(self.ext_online.output_layer.mu)
        target_values = self.ext_online.output_layer.normalize(target_values)

        pred_normalized = True
        taus_pred = self._sample_taus(batch_size, self.n_quantiles, device)
        quantiles_pred = self.ext_online.forward(b_obs, taus_pred, normalized=pred_normalized) # [B, N, D, Bins]

        gather_index_pred = b_act_view.unsqueeze(1).unsqueeze(-1).expand(-1, self.n_quantiles, -1, 1)
        pred_chosen_per_dim = torch.gather(quantiles_pred, 3, gather_index_pred).squeeze(-1)
        pred_chosen = pred_chosen_per_dim.sum(dim=-1) # [B, N]

        loss = self._quantile_huber_loss(pred_chosen, target_values.detach(), taus_pred.detach())

        # Execute entropy calculation on newly fresh policy outputs post PopArt adjustment
        e_loss = 0.0
        if self.ent_reg_coef > 0.0:
            q_ext_now_fresh = quantiles_pred.mean(dim=1)
            if self.Beta > 0.0:
                with torch.no_grad():
                    int_taus_fresh = self._sample_taus(batch_size, self.n_quantiles, device)
                    q_int_now_fresh = self.int_online.forward(b_obs, int_taus_fresh, normalized=pred_normalized).mean(dim=1)
                q_mixed_fresh = (1.0 - self.Beta) * q_ext_now_fresh + self.Beta * q_int_now_fresh.detach()
            else:
                q_mixed_fresh = q_ext_now_fresh

            logpi_fresh = torch.clamp(torch.log_softmax(q_mixed_fresh / self.tau, dim=-1), min=-1e8)
            pi_fresh = torch.exp(logpi_fresh)
            entropy_per_dim = (pi_fresh * logpi_fresh).sum(dim=-1)
            entropy = entropy_per_dim.mean()
            e_loss = self.ent_reg_coef * entropy
        
        loss = loss + e_loss

        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ext_online.parameters(), max_norm=10.0)
        self.optim.step()

        if isinstance(m_r, torch.Tensor): m_r = m_r.mean()
        return loss, m_r, e_loss

    def _rl_update_intrinsic(self, b_obs, b_actions_idx, b_r_int, b_next_obs, batch_size, b_term):
        device = b_obs.device

        with torch.no_grad():
            int_taus = self._sample_taus(batch_size, self.n_quantiles, device)
            int_target_taus = self._sample_taus(batch_size, self.n_target_quantiles, device)

            online_next_q_int_norm = self.int_online.forward(b_next_obs, int_taus, normalized=True).mean(dim=1)
            target_actions = online_next_q_int_norm.argmax(dim=-1)

            t_net_int = self.int_target if self.delayed_target else self.int_online
            int_target_all = t_net_int.forward(b_next_obs, int_target_taus, normalized=False) # [B, Nt, D, Bins]

            action_idx = target_actions.unsqueeze(1).unsqueeze(-1).expand(-1, self.n_target_quantiles, -1, 1)
            mixed_target = torch.gather(int_target_all, -1, action_idx).squeeze(-1).sum(dim=-1) # [B, Nt]
            
            int_target_values = b_r_int.unsqueeze(1) + self.gamma * mixed_target

        self.int_online.output_layer.update_stats(int_target_values.detach())
        if self.delayed_target:
            self.int_target.output_layer.sigma.copy_(self.int_online.output_layer.sigma)
            self.int_target.output_layer.mu.copy_(self.int_online.output_layer.mu)
        int_target_values = self.int_online.output_layer.normalize(int_target_values)

        pred_normalized = True
        int_taus_pred = self._sample_taus(batch_size, self.n_quantiles, device)
        int_quantiles = self.int_online.forward(b_obs, int_taus_pred, normalized=pred_normalized)

        if b_actions_idx.ndim == 1:
            b_act_view = b_actions_idx.view(-1, 1)
        else:
            b_act_view = b_actions_idx
        
        gather_index_int_pred = b_act_view.unsqueeze(1).unsqueeze(-1).expand(-1, self.n_quantiles, -1, 1)
        int_pred_per_dim = torch.gather(int_quantiles, 3, gather_index_int_pred).squeeze(-1)
        int_pred_chosen = int_pred_per_dim.sum(dim=-1)

        int_loss = self._quantile_huber_loss(int_pred_chosen, int_target_values.detach(), int_taus_pred.detach())

        self.int_optim.zero_grad()
        int_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.int_online.parameters(), max_norm=10.0)
        self.int_optim.step()

        return int_loss

    def sample_action(
        self,
        obs: torch.Tensor,
        eps: float,
        step: int,
        n_steps: int = 100000,
        min_eps=0.01,
        verbose: bool = False,
    ):
        self.last_eps = eps
        is_batched = obs.ndim > 1
        obs_b = obs if is_batched else obs.unsqueeze(0)
        batch_size = obs_b.size(0)

        with torch.no_grad():
            taus = self._sample_taus(batch_size, self.n_quantiles, obs_b.device)
            ext_q = self.ext_online(obs_b, taus, normalized=True).mean(dim=1)  # [B,D,Bins]

            rand_vals = torch.rand(batch_size, device=obs_b.device)
            explore_mask = (rand_vals < min_eps) | (rand_vals < eps)

            if self.Beta > 0.0:
                int_taus = self._sample_taus(batch_size, self.n_quantiles, obs_b.device)
                int_q = self.int_online(obs_b, int_taus, normalized=True).mean(dim=1)
                ext_q = (1.0 - self.Beta) * ext_q + self.Beta * int_q

            if self.soft or self.munchausen:
                logits = ext_q / self.tau
                actions = torch.distributions.Categorical(logits=logits).sample()  # [B,D]
                if verbose:
                    print(f"Logits: {logits.cpu().numpy()}")
            else:
                actions = torch.argmax(ext_q, dim=-1)  # [B,D]
                
            if explore_mask.any():
                explore_actions = torch.randint(
                    0,
                    self.n_action_bins,
                    (batch_size, self.n_action_dims),
                    device=obs_b.device,
                )
                actions = torch.where(
                    explore_mask.unsqueeze(1) if actions.ndim > 1 else explore_mask, explore_actions, actions
                )

            if verbose:
                print(f"Q-values ext: {ext_q.cpu().numpy()}")

            if is_batched:
                return actions.tolist()
            else:
                return actions.squeeze(0).tolist()