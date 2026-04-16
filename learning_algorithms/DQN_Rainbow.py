import numpy as np
from learning_algorithms.cleanrl_buffers import ReplayBuffer
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
        n_envs=1,
        buffer_size: int = int(1e5),
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
        device = 'cpu',
        buffer_device = 'cpu',
    ):
        super().__init__()
        self.device = device
        self.buffer_device = buffer_device
        self.buffer_size = buffer_size
        if not torch.cuda.is_available():
            self.device = "cpu"
            self.buffer_device = "cpu"
        self.input_dim = input_dim
        if isinstance(self.input_dim, (int, np.integer)):
            self.obs_ndim = 1
        elif hasattr(self.input_dim, '__len__'):
            self.obs_ndim = len(self.input_dim)
        else:
            raise TypeError(f"Unsupported input_dim type: {type(self.input_dim)}. Expected int or array-like.")
        self.timing = {}
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
        # Determine number of environments
        self.n_envs = n_envs# = len(self.envs.remotes) if self.envs is not None and hasattr(self.envs, 'remotes') else getattr(self.envs, 'num_envs', 1)

        
        self.step = 0
        self.last_eps = 1.0
        self.update_timings = None
        self.buffer = self._init_buffer()

        # RND and running stats setup
        rnd_target_encoder = encoder_factory() if encoder_factory is not None else None
        rnd_predictor_encoder = encoder_factory() if encoder_factory is not None else None
        
        self.rnd = RNDModel(
            input_dim,
            rnd_output_dim,
            encoder_target=rnd_target_encoder,
            encoder_predictor=rnd_predictor_encoder,
        ).float()
        self.rnd.to(self.device)
        self.rnd_optim = torch.optim.Adam(self.rnd.predictor.parameters(), lr=rnd_lr)
        self.obs_rms = RunningMeanStd(shape=(input_dim,))
        
    def _init_buffer(self) -> ReplayBuffer:
        """
        Initializes the pinned memory ReplayBuffer based on RainbowBase parameters.
        """
        # Determine observation shape (handling both tuple and int inputs)
        obs_shape = self.input_dim if isinstance(self.input_dim, tuple) else (self.input_dim,)
        
        # In DQN/Rainbow, we usually store the action index.
        # If n_action_dims > 1, it's a MultiDiscrete setup.
        # We store them as a vector of length n_action_dims.
        action_dim = self.n_action_dims
        return ReplayBuffer(
            buffer_size=self.buffer_size,
            n_envs=self.n_envs,
            obs_shape=obs_shape,
            action_dim=action_dim,
            device=self.buffer_device,
            optimize_memory_usage=False,
            handle_timeout_termination=True,
            action_dtype=torch.int32,
        )

    def observe(self, obs, action, reward, next_obs, terminated, truncated, info=None):
        # Update random network distillation running mean and std norm
        # if we are using intrinsic rewards. 
        if self.Beta > 0.0:
            if isinstance(next_obs, np.ndarray):
                b_next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=self.buffer_device)
            else:
                b_next_obs = next_obs.clone().detach().to(device=self.buffer_device, dtype=torch.float32)
            
            # Add batch dimension if it is missing (e.g., n_envs=1 returning unbatched states)
            if b_next_obs.ndim == self.obs_ndim:
                b_next_obs = b_next_obs.unsqueeze(0)
                
            self.obs_rms.update(b_next_obs)
        # The buffer natively handles the shapes and tensor casting for memory efficiency
        self.buffer.add(
            obs=obs,
            next_obs=next_obs,
            action=action,
            reward=reward,
            term=terminated, # Fixed to match function signature
            trunc=truncated, # Fixed to match function signature
        )

    # def old_popart()
    #     # ONLINE TD TARGET AND UPDATE_STATS:
    #     if hasattr(self, 'ext_online') and getattr(self, "popart", True):
    #         with torch.no_grad():
    #             # Extrinsic Target
    #             if hasattr(self.ext_online, "normalize"): 
    #                 pass # We will do a forward pass

    #             # Intrinsic Reward Generation
    #             rnd_errors, _ = self._update_RND(no, batch_norm=False)
    #             norm_int = self.int_rms.scale(rnd_errors.detach().to(dtype=torch.float64))
    #             if getattr(self, "int_r_clip", None) is not None and self.int_r_clip > 0.0:
    #                 norm_int = torch.clamp(norm_int, min=-self.int_r_clip, max=self.int_r_clip)
    #             int_r = norm_int.to(dtype=torch.float32)

    #             # Q-values depends on distributional or EV
    #             distributional = getattr(self, "n_quantiles", None) is not None
                
    #             if distributional:
    #                 # IQN
    #                 taus = self._sample_taus(no.shape[0], self.n_target_quantiles, device)
    #                 next_q_norm = self.ext_target.forward(no, taus, normalized=True).mean(dim=1)
    #                 next_q_int_norm = self.int_target.forward(no, taus, normalized=True).mean(dim=1)
    #             else:
    #                 # EV
    #                 next_q_norm = self.ext_target(no, normalized=True)
    #                 next_q_int_norm = self.int_target(no, normalized=True)

    #             if self.dueling:
    #                 if distributional:
    #                     online_next_q_norm = self.ext_online.forward(no, taus, normalized=True).mean(dim=1)
    #                 else:
    #                     online_next_q_norm = self.ext_online(no, normalized=True)
    #                 next_actions = online_next_q_norm.argmax(dim=-1, keepdim=True)
    #             else:
    #                 next_actions = next_q_norm.argmax(dim=-1, keepdim=True)

    #             if distributional:
    #                 next_q_unnorm = self.ext_target(no, taus, normalized=False).mean(dim=1)
    #                 target_q = next_q_unnorm.gather(-1, next_actions).squeeze(-1)
                    
    #                 next_q_int_unnorm = self.int_target(no, taus, normalized=False).mean(dim=1)
    #                 target_q_int = next_q_int_unnorm.gather(-1, next_actions).squeeze(-1)
    #             else:
    #                 target_q = next_q_norm.gather(-1, next_actions).squeeze(-1)
    #                 if hasattr(self.ext_target.output_layer, "unnormalize"):
    #                     target_q = self.ext_target.output_layer.unnormalize(target_q)
    #                 target_q_int = next_q_int_norm.gather(-1, next_actions).squeeze(-1)
    #                 if hasattr(self.int_target.output_layer, "unnormalize"):
    #                     target_q_int = self.int_target.output_layer.unnormalize(target_q_int)

    #             r_ext_view = r.view(-1, 1) if r.ndim == 1 else r
    #             d_view = d.view(-1, 1) if d.ndim == 1 else d
    #             int_r_view = int_r.view(-1, 1) if int_r.ndim == 1 else int_r

    #             ext_target_val = r_ext_view + (1 - d_view) * self.gamma * target_q
    #             int_target_val = int_r_view + (1 - d_view) * self.gamma * target_q_int

    #             if hasattr(self.ext_online, "output_layer") and hasattr(self.ext_online.output_layer, "update_stats"):
    #                 if not distributional: ext_target_val = ext_target_val.view(-1, 1)
    #                 self.ext_online.output_layer.normalize(ext_target_val)
    #                 if not distributional: int_target_val = int_target_val.view(-1, 1)
    #                 self.int_online.output_layer.normalize(int_target_val)

    #     # Format tensors for buffer
    #     if isinstance(obs, np.ndarray):
    #         o = torch.as_tensor(obs, dtype=torch.float32, device=device)
    #         a = torch.as_tensor(action, dtype=torch.float32, device=device)
    #         if a.ndim == 1:
    #             a = a.view(-1, 1)
    #         r = torch.as_tensor(reward, dtype=torch.float32, device=device)
    #         no = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    #         d = torch.as_tensor(done, dtype=torch.float32, device=device)
    #     else:
    #         o = obs.to(device=device, dtype=torch.float32)
    #         a = action.to(device=device, dtype=torch.float32)
    #         if a.ndim == 1:
    #             a = a.view(-1, 1)
    #         r = reward.to(device=device, dtype=torch.float32)
    #         no = next_obs.to(device=device, dtype=torch.float32)
    #         d = done.to(device=device, dtype=torch.float32)
            
    #     if self.buffer is not None:
    #         self.buffer.add(o.cpu().numpy(), no.cpu().numpy(), a.cpu().numpy(), r.cpu().numpy(), d.cpu().numpy(), info)

    def to(self, device):
        """Move the agent and all its subcomponents to a specific device."""
        device = torch.device(device)
        self.device = device
        main_lr = self.optim.param_groups[0]["lr"] if hasattr(self, "optim") else self.lr
        int_lr = self.int_optim.param_groups[0]["lr"] if hasattr(self, "int_optim") else self.intrinsic_lr
        rnd_lr = self.rnd_optim.param_groups[0]["lr"] if hasattr(self, "rnd_optim") else self.rnd_lr

        if hasattr(self, "ext_online"): self.ext_online.to(device)
        if hasattr(self, "ext_target"): self.ext_target.to(device)
        if hasattr(self, "int_online"): self.int_online.to(device)
        if hasattr(self, "int_target"): self.int_target.to(device)
        if hasattr(self, "rnd"): self.rnd.to(device)

        if hasattr(self, "ext_online"):
            self.optim = torch.optim.Adam(self.ext_online.parameters(), lr=main_lr)
        if hasattr(self, "int_online"):
            self.int_optim = torch.optim.Adam(self.int_online.parameters(), lr=int_lr)
        if hasattr(self, "rnd"):
            self.rnd_optim = torch.optim.Adam(self.rnd.predictor.parameters(), lr=rnd_lr)
        return self

    def buffer_to(self,device):
        self.buffer_device = device
        if hasattr(self, 'buffer') and self.buffer is not None:
            self.buffer.device = device
        if hasattr(self, "obs_rms") and hasattr(self.obs_rms, "to"): 
            self.obs_rms.to(device)

    def update_target(self):
        """Polyak averaging: target = (1 - tau) * target + tau * online."""
        if not self.delayed_target:
            return
        with torch.no_grad():
            for tp, op in zip(self.ext_target.parameters(), self.ext_online.parameters()):
                tp.data.mul_(1.0 - self.polyak_tau).add_(op.data, alpha=self.polyak_tau)
            for tp, op in zip(self.int_target.parameters(), self.int_online.parameters()):
                tp.data.mul_(1.0 - self.polyak_tau).add_(op.data, alpha=self.polyak_tau)

    def _update_RND(self, next_obs: torch.Tensor):
        with torch.no_grad():        
            norm_next_obs = self.obs_rms.normalize(next_obs).float()

        with torch.enable_grad():
            rnd_errors = self.rnd(norm_next_obs)
            rnd_loss = rnd_errors.mean()
            self.rnd_optim.zero_grad()
            rnd_loss.backward()
            self.rnd_optim.step()
        return rnd_errors.detach(), rnd_loss.detach()


class EVRainbowDQN(RainbowBase):
    """Non-distributional counterpart to RainbowDQN with optional dueling and all five pillars."""

    def __init__(
        self,
        input_dim,
        n_action_dims,
        n_action_bins,
        n_envs:int=1,
        buffer_size: int = int(1e5),
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
            input_dim=input_dim, n_action_dims=n_action_dims, n_action_bins=n_action_bins, n_envs=n_envs, buffer_size=buffer_size,
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
            dueling=dueling, popart=True, min_std=0.01, **int_online_kwargs,
        ).float()
        self.int_target = EV_Q_Network(
            input_dim, n_action_dims, n_action_bins, hidden_layer_sizes=hidden_layer_sizes,
            dueling=dueling, popart=True, min_std=0.01, **int_target_kwargs,
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

    def update(self, batch_size=None, step=None):
        self.step += 1

        # Get batch data from buffer
        if batch_size is None:
            batch_size = 128
        (
            b_obs, b_a, b_next_obs,b_term, b_trunc, b_r_ext
        ) = self.buffer.sample(batch_size)
        # Get the batch to the gpu
        b_next_obs = b_next_obs.to(self.device, non_blocking=True)
        
        if self.step < self.burn_in_updates:
            if self.Beta > 0.0 or getattr(self, 'always_update_rnd', False):
                rnd_errors, rnd_loss = self._update_RND(b_next_obs)
                return 0.0
            return 0.0

        b_obs = b_obs.to(self.device, non_blocking=True)
        b_a = b_a.to(self.device, non_blocking=True)
        b_term = b_term.to(self.device, non_blocking=True).view(-1)
        b_trunc = b_trunc.to(self.device, non_blocking=True).view(-1)
        b_r_ext = b_r_ext.to(self.device, non_blocking=True).view(-1)
        # Need the extra trailing dim for torch.gather later
        b_actions_idx = b_a.view(batch_size, self.n_action_dims, 1)
        #Get intrinsic errors if we are going to use them
        if self.Beta > 0.0:
            rnd_errors, rnd_loss = self._update_RND(b_next_obs)
            if self.beta_half_life_steps is not None and self.beta_half_life_steps > 0:
                self.Beta = self.start_Beta * (
                    0.5 ** (self.step / self.beta_half_life_steps)
                )
            b_r_int = rnd_errors.detach()
        else:
            rnd_errors, rnd_loss, b_r_int = torch.zeros_like(b_r_ext), 0, torch.zeros_like(b_r_ext)


        logpi_now = None
        pi_now = None
        entropy_loss = 0
        current_sigma = self.ext_online.output_layer.sigma.detach()

        # Get target
        with torch.no_grad():
            q_ext_norm = self.ext_online(b_obs, normalized=True)  # [B,D,Bins]
            q_next_online_norm = self.ext_online(b_next_obs, normalized=True)
            q_next_target_raw = self.ext_target(b_next_obs, normalized=False) if self.delayed_target else self.ext_online(b_next_obs, normalized=False)   
            
            # Munchausen loss only for the exploiter 
            if self.munchausen:
                if logpi_now is None:
                    logpi_now = torch.clamp(
                        torch.log_softmax(q_ext_norm / self.tau, dim=-1), min=-1e8
                    )
                selected_logpi = torch.gather(logpi_now, -1, b_actions_idx).squeeze(-1)  # [B,D]
                r_kl = torch.clamp(selected_logpi, min=self.l_clip)
                if r_kl.ndim > 1:
                    r_kl = r_kl.sum(-1)
                assert b_r_ext.ndim == r_kl.ndim
                b_r_ext += current_sigma * self.alpha * self.tau * r_kl 

            # Next value with entropy and weighted sum over q values
            if self.munchausen or self.soft:
                logpi_next = torch.clamp(
                    torch.log_softmax(q_next_online_norm / self.tau, dim=-1), min=-1e8
                )
                pi_next = torch.exp(logpi_next)
                next_head_vals = (pi_next * (current_sigma * self.alpha * logpi_next + q_next_target_raw)).sum(-1)
            # Next value with no entropy or weighted sum, using argmax policy
            else:
                target_actions_next = q_next_online_norm.argmax(dim=-1, keepdim=True).detach()
                next_head_vals = torch.gather(q_next_target_raw, -1, target_actions_next).squeeze(-1)
            # vdn sum the vals 
            if next_head_vals.ndim>1:
                next_head_vals=next_head_vals.sum(-1)
            assert b_r_ext.shape == b_term.shape == next_head_vals.shape, \
                f"Shape mismatch: {b_r_ext.shape}, {b_term.shape}, {next_head_vals.shape}"
            online_ext_target = b_r_ext.view(-1) + self.gamma * (1 - b_term).view(-1) * next_head_vals
        
        target_for_stats_ext = online_ext_target.detach() # maintain [B, D]
        self.ext_online.output_layer.update_stats(target_for_stats_ext)
        if self.delayed_target:
            self.ext_target.output_layer.sigma.copy_(self.ext_online.output_layer.sigma)
            self.ext_target.output_layer.mu.copy_(self.ext_online.output_layer.mu)
        td_target_norm = self.ext_online.output_layer.normalize(target_for_stats_ext)
        q_ext_now_norm = self.ext_online(b_obs, normalized=True)
        q_selected_norm = torch.gather(q_ext_now_norm, -1, b_actions_idx).squeeze(-1) # [B, D]
        if q_selected_norm.ndim>1:
            q_selected_norm=q_selected_norm.sum(-1)
        assert q_selected_norm.shape == td_target_norm.shape, \
            f"Shape mismatch: q_selected_norm {q_selected_norm.shape}, td_target_norm {td_target_norm.shape}"
        extrinsic_loss = torch.nn.functional.mse_loss(q_selected_norm, td_target_norm)
        
        if self.ent_reg_coef > 0.0:
            logpi_now = torch.clamp(
                torch.log_softmax(q_ext_now_norm / self.tau, dim=-1), min=-1e8
            )
            if torch.isnan(logpi_now).any():
                print("NaN detected in logpi_now!")
                raise("fuck")
            pi_now = torch.exp(logpi_now)
            entropy_loss = (pi_now * logpi_now).sum(dim=-1).mean()
            extrinsic_loss = extrinsic_loss + self.ent_reg_coef * entropy_loss
        
        self.optim.zero_grad()
        extrinsic_loss.backward()
        if hasattr(self, "ext_online"):
            torch.nn.utils.clip_grad_norm_(self.ext_online.parameters(), max_norm=10.0)
        self.optim.step()



        # Intrinsic Q update
        with torch.no_grad():
            int_q_next = (self.int_target if self.delayed_target else self.int_online)(
                b_next_obs, normalized=False
            )
            # Double q actions if delayed. We grab actions from online and vals from target
            if self.delayed_target:
                next_int_actions = self.int_online(
                    b_next_obs, normalized=True
                ).argmax(-1,keepdim=True).detach()
            else:
                next_int_actions = int_q_next.argmax(-1,keepdim=True).detach()
            int_q_next_target = torch.gather(int_q_next, -1, next_int_actions).squeeze(-1)
            if int_q_next_target.ndim>1:
                int_q_next_target=int_q_next_target.sum(-1)
            assert b_r_int.view(-1).shape == int_q_next_target.shape, \
                f"Shape mismatch: b_r_int {b_r_int.view(-1).shape}, int_q_next_target {int_q_next_target.shape}"
            int_td_target = b_r_int.view(-1) + self.gamma * int_q_next_target

        target_for_stats_int = int_td_target.detach()
        self.int_online.output_layer.update_stats(target_for_stats_int)
        if self.delayed_target:
            self.int_target.output_layer.sigma.copy_(self.int_online.output_layer.sigma)
            self.int_target.output_layer.mu.copy_(self.int_online.output_layer.mu)
        
        int_td_target_norm = self.int_online.output_layer.normalize(target_for_stats_int)
        int_q_now_norm = self.int_online(b_obs, normalized=True)
        int_q_selected_norm = torch.gather(int_q_now_norm, -1, b_actions_idx).squeeze(-1)
        if int_q_selected_norm.ndim>1:
            int_q_selected_norm=int_q_selected_norm.sum(-1)
        assert int_q_selected_norm.shape == int_td_target_norm.shape, \
            f"Shape mismatch: int_q_selected_norm {int_q_selected_norm.shape}, int_td_target_norm {int_td_target_norm.shape}"
        intrinsic_loss = torch.nn.functional.mse_loss(
            int_q_selected_norm, int_td_target_norm
        )

        self.int_optim.zero_grad()
        intrinsic_loss.backward()
        
        # Update target network
        if self.delayed_target:
            self.update_target()
        
        # tracking
        if hasattr(self, "int_online"):
            torch.nn.utils.clip_grad_norm_(self.int_online.parameters(), max_norm=10.0)
        self.int_optim.step()

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
        is_batched = obs.ndim > self.obs_ndim
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
        n_envs=1,
        buffer_size: int = int(1e5),
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
            input_dim=input_dim, n_action_dims=n_action_dims, n_action_bins=n_action_bins, n_envs=n_envs, buffer_size=buffer_size,
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
            dueling=dueling, popart=True, min_std=0.01, **int_online_kwargs,
        ).float()
        self.int_target = IQN_Network(
            input_dim, n_action_dims, n_action_bins, hidden_layer_sizes=hidden_layer_sizes,
            dueling=dueling, popart=True, min_std=0.01, **int_target_kwargs,
        ).float()

        self.ext_target.requires_grad_(False)
        self.ext_target.load_state_dict(self.ext_online.state_dict())
        self.int_target.requires_grad_(False)
        self.int_target.load_state_dict(self.int_online.state_dict())

        self.optim = torch.optim.Adam(self.ext_online.parameters(), lr=lr)
        self.int_optim = torch.optim.Adam(self.int_online.parameters(), lr=intrinsic_lr)

        self.n_quantiles = 32
        self.n_target_quantiles = 32

    def _sample_taus(self, batch_size: int, n: int, device: torch.device) -> torch.Tensor:
        return torch.rand(batch_size, n, device=device)

    def _quantile_huber_loss(self, pred: torch.Tensor, target: torch.Tensor, taus: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
        B, N = pred.shape
        td = target.unsqueeze(1) - pred.unsqueeze(2)  # [B, N, Nt]
        abs_td = torch.abs(td)
        huber = torch.where(abs_td <= kappa, 0.5 * td.pow(2), kappa * (abs_td - 0.5 * kappa))
        I_ = (td < 0).float()
        taus_expanded = taus.unsqueeze(2)  # [B,N,1]
        loss = (torch.abs(taus_expanded - I_) * huber).mean()
        return loss

    def update(self, batch_size=None, step=None):
        self.step += 1

        # Get batch data from buffer
        if batch_size is None:
            batch_size = 128
        (
            b_obs, b_a, b_next_obs, b_term, b_trunc, b_r_ext
        ) = self.buffer.sample(batch_size)
        
        # Get the batch to the gpu
        b_next_obs = b_next_obs.to(self.device, non_blocking=True)
        
        # Burn-in logic identical to EVRainbowDQN
        if self.step < self.burn_in_updates:
            if self.Beta > 0.0 or getattr(self, 'always_update_rnd', False):
                rnd_errors, rnd_loss = self._update_RND(b_next_obs)
            return 0.0

        b_obs = b_obs.to(self.device, non_blocking=True)
        b_a = b_a.to(self.device, non_blocking=True)
        b_term = b_term.to(self.device, non_blocking=True).view(-1)
        b_trunc = b_trunc.to(self.device, non_blocking=True).view(-1)
        b_r_ext = b_r_ext.to(self.device, non_blocking=True).view(-1)
        b_actions_idx = b_a.view(batch_size, self.n_action_dims, 1)

        # Get intrinsic errors if we are going to use them
        if self.Beta > 0.0:
            rnd_errors, rnd_loss = self._update_RND(b_next_obs)
            if self.beta_half_life_steps is not None and self.beta_half_life_steps > 0:
                self.Beta = self.start_Beta * (
                    0.5 ** (self.step / self.beta_half_life_steps)
                )
            b_r_int = rnd_errors.detach()
        else:
            rnd_errors, rnd_loss, b_r_int = torch.zeros_like(b_r_ext), 0, torch.zeros_like(b_r_ext)

        entropy_loss = 0.0
        current_sigma = self.ext_online.output_layer.sigma.detach()

        # ========================================================
        # Extrinsic Q update
        # ========================================================
        with torch.no_grad():
            taus = self._sample_taus(batch_size, self.n_quantiles, self.device)
            target_taus = self._sample_taus(batch_size, self.n_target_quantiles, self.device)

            # Online Next Q -> For action selection
            online_next_q_norm = self.ext_online(b_next_obs, taus, normalized=True).mean(dim=1)

            # Target Net Quantiles -> For target values
            t_net = self.ext_target if self.delayed_target else self.ext_online
            target_quantiles_all = t_net(b_next_obs, target_taus, normalized=False) # [B, Nt, D, Bins]

            m_r = 0.0
            ent_bonus = 0.0

            if self.munchausen or self.soft:
                logpi_next = torch.clamp(torch.log_softmax(online_next_q_norm / self.tau, dim=-1), min=-1e8)
                pi_next = torch.exp(logpi_next)
                # Entropy bonus: sum over D
                ent_bonus = -(pi_next * logpi_next).sum(dim=-1) # sum over bins
                if ent_bonus.ndim > 1:
                    ent_bonus = ent_bonus.sum(dim=-1).unsqueeze(1) # sum over D, then [B, 1]
                else:
                    ent_bonus = ent_bonus.unsqueeze(1)
                # Mixed target values
                mixed_target = (pi_next.unsqueeze(1) * target_quantiles_all).sum(dim=-1) # sum over bins -> [B, Nt, D]
                if mixed_target.ndim > 2:
                    mixed_target = mixed_target.sum(dim=-1) # sum over D -> [B, Nt]
            else:
                target_actions = online_next_q_norm.argmax(dim=-1)
                action_idx = target_actions.unsqueeze(1).unsqueeze(-1).expand(-1, self.n_target_quantiles, -1, 1)
                mixed_target = torch.gather(target_quantiles_all, -1, action_idx).squeeze(-1) # [B, Nt, D]
                if mixed_target.ndim > 2:
                    mixed_target = mixed_target.sum(dim=-1) # sum over D -> [B, Nt]

            if self.munchausen:
                q_ext_norm_now = self.ext_online(b_obs, taus, normalized=True).mean(dim=1)
                logpi_now = torch.clamp(torch.log_softmax(q_ext_norm_now / self.tau, dim=-1), min=-1e8)
                selected_logpi = torch.gather(logpi_now, -1, b_actions_idx).squeeze(-1)
                
                # sum r_kl over D if needed
                r_kl = torch.clamp(selected_logpi, min=self.l_clip)
                if r_kl.ndim > 1:
                    r_kl = r_kl.sum(dim=-1)
                m_r = current_sigma * self.alpha * self.tau * r_kl

            b_r_final = b_r_ext + m_r
            # Target Q-distribution
            assert b_r_final.shape == b_term.shape, \
                f"Shape mismatch: b_r_final {b_r_final.shape}, b_term {b_term.shape}"
            target_values = b_r_final.unsqueeze(1) + (1 - b_term).unsqueeze(1) * self.gamma * (mixed_target + current_sigma * ent_bonus)

        # Apply PopArt stats tracking over target distributions
        self.ext_online.output_layer.update_stats(target_values.detach())
        if self.delayed_target:
            self.ext_target.output_layer.sigma.copy_(self.ext_online.output_layer.sigma)
            self.ext_target.output_layer.mu.copy_(self.ext_online.output_layer.mu)
        target_values_norm = self.ext_online.output_layer.normalize(target_values.detach())

        # Current normalized quantile predictions
        taus_pred = self._sample_taus(batch_size, self.n_quantiles, self.device)
        quantiles_pred = self.ext_online(b_obs, taus_pred, normalized=True) # [B, N, D, Bins]

        gather_index_pred = b_actions_idx.unsqueeze(1).expand(-1, self.n_quantiles, -1, 1)
        pred_chosen = torch.gather(quantiles_pred, -1, gather_index_pred).squeeze(-1) # [B, N, D]
        if pred_chosen.ndim > 2:
            pred_chosen = pred_chosen.sum(dim=-1) # [B, N]
        assert pred_chosen.shape[0] == target_values_norm.shape[0] and pred_chosen.ndim == target_values_norm.ndim == 2, \
            f"Shape mismatch: pred_chosen {pred_chosen.shape}, target_values_norm {target_values_norm.shape}"

        extrinsic_loss = self._quantile_huber_loss(pred_chosen, target_values_norm, taus_pred)

        # Add pure entropy reg if needed
        if self.ent_reg_coef > 0.0:
            q_ext_now_fresh = quantiles_pred.mean(dim=1)
            logpi_fresh = torch.clamp(torch.log_softmax(q_ext_now_fresh / self.tau, dim=-1), min=-1e8)
            pi_fresh = torch.exp(logpi_fresh)
            entropy = (pi_fresh * logpi_fresh).sum(dim=-1).mean()
            entropy_loss = self.ent_reg_coef * entropy
            extrinsic_loss = extrinsic_loss + entropy_loss

        self.optim.zero_grad()
        extrinsic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ext_online.parameters(), max_norm=10.0)
        self.optim.step()

        # ========================================================
        # Intrinsic Q update
        # ========================================================
        if self.Beta > 0.0:
            with torch.no_grad():
                int_taus = self._sample_taus(batch_size, self.n_quantiles, self.device)
                int_target_taus = self._sample_taus(batch_size, self.n_target_quantiles, self.device)

                online_next_q_int_norm = self.int_online(b_next_obs, int_taus, normalized=True).mean(dim=1)
                target_actions_int = online_next_q_int_norm.argmax(dim=-1)

                t_net_int = self.int_target if self.delayed_target else self.int_online
                int_target_all = t_net_int(b_next_obs, int_target_taus, normalized=False)

                action_idx_int = target_actions_int.unsqueeze(1).unsqueeze(-1).expand(-1, self.n_target_quantiles, -1, 1)
                mixed_target_int = torch.gather(int_target_all, -1, action_idx_int).squeeze(-1) # [B, Nt, D]
                if mixed_target_int.ndim > 2:
                    mixed_target_int = mixed_target_int.sum(dim=-1) # [B, Nt]
                
                # No terminal mask for intrinsic reward
                assert b_r_int.unsqueeze(1).shape[0] == mixed_target_int.shape[0], \
                    f"Shape mismatch: b_r_int {b_r_int.unsqueeze(1).shape}, mixed_target_int {mixed_target_int.shape}"
                int_target_values = b_r_int.unsqueeze(1) + self.gamma * mixed_target_int

            self.int_online.output_layer.update_stats(int_target_values.detach())
            if self.delayed_target:
                self.int_target.output_layer.sigma.copy_(self.int_online.output_layer.sigma)
                self.int_target.output_layer.mu.copy_(self.int_online.output_layer.mu)
            int_target_values_norm = self.int_online.output_layer.normalize(int_target_values.detach())

            int_taus_pred = self._sample_taus(batch_size, self.n_quantiles, self.device)
            int_quantiles = self.int_online(b_obs, int_taus_pred, normalized=True)
            
            gather_index_int_pred = b_actions_idx.unsqueeze(1).expand(-1, self.n_quantiles, -1, 1)
            int_pred_chosen = torch.gather(int_quantiles, -1, gather_index_int_pred).squeeze(-1) # [B, N, D]
            if int_pred_chosen.ndim > 2:
                int_pred_chosen = int_pred_chosen.sum(dim=-1) # [B, N]
                
            assert int_pred_chosen.shape[0] == int_target_values_norm.shape[0] and int_pred_chosen.ndim == int_target_values_norm.ndim == 2, \
                f"Shape mismatch: int_pred_chosen {int_pred_chosen.shape}, int_target_values_norm {int_target_values_norm.shape}"

            intrinsic_loss = self._quantile_huber_loss(int_pred_chosen, int_target_values_norm, int_taus_pred)

            self.int_optim.zero_grad()
            intrinsic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.int_online.parameters(), max_norm=10.0)
            self.int_optim.step()
        else:
            intrinsic_loss = torch.tensor(0.0)

        # Update target network
        if self.delayed_target:
            self.update_target()

        # ========================================================
        # Tracking identical to EV
        # ========================================================
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

        q_ext_now_norm = quantiles_pred.mean(dim=1) if 'quantiles_pred' in locals() else torch.tensor(0.0)

        self.last_losses = {
            "extrinsic": float(extrinsic_loss.item()),
            "intrinsic": float(intrinsic_loss.item()),
            "rnd": float(rnd_loss),
            "avg_r_int": r_int_log,
            "entropy_reg": entropy_val,
            "batch_nonzero_r_frac": float((b_r_ext != 0).float().mean().item()),
            "target_mean": float(target_values.mean().item()) if 'target_values' in locals() else 0.0,
            "entropy_loss": entropy_val,
            "Beta": float(self.Beta),
            "Q_ext_mean": float(q_ext_now_norm.mean().item()),
            "Q_int_mean": float(int_quantiles.mean().item()) if 'int_quantiles' in locals() else 0.0,
            "last_eps": float(self.last_eps),
        }
        return float(extrinsic_loss.item())

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
        is_batched = obs.ndim > self.obs_ndim
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

            if is_batched:
                return actions.tolist()
            else:
                return actions.squeeze(0).tolist()