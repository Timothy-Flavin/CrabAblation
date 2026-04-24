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
        alpha: float = 0.001,
        munchausen_constant: float = 0.1,
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
        device="cpu",
        buffer_device="cpu",
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
        elif hasattr(self.input_dim, "__len__"):
            self.obs_ndim = len(self.input_dim)
        else:
            raise TypeError(
                f"Unsupported input_dim type: {type(self.input_dim)}. Expected int or array-like."
            )
        self.timing = {}
        self.n_action_dims = n_action_dims
        self.n_action_bins = n_action_bins
        self.n_actions = n_action_dims * n_action_bins
        self.hidden_layer_sizes = hidden_layer_sizes

        self.lr = lr
        self.gamma = gamma
        self.alpha = alpha
        self.polyak_tau = polyak_tau
        self.l_clip = l_clip
        self.soft = soft
        self.munchausen = munchausen_constant > 0.0
        self.munchausen_constant = munchausen_constant
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
        self.n_envs = n_envs  # = len(self.envs.remotes) if self.envs is not None and hasattr(self.envs, 'remotes') else getattr(self.envs, 'num_envs', 1)

        self.step = 0
        self.last_eps = 1.0
        self.update_timings = None
        self.buffer = self._init_buffer()

        # RND and running stats setup
        rnd_target_encoder = encoder_factory() if encoder_factory is not None else None
        rnd_predictor_encoder = (
            encoder_factory() if encoder_factory is not None else None
        )

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
        obs_shape = (
            self.input_dim if isinstance(self.input_dim, tuple) else (self.input_dim,)
        )

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

        real_next_obs = next_obs
        if info is not None and "final_observation" in info:
            if isinstance(next_obs, torch.Tensor):
                real_next_obs = next_obs.clone()
            elif isinstance(next_obs, np.ndarray):
                real_next_obs = next_obs.copy()
            else:
                real_next_obs = next_obs  # Fallback
            if "_final_observation" in info:
                # Vectorized environment handling
                for idx, is_final in enumerate(info["_final_observation"]):
                    if is_final:
                        real_next_obs[idx] = info["final_observation"][idx]
            elif "final_observation" in info:
                # Single environment handling
                if terminated or truncated:
                    real_next_obs = info["final_observation"]

        if self.Beta > 0.0:
            if isinstance(next_obs, np.ndarray):
                b_next_obs = torch.as_tensor(
                    real_next_obs, dtype=torch.float32, device=self.buffer_device
                )
            else:
                b_next_obs = (
                    real_next_obs.clone()
                    .detach()
                    .to(device=self.buffer_device, dtype=torch.float32)
                )

            # Add batch dimension if it is missing (e.g., n_envs=1 returning unbatched states)
            if b_next_obs.ndim == self.obs_ndim:
                b_next_obs = b_next_obs.unsqueeze(0)

            self.obs_rms.update(b_next_obs)
        # The buffer natively handles the shapes and tensor casting for memory efficiency
        self.buffer.add(
            obs=obs,
            next_obs=real_next_obs,
            action=action,
            reward=reward,
            term=terminated,  # Fixed to match function signature
            trunc=truncated,  # Fixed to match function signature
        )

    def to(self, device):
        """Move the agent and all its subcomponents to a specific device."""
        device = torch.device(device)
        self.device = device
        main_lr = (
            self.optim.param_groups[0]["lr"] if hasattr(self, "optim") else self.lr
        )
        int_lr = (
            self.int_optim.param_groups[0]["lr"]
            if hasattr(self, "int_optim")
            else self.intrinsic_lr
        )
        rnd_lr = (
            self.rnd_optim.param_groups[0]["lr"]
            if hasattr(self, "rnd_optim")
            else self.rnd_lr
        )

        if hasattr(self, "ext_online"):
            self.ext_online.to(device)
        if hasattr(self, "ext_target"):
            self.ext_target.to(device)
        if hasattr(self, "int_online"):
            self.int_online.to(device)
        if hasattr(self, "int_target"):
            self.int_target.to(device)
        if hasattr(self, "rnd"):
            self.rnd.to(device)

        if hasattr(self, "ext_online"):
            self.optim = torch.optim.Adam(self.ext_online.parameters(), lr=main_lr)
        if hasattr(self, "int_online"):
            self.int_optim = torch.optim.Adam(self.int_online.parameters(), lr=int_lr)
        if hasattr(self, "rnd"):
            self.rnd_optim = torch.optim.Adam(
                self.rnd.predictor.parameters(), lr=rnd_lr
            )
        return self

    def buffer_to(self, device):
        self.buffer_device = device
        if hasattr(self, "buffer") and self.buffer is not None:
            self.buffer.device = device
        if hasattr(self, "obs_rms") and hasattr(self.obs_rms, "to"):
            self.obs_rms.to(device)

    @torch.no_grad()
    def update_target(self):
        if not self.delayed_target:
            return

        for online, target in [
            (self.ext_online, self.ext_target),
            (self.int_online, self.int_target),
        ]:
            # Multiplies target weights by (1 - tau) in-place
            torch._foreach_mul_(list(target.parameters()), 1.0 - self.polyak_tau)
            # Adds online weights * tau in-place
            torch._foreach_add_(
                list(target.parameters()),
                list(online.parameters()),
                alpha=self.polyak_tau,
            )

            target_buffers = list(target.buffers())
            online_buffers = list(online.buffers())
            if len(target_buffers) > 0:
                torch._foreach_mul_(target_buffers, 1.0 - self.polyak_tau)
                torch._foreach_add_(
                    target_buffers, online_buffers, alpha=self.polyak_tau
                )

    # def update_target(self):
    #     """Hard update: copy online to target every K steps."""
    #     if not self.delayed_target:
    #         return

    #     # Assuming you trigger this every K steps instead of every frame
    #     self.ext_target.load_state_dict(self.ext_online.state_dict())
    #     self.int_target.load_state_dict(self.int_online.state_dict())
    #     self.ext_target.output_layer.sigma.copy_(self.ext_online.output_layer.sigma)
    #     self.ext_target.output_layer.mu.copy_(self.ext_online.output_layer.mu)
    #     self.int_target.output_layer.sigma.copy_(self.int_online.output_layer.sigma)
    #     self.int_target.output_layer.mu.copy_(self.int_online.output_layer.mu)
    #     # """Polyak averaging: target = (1 - tau) * target + tau * online."""
    #     # if not self.delayed_target:
    #     #     return
    #     # with torch.no_grad():
    #     #     for tp, op in zip(
    #     #         self.ext_target.parameters(), self.ext_online.parameters()
    #     #     ):
    #     #         tp.data.mul_(1.0 - self.polyak_tau).add_(op.data, alpha=self.polyak_tau)
    #     #     for tp, op in zip(
    #     #         self.int_target.parameters(), self.int_online.parameters()
    #     #     ):
    #     #         tp.data.mul_(1.0 - self.polyak_tau).add_(op.data, alpha=self.polyak_tau)

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
        n_envs: int = 1,
        buffer_size: int = int(1e5),
        hidden_layer_sizes=[128, 128],
        lr: float = 1e-3,
        gamma: float = 0.99,
        alpha: float = 0.9,
        munchausen_constant: float = 0.1,
        polyak_tau: float = 0.005,
        l_clip: float = -1.0,
        soft: bool = False,
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
        min_std: float = 0.01,
    ):
        super().__init__(
            input_dim=input_dim,
            n_action_dims=n_action_dims,
            n_action_bins=n_action_bins,
            n_envs=n_envs,
            buffer_size=buffer_size,
            hidden_layer_sizes=hidden_layer_sizes,
            lr=lr,
            gamma=gamma,
            alpha=alpha,
            munchausen_constant=munchausen_constant,
            polyak_tau=polyak_tau,
            l_clip=l_clip,
            soft=soft,
            Thompson=Thompson,
            dueling=dueling,
            Beta=Beta,
            delayed=delayed,
            ent_reg_coef=ent_reg_coef,
            rnd_output_dim=rnd_output_dim,
            rnd_lr=rnd_lr,
            intrinsic_lr=intrinsic_lr,
            int_r_clip=int_r_clip,
            ext_r_clip=ext_r_clip,
            beta_half_life_steps=beta_half_life_steps,
            norm_obs=norm_obs,
            burn_in_updates=burn_in_updates,
            encoder_factory=encoder_factory,
        )

        def _encoder_kwargs():
            if encoder_factory is None:
                return {}
            encoder = encoder_factory()
            return {
                "encoder": encoder,
                "encoder_out_dim": infer_encoder_out_dim(encoder, int(input_dim)),
            }

        ext_online_kwargs = _encoder_kwargs()
        ext_target_kwargs = _encoder_kwargs()
        int_online_kwargs = _encoder_kwargs()
        int_target_kwargs = _encoder_kwargs()

        self.ext_online = EV_Q_Network(
            input_dim,
            n_action_dims,
            n_action_bins,
            hidden_layer_sizes=hidden_layer_sizes,
            dueling=dueling,
            popart=True,
            min_std=min_std,
            **ext_online_kwargs,
        ).float()
        self.ext_target = EV_Q_Network(
            input_dim,
            n_action_dims,
            n_action_bins,
            hidden_layer_sizes=hidden_layer_sizes,
            dueling=dueling,
            popart=True,
            min_std=min_std,
            **ext_target_kwargs,
        ).float()
        self.int_online = EV_Q_Network(
            input_dim,
            n_action_dims,
            n_action_bins,
            hidden_layer_sizes=hidden_layer_sizes,
            dueling=dueling,
            popart=True,
            min_std=0.01,
            **int_online_kwargs,
        ).float()
        self.int_target = EV_Q_Network(
            input_dim,
            n_action_dims,
            n_action_bins,
            hidden_layer_sizes=hidden_layer_sizes,
            dueling=dueling,
            popart=True,
            min_std=0.01,
            **int_target_kwargs,
        ).float()

        self.ext_target.requires_grad_(False)
        self.ext_target.load_state_dict(self.ext_online.state_dict())
        self.int_target.requires_grad_(False)
        self.int_target.load_state_dict(self.int_online.state_dict())

        self.optim = torch.optim.Adam(self.ext_online.parameters(), lr=lr)
        self.int_optim = torch.optim.Adam(self.int_online.parameters(), lr=intrinsic_lr)

        # --- ALPHA AUTOTUNER SETUP ---
        if self.soft:
            # Max entropy per dim is ln(bins). Target ~80% of max entropy across all dims.
            max_ent = np.log(self.n_action_bins)  # self.n_action_dims *
            self.target_entropy = 0.8 * max_ent
            # Start alpha small so the penalty doesn't immediately crush Q-values
            self.log_alpha = nn.Parameter(torch.tensor([-3.0], device=self.device))
            # Use a slightly lower LR for alpha to prevent temperature whiplash
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr * 0.1)
            self.alpha = self.log_alpha.exp().item()

    @torch.no_grad()
    def _soft_policy(self, q_values: torch.Tensor):
        logpi = torch.clamp(torch.log_softmax(q_values / self.alpha, dim=-1), min=-1e8)
        pi = torch.exp(logpi)
        ent = -(pi * logpi).sum(dim=-1)  # [B]
        return pi, logpi, ent

    def update(self, batch_size=None, step=None):
        self.step += 1

        # Get batch data from buffer
        if batch_size is None:
            batch_size = 256
        (b_obs, b_a, b_next_obs, b_term, b_trunc, b_r_ext) = self.buffer.sample(
            batch_size
        )
        # Get the batch to the gpu
        b_next_obs = b_next_obs.to(self.device, non_blocking=True)

        if self.step < self.burn_in_updates:
            if self.Beta > 0.0 or getattr(self, "always_update_rnd", False):
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
        # Get intrinsic errors if we are going to use them
        if self.Beta > 0.0:
            rnd_errors, rnd_loss = self._update_RND(b_next_obs)
            if self.beta_half_life_steps is not None and self.beta_half_life_steps > 0:
                self.Beta = self.start_Beta * (
                    0.5 ** (self.step / self.beta_half_life_steps)
                )
            b_r_int = rnd_errors.detach()
        else:
            rnd_errors, rnd_loss, b_r_int = (
                torch.zeros_like(b_r_ext),
                0,
                torch.zeros_like(b_r_ext),
            )

        logpi_now = None
        pi_now = None
        entropy_loss = 0
        current_sigma = self.ext_target.output_layer.sigma.detach()

        # Get target
        with torch.no_grad():
            q_ext_norm = self.ext_online(b_obs, normalized=True)  # [B,D,Bins]
            q_next_online_norm = self.ext_online(b_next_obs, normalized=True)
            q_next_target_raw = (
                self.ext_target(b_next_obs, normalized=False)
                if self.delayed_target
                else self.ext_online(b_next_obs, normalized=False)
            )

            # Munchausen loss only for the exploiter
            if self.munchausen:
                if logpi_now is None:
                    logpi_now = torch.clamp(
                        torch.log_softmax(q_ext_norm / self.alpha, dim=-1), min=-1e8
                    )
                selected_logpi = torch.gather(logpi_now, -1, b_actions_idx).squeeze(
                    -1
                )  # [B,D]
                r_kl = torch.clamp(selected_logpi, min=self.l_clip)
                if r_kl.ndim > 1:
                    r_kl = r_kl.mean(-1)
                assert b_r_ext.ndim == r_kl.ndim
                b_r_ext += current_sigma * self.alpha * self.munchausen_constant * r_kl

            # Next value with entropy and weighted sum over q values
            if self.munchausen or self.soft:
                logpi_next = torch.clamp(
                    torch.log_softmax(q_next_online_norm / self.alpha, dim=-1), min=-1e8
                )
                pi_next = torch.exp(logpi_next)
                next_head_vals = (
                    pi_next
                    * (q_next_target_raw - current_sigma * self.alpha * logpi_next)
                ).sum(-1)
            # Next value with no entropy or weighted sum, using argmax policy
            else:
                target_actions_next = q_next_online_norm.argmax(
                    dim=-1, keepdim=True
                ).detach()
                next_head_vals = torch.gather(
                    q_next_target_raw, -1, target_actions_next
                ).squeeze(-1)
            # vdn sum the vals
            if next_head_vals.ndim > 1:
                next_head_vals = next_head_vals.mean(-1)
            assert (
                b_r_ext.shape == b_term.shape == next_head_vals.shape
            ), f"Shape mismatch: {b_r_ext.shape}, {b_term.shape}, {next_head_vals.shape}"
            online_ext_target = (
                b_r_ext.view(-1) + self.gamma * (1 - b_term).view(-1) * next_head_vals
            )

        target_for_stats_ext = online_ext_target.detach()  # maintain [B, D]
        self.ext_online.output_layer.update_stats(target_for_stats_ext)
        # if self.delayed_target:

        td_target_norm = self.ext_online.output_layer.normalize(target_for_stats_ext)
        q_ext_now_norm = self.ext_online(b_obs, normalized=True)
        q_selected_norm = torch.gather(q_ext_now_norm, -1, b_actions_idx).squeeze(
            -1
        )  # [B, D]
        drift_penalty = 0.0
        if q_selected_norm.ndim > 1:
            drift_penalty = q_selected_norm.pow(2).mean() * 1e-3
            q_selected_norm = q_selected_norm.mean(-1)
        assert (
            q_selected_norm.shape == td_target_norm.shape
        ), f"Shape mismatch: q_selected_norm {q_selected_norm.shape}, td_target_norm {td_target_norm.shape}"
        extrinsic_loss = (
            torch.nn.functional.mse_loss(q_selected_norm, td_target_norm)
            + drift_penalty
        )

        self.optim.zero_grad()
        extrinsic_loss.backward()
        if hasattr(self, "ext_online"):
            torch.nn.utils.clip_grad_norm_(self.ext_online.parameters(), max_norm=10.0)
        self.optim.step()

        # --- ALPHA AUTO-TUNING UPDATE ---
        alpha_loss = torch.tensor(0.0, device=self.log_alpha.device)
        if self.soft:
            with torch.no_grad():
                q_fresh = self.ext_online(b_obs, normalized=True).detach()
                logpi_fresh = torch.clamp(
                    torch.log_softmax(q_fresh / self.alpha, dim=-1), min=-1e8
                )
                pi_fresh = torch.exp(logpi_fresh)
                current_entropy = -(pi_fresh * logpi_fresh).sum(dim=-1).mean()
                # Ensure current_entropy and target_entropy are on the same device
                current_entropy = current_entropy.to(self.log_alpha.device)
                target_entropy = torch.tensor(
                    self.target_entropy,
                    device=self.log_alpha.device,
                    dtype=current_entropy.dtype,
                )
            # REMOVED THE NEGATIVE SIGN at the front
            alpha_loss = (
                self.log_alpha.exp() * (current_entropy - target_entropy).detach()
            )
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().item()

        # Intrinsic Q update
        with torch.no_grad():
            int_q_next = (self.int_target if self.delayed_target else self.int_online)(
                b_next_obs, normalized=False
            )
            # Double q actions if delayed. We grab actions from online and vals from target
            if self.delayed_target:
                next_int_actions = (
                    self.int_online(b_next_obs, normalized=True)
                    .argmax(-1, keepdim=True)
                    .detach()
                )
            else:
                next_int_actions = int_q_next.argmax(-1, keepdim=True).detach()
            int_q_next_target = torch.gather(int_q_next, -1, next_int_actions).squeeze(
                -1
            )
            if int_q_next_target.ndim > 1:
                int_q_next_target = int_q_next_target.mean(-1)
            assert (
                b_r_int.view(-1).shape == int_q_next_target.shape
            ), f"Shape mismatch: b_r_int {b_r_int.view(-1).shape}, int_q_next_target {int_q_next_target.shape}"
            int_td_target = b_r_int.view(-1) + self.gamma * int_q_next_target

        target_for_stats_int = int_td_target.detach()
        self.int_online.output_layer.update_stats(target_for_stats_int)
        # if self.delayed_target:
        #    self.int_target.output_layer.sigma.copy_(self.int_online.output_layer.sigma)
        #    self.int_target.output_layer.mu.copy_(self.int_online.output_layer.mu)

        int_td_target_norm = self.int_online.output_layer.normalize(
            target_for_stats_int
        )
        int_q_now_norm = self.int_online(b_obs, normalized=True)
        int_q_selected_norm = torch.gather(int_q_now_norm, -1, b_actions_idx).squeeze(
            -1
        )
        drift_penalty_int = 0.0
        if int_q_selected_norm.ndim > 1:
            drift_penalty_int = int_q_selected_norm.pow(2).mean() * 1e-3
            int_q_selected_norm = int_q_selected_norm.mean(-1)
        assert (
            int_q_selected_norm.shape == int_td_target_norm.shape
        ), f"Shape mismatch: int_q_selected_norm {int_q_selected_norm.shape}, int_td_target_norm {int_td_target_norm.shape}"
        intrinsic_loss = (
            torch.nn.functional.mse_loss(int_q_selected_norm, int_td_target_norm)
            + drift_penalty_int
        )

        self.int_optim.zero_grad()
        intrinsic_loss.backward()

        # Update target network
        # if self.delayed_target and self.step % 200 == 0:
        self.update_target()

        if self.step % 1000 == 0:
            # tracking
            if hasattr(self, "int_online"):
                torch.nn.utils.clip_grad_norm_(
                    self.int_online.parameters(), max_norm=10.0
                )
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
                "alpha_loss": (
                    float(alpha_loss.item())
                    if isinstance(alpha_loss, torch.Tensor)
                    else float(alpha_loss)
                ),
                "batch_nonzero_r_frac": float((b_r_ext != 0).float().mean().item()),
                "target_mean": float(target_for_stats_ext.mean().item()),
                "td_target_norm_mean": float(
                    td_target_norm.abs().mean().detach().item()
                ),
                "Beta": float(self.Beta),
                "Q_ext_mean": float(q_ext_now_norm.mean().item()),
                "Q_int_mean": (
                    float(int_q_now_norm.mean().item())
                    if "int_q_now_norm" in locals()
                    else 0.0
                ),
                "last_eps": float(self.last_eps),
            }
            print(self.last_losses)

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
            if self.Beta > 0.0:
                int_q = self.int_online(obs_b, normalized=True)
                q_ext = (1.0 - self.Beta) * q_ext + self.Beta * int_q

            if self.soft or self.munchausen:
                actions = torch.distributions.Categorical(
                    logits=q_ext / self.alpha
                ).sample()
            else:
                actions = torch.argmax(q_ext, dim=-1)
                rand_vals = torch.rand(batch_size, device=obs_b.device)
                explore_mask = (rand_vals < min_ent) | (rand_vals < eps)

                if explore_mask.any():
                    explore_actions = torch.randint(
                        0,
                        self.n_action_bins,
                        (batch_size, self.n_action_dims),
                        device=obs_b.device,
                    )
                    actions = torch.where(
                        explore_mask.unsqueeze(1) if actions.ndim > 1 else explore_mask,
                        explore_actions,
                        actions,
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
        polyak_tau: float = 0.03,
        l_clip: float = -1.0,
        soft: bool = False,
        munchausen_constant: float = 0.1,
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
        min_std: float = 0.01,
    ):
        super().__init__(
            input_dim=input_dim,
            n_action_dims=n_action_dims,
            n_action_bins=n_action_bins,
            n_envs=n_envs,
            buffer_size=buffer_size,
            hidden_layer_sizes=hidden_layer_sizes,
            lr=lr,
            gamma=gamma,
            alpha=alpha,
            munchausen_constant=munchausen_constant,
            polyak_tau=polyak_tau,
            l_clip=l_clip,
            soft=soft,
            Thompson=Thompson,
            dueling=dueling,
            Beta=Beta,
            delayed=delayed,
            ent_reg_coef=ent_reg_coef,
            rnd_output_dim=rnd_output_dim,
            rnd_lr=rnd_lr,
            intrinsic_lr=intrinsic_lr,
            int_r_clip=int_r_clip,
            ext_r_clip=ext_r_clip,
            beta_half_life_steps=beta_half_life_steps,
            norm_obs=norm_obs,
            burn_in_updates=burn_in_updates,
            encoder_factory=encoder_factory,
        )

        def _encoder_kwargs():
            if encoder_factory is None:
                return {}
            encoder = encoder_factory()
            return {
                "encoder": encoder,
                "encoder_out_dim": infer_encoder_out_dim(encoder, int(input_dim)),
            }

        ext_online_kwargs = _encoder_kwargs()
        ext_target_kwargs = _encoder_kwargs()
        int_online_kwargs = _encoder_kwargs()
        int_target_kwargs = _encoder_kwargs()

        self.ext_online = IQN_Network(
            input_dim,
            n_action_dims,
            n_action_bins,
            hidden_layer_sizes=hidden_layer_sizes,
            dueling=dueling,
            popart=True,
            min_std=min_std,
            **ext_online_kwargs,
        ).float()
        self.ext_target = IQN_Network(
            input_dim,
            n_action_dims,
            n_action_bins,
            hidden_layer_sizes=hidden_layer_sizes,
            dueling=dueling,
            popart=True,
            min_std=min_std,
            **ext_target_kwargs,
        ).float()
        self.int_online = IQN_Network(
            input_dim,
            n_action_dims,
            n_action_bins,
            hidden_layer_sizes=hidden_layer_sizes,
            dueling=dueling,
            popart=True,
            min_std=0.01,
            **int_online_kwargs,
        ).float()
        self.int_target = IQN_Network(
            input_dim,
            n_action_dims,
            n_action_bins,
            hidden_layer_sizes=hidden_layer_sizes,
            dueling=dueling,
            popart=True,
            min_std=0.01,
            **int_target_kwargs,
        ).float()

        self.ext_target.requires_grad_(False)
        self.ext_target.load_state_dict(self.ext_online.state_dict())
        self.int_target.requires_grad_(False)
        self.int_target.load_state_dict(self.int_online.state_dict())

        self.optim = torch.optim.Adam(self.ext_online.parameters(), lr=lr)
        self.int_optim = torch.optim.Adam(self.int_online.parameters(), lr=intrinsic_lr)

        # --- ALPHA AUTOTUNER SETUP ---
        if self.soft:
            max_ent = np.log(self.n_action_bins)  # self.n_action_dims *
            self.target_entropy = 0.8 * max_ent
            self.log_alpha = nn.Parameter(torch.tensor([-3.0], device=self.device))
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr * 0.1)
            self.alpha = self.log_alpha.exp().item()

        self.n_quantiles = 32
        self.n_target_quantiles = 32

    def _sample_taus(
        self, batch_size: int, n: int, device: torch.device
    ) -> torch.Tensor:
        return torch.rand(batch_size, n, device=device)

    def _quantile_huber_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        taus: torch.Tensor,
        kappa: float = 1.0,
    ) -> torch.Tensor:
        B, N = pred.shape
        td = target.unsqueeze(1) - pred.unsqueeze(2)  # [B, N, Nt]
        abs_td = torch.abs(td)
        huber = torch.where(
            abs_td <= kappa, 0.5 * td.pow(2), kappa * (abs_td - 0.5 * kappa)
        )
        I_ = (td < 0).float()
        taus_expanded = taus.unsqueeze(2)  # [B,N,1]
        loss = (torch.abs(taus_expanded - I_) * huber).mean()
        return loss

    def update(self, batch_size=None, step=None):
        self.step += 1

        # Get batch data from buffer
        if batch_size is None:
            batch_size = 256
        (b_obs, b_a, b_next_obs, b_term, b_trunc, b_r_ext) = self.buffer.sample(
            batch_size
        )

        # Get the batch to the gpu
        b_next_obs = b_next_obs.to(self.device, non_blocking=True)

        # Burn-in logic identical to EVRainbowDQN
        if self.step < self.burn_in_updates:
            if self.Beta > 0.0 or getattr(self, "always_update_rnd", False):
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
            rnd_errors, rnd_loss, b_r_int = (
                torch.zeros_like(b_r_ext),
                0,
                torch.zeros_like(b_r_ext),
            )

        entropy_loss = 0.0
        current_sigma = self.ext_online.output_layer.sigma.detach()

        # ========================================================
        # Extrinsic Q update
        # ========================================================
        with torch.no_grad():
            dist_q_shape = (
                batch_size,
                self.n_quantiles,
                self.n_action_dims,
                self.n_action_bins,
            )
            taus = self._sample_taus(batch_size, self.n_quantiles, self.device)
            target_taus = self._sample_taus(
                batch_size, self.n_target_quantiles, self.device
            )

            # Online Next Q -> For action selection
            online_next_q_norm = self.ext_online(b_next_obs, taus, normalized=True)
            # print(online_next_q_norm.shape)
            online_next_q_norm = online_next_q_norm.view(dist_q_shape).mean(dim=1)
            # print(f"online next q nrm [0] {online_next_q_norm[0]}")

            # Target Net Quantiles -> For target values
            t_net = self.ext_target if self.delayed_target else self.ext_online
            target_quantiles_all = t_net(
                b_next_obs, target_taus, normalized=False
            )  # [B, Nt, D, Bins]
            # print(f"Target quantils shape: {target_quantiles_all.shape}")

            m_r = 0.0
            ent_bonus = 0.0

            if self.munchausen or self.soft:
                logpi_next = torch.clamp(
                    torch.log_softmax(online_next_q_norm / self.alpha, dim=-1),
                    min=-1e8,
                )
                pi_next = torch.exp(logpi_next)
                # Entropy bonus: sum over D
                ent_bonus = -(pi_next * logpi_next).mean(dim=-1)  # sum over bins
                if ent_bonus.ndim > 1:
                    ent_bonus = ent_bonus.mean(dim=-1).unsqueeze(
                        1
                    )  # sum over D, then [B, 1]
                else:
                    ent_bonus = ent_bonus.unsqueeze(1)
                # print(f"pi next: {pi_next.shape} logpinext: {logpi_next.shape}, ")
                # if self.soft:
                #     print(f"ent: {ent_bonus.shape}")
                # Mixed target values
                mixed_target = (pi_next.unsqueeze(1) * target_quantiles_all).sum(
                    dim=-1
                )  # sum over bins -> [B, Nt, D]
                if mixed_target.ndim > 2:
                    mixed_target = mixed_target.mean(dim=-1)  # sum over D -> [B, Nt]
            else:
                target_actions = online_next_q_norm.argmax(dim=-1)
                action_idx = (
                    target_actions.unsqueeze(1)
                    .unsqueeze(-1)
                    .expand(-1, self.n_target_quantiles, -1, 1)
                )
                mixed_target = torch.gather(
                    target_quantiles_all, -1, action_idx
                ).squeeze(
                    -1
                )  # [B, Nt, D]
                if mixed_target.ndim > 2:
                    mixed_target = mixed_target.mean(dim=-1)  # sum over D -> [B, Nt]

            # print(f"mixed target dim: {mixed_target.shape}")
            if self.munchausen:
                t_expected = torch.linspace(
                    0.01, 0.99, self.n_quantiles, device=self.device
                )
                t_expected = t_expected.unsqueeze(0).expand(batch_size, -1)
                q_ext_norm_now = self.ext_online(
                    b_obs, t_expected, normalized=True
                ).mean(dim=1)
                # print(q_ext_norm_now.shape)
                # print(q_ext_norm_now[0])
                logpi_now = torch.log_softmax(q_ext_norm_now / self.alpha, dim=-1)
                # print(f"logpi now: {logpi_now[0]} pi now: {torch.exp(logpi_now[0])}")
                # print(f"logpi shape: {logpi_now.shape} actions shape: {b_actions_idx.shape}")
                selected_logpi = torch.gather(logpi_now, -1, b_actions_idx).squeeze(-1)
                # print(f"actins: {b_actions_idx[0]} \nselected {selected_logpi[0]} \nlogpis {logpi_now[0]}")
                # print(selected_logpi.shape)
                # sum r_kl over D if needed

                if selected_logpi.ndim > 1:
                    # print("summed dim 1 logpis for multiple action dims")
                    r_kl = torch.clamp(
                        selected_logpi.mean(dim=-1), min=self.l_clip
                    ).view(-1)
                else:
                    r_kl = torch.clamp(selected_logpi, min=self.l_clip).view(-1)
                m_r = (
                    current_sigma * self.alpha * self.munchausen_constant * r_kl
                ).view(-1)
            b_r_final = b_r_ext.view(-1) + m_r
            # Target Q-distribution
            assert (
                b_r_final.shape == b_term.shape
            ), f"Shape mismatch: b_r_final {b_r_final.shape}, b_term {b_term.shape}"
            assert (
                b_term.ndim == mixed_target.ndim - 1
            ), "mixed target is going to broadcast bad"
            target_values = b_r_final.unsqueeze(1) + (1 - b_term).unsqueeze(
                1
            ) * self.gamma * (mixed_target + current_sigma * self.alpha * ent_bonus)
        # print(f"We made it past the target calculation br {b_r_final.unsqueeze(1).shape} bterm {(1 - b_term).unsqueeze(1).shape} mix_target {mixed_target.shape}, ")
        # if self.soft:
        #     print(f"ent: {ent_bonus.shape}")
        # Apply PopArt stats tracking over target distributions
        # Mean to get the expected value and stop popart from thinking taus are
        self.ext_online.output_layer.update_stats(target_values.detach().mean(-1))
        # if self.delayed_target:
        #    self.ext_target.output_layer.sigma.copy_(self.ext_online.output_layer.sigma)
        #    self.ext_target.output_layer.mu.copy_(self.ext_online.output_layer.mu)
        target_values_norm = self.ext_online.output_layer.normalize(
            target_values.detach()
        )

        # Current normalized quantile predictions
        taus_pred = self._sample_taus(batch_size, self.n_quantiles, self.device)
        quantiles_pred = self.ext_online(
            b_obs, taus_pred, normalized=True
        )  # [B, N, D, Bins]

        gather_index_pred = b_actions_idx.unsqueeze(1).expand(
            -1, self.n_quantiles, -1, 1
        )
        pred_chosen = torch.gather(quantiles_pred, -1, gather_index_pred).squeeze(
            -1
        )  # [B, N, D]
        int_drift_penalty = 0.0
        if pred_chosen.ndim > 2:
            int_drift_penalty = pred_chosen.pow(2).mean() * 1e-3
            pred_chosen = pred_chosen.mean(dim=-1)  # [B, N]
        assert (
            pred_chosen.shape[0] == target_values_norm.shape[0]
            and pred_chosen.ndim == target_values_norm.ndim == 2
        ), f"Shape mismatch: pred_chosen {pred_chosen.shape}, target_values_norm {target_values_norm.shape}"

        extrinsic_loss = (
            self._quantile_huber_loss(pred_chosen, target_values_norm, taus_pred)
            + int_drift_penalty
        )

        self.optim.zero_grad()
        extrinsic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ext_online.parameters(), max_norm=10.0)
        self.optim.step()

        # --- ALPHA AUTO-TUNING UPDATE ---
        alpha_loss = torch.tensor(0.0, device=self.log_alpha.device)
        if self.soft:
            with torch.no_grad():
                q_fresh = (
                    quantiles_pred.mean(dim=1).detach()
                    if "quantiles_pred" in locals()
                    else self.ext_online(
                        b_obs,
                        self._sample_taus(batch_size, self.n_quantiles, self.device),
                        normalized=True,
                    )
                    .mean(dim=1)
                    .detach()
                )
                logpi_fresh = torch.clamp(
                    torch.log_softmax(q_fresh / self.alpha, dim=-1), min=-1e8
                )
                pi_fresh = torch.exp(logpi_fresh)
                current_entropy = -(pi_fresh * logpi_fresh).sum(dim=-1).mean()
                current_entropy = current_entropy.to(self.log_alpha.device)
                target_entropy = torch.tensor(
                    self.target_entropy,
                    device=self.log_alpha.device,
                    dtype=current_entropy.dtype,
                )
            alpha_loss = (
                self.log_alpha.exp() * (current_entropy - target_entropy).detach()
            )
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().item()

        # ========================================================
        # Intrinsic Q update
        # ========================================================
        if self.Beta > 0.0:
            with torch.no_grad():
                int_taus = self._sample_taus(batch_size, self.n_quantiles, self.device)
                int_target_taus = self._sample_taus(
                    batch_size, self.n_target_quantiles, self.device
                )

                online_next_q_int_norm = self.int_online(
                    b_next_obs, int_taus, normalized=True
                ).mean(dim=1)
                target_actions_int = online_next_q_int_norm.argmax(dim=-1)

                t_net_int = self.int_target if self.delayed_target else self.int_online
                int_target_all = t_net_int(
                    b_next_obs, int_target_taus, normalized=False
                )

                action_idx_int = (
                    target_actions_int.unsqueeze(1)
                    .unsqueeze(-1)
                    .expand(-1, self.n_target_quantiles, -1, 1)
                )
                mixed_target_int = torch.gather(
                    int_target_all, -1, action_idx_int
                ).squeeze(
                    -1
                )  # [B, Nt, D]

                if mixed_target_int.ndim > 2:
                    mixed_target_int = mixed_target_int.mean(dim=-1)  # [B, Nt]

                # No terminal mask for intrinsic reward
                assert (
                    b_r_int.unsqueeze(1).shape[0] == mixed_target_int.shape[0]
                ), f"Shape mismatch: b_r_int {b_r_int.unsqueeze(1).shape}, mixed_target_int {mixed_target_int.shape}"
                int_target_values = b_r_int.unsqueeze(1) + self.gamma * mixed_target_int

            # Mean -1 to get expected value so popart tracks target variance not env varaince
            self.int_online.output_layer.update_stats(
                int_target_values.detach().mean(-1)
            )
            # if self.delayed_target:
            # self.int_target.output_layer.sigma.copy_(
            #     self.int_online.output_layer.sigma
            # )
            # self.int_target.output_layer.mu.copy_(self.int_online.output_layer.mu)
            int_target_values_norm = self.int_online.output_layer.normalize(
                int_target_values.detach()
            )

            int_taus_pred = self._sample_taus(batch_size, self.n_quantiles, self.device)
            int_quantiles = self.int_online(b_obs, int_taus_pred, normalized=True)

            gather_index_int_pred = b_actions_idx.unsqueeze(1).expand(
                -1, self.n_quantiles, -1, 1
            )
            int_pred_chosen = torch.gather(
                int_quantiles, -1, gather_index_int_pred
            ).squeeze(
                -1
            )  # [B, N, D]
            int_drift_penalty = 0.0
            if int_pred_chosen.ndim > 2:
                int_drift_penalty = int_pred_chosen.pow(2).mean() * 1e-3
                int_pred_chosen = int_pred_chosen.mean(dim=-1)  # [B, N]

            assert (
                int_pred_chosen.shape[0] == int_target_values_norm.shape[0]
                and int_pred_chosen.ndim == int_target_values_norm.ndim == 2
            ), f"Shape mismatch: int_pred_chosen {int_pred_chosen.shape}, int_target_values_norm {int_target_values_norm.shape}"

            intrinsic_loss = (
                self._quantile_huber_loss(
                    int_pred_chosen, int_target_values_norm, int_taus_pred
                )
                + int_drift_penalty
            )

            self.int_optim.zero_grad()
            intrinsic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.int_online.parameters(), max_norm=10.0)
            self.int_optim.step()
        else:
            intrinsic_loss = torch.tensor(0.0)

        # Update target network
        # if self.delayed_target and self.step % 200 == 0:
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

        q_ext_now_norm = (
            quantiles_pred.mean(dim=1)
            if "quantiles_pred" in locals()
            else torch.tensor(0.0)
        )

        self.last_losses = {
            "extrinsic": float(extrinsic_loss.item()),
            "intrinsic": float(intrinsic_loss.item()),
            "rnd": float(rnd_loss),
            "avg_r_int": r_int_log,
            "alpha_loss": (
                float(alpha_loss.item())
                if isinstance(alpha_loss, torch.Tensor)
                else float(alpha_loss)
            ),
            "batch_nonzero_r_frac": float((b_r_ext != 0).float().mean().item()),
            "target_mean": (
                float(target_values.mean().item())
                if "target_values" in locals()
                else 0.0
            ),
            "Beta": float(self.Beta),
            "Q_ext_mean": float(q_ext_now_norm.mean().item()),
            "Q_int_mean": (
                float(int_quantiles.mean().item())
                if "int_quantiles" in locals()
                else 0.0
            ),
            "last_eps": float(self.last_eps),
        }
        # print(self.last_losses)
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
            ext_q = self.ext_online(obs_b, taus, normalized=True).mean(
                dim=1
            )  # [B,D,Bins]
            if self.Beta > 0.0:
                int_taus = self._sample_taus(batch_size, self.n_quantiles, obs_b.device)
                int_q = self.int_online(obs_b, int_taus, normalized=True).mean(dim=1)
                ext_q = (1.0 - self.Beta) * ext_q + self.Beta * int_q

            if self.soft or self.munchausen:
                logits = ext_q / self.alpha
                actions = torch.distributions.Categorical(
                    logits=logits
                ).sample()  # [B,D]
            else:
                actions = torch.argmax(ext_q, dim=-1)  # [B,D]
                rand_vals = torch.rand(batch_size, device=obs_b.device)
                explore_mask = (rand_vals < min_eps) | (rand_vals < eps)
                if explore_mask.any():
                    explore_actions = torch.randint(
                        0,
                        self.n_action_bins,
                        (batch_size, self.n_action_dims),
                        device=obs_b.device,
                    )
                    actions = torch.where(
                        explore_mask.unsqueeze(1) if actions.ndim > 1 else explore_mask,
                        explore_actions,
                        actions,
                    )

            if is_batched:
                return actions.tolist()
            else:
                return actions.squeeze(0).tolist()
