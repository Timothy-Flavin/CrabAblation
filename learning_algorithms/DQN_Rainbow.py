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
        alpha: float = 0.03,
        munchausen_constant: float = 0.9,
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

        # Hard-start PopArt at burn-in end from raw reward stats (see
        # _seed_popart_from_buffer). This flag tracks one-shot seeding.
        self._popart_burn_in_seeded = False

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

    @torch.no_grad()
    def _seed_popart_from_buffer(self, ext_layer=None, int_layer=None, sample_size: int = 8192):
        """Hard-start calibration of PopArt from raw reward stats, scaled to
        infinite-horizon returns (μ_R/(1-γ), σ_R/√(1-γ²)).

        Avoids contaminating PopArt with random Q_target noise from an untrained
        critic, which would normalize the real reward signal toward zero.
        """
        if self._popart_burn_in_seeded:
            return
        buffer = getattr(self, "buffer", None)
        if buffer is None or buffer.size() == 0:
            self._popart_burn_in_seeded = True
            return

        valid_pos = int(buffer.size())
        n = min(int(sample_size), valid_pos * int(buffer.n_envs))
        batch_inds = torch.randint(0, valid_pos, (n,))
        env_inds = torch.randint(0, buffer.n_envs, (n,))
        rewards = buffer.rewards[batch_inds, env_inds].float().view(-1)

        gamma = float(self.gamma)
        scale_mu = 1.0 / max(1.0 - gamma, 1e-6)
        scale_sigma = 1.0 / max((1.0 - gamma * gamma) ** 0.5, 1e-6)
        eps = 1e-4

        if ext_layer is not None and rewards.numel() > 0:
            mu_R = float(rewards.mean().item())
            sigma_R = float(rewards.std(unbiased=False).item())
            seeded_mu = mu_R * scale_mu
            seeded_sigma = max(sigma_R * scale_sigma, eps)
            ext_layer.set_initial_stats(seeded_mu, seeded_sigma)
            self._seeded_ext_sigma = seeded_sigma

        # Intrinsic head: compute RND error stats on a fresh batch and scale.
        if int_layer is not None and self.Beta > 0.0 and rewards.numel() > 0:
            next_obs = buffer.next_observations[batch_inds, env_inds].to(self.device).float()
            norm_next_obs = self.obs_rms.normalize(next_obs).float()
            rnd_errors = self.rnd(norm_next_obs).detach().reshape(-1)
            rnd_errors = torch.clamp(rnd_errors, -float(self.int_r_clip), float(self.int_r_clip))
            mu_I = float(rnd_errors.mean().item())
            sigma_I = float(rnd_errors.std(unbiased=False).item())
            seeded_mu_I = mu_I * scale_mu
            seeded_sigma_I = max(sigma_I * scale_sigma, eps)
            int_layer.set_initial_stats(seeded_mu_I, seeded_sigma_I)
            self._seeded_int_sigma = seeded_sigma_I

        self._popart_burn_in_seeded = True

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
        alpha: float = 0.03,
        munchausen_constant: float = 0.9,
        polyak_tau: float = 0.03,
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
        autotune: bool = True,
        encoder_factory: Optional[Callable[[], nn.Module]] = None,
        min_std: float = 0.01,
        target_entropy_frac: float = 0.2,
    ):
        # Fraction of max entropy (ln(bins)) the soft-Q alpha autotuner targets. 0.2 is
        # an exploitative single-agent default; multi-agent Nash on symmetric games (e.g.
        # RPS) needs near-max entropy, so the MA runner raises this toward ~1.0.
        self._target_entropy_frac = float(target_entropy_frac)
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
        self.autotune = autotune

        def _encoder_kwargs():
            if encoder_factory is None:
                return {}
            encoder = encoder_factory()
            return {
                "encoder": encoder,
                "encoder_out_dim": infer_encoder_out_dim(encoder, int(input_dim)),
            }

        self.ext_online = EV_Q_Network(
            input_dim,
            n_action_dims,
            n_action_bins,
            hidden_layer_sizes=hidden_layer_sizes,
            dueling=dueling,
            popart=True,
            min_std=min_std,
            **_encoder_kwargs(),
        ).float()
        self.ext_target = EV_Q_Network(
            input_dim,
            n_action_dims,
            n_action_bins,
            hidden_layer_sizes=hidden_layer_sizes,
            dueling=dueling,
            popart=True,
            min_std=min_std,
            **_encoder_kwargs(),
        ).float()
        self.int_online = EV_Q_Network(
            input_dim,
            n_action_dims,
            n_action_bins,
            hidden_layer_sizes=hidden_layer_sizes,
            dueling=dueling,
            popart=True,
            min_std=0.01,
            **_encoder_kwargs(),
        ).float()
        self.int_target = EV_Q_Network(
            input_dim,
            n_action_dims,
            n_action_bins,
            hidden_layer_sizes=hidden_layer_sizes,
            dueling=dueling,
            popart=True,
            min_std=0.01,
            **_encoder_kwargs(),
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
            max_ent = np.log(self.n_action_bins)
            self.target_entropy = self._target_entropy_frac * max_ent
            # Start alpha small so the penalty doesn't immediately crush Q-values
            # Honor the constructor alpha as the starting temperature (was hardcoded
            # 0.03/0.05). The MA runner passes a higher --dqn_alpha so the soft policy
            # is actually anchored; single-agent default stays 0.03.
            initial_alpha = float(alpha)
            self.log_alpha = nn.Parameter(torch.tensor([np.log(initial_alpha)], device=self.device))
            # Use a slightly lower LR for alpha to prevent temperature whiplash
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr * 0.1)
            self.alpha = self.log_alpha.exp().item()

        self.autotune = True
        # if self.munchausen:
        #     self.autotune = False
        #     # self.alpha = 0.03 # This will be set by log_alpha.exp() if soft=True
        # else:
        #     self.autotune = True

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

        # First post-burn-in step: hard-start PopArt from raw reward stats
        # scaled to infinite-horizon returns. Avoids being polluted by an
        # untrained Q_target's massive noise.
        if not self._popart_burn_in_seeded and self.burn_in_updates > 0:
            self._seed_popart_from_buffer(
                ext_layer=self.ext_online.output_layer,
                int_layer=self.int_online.output_layer,
            )
            if self.delayed_target:
                self.ext_target.load_state_dict(self.ext_online.state_dict())
                self.int_target.load_state_dict(self.int_online.state_dict())

        b_obs = b_obs.to(self.device, non_blocking=True)
        b_a = b_a.to(self.device, non_blocking=True).long()
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
        current_sigma = self.ext_target.output_layer.sigma.detach()

        # Get target
        with torch.no_grad():
            q_ext_norm = ( # Change to target network for everything
                self.ext_target(b_obs, normalized=True)
                if self.delayed_target
                else self.ext_online(b_obs, normalized=True)  # [B,D,Bins]
            )
            q_next_online_norm = self.ext_online(b_next_obs, normalized=True)
            q_next_target_raw = (
                self.ext_target(b_next_obs, normalized=False)
                if self.delayed_target
                else self.ext_online(b_next_obs, normalized=False)
            )

            # 1. Calculate the Munchausen Penalty separately (Do NOT modify b_r_ext!)
            m_r = 0.0
            if self.munchausen:
                if logpi_now is None:
                    logpi_now = torch.clamp(
                        torch.log_softmax(q_ext_norm / self.alpha, dim=-1), min=-1e8
                    )
                selected_logpi = torch.gather(logpi_now, -1, b_actions_idx).squeeze(
                    -1
                )  # [B,D]
                # Sigma remains outside the clamp for perfect scale-free behavior
                m_r = current_sigma * self.munchausen_constant * torch.clamp(self.alpha * selected_logpi, min=self.l_clip, max=0.0)

            # 2. Split Next Values into Env-Only and Entropy Bonus
            if self.munchausen or self.soft:
                logpi_next = torch.clamp(
                    torch.log_softmax(q_next_online_norm / self.alpha, dim=-1), min=-1e8
                )
                pi_next = torch.exp(logpi_next)
                
                # To break the feedback loop, we need an environment-only estimate for Q_next.
                # Since the network predicts Q_full = Q_env + Q_penalty, we subtract the
                # immediate next-state penalty to approximate Q_env.
                # removing the immediate one reduces the loop multiplier below 1.0.
                
                # Entropy bonus is -sigma * alpha * logpi.
                next_ent_bonus_raw = (pi_next * (-self.alpha * logpi_next)).sum(-1) # [B,D]
                next_ent_bonus = current_sigma * next_ent_bonus_raw

                # Munchausen bonus at the next state
                if self.munchausen:
                    next_m_r_raw = (pi_next * (self.munchausen_constant * torch.clamp(self.alpha * logpi_next, min=self.l_clip, max=0.0))).sum(-1)
                    next_m_r = current_sigma * next_m_r_raw
                else:
                    next_m_r = 0.0

                next_q_full = (pi_next * q_next_target_raw).sum(-1)
                # Subtract BOTH bonuses to get back to environment part (1st order approximation)
                next_q_env = next_q_full - next_ent_bonus - next_m_r
            # Next value with no entropy or weighted sum, using argmax policy
            else:
                target_actions_next = q_next_online_norm.argmax(
                    dim=-1, keepdim=True
                ).detach()
                next_q_env = torch.gather(
                    q_next_target_raw, -1, target_actions_next
                ).squeeze(-1)
                next_ent_bonus = 0.0

            if next_q_env.ndim > 1:
                next_q_env = next_q_env.mean(-1)
            if isinstance(next_ent_bonus, torch.Tensor) and next_ent_bonus.ndim > 1:
                next_ent_bonus = next_ent_bonus.mean(-1)

            # 3. DECOUPLED POPART: Track ONLY the environment returns
            # To break the geometric feedback loop, we update statistics using a target
            # that uses a fixed reference scale for the future value contribution.
            sigma_ref = getattr(self, "_seeded_ext_sigma", 1.0)
            mu_ref = self.ext_online.output_layer.mu.detach()
            next_q_stable = mu_ref + sigma_ref * q_next_online_norm.mean(dim=-1)
            
            env_target_stats = b_r_ext.unsqueeze(-1) + self.gamma * (1 - b_term).unsqueeze(-1) * next_q_stable
            self.ext_online.output_layer.update_stats(env_target_stats.detach())

            # 4. Construct Full Target for Network Training
            if self.munchausen or self.soft:
                next_v_soft = (pi_next * (q_next_target_raw - current_sigma * self.alpha * logpi_next)).sum(-1)
            else:
                next_v_soft = torch.gather(q_next_target_raw, -1, target_actions_next).squeeze(-1)
            
            if next_v_soft.ndim > 1:
                pass # Delay mean to allow per-head PopArt stats
                
            m_r_view = m_r
            online_ext_target = (
                b_r_ext.unsqueeze(-1) + m_r_view + self.gamma * (1 - b_term).unsqueeze(-1) * next_v_soft
            )

        target_for_stats_ext = online_ext_target.detach()  # maintain [B, D]
        # self.ext_online.output_layer.update_stats(target_for_stats_ext) # Removed: already updated with env_target
        # if self.delayed_target:

        td_target_norm = self.ext_online.output_layer.normalize(target_for_stats_ext)
        if td_target_norm.ndim > 1:
            td_target_norm = td_target_norm.mean(-1)

        q_ext_now_norm = self.ext_online(b_obs, normalized=True)
        q_selected_norm = torch.gather(q_ext_now_norm, -1, b_actions_idx).squeeze(
            -1
        )  # [B, D]
        drift_penalty = 0.0
        if q_selected_norm.ndim > 1:
            # drift_penalty = q_selected_norm.pow(2).mean() * 1e-3
            q_selected_norm = q_selected_norm.mean(-1) # Changed to sum
        assert (
            q_selected_norm.shape == td_target_norm.shape
        ), f"Shape mismatch: q_selected_norm {q_selected_norm.shape}, td_target_norm {td_target_norm.shape}"
        extrinsic_loss = (
            torch.nn.functional.mse_loss(q_selected_norm, td_target_norm)
            #+ drift_penalty
        )

        self.optim.zero_grad()
        extrinsic_loss.backward()
        if hasattr(self, "ext_online"):
            torch.nn.utils.clip_grad_norm_(self.ext_online.parameters(), max_norm=10.0)
        self.optim.step()

        # --- ALPHA AUTO-TUNING UPDATE ---
        alpha_loss = torch.tensor(0.0, device=self.device)
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
            alpha_loss = 0.0
            if self.autotune:
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
                pass # Delay mean to allow per-head PopArt stats
            # Since b_r_int might be [B] or [B,1] and int_q_next_target might be [B, D]
            b_r_int_v = b_r_int.view(-1).unsqueeze(-1) if b_r_int.ndim == 1 else b_r_int
            int_td_target = b_r_int_v + self.gamma * int_q_next_target

        target_for_stats_int = int_td_target.detach()
        self.int_online.output_layer.update_stats(target_for_stats_int)
        # if self.delayed_target:
        #    self.int_target.output_layer.sigma.copy_(self.int_online.output_layer.sigma)
        #    self.int_target.output_layer.mu.copy_(self.int_online.output_layer.mu)

        int_td_target_norm = self.int_online.output_layer.normalize(
            target_for_stats_int
        )
        if int_td_target_norm.ndim > 1:
            int_td_target_norm = int_td_target_norm.mean(-1)
            
        int_q_now_norm = self.int_online(b_obs, normalized=True)
        int_q_selected_norm = torch.gather(int_q_now_norm, -1, b_actions_idx).squeeze(
            -1
        )
        drift_penalty_int = 0.0
        if int_q_selected_norm.ndim > 1:
            #drift_penalty_int = int_q_selected_norm.pow(2).mean() * 1e-3
            int_q_selected_norm = int_q_selected_norm.mean(-1) # Changed to Sum
        assert (
            int_q_selected_norm.shape == int_td_target_norm.shape
        ), f"Shape mismatch: int_q_selected_norm {int_q_selected_norm.shape}, int_td_target_norm {int_td_target_norm.shape}"
        intrinsic_loss = (
            torch.nn.functional.mse_loss(int_q_selected_norm, int_td_target_norm)
            #+ drift_penalty_int
        )

        self.int_optim.zero_grad()
        intrinsic_loss.backward()
        self.int_optim.step()
        self.update_target()

        if self.step % 1000 == 0:
            # tracking
            if hasattr(self, "int_online"):
                torch.nn.utils.clip_grad_norm_(
                    self.int_online.parameters(), max_norm=10.0
                )

            if isinstance(b_r_int, torch.Tensor):
                r_int_log = float(b_r_int.mean().item())
            else:
                r_int_log = 0.0

            if isinstance(rnd_loss, torch.Tensor):
                rnd_loss = rnd_loss.item()
            self.last_losses = {
                "extrinsic": float(extrinsic_loss.item()),
                "intrinsic": float(intrinsic_loss.item()),
                "rnd": float(rnd_loss),
                "avg_r_int": r_int_log,
                "alpha": self.alpha,
                "alpha_loss": (float(alpha_loss.item()) if isinstance(alpha_loss, torch.Tensor) else float(alpha_loss)),
                "batch_nonzero_r_frac": float((b_r_ext != 0).float().mean().item()),
                "target_mean": float(target_for_stats_ext.mean().item()),
                "env_target_stats_std": float(env_target_stats.std().item()),
                "td_target_norm_mean": float(td_target_norm.abs().mean().detach().item()),
                "Beta": float(self.Beta),
                "Q_ext_mean": float(q_ext_now_norm.mean().item()),
                "Q_int_mean": (
                    float(int_q_now_norm.mean().item())
                    if "int_q_now_norm" in locals()
                    else 0.0
                ),
                "last_eps": float(self.last_eps),
                "entropy": current_entropy if self.soft else 0.0,
                "cur_sigma": float(current_sigma.item()),
                "cur_mu": float(self.ext_target.output_layer.mu.item()),
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
        action_mask: Optional[torch.Tensor] = None,
    ):
        self.last_eps = eps
        is_batched = obs.ndim > self.obs_ndim
        obs_b = obs if is_batched else obs.unsqueeze(0)
        batch_size = obs_b.size(0)

        # Force random actions during burn-in to ensure diverse buffer data
        if self.step < self.burn_in_updates:
            if action_mask is not None:
                mask = action_mask if action_mask.ndim > 1 else action_mask.unsqueeze(0)
                actions = []
                for i in range(batch_size):
                    valid_indices = torch.where(mask[i] == 1)[0]
                    if valid_indices.numel() > 0:
                        idx = torch.randint(0, valid_indices.numel(), (1,), device=obs_b.device)
                        actions.append(valid_indices[idx])
                    else:
                        actions.append(torch.tensor([0], device=obs_b.device))
                actions = torch.cat(actions)
            else:
                actions = torch.randint(
                    0,
                    self.n_action_bins,
                    (batch_size, self.n_action_dims),
                    device=obs_b.device,
                )
            if is_batched:
                return actions.tolist()
            else:
                return actions.squeeze(0).tolist()

        with torch.no_grad():
            q_ext = self.ext_online(obs_b, normalized=True)  # [B,D,Bins] or [B,n_actions]
            
            if self.Beta > 0.0:
                int_q = self.int_online(obs_b, normalized=True)
                q_ext = (1.0 - self.Beta) * q_ext + self.Beta * int_q

            # Apply action mask if provided
            if action_mask is not None:
                mask = action_mask if action_mask.ndim > 1 else action_mask.unsqueeze(0)
                # Ensure mask is same device as q_ext
                mask = mask.to(q_ext.device)
                q_ext = q_ext.clone()
                # For categorical sampling (soft/munchausen) and argmax, -1e9 works.
                if q_ext.ndim == 3:
                    # If mask is [B, Bins] and q_ext is [B, 1, Bins], this expands correctly.
                    # For true MultiDiscrete, mask would need to be [B, D, Bins].
                    if mask.ndim == 2 and q_ext.shape[1] == 1:
                        mask = mask.unsqueeze(1)
                    q_ext[mask.expand_as(q_ext) == 0] = -1e9
                else:
                    q_ext[mask == 0] = -1e9

            if self.soft or self.munchausen:
                actions = torch.distributions.Categorical(
                    logits=q_ext / self.alpha
                ).sample()
            else:
                actions = torch.argmax(q_ext, dim=-1)
                rand_vals = torch.rand(batch_size, device=obs_b.device)
                explore_mask = (rand_vals < min_ent) | (rand_vals < eps)

                if explore_mask.any():
                    if action_mask is not None:
                        # Explore only among valid actions
                        mask = action_mask if action_mask.ndim > 1 else action_mask.unsqueeze(0)
                        mask = mask.to(q_ext.device)
                        explore_actions = []
                        for i in range(batch_size):
                            valid_indices = torch.where(mask[i] == 1)[0]
                            if valid_indices.numel() > 0:
                                idx = torch.randint(0, valid_indices.numel(), (1,), device=obs_b.device)
                                explore_actions.append(valid_indices[idx])
                            else:
                                # Fallback if mask is all zeros
                                explore_actions.append(torch.tensor([0], device=obs_b.device))
                        explore_actions = torch.cat(explore_actions)
                    else:
                        explore_actions = torch.randint(
                            0,
                            self.n_action_bins,
                            (batch_size, self.n_action_dims),
                            device=obs_b.device,
                        )

                    # The masked-explore branch builds explore_actions as [B], while
                    # argmax over a 3D q_ext [B,D,Bins] makes `actions` [B,D]. Align the
                    # shapes so torch.where doesn't broadcast to [B,B].
                    if explore_actions.shape != actions.shape:
                        explore_actions = explore_actions.reshape(actions.shape)

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
        alpha: float = 0.03,
        polyak_tau: float = 0.03,
        l_clip: float = -1.0,
        soft: bool = False,
        munchausen_constant: float = 0.9,
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
        target_entropy_frac: float = 0.2,
    ):
        # Fraction of max entropy (ln(bins)) the soft-Q alpha autotuner targets. 0.2 is
        # an exploitative single-agent default; multi-agent Nash on symmetric games (e.g.
        # RPS) needs near-max entropy, so the MA runner raises this toward ~1.0.
        self._target_entropy_frac = float(target_entropy_frac)
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

        self.n_quantiles = 32
        self.n_target_quantiles = 32
        self.autotune = True
        # if self.munchausen:
        #     self.autotune = False
        #     self.alpha = 0.03
        # else:
        #     self.autotune = True

        # --- ALPHA AUTOTUNER SETUP ---
        if self.soft:
            max_ent = np.log(self.n_action_bins)  # self.n_action_dims *
            self.target_entropy = self._target_entropy_frac * max_ent
            # Honor the constructor alpha as the starting temperature (was hardcoded
            # 0.03/0.05). The MA runner passes a higher --dqn_alpha so the soft policy
            # is actually anchored; single-agent default stays 0.03.
            initial_alpha = float(alpha)
            self.log_alpha = nn.Parameter(torch.tensor([np.log(initial_alpha)], device=self.device, requires_grad=True))
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr * 0.1)
            self.alpha = self.log_alpha.exp().item()

        print(f"self soft {self.soft} selfmunch: {self.munchausen} mc {self.munchausen_constant}")

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
        # Try loss times action dims to stop it from scaling down. 
        #loss = (torch.abs(taus_expanded - I_) * huber).mean()
        loss = (torch.abs(taus_expanded - I_) * huber).sum(dim=1).mean()
        return loss

    def update(self, batch_size=None, step=None):
        self.step += 1
        current_entropy = 0.0

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

        # First post-burn-in step: hard-start PopArt from raw reward stats
        # scaled to infinite-horizon returns.
        if not self._popart_burn_in_seeded and self.burn_in_updates > 0:
            self._seed_popart_from_buffer(
                ext_layer=self.ext_online.output_layer,
                int_layer=self.int_online.output_layer,
            )
            if self.delayed_target:
                self.ext_target.load_state_dict(self.ext_online.state_dict())
                self.int_target.load_state_dict(self.int_online.state_dict())

        b_obs = b_obs.to(self.device, non_blocking=True)
        b_a = b_a.to(self.device, non_blocking=True).long()
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

        current_sigma = self.ext_target.output_layer.sigma.detach()

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
            online_next_q_norm = self.ext_online(b_next_obs, taus, normalized=True)#self.ext_target(b_next_obs, taus, normalized=True) if self.delayed_target else self.ext_online(b_next_obs, taus, normalized=True)
            online_next_q_norm = online_next_q_norm.view(dist_q_shape).mean(dim=1) # [B, D, Bins]

            # Target Net Quantiles -> For target values
            t_net = self.ext_target if self.delayed_target else self.ext_online
            target_quantiles_all = t_net(
                b_next_obs, target_taus, normalized=False
            )  # [B, Nt, D, Bins]

            m_r = 0.0
            ent_bonus = 0.0

            if self.munchausen or self.soft:
                logpi_next = torch.clamp(
                    torch.log_softmax(online_next_q_norm / self.alpha, dim=-1),
                    min=-1e8,
                )
                pi_next = torch.exp(logpi_next)
                # Entropy bonus per head: [B, D]
                ent_bonus_raw = -(pi_next * logpi_next).sum(dim=-1)
                ent_bonus = current_sigma * self.alpha * ent_bonus_raw
                current_entropy = ent_bonus_raw.mean()

                # ENV-ONLY mixed target (Approximation by subtracting immediate next penalty)
                mixed_target_full = (pi_next.unsqueeze(1) * target_quantiles_all).sum(
                    dim=-1
                )
                mixed_target_env = mixed_target_full - ent_bonus.unsqueeze(1)
            else:
                target_actions = online_next_q_norm.argmax(dim=-1) # [B, D]
                action_idx = (
                    target_actions.unsqueeze(1)
                    .unsqueeze(-1)
                    .expand(-1, self.n_target_quantiles, -1, 1)
                )
                mixed_target_env = torch.gather(
                    target_quantiles_all, -1, action_idx
                ).squeeze(
                    -1
                )  # [B, Nt, D]
                ent_bonus = 0.0

            if self.munchausen:
                t_expected = torch.linspace(
                    0.01, 0.99, self.n_quantiles, device=self.device
                )
                t_expected = t_expected.unsqueeze(0).expand(batch_size, -1)
                q_ext_norm_now = self.ext_online(
                    b_obs, t_expected, normalized=True
                ).mean(dim=1) if not self.delayed_target else self.ext_target(
                    b_obs, t_expected, normalized=True
                ).mean(dim=1)# [B, D, Bins]

                logpi_now = torch.clamp(torch.log_softmax(q_ext_norm_now / self.alpha, dim=-1),min=-1e8)
                selected_logpi = torch.gather(logpi_now, -1, b_actions_idx).squeeze(-1) # [B, D]

                # Munchausen reward per head: [B, D]
                # Sigma remains outside clamp
                m_r = (
                    current_sigma * self.munchausen_constant * torch.clamp(self.alpha * selected_logpi, min=self.l_clip, max=0)
                )

            # Ensure m_r and ent_bonus are tensors of correct shape for broadcasting
            if not isinstance(m_r, torch.Tensor):
                m_r = torch.zeros(batch_size, self.n_action_dims, device=self.device)
            if not isinstance(ent_bonus, torch.Tensor):
                ent_bonus = torch.zeros(batch_size, self.n_action_dims, device=self.device)

            # 2. DECOUPLED POPART: Track ONLY the environment returns
            # To break the geometric feedback loop, update stats using a stable target
            # independent of current explosive sigma.
            sigma_ref = getattr(self, "_seeded_ext_sigma", 1.0)
            mu_ref = self.ext_online.output_layer.mu.detach()
            # online_next_q_norm is [B, D, Bins]. Mean over bins for expected value.
            next_q_stable = mu_ref + sigma_ref * online_next_q_norm.mean(dim=-1) # [B, D]
            
            env_target_stats = b_r_ext.unsqueeze(-1) + (1 - b_term).unsqueeze(-1) * self.gamma * next_q_stable
            self.ext_online.output_layer.update_stats(env_target_stats.detach())

            # 3. Compute Full Target & Normalize
            # Full target = r_env + m_r + gamma * (Q_env_next + Ent_next)
            # Use the most accurate recursive target for network training.
            if self.munchausen or self.soft:
                # Value including entropy: [B, Nt, D]
                # ent_bonus_raw is -(pi_next * logpi_next).sum(-1) -> [B, D]
                next_v_soft = target_quantiles_all + current_sigma * self.alpha * (-logpi_next).unsqueeze(1)
                mixed_target_full = (pi_next.unsqueeze(1) * next_v_soft).sum(dim=-1)
            else:
                target_actions = online_next_q_norm.argmax(dim=-1)
                action_idx = (
                    target_actions.unsqueeze(1)
                    .unsqueeze(-1)
                    .expand(-1, self.n_target_quantiles, -1, 1)
                )
                mixed_target_full = torch.gather(
                    target_quantiles_all, -1, action_idx
                ).squeeze(-1)

            b_r_final = b_r_ext.unsqueeze(-1) + m_r # [B, D]
            target_values = b_r_final.unsqueeze(1) + (1 - b_term).view(batch_size, 1, 1) * self.gamma * mixed_target_full
        
        # Apply PopArt stats tracking over target distributions
        # self.ext_online.output_layer.update_stats(target_values.detach().mean(dim=1)) # Removed: already updated with env_target
        cur_sigma = self.ext_online.output_layer.sigma
        cur_mu = self.ext_online.output_layer.mu
        #self.ext_online.output_layer.update_stats(target_values.detach())
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
        
        # Reshape to treat heads as independent samples for quantile loss
        pred_chosen_flat = pred_chosen.transpose(1, 2).reshape(batch_size * self.n_action_dims, self.n_quantiles)
        target_values_norm_flat = target_values_norm.transpose(1, 2).reshape(batch_size * self.n_action_dims, self.n_target_quantiles)
        taus_pred_flat = taus_pred.unsqueeze(1).expand(-1, self.n_action_dims, -1).reshape(batch_size * self.n_action_dims, self.n_quantiles)

        extrinsic_loss = self._quantile_huber_loss(pred_chosen_flat, target_values_norm_flat, taus_pred_flat)

        self.optim.zero_grad()
        extrinsic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ext_online.parameters(), max_norm=10.0)
        self.optim.step()

        # --- ALPHA AUTO-TUNING UPDATE ---
        if self.soft and self.autotune:
            alpha_loss = torch.tensor(0.0, device=self.log_alpha.device)
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
                ).mean(dim=1) # [B, D, Bins]
                target_actions_int = online_next_q_int_norm.argmax(dim=-1) # [B, D]

                t_net_int = self.int_target if self.delayed_target else self.int_online
                int_target_all = t_net_int(
                    b_next_obs, int_target_taus, normalized=False
                ) # [B, Nt, D, Bins]

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

                # No terminal mask for intrinsic reward
                int_target_values = b_r_int.unsqueeze(1).unsqueeze(2) + self.gamma * mixed_target_int # [B, Nt, D]

            # Mean -1 to get expected value so popart tracks target variance not env varaince
            self.int_online.output_layer.update_stats(
                int_target_values.detach().mean(dim=1)
            )
            #self.ext_online.output_layer.update_stats(target_values.detach().mean(dim=1))
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
            
            # Reshape for head-wise quantile loss
            int_pred_chosen_flat = int_pred_chosen.transpose(1, 2).reshape(batch_size * self.n_action_dims, self.n_quantiles)
            int_target_values_norm_flat = int_target_values_norm.transpose(1, 2).reshape(batch_size * self.n_action_dims, self.n_target_quantiles)
            int_taus_pred_flat = int_taus_pred.unsqueeze(1).expand(-1, self.n_action_dims, -1).reshape(batch_size * self.n_action_dims, self.n_quantiles)

            intrinsic_loss = self._quantile_huber_loss(
                int_pred_chosen_flat, int_target_values_norm_flat, int_taus_pred_flat
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

        if self.step%1000==0:
            # ========================================================
            # Tracking identical to EV
            # ========================================================
            if isinstance(b_r_int, torch.Tensor):
                r_int_log = float(b_r_int.mean().item())
            else:
                r_int_log = 0.0

            if isinstance(rnd_loss, torch.Tensor):
                rnd_loss = rnd_loss.item()

            q_ext_now_norm = (
                quantiles_pred.mean(dim=1)
                if "quantiles_pred" in locals()
                else torch.tensor(0.0)
            )
            if self.soft and self.autotune:
                alpha_loss_val = float(alpha_loss.item())
            else:
                alpha_loss_val = 0.0
            self.last_losses = {
                "extrinsic": float(extrinsic_loss.item()),
                "intrinsic": float(intrinsic_loss.item()),
                "rnd": float(rnd_loss),
                "avg_r_int": r_int_log,
                "alpha_loss": alpha_loss_val,
                "alpha":self.alpha,
                "mr":m_r.mean() if self.munchausen else 0.0,
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
                "entropy": current_entropy,
                "cur_sigma": cur_sigma.item(),
                "cur_mu": cur_mu.item(),
            }
            print(self.last_losses)
        return float(extrinsic_loss.item())

    def sample_action(
        self,
        obs: torch.Tensor,
        eps: float,
        step: int,
        n_steps: int = 100000,
        min_eps=0.01,
        verbose: bool = False,
        action_mask: Optional[torch.Tensor] = None,
    ):
        self.last_eps = eps
        is_batched = obs.ndim > self.obs_ndim
        obs_b = obs if is_batched else obs.unsqueeze(0)
        batch_size = obs_b.size(0)

        # Force random actions during burn-in to ensure diverse buffer data
        if self.step < self.burn_in_updates:
            if action_mask is not None:
                mask = action_mask if action_mask.ndim > 1 else action_mask.unsqueeze(0)
                actions = []
                for i in range(batch_size):
                    valid_indices = torch.where(mask[i] == 1)[0]
                    if valid_indices.numel() > 0:
                        idx = torch.randint(0, valid_indices.numel(), (1,), device=obs_b.device)
                        actions.append(valid_indices[idx])
                    else:
                        actions.append(torch.tensor([0], device=obs_b.device))
                actions = torch.cat(actions)
            else:
                actions = torch.randint(
                    0,
                    self.n_action_bins,
                    (batch_size, self.n_action_dims),
                    device=obs_b.device,
                )
            if is_batched:
                return actions.tolist()
            else:
                return actions.squeeze(0).tolist()

        with torch.no_grad():
            taus = self._sample_taus(batch_size, self.n_quantiles, obs_b.device)
            ext_q = self.ext_online(obs_b, taus, normalized=True).mean(
                dim=1
            )  # [B,D,Bins]
                
            if self.Beta > 0.0:
                int_taus = self._sample_taus(batch_size, self.n_quantiles, obs_b.device)
                int_q = self.int_online(obs_b, int_taus, normalized=True).mean(dim=1)
                ext_q = (1.0 - self.Beta) * ext_q + self.Beta * int_q

            # Apply action mask if provided
            if action_mask is not None:
                mask = action_mask if action_mask.ndim > 1 else action_mask.unsqueeze(0)
                mask = mask.to(ext_q.device)
                ext_q = ext_q.clone()
                # ext_q can be [B, n_quantiles, n_actions] or [B, n_actions]
                if ext_q.ndim == 3:
                    # Broadcast mask to quantiles
                    if mask.ndim == 2 and ext_q.shape[1] == 1:
                        mask = mask.unsqueeze(1)
                    ext_q[mask.expand_as(ext_q) == 0] = -1e9
                else:
                    ext_q[mask == 0] = -1e9

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
                    if action_mask is not None:
                        mask = action_mask if action_mask.ndim > 1 else action_mask.unsqueeze(0)
                        mask = mask.to(ext_q.device)
                        explore_actions = []
                        for i in range(batch_size):
                            valid_indices = torch.where(mask[i] == 1)[0]
                            if valid_indices.numel() > 0:
                                idx = torch.randint(0, valid_indices.numel(), (1,), device=obs_b.device)
                                explore_actions.append(valid_indices[idx])
                            else:
                                explore_actions.append(torch.tensor([0], device=obs_b.device))
                        explore_actions = torch.cat(explore_actions)
                    else:
                        explore_actions = torch.randint(
                            0,
                            self.n_action_bins,
                            (batch_size, self.n_action_dims),
                            device=obs_b.device,
                        )
                    # The masked-explore branch builds explore_actions as [B], while
                    # argmax over a 3D ext_q [B,D,Bins] makes `actions` [B,D]. Align the
                    # shapes so torch.where doesn't broadcast to [B,B].
                    if explore_actions.shape != actions.shape:
                        explore_actions = explore_actions.reshape(actions.shape)
                    actions = torch.where(
                        explore_mask.unsqueeze(1) if actions.ndim > 1 else explore_mask,
                        explore_actions,
                        actions,
                    )

            if is_batched:
                return actions.tolist()
            else:
                return actions.squeeze(0).tolist()
