import torch
import random
from typing import Optional
from torch.utils.tensorboard import SummaryWriter
from RandomDistilation import RNDModel, RunningMeanStd
from RainbowNetworks import EV_Q_Network, IQN_Network


class RainbowDQN:
    """Maintains online (current) and target Q_Networks and training logic."""

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
        popart: bool = False,
        Beta: float = 0.0,
        # Delayed target usage (pillar 5). If False, use online net as target.
        delayed: bool = True,
        # Entropy regularization on current policy (online net)
        ent_reg_coef: float = 0.0,
        # Intrinsic/RND configs
        rnd_output_dim: int = 128,
        rnd_lr: float = 1e-3,
        intrinsic_lr: float = 1e-3,
        int_r_clamp: float = 5.0,
        ext_r_clamp: float = 5.0,
        Beta_half_life_steps: Optional[int] = None,
        norm_obs: bool = True,
        burn_in_updates: int = 0,
    ):
        self.popart = popart
        self.ext_online = IQN_Network(
            input_dim,
            n_action_dims,
            n_action_bins,
            hidden_layer_sizes=hidden_layer_sizes,
            dueling=dueling,
            popart=popart,
        ).float()
        self.ext_target = IQN_Network(
            input_dim,
            n_action_dims,
            n_action_bins,
            hidden_layer_sizes=hidden_layer_sizes,
            dueling=dueling,
            popart=popart,
        ).float()
        # Intrinsic Q networks (separate head trained on intrinsic rewards only)
        self.int_online = IQN_Network(
            input_dim,
            n_action_dims,
            n_action_bins,
            hidden_layer_sizes=hidden_layer_sizes,
            dueling=dueling,
            popart=popart,
        ).float()
        self.int_target = IQN_Network(
            input_dim,
            n_action_dims,
            n_action_bins,
            hidden_layer_sizes=hidden_layer_sizes,
            dueling=dueling,
            popart=popart,
        ).float()
        self.beta_half_life_steps = Beta_half_life_steps
        self.burn_in_updates = burn_in_updates
        self.norm_obs = norm_obs
        self.int_r_clamp = int_r_clamp
        self.ext_r_clamp = ext_r_clamp
        self.alpha = alpha
        self.tau = tau
        self.polyak_tau = polyak_tau
        self.l_clip = l_clip
        self.soft = soft
        self.munchausen = munchausen
        self.Thompson = Thompson
        self.ext_target.requires_grad_(False)
        self.ext_target.load_state_dict(self.ext_online.state_dict())
        self.int_target.requires_grad_(False)
        self.int_target.load_state_dict(self.int_online.state_dict())

        self.optim = torch.optim.Adam(self.ext_online.parameters(), lr=lr)
        self.int_optim = torch.optim.Adam(self.int_online.parameters(), lr=intrinsic_lr)
        self.gamma = gamma
        # IQN specific hyperparameters (kept static; could be exposed later)
        self.n_quantiles = 32
        self.n_target_quantiles = 32
        self.n_cosines = 64
        self.n_action_dims = n_action_dims
        self.n_action_bins = n_action_bins
        self.n_actions = n_action_dims * n_action_bins  # legacy single number
        self.Beta = Beta
        self.start_Beta = Beta
        self.delayed_target = delayed
        self.ent_reg_coef = ent_reg_coef

        # RND intrinsic reward model and normalizers
        self.rnd = RNDModel(input_dim, rnd_output_dim).float()
        self.rnd_optim = torch.optim.Adam(self.rnd.predictor.parameters(), lr=rnd_lr)
        # Running stats: observations and intrinsic reward magnitude
        self.obs_rms = RunningMeanStd(shape=(input_dim,))
        # Scalar running stats for intrinsic reward
        self.int_rms = RunningMeanStd(shape=())
        self.ext_rms = RunningMeanStd(shape=())
        # Optional TensorBoard writer
        self.tb_writer = None
        self.tb_prefix = "agent"
        self.step = 0
        self.last_eps = 1.0
        self.last_losses = {}

    def attach_tensorboard(self, writer: SummaryWriter, prefix: str = "agent"):
        """Attach a TensorBoard SummaryWriter to enable internal logging during updates."""
        self.tb_writer = writer
        self.tb_prefix = prefix

    def update_target(self):
        """Polyak averaging: target = (1 - tau) * target + tau * online."""
        if not self.delayed_target:
            return  # no delayed targets requested
        with torch.no_grad():
            for tp, op in zip(
                self.ext_target.parameters(), self.ext_online.parameters()
            ):
                tp.data.mul_(1.0 - self.polyak_tau).add_(op.data, alpha=self.polyak_tau)
            for tp, op in zip(
                self.int_target.parameters(), self.int_online.parameters()
            ):
                tp.data.mul_(1.0 - self.polyak_tau).add_(op.data, alpha=self.polyak_tau)

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
        """Quantile Huber loss.
        pred, target: [B, N] (pred quantiles for chosen actions, target quantiles)
        taus: [B, N] sampled quantile fractions corresponding to pred.
        We broadcast to pairwise then average.
        """
        B, N = pred.shape
        # Pairwise differences δ_{ij} = target_j - pred_i
        td = target.unsqueeze(1) - pred.unsqueeze(2)  # [B, N, N]
        abs_td = torch.abs(td)
        huber = torch.where(
            abs_td <= kappa, 0.5 * td.pow(2), kappa * (abs_td - 0.5 * kappa)
        )
        # Indicator for td < 0 (for quantile weighting)
        I = (td < 0).float()
        taus_expanded = taus.unsqueeze(2)  # [B,N,1]
        loss = (torch.abs(taus_expanded - I) * huber).mean()
        return loss

    def update(
        self, obs, a, r, next_obs, term, batch_size=None, step=0, extrinsic_only=False
    ):
        self.update_running_stats(next_obs, r)
        r = self.ext_rms.normalize(r, clip_range=self.ext_r_clamp).to(device=obs.device)
        # Sample a random minibatch
        if batch_size is None:
            idx = torch.arange(0, len(r))
            batch_size = len(r)
        else:
            idx = torch.randint(low=0, high=step, size=(batch_size,))
        b_next_obs = next_obs[idx]

        self.step += 1
        if self.step < self.burn_in_updates:
            # During burn-in, do not train RL networks; only RND and running stats
            rnd_errors, rnd_loss = self._update_RND(b_next_obs, batch_norm=True)
            return 0.0
        else:
            rnd_errors, rnd_loss = self._update_RND(b_next_obs)
        b_obs = obs[idx]
        b_r = r[idx]
        b_term = term[idx]
        b_actions = a[idx]

        if self.beta_half_life_steps is not None and self.beta_half_life_steps > 0:
            # Exponentially decay Beta towards zero
            self.Beta = self.start_Beta * (
                0.5 ** (self.step / self.beta_half_life_steps)
            )

        b_r_int = self._int_reward(b_r, rnd_errors)
        extrinsic_loss, munchausen_reward, entropy_loss = self._rl_update_extrinsic(
            b_obs, b_actions, b_r, b_next_obs, b_term, batch_size
        )
        intrinsic_loss = torch.tensor(0.0)
        if self.Beta > 0.0:
            assert isinstance(b_r_int, torch.Tensor)
            intrinsic_loss = self._rl_update_intrinsic(
                b_obs, b_actions, b_r_int, b_next_obs, batch_size, b_term=b_term
            )
        if isinstance(entropy_loss, torch.Tensor):
            entropy_loss = float(entropy_loss.item())
        with torch.no_grad():
            taus = self._sample_taus(b_obs.shape[0], self.n_quantiles, b_obs.device)
            Q_ext = self.ext_online(b_obs, taus).mean(dim=1)
            Q_int = self.int_online(b_obs, taus).mean(dim=1)
        # Store last auxiliary losses for logging (optional)
        self.last_losses = {
            "extrinsic": float(extrinsic_loss.item()),
            "intrinsic": float(intrinsic_loss.item()),
            "rnd": float(rnd_loss.item()),
            "entropy_reg": entropy_loss,
            "abs_r_ext": float(b_r.abs().mean().item()),
            "avg_r_int": (
                b_r_int.mean().item() if isinstance(b_r_int, torch.Tensor) else 0.0
            ),
            "munchausen_r": float(munchausen_reward),
            "Beta": float(self.Beta),
            "Q_ext_mean": float(Q_ext.mean().item()),
            "Q_int_mean": float(Q_int.mean().item()),
            "last_eps": float(self.last_eps),
        }

        # Inline TensorBoard logging if writer attached
        if self.tb_writer is not None:
            try:
                for k, v in self.last_losses.items():
                    if isinstance(v, (int, float)):
                        self.tb_writer.add_scalar(
                            f"{self.tb_prefix}/{k}", float(v), self.step
                        )
            except Exception:
                pass

        return extrinsic_loss.item()

    @torch.no_grad()
    def update_running_stats(self, next_obs: torch.Tensor, r: torch.Tensor):
        """Update observation and intrinsic reward running stats with a single environment step.

        This does not train any networks; it only updates the normalizers so that
        later batch updates can use stable statistics.
        """
        # Update observation stats
        x64 = next_obs.to(dtype=torch.float64, device=self.obs_rms.mean.device)
        self.obs_rms.update(x64)
        # Compute a single-step intrinsic error to update intrinsic RMS
        norm_x64 = self.obs_rms.normalize(x64)
        # Ensure batch dimension for RND
        if norm_x64.ndim == 1:
            norm_x64 = norm_x64.unsqueeze(0)
        rnd_err = self.rnd(norm_x64.to(dtype=torch.float32)).squeeze()
        self.int_rms.update(rnd_err.to(dtype=torch.float64))
        self.ext_rms.update(r.to(dtype=torch.float64))

    def sample_action(
        self,
        obs: torch.Tensor,
        eps: float,
        step: int,
        n_steps: int = 100000,
        min_eps=0.01,
    ):
        """Return vector of length D with selected bin indices for each action dimension."""
        r = random.random()
        self.last_eps = eps
        if r < min_eps or ((not (self.soft or self.munchausen)) and r < eps):
            return torch.randint(0, self.n_action_bins, (self.n_action_dims,)).tolist()
        with torch.no_grad():
            obs_b = obs.unsqueeze(0) if obs.ndim == 1 else obs
            taus = self._sample_taus(obs_b.shape[0], self.n_quantiles, obs_b.device)
            ext_q = self.ext_online(obs_b, taus).mean(dim=1)  # [B,D,Bins]
            if self.Beta > 0.0:
                int_taus = self._sample_taus(
                    obs_b.shape[0], self.n_quantiles, obs_b.device
                )
                int_q = self.int_online(obs_b, int_taus).mean(dim=1)  # [B,D,Bins]
                q_comb = ext_q + self.Beta * int_q

            else:
                q_comb = ext_q
            if self.Thompson:
                g = -torch.log(-torch.log(torch.rand_like(q_comb)))
                q_comb = q_comb + g
            # eps_curr = 1 - step / n_steps

            if self.soft or self.munchausen:
                # Sample per dimension from softmax policy
                actions = []
                for d in range(self.n_action_dims):
                    logits_d = q_comb[0, d] / self.tau
                    dst = torch.distributions.Categorical(logits=logits_d)
                    a_d = dst.sample().item()
                    actions.append(a_d)
                return actions
            # Greedy per dimension
            return torch.argmax(q_comb.squeeze(0), dim=-1).tolist()

    def _update_RND(self, next_obs: torch.Tensor, batch_norm=True):
        # 1) Intrinsic reward via RND (train predictor to reduce novelty on visited states)
        # Use existing running stats (updated per-environment step) to normalize inputs for RND
        if batch_norm:
            norm_next_obs_f32 = (next_obs - next_obs.mean(dim=0, keepdim=True)) / (
                next_obs.std(dim=0, keepdim=True) + 1e-6
            )
        elif self.norm_obs:
            with torch.no_grad():
                norm_next_obs = self.obs_rms.normalize(next_obs.to(dtype=torch.float64))
            norm_next_obs_f32 = norm_next_obs.to(dtype=torch.float32)
        else:
            norm_next_obs_f32 = next_obs.to(dtype=torch.float32)

        # Train RND predictor
        rnd_errors = self.rnd(norm_next_obs_f32)  # [B]
        rnd_loss = rnd_errors.mean()
        self.rnd_optim.zero_grad()
        rnd_loss.backward()
        self.rnd_optim.step()
        return rnd_errors, rnd_loss

    def _int_reward(self, b_r: torch.Tensor, rnd_errors: torch.Tensor):
        # Normalize intrinsic reward magnitude using running stats (updated per step)
        with torch.no_grad():
            norm_rnd_err = self.int_rms.normalize(rnd_errors.to(dtype=torch.float64))
            # Clamp to avoid extreme intrinsic rewards destabilizing training
            if self.int_r_clamp is not None and self.int_r_clamp > 0.0:
                norm_rnd_err = torch.clamp(
                    norm_rnd_err, max=self.int_r_clamp, min=-self.int_r_clamp
                )
            return norm_rnd_err.to(dtype=b_r.dtype)

    def _rl_update_extrinsic(
        self,
        b_obs: torch.Tensor,
        b_actions: torch.Tensor,
        b_r: torch.Tensor,
        b_next_obs: torch.Tensor,
        b_term: torch.Tensor,
        batch_size: int,
    ):
        b_r_final = b_r
        device = b_obs.device
        taus = self._sample_taus(batch_size, self.n_quantiles, device)

        with torch.no_grad():
            target_taus = self._sample_taus(batch_size, self.n_target_quantiles, device)
            # [B,Nt,D,Bins]
            t_net = self.ext_target if self.delayed_target else self.ext_online
            target_quantiles_all = t_net.forward(
                b_next_obs, target_taus, normalized=False
            )

            online_next_q_norm = self.ext_online.forward(
                b_next_obs, taus, normalized=True
            ).mean(
                dim=1
            )  # [B,D,Bins]
            if self.soft or self.munchausen:
                # Soft reward for future policy entropy
                logpi_next = torch.log_softmax(online_next_q_norm / self.tau, dim=-1)
                pi_next = torch.exp(logpi_next)  # [B,D,Bins]
                ent_bonus_per_dim = -(pi_next * logpi_next).sum(dim=-1)  # [B,D]
                ent_bonus = ent_bonus_per_dim.sum(dim=-1).unsqueeze(1)  # [B,1]

                # Quantile target expected values
                mixed_target = (
                    (pi_next.unsqueeze(1) * target_quantiles_all)
                    .sum(dim=-1)  # over action q values
                    .sum(dim=-1)  # over action dims
                )  # [B,Nt]s

            else:
                # Simplified target selection: directly take max over bins per dimension
                # 1. Calculate Mean Q-values for selection (average over quantiles/tau dim 1)
                # shape: [B, D, Bins]
                # Double DQN Logic: Use Online Network for selection
                target_actions = online_next_q_norm.argmax(dim=-1)  # [B, D]

                # 3. Gather the quantiles corresponding to the best action
                # Expand indices to match target_quantiles_all shape: [B, Nt, D, 1]
                action_idx = (
                    target_actions.unsqueeze(1)
                    .unsqueeze(-1)
                    .expand(-1, self.n_target_quantiles, -1, 1)
                )
                # Gather along the bins dimension
                mixed_target = torch.gather(
                    target_quantiles_all, 3, action_idx
                ).squeeze(
                    -1
                )  # [B, Nt, D]
                # Sum over action dimensions (if multi-dim actions)
                mixed_target = mixed_target.sum(dim=-1)  # [B, Nt]
                ent_bonus = 0.0

            # Munchausen augmentation on reward
            logpi_a = 0
            if b_actions.ndim == 1:
                b_actions_view = b_actions.view(batch_size, 1)
            else:
                b_actions_view = b_actions  # [B,D]

            # Do entropy loss and cache pi/logpi
            pi, logpi = None, None
            quantiles_norm = self.ext_online.forward(
                b_obs, taus, normalized=True
            )  # [B,N,D,Bins]
            e_loss = 0.0
            if self.ent_reg_coef > 0.0:
                # Policy over actions from expected values (mean quantiles)
                qm = quantiles_norm.mean(dim=1)
                logpi = torch.log_softmax(qm / self.tau, dim=-1)
                pi = torch.exp(logpi)  # [B,D,Bins]
                entropy_per_dim = (pi * logpi).sum(dim=-1)  # [B,D]
                entropy = entropy_per_dim.mean()  # average over dims
                e_loss = self.ent_reg_coef * entropy

            m_r = 0
            if self.munchausen or self.soft:
                if logpi is None:
                    qm = quantiles_norm.mean(dim=1)
                    logpi = torch.log_softmax(qm / self.tau, dim=-1)

                if self.munchausen:
                    with torch.no_grad():
                        # Gather logpi for taken actions
                        gather_m_hist = b_actions_view.unsqueeze(-1)
                        selected_logpi = torch.gather(logpi, -1, gather_m_hist).squeeze(
                            -1
                        )  # [B,D]
                        logpi_a = torch.clamp(
                            selected_logpi.mean(dim=-1), min=self.l_clip
                        )  # average over dims
                        m_r = self.alpha * self.tau * logpi_a
                        # print(b_r_final.device, m_r.device)
                        b_r_final = b_r_final + m_r
            # print(b_term.device, mixed_target.device, b_r_final.device)
            target_values = b_r_final.unsqueeze(1) + (1 - b_term).unsqueeze(
                1
            ) * self.gamma * (mixed_target + ent_bonus)
            # [B,Nt]

            # Popart happens before loss step to rescale properly
            if self.popart:
                self.ext_online.output_layer.update_stats(target_values)
                target_values = self.ext_online.output_layer.normalize(target_values)

        # Gather predicted quantiles for taken actions
        quantiles = self.ext_online.forward(
            b_obs, taus, normalized=True
        )  # [B,N,D,Bins]
        # Gather predicted quantiles per dimension and aggregate
        gather_index_pred = (
            b_actions_view.unsqueeze(1)
            .unsqueeze(-1)
            .expand(-1, self.n_quantiles, -1, 1)
        )
        pred_chosen_per_dim = torch.gather(quantiles, 3, gather_index_pred).squeeze(
            -1
        )  # [B,N,D]
        pred_chosen = pred_chosen_per_dim.sum(
            dim=-1
        )  # [B,N] aggregated joint quantiles
        # Quantile Huber loss
        loss = (
            self._quantile_huber_loss(
                pred_chosen, target_values.detach(), taus.detach()
            )
            + e_loss
        )
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        if isinstance(m_r, torch.Tensor):
            m_r = m_r.mean()
        return loss, m_r, e_loss

    def _rl_update_intrinsic(
        self,
        b_obs: torch.Tensor,
        b_actions: torch.Tensor,
        b_r_int: torch.Tensor,
        b_next_obs: torch.Tensor,
        batch_size: int,
        b_term: torch.Tensor,
    ):
        # Update intrinsic Q network on intrinsic rewards only
        device = b_obs.device
        int_taus = self._sample_taus(batch_size, self.n_quantiles, device)
        int_quantiles = self.int_online(b_obs, int_taus)  # [B,N,D,Bins]
        with torch.no_grad():
            int_target_taus = self._sample_taus(
                batch_size, self.n_target_quantiles, device
            )
            int_target_all = (
                self.int_target if self.delayed_target else self.int_online
            )(
                b_next_obs, int_target_taus
            )  # [B,Nt,D,Bins]
            # 1. Calculate Mean Q-values for selection (average over quantiles/tau dim 1)
            # shape: [B, D, Bins]
            # target_q_means = int_target_all.mean(dim=1)
            # online_q_means = int_quantiles.mean(dim=1)
            online_q_means = self.int_online(b_next_obs, int_taus).mean(dim=1)
            # 2. Select best action based on expected value
            # shape: [B, D]
            target_actions = online_q_means.argmax(dim=-1)
            # 3. Gather the quantiles corresponding to the best action
            # Expand indices to match target_quantiles_all shape: [B, Nt, D, 1]
            action_idx = (
                target_actions.unsqueeze(1)
                .unsqueeze(-1)
                .expand(-1, self.n_target_quantiles, -1, 1)
            )
            # Gather along the bins dimension
            mixed_target = torch.gather(int_target_all, -1, action_idx).squeeze(
                -1
            )  # [B, Nt, D]
            # Sum over action dimensions (if multi-dim actions)
            mixed_target = mixed_target.sum(dim=-1)  # [B, Nt]
            int_target_values = (
                b_r_int.unsqueeze(1)
                + self.gamma * (1 - b_term).unsqueeze(1) * mixed_target
            )

        # Gather current quantile values from taken actions
        if b_actions.ndim == 1:
            b_actions_view = b_actions.view(batch_size, 1)
        else:
            b_actions_view = b_actions  # [B,D]
        gather_index_int_pred = (
            b_actions_view.unsqueeze(1)
            .unsqueeze(-1)
            .expand(-1, self.n_quantiles, -1, 1)
        )
        int_pred_per_dim = torch.gather(
            int_quantiles, 3, gather_index_int_pred
        ).squeeze(
            -1
        )  # [B,N,D]
        int_pred_chosen = int_pred_per_dim.sum(dim=-1)  # [B,N]
        int_loss = self._quantile_huber_loss(
            int_pred_chosen, int_target_values.detach(), int_taus.detach()
        )

        self.int_optim.zero_grad()
        int_loss.backward()
        self.int_optim.step()
        return int_loss


class EVRainbowDQN:
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
        popart: bool = False,
        Beta: float = 0.0,
        delayed: bool = True,
        ent_reg_coef: float = 0.0,
        # Intrinsic/RND configs
        rnd_output_dim: int = 128,
        rnd_lr: float = 1e-3,
        intrinsic_lr: float = 1e-3,
        norm_obs: bool = True,
        burn_in_updates: int = 0,
        int_r_clip=5,
        ext_r_clip=5,
    ):
        self.last_losses = {}
        self.popart = popart
        self.int_r_clip = int_r_clip
        self.ext_r_clip = ext_r_clip
        self.ext_online = EV_Q_Network(
            input_dim,
            n_action_dims,
            n_action_bins,
            hidden_layer_sizes=hidden_layer_sizes,
            dueling=dueling,
            popart=popart,
        ).float()
        self.ext_target = EV_Q_Network(
            input_dim,
            n_action_dims,
            n_action_bins,
            hidden_layer_sizes=hidden_layer_sizes,
            dueling=dueling,
            popart=popart,
        ).float()
        # Intrinsic Q networks (separate head trained on intrinsic rewards only)
        self.int_online = EV_Q_Network(
            input_dim,
            n_action_dims,
            n_action_bins,
            hidden_layer_sizes=hidden_layer_sizes,
            dueling=dueling,
            popart=popart,
        ).float()
        self.int_target = EV_Q_Network(
            input_dim,
            n_action_dims,
            n_action_bins,
            hidden_layer_sizes=hidden_layer_sizes,
            dueling=dueling,
            popart=popart,
        ).float()
        self.norm_obs = norm_obs
        self.burn_in_updates = burn_in_updates
        self.alpha = alpha
        self.tau = tau
        self.polyak_tau = polyak_tau
        self.l_clip = l_clip
        self.soft = soft
        self.munchausen = munchausen
        self.Thompson = Thompson
        self.delayed_target = delayed
        self.Beta = Beta
        self.ent_reg_coef = ent_reg_coef
        self.gamma = gamma
        self.n_action_dims = n_action_dims
        self.n_action_bins = n_action_bins
        self.n_actions = n_action_dims * n_action_bins

        self.ext_target.requires_grad_(False)
        self.ext_target.load_state_dict(self.ext_online.state_dict())
        self.int_target.requires_grad_(False)
        self.int_target.load_state_dict(self.int_online.state_dict())

        self.optim = torch.optim.Adam(self.ext_online.parameters(), lr=lr)
        self.int_optim = torch.optim.Adam(self.int_online.parameters(), lr=intrinsic_lr)

        # RND and running stats
        self.rnd = RNDModel(input_dim, rnd_output_dim).float()
        self.rnd_optim = torch.optim.Adam(self.rnd.predictor.parameters(), lr=rnd_lr)
        self.obs_rms = RunningMeanStd(shape=(input_dim,))
        self.int_rms = RunningMeanStd(shape=())
        self.ext_rms = RunningMeanStd(shape=())
        self.tb_writer = None
        self.tb_prefix = "agent"

    def attach_tensorboard(self, writer: SummaryWriter, prefix: str = "agent"):
        self.tb_writer = writer
        self.tb_prefix = prefix

    def update_target(self):
        if not self.delayed_target:
            return
        with torch.no_grad():
            for tp, op in zip(
                self.ext_target.parameters(), self.ext_online.parameters()
            ):
                tp.data.mul_(1.0 - self.polyak_tau).add_(op.data, alpha=self.polyak_tau)
            for tp, op in zip(
                self.int_target.parameters(), self.int_online.parameters()
            ):
                tp.data.mul_(1.0 - self.polyak_tau).add_(op.data, alpha=self.polyak_tau)

    @torch.no_grad()
    def _soft_policy(self, q_values: torch.Tensor):
        pi = torch.softmax(q_values / self.tau, dim=-1)
        logpi = torch.log_softmax(q_values / self.tau, dim=-1)
        ent = -(pi * logpi).sum(dim=-1)  # [B]
        return pi, logpi, ent

    def update(
        self, obs, a, r, next_obs, term, batch_size, step=0, extrinsic_only=False
    ):
        self.update_running_stats(next_obs, r)
        r = self.ext_rms.normalize(r, self.ext_r_clip).to(
            dtype=torch.float32, device=obs.device
        )
        # Get Batch items
        if batch_size is None:
            idx = torch.arange(0, len(r))
            batch_size = len(r)
        else:
            idx = torch.randint(low=0, high=step, size=(batch_size,))
        b_obs = obs[idx]
        b_r = r[idx]
        b_next_obs = next_obs[idx]
        b_term = term[idx]
        b_actions = a[idx]
        # RND intrinsic reward
        with torch.no_grad():
            norm_next_obs = self.obs_rms.normalize(b_next_obs.to(dtype=torch.float64))

        # Get rnd errors and step optimizer
        rnd_errors = self.rnd(norm_next_obs.to(dtype=torch.float32))
        rnd_loss = rnd_errors.mean()
        self.rnd_optim.zero_grad()
        rnd_loss.backward()
        self.rnd_optim.step()

        # Collect intrinsic reward
        r_int = 0
        if self.Beta > 0.0:
            with torch.no_grad():
                norm_int = self.int_rms.normalize(
                    rnd_errors.detach().to(dtype=torch.float64)
                )
                r_int = norm_int.to(dtype=torch.float32)
        # TODO: dont merge reward channels
        b_r_total = b_r + self.Beta * r_int

        # Current and next Q-values
        q_now = self.ext_online(b_obs)  # [B,D,Bins]
        logpi_now = None
        pi_now = None
        entropy_loss = 0

        # Grab entropy loss here if we need to so that we can reuse
        # logpi and pi or calculate then later without grad if ent
        # reg is not being used
        if self.ent_reg_coef > 0.0:
            logpi_now = torch.log_softmax(q_now / self.tau, dim=-1)
            pi_now = torch.exp(logpi_now)
            entropy_loss = (pi_now * logpi_now).sum(dim=-1).mean()

        with torch.no_grad():
            if b_actions.ndim == 1:
                b_act_view = b_actions.view(-1, 1)
            else:
                b_act_view = b_actions  # [B,D]
            action_idx_now = b_act_view.unsqueeze(-1)
            # q_next_online = self.ext_online(b_next_obs)  # [B,D,Bins]
            q_next = (self.ext_target if self.delayed_target else self.ext_online)(
                b_next_obs
            )  # [B,D,Bins]
            if self.munchausen:
                # only need this right now for ln(pi(a)) for munchausen loss
                if logpi_now is None:
                    logpi_now = torch.log_softmax(q_now / self.tau, dim=-1)

                logpi_now = torch.gather(logpi_now, -1, action_idx_now).squeeze(
                    -1
                )  # [B,D]
                r_kl = torch.clamp(logpi_now, min=self.l_clip)
                if r_kl.ndim > 1:
                    r_kl = r_kl.sum(-1)
                b_r_total += self.alpha * self.tau * r_kl

            if self.munchausen or self.soft:
                # Need next probs for next entropy if soft
                logpi_next = torch.log_softmax(q_next / self.tau, dim=-1)
                pi_next = torch.exp(logpi_next)
                next_head_vals = (pi_next * (self.alpha * logpi_next + q_next)).sum(-1)
            else:
                next_head_vals = torch.max(q_next, dim=-1).values

            assert isinstance(next_head_vals, torch.Tensor)
            # Sum over action heads to perform vdn
            if next_head_vals.ndim > 1:
                next_head_vals = next_head_vals.sum(-1)
            td_target = b_r_total + self.gamma * (1 - b_term) * next_head_vals

        q_selected = torch.gather(q_now, -1, action_idx_now).squeeze(
            -1
        )  # joint value as sum
        # if multiple actions sum for vdn
        if q_selected.ndim > 1:
            q_selected = q_selected.sum(-1)

        extrinsic_loss = torch.nn.functional.smooth_l1_loss(q_selected, td_target)
        extrinsic_loss = extrinsic_loss + self.ent_reg_coef * entropy_loss
        self.optim.zero_grad()
        extrinsic_loss.backward()
        self.optim.step()
        if extrinsic_only:
            return float(extrinsic_loss.item())
        # Intrinsic Q update
        int_q_now = self.int_online(b_obs)  # [B,D,Bins]
        int_q_selected = torch.gather(int_q_now, -1, action_idx_now).squeeze(-1)
        if int_q_selected.ndim > 1:
            int_q_selected = int_q_selected.sum(-1)

        with torch.no_grad():
            int_q_next_target = (
                (self.int_target if self.delayed_target else self.int_online)(
                    b_next_obs
                )
                .max(dim=-1)
                .values
            )
            # sum for vdn q
            if int_q_next_target.ndim > 1:
                int_q_next_target = int_q_next_target.sum(-1)
            r_int_only = (
                r_int if isinstance(r_int, torch.Tensor) else torch.zeros_like(b_r)
            )
            int_td_target = r_int_only + self.gamma * (1 - b_term) * int_q_next_target

        intrinsic_loss = torch.nn.functional.smooth_l1_loss(
            int_q_selected, int_td_target
        )
        self.int_optim.zero_grad()
        intrinsic_loss.backward()
        self.int_optim.step()

        # collecting values for tensorboard
        if isinstance(r_int, torch.Tensor):
            r_int_log = float(r_int.mean().item())
        else:
            r_int_log = 0.0

        if isinstance(entropy_loss, torch.Tensor):
            entropy_val = float(entropy_loss.item())
        else:
            entropy_val = entropy_loss

        self.last_losses = {
            "extrinsic": float(extrinsic_loss.item()),
            "intrinsic": float(intrinsic_loss.item()),
            "rnd": float(rnd_loss.item()),
            "avg_r_int": r_int_log,
            "entropy_reg": entropy_val,
            "batch_nonzero_r_frac": float((b_r != 0).float().mean().item()),
            "target_mean": float(td_target.mean().item()),
            "entropy_loss": entropy_val,
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
        return float(extrinsic_loss.item())

    @torch.no_grad()
    def update_running_stats(self, next_obs: torch.Tensor, r: torch.Tensor):
        x64 = next_obs.to(dtype=torch.float64, device=self.obs_rms.mean.device)
        self.obs_rms.update(x64)
        if self.norm_obs:
            norm_x64 = self.obs_rms.normalize(x64)
        else:
            norm_x64 = x64
        if norm_x64.ndim == 1:
            norm_x64 = norm_x64.unsqueeze(0)
        rnd_err = self.rnd(norm_x64.to(dtype=torch.float32)).squeeze()
        self.int_rms.update(rnd_err.to(dtype=torch.float64))
        self.ext_rms.update(r.to(dtype=torch.float64))

    def sample_action(
        self,
        obs: torch.Tensor,
        eps: float,
        step: int,
        n_steps: int = 100000,
        min_ent=0.01,
        verbose: bool = False,
    ):
        with torch.no_grad():
            q_ext = self.ext_online(obs.unsqueeze(0) if obs.ndim == 1 else obs).squeeze(
                0
            )  # [D,Bins]
            if self.Beta > 0.0:
                if random.random() < self.Beta:
                    q_comb = self.int_online(
                        obs.unsqueeze(0) if obs.ndim == 1 else obs
                    ).squeeze(0)
                else:
                    q_comb = q_ext

            else:
                q_comb = q_ext
            eps_curr = 1 - step / n_steps
            if self.soft or self.munchausen:
                actions = []
                for d in range(self.n_action_dims):
                    logits_d = q_comb[d] / self.tau
                    dst = torch.distributions.Categorical(logits=logits_d)
                    a_d = dst.sample().item()
                    if verbose:
                        print(
                            f"Action dim {d}: logits {logits_d.cpu().numpy()}, probs {dst.probs.cpu().numpy()}"
                        )
                    actions.append(a_d)
                return actions
            if random.random() < eps_curr:
                return torch.randint(
                    0, self.n_action_bins, (self.n_action_dims,)
                ).tolist()
            if verbose:
                print(f"Q-values combined: {q_comb.cpu().numpy()}")
            return torch.argmax(q_comb, dim=-1).tolist()
