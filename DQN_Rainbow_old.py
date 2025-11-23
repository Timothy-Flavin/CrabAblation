import torch
import torch.nn as nn
import random
from typing import Optional
from torch.utils.tensorboard import SummaryWriter
from RandomDistilation import RNDModel, RunningMeanStd


class EV_Q_Network(nn.Module):
    """Single-valued (non-distributional) multi-dimensional Q network with optional dueling.

    Outputs Q-values per action dimension and per bin: [B, D, Bins].
    Dueling implemented by producing a single value head plus an advantage head of size D*Bins,
    which is reshaped and mean-subtracted across all (D*Bins) entries.
    """

    def __init__(
        self,
        input_dim: int,
        n_action_dims: int,
        n_action_bins: int,
        hidden_layer_sizes=[128, 128],
        dueling: bool = False,
    ):
        super().__init__()
        self.dueling = dueling
        self.n_action_dims = n_action_dims
        self.n_action_bins = n_action_bins
        self.n_actions_total = n_action_dims * n_action_bins  # legacy convenience
        layers = [nn.Linear(input_dim, hidden_layer_sizes[0])]
        for li in range(len(hidden_layer_sizes) - 1):
            layers.append(nn.Linear(hidden_layer_sizes[li], hidden_layer_sizes[li + 1]))
        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()
        last_hidden = hidden_layer_sizes[-1]
        if self.dueling:
            self.value_layer = nn.Linear(last_hidden, 1)
            self.advantage_layer = nn.Linear(last_hidden, self.n_actions_total)
        else:
            self.out_layer = nn.Linear(last_hidden, self.n_actions_total)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = self.relu(layer(h))
        if not self.dueling:
            out = self.out_layer(h)
            if h.ndim == 1:
                return out.view(self.n_action_dims, self.n_action_bins)
            return out.view(h.shape[0], self.n_action_dims, self.n_action_bins)
        if h.ndim == 1:
            v = self.value_layer(h).view(1)  # scalar value
            a = self.advantage_layer(h).view(self.n_action_dims, self.n_action_bins)
            a_mean = a.mean()
            q = a - a_mean + v
            return q
        bsz = h.shape[0]
        v = self.value_layer(h).view(bsz, 1, 1)
        a = self.advantage_layer(h).view(bsz, self.n_action_dims, self.n_action_bins)
        a_mean = a.mean(dim=(1, 2), keepdim=True)
        q = a - a_mean + v
        return q

    def expected_value(self, x: torch.Tensor, probs: bool = False):
        # For EV network, expected value per dimension/bin is x itself.
        return x


class IQN_Network(nn.Module):
    """Implicit Quantile Network supporting multi-dimensional discrete actions.

    Output shape: [B, Nq, D, Bins].
    Dueling implemented similarly to EV_Q_Network using a shared advantage head sized D*Bins.
    """

    def __init__(
        self,
        input_dim: int,
        n_action_dims: int,
        n_action_bins: int,
        hidden_layer_sizes=[128, 128],
        n_cosines: int = 64,
        dueling: bool = False,
    ):
        super().__init__()
        self.dueling = dueling
        self.n_action_dims = n_action_dims
        self.n_action_bins = n_action_bins
        self.n_actions_total = n_action_dims * n_action_bins
        self.n_cosines = n_cosines
        base = [nn.Linear(input_dim, hidden_layer_sizes[0])]
        for li in range(len(hidden_layer_sizes) - 1):
            base.append(nn.Linear(hidden_layer_sizes[li], hidden_layer_sizes[li + 1]))
        self.base_layers = nn.ModuleList(base)
        self.relu = nn.ReLU()
        last_hidden = hidden_layer_sizes[-1]
        self.cosine_layer = nn.Linear(n_cosines, last_hidden)
        if dueling:
            self.value_head = nn.Linear(last_hidden, 1)
            self.adv_head = nn.Linear(last_hidden, self.n_actions_total)
        else:
            self.out_head = nn.Linear(last_hidden, self.n_actions_total)

    def _phi(self, tau: torch.Tensor) -> torch.Tensor:
        i = torch.arange(self.n_cosines, device=tau.device, dtype=tau.dtype).view(
            1, 1, -1
        )
        cosines = torch.cos(i * torch.pi * tau.unsqueeze(-1))
        emb = self.relu(self.cosine_layer(cosines))  # [B,N,H]
        return emb

    def forward(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.base_layers:
            h = self.relu(layer(h))  # [B,H]
        h = h.unsqueeze(1)  # [B,1,H]
        tau_emb = self._phi(tau)  # [B,N,H]
        h = h * tau_emb  # [B,N,H]
        if not self.dueling:
            out = self.out_head(h)  # [B,N,D*Bins]
            return out.view(
                out.shape[0], out.shape[1], self.n_action_dims, self.n_action_bins
            )
        v = self.value_head(h)  # [B,N,1]
        a = self.adv_head(h)  # [B,N,D*Bins]
        a = a.view(a.shape[0], a.shape[1], self.n_action_dims, self.n_action_bins)
        a_mean = a.mean(dim=(2, 3), keepdim=True)
        q = a - a_mean + v.unsqueeze(-1)  # broadcast v
        return q  # [B,N,D,Bins]

    def expected_value(self, quantiles: torch.Tensor) -> torch.Tensor:
        # Mean over quantile dimension -> [B,D,Bins]
        return quantiles.mean(dim=1)


class RainbowDQN:
    """Maintains online (current) and target Q_Networks and training logic."""

    def __init__(
        self,
        input_dim,
        n_action_dims,
        n_action_bins,
        hidden_layer_sizes=[128, 128],
        n_atoms=101,
        zmin=-50,
        zmax=50,
        lr: float = 1e-3,
        gamma: float = 0.99,
        alpha: float = 0.9,
        tau: float = 0.03,
        l_clip: float = -1.0,
        soft: bool = False,
        munchausen: bool = False,
        Thompson: bool = False,
        dueling: bool = False,
        Beta: float = 0.0,
        # Delayed target usage (pillar 5). If False, use online net as target.
        delayed: bool = True,
        # Entropy regularization on current policy (online net)
        ent_reg_coef: float = 0.0,
        # Intrinsic/RND configs
        rnd_output_dim: int = 128,
        rnd_lr: float = 1e-3,
        intrinsic_lr: float = 1e-3,
    ):
        self.ext_online = IQN_Network(
            input_dim,
            n_action_dims,
            n_action_bins,
            hidden_layer_sizes=hidden_layer_sizes,
            dueling=dueling,
        ).float()
        self.ext_target = IQN_Network(
            input_dim,
            n_action_dims,
            n_action_bins,
            hidden_layer_sizes=hidden_layer_sizes,
            dueling=dueling,
        ).float()
        # Intrinsic Q networks (separate head trained on intrinsic rewards only)
        self.int_online = IQN_Network(
            input_dim,
            n_action_dims,
            n_action_bins,
            hidden_layer_sizes=hidden_layer_sizes,
            dueling=dueling,
        ).float()
        self.int_target = IQN_Network(
            input_dim,
            n_action_dims,
            n_action_bins,
            hidden_layer_sizes=hidden_layer_sizes,
            dueling=dueling,
        ).float()
        self.alpha = alpha
        self.tau = tau
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
        self.delayed_target = delayed
        self.ent_reg_coef = ent_reg_coef

        # RND intrinsic reward model and normalizers
        self.rnd = RNDModel(input_dim, rnd_output_dim).float()
        self.rnd_optim = torch.optim.Adam(self.rnd.predictor.parameters(), lr=rnd_lr)
        # Running stats: observations and intrinsic reward magnitude
        self.obs_rms = RunningMeanStd(shape=(input_dim,))
        # Scalar running stats for intrinsic reward
        self.int_rms = RunningMeanStd(shape=())
        # Optional TensorBoard writer
        self.tb_writer = None
        self.tb_prefix = "agent"

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
                tp.data.mul_(1.0 - self.tau).add_(op.data, alpha=self.tau)
            for tp, op in zip(
                self.int_target.parameters(), self.int_online.parameters()
            ):
                tp.data.mul_(1.0 - self.tau).add_(op.data, alpha=self.tau)

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

    def update(self, obs, a, r, next_obs, term, batch_size, step=0):
        # Sample a random minibatch
        idx = torch.randint(low=0, high=step, size=(batch_size,))
        b_obs = obs[idx]
        b_r = r[idx]
        b_next_obs = next_obs[idx]
        b_term = term[idx]
        b_actions = a[idx]

        # 1) Intrinsic reward via RND (train predictor to reduce novelty on visited states)
        # Use existing running stats (updated per-environment step) to normalize inputs for RND
        with torch.no_grad():
            norm_next_obs = self.obs_rms.normalize(b_next_obs.to(dtype=torch.float64))
        norm_next_obs_f32 = norm_next_obs.to(dtype=torch.float32)

        # Train RND predictor
        rnd_errors = self.rnd(norm_next_obs_f32)  # [B]
        rnd_loss = rnd_errors.mean()
        self.rnd_optim.zero_grad()
        rnd_loss.backward()
        self.rnd_optim.step()

        # Normalize intrinsic reward magnitude using running stats (updated per step)
        r_int = 0
        if self.Beta > 0.0:
            with torch.no_grad():
                norm_int = self.int_rms.normalize(
                    rnd_errors.detach().to(dtype=torch.float64)
                )
                r_int = norm_int.to(dtype=torch.float32)
                r_int = r_int.clamp(-10.0, 10.0)

        # Combine rewards for extrinsic head

        b_r_total = b_r + self.Beta * r_int

        # 2) IQN forward passes
        device = b_obs.device
        taus = self._sample_taus(batch_size, self.n_quantiles, device)
        quantiles = self.ext_online(b_obs, taus)  # [B,N,D,Bins]
        with torch.no_grad():
            target_taus = self._sample_taus(batch_size, self.n_target_quantiles, device)
            target_quantiles_all = (
                self.ext_target if self.delayed_target else self.ext_online
            )(
                b_next_obs, target_taus
            )  # [B,Nt,D,Bins]
            online_next_quantiles = self.ext_online(b_next_obs, taus)  # [B,N,D,Bins]
            mean_next = online_next_quantiles.mean(dim=1)  # [B,D,Bins]
            if self.soft or self.munchausen:
                pi_next = torch.softmax(mean_next / self.tau, dim=-1)  # [B,D,Bins]
                mixed_target_per_dim = (
                    pi_next.unsqueeze(1) * target_quantiles_all
                ).sum(
                    dim=-1
                )  # [B,Nt,D]
                mixed_target = mixed_target_per_dim.sum(
                    dim=-1
                )  # [B,Nt] aggregate across dims
                logpi_next = torch.log_softmax(mean_next / self.tau, dim=-1)
                ent_bonus_per_dim = -(pi_next * logpi_next).sum(dim=-1)  # [B,D]
                ent_bonus = ent_bonus_per_dim.sum(dim=-1)  # [B]
            else:
                next_actions_per_dim = torch.argmax(mean_next, dim=-1)  # [B,D]
                gather_index = (
                    next_actions_per_dim.unsqueeze(1)
                    .unsqueeze(-1)
                    .expand(-1, self.n_target_quantiles, -1, 1)
                )
                chosen_target_per_dim = torch.gather(
                    target_quantiles_all, 3, gather_index
                ).squeeze(
                    -1
                )  # [B,Nt,D]
                mixed_target = chosen_target_per_dim.sum(dim=-1)  # [B,Nt]
                ent_bonus = torch.zeros(batch_size, device=device)

            # Munchausen augmentation on reward
            r_aug = b_r_total
            if self.munchausen:
                ev_hist = quantiles.mean(dim=1)  # [B,D,Bins]
                logpi_hist = torch.log_softmax(ev_hist / self.tau, dim=-1)
                # b_actions shape [B,D]
                if b_actions.ndim == 1:
                    b_actions_hist = b_actions.view(-1, 1)
                else:
                    b_actions_hist = b_actions
                gather_m_hist = b_actions_hist.unsqueeze(-1)
                selected_logpi = torch.gather(logpi_hist, -1, gather_m_hist).squeeze(
                    -1
                )  # [B,D]
                logpi_a = torch.clamp(
                    selected_logpi.mean(dim=-1), min=self.l_clip
                )  # average over dims
                r_aug = r_aug + self.alpha * self.tau * logpi_a

            target_values = r_aug.unsqueeze(1) + (1 - b_term).unsqueeze(
                1
            ) * self.gamma * (
                mixed_target + ent_bonus.unsqueeze(1)
            )  # [B,Nt]

        # Gather predicted quantiles for taken actions
        # Gather predicted quantiles per dimension and aggregate
        if b_actions.ndim == 1:
            b_actions_view = b_actions.view(batch_size, 1)  # assume single dimension
        else:
            b_actions_view = b_actions  # [B,D]
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
        loss = self._quantile_huber_loss(
            pred_chosen, target_values.detach(), taus.detach()
        )

        # Optional entropy regularization for current policy over actions at b_obs
        entropy_val = 0.0
        if self.ent_reg_coef > 0.0:
            # Policy over actions from expected values (mean quantiles)
            ev_all = quantiles.mean(dim=1)  # [B,D,Bins]
            pi = torch.softmax(ev_all / self.tau, dim=-1)
            logpi = torch.log_softmax(ev_all / self.tau, dim=-1)
            entropy_per_dim = -(pi * logpi).sum(dim=-1)  # [B,D]
            entropy = entropy_per_dim.mean()  # average over dims
            entropy_val = float(entropy.item())
            loss = loss - self.ent_reg_coef * entropy

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # 3) Update intrinsic Q network on intrinsic rewards only
        # Intrinsic IQN update (on intrinsic rewards only)
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
            int_online_next = self.int_online(b_next_obs, int_taus).mean(
                dim=1
            )  # [B,D,Bins]
            next_a_int = torch.argmax(int_online_next, dim=-1)  # [B,D]
            gather_index_int = (
                next_a_int.unsqueeze(1)
                .unsqueeze(-1)
                .expand(-1, self.n_target_quantiles, -1, 1)
            )
            chosen_int_target_per_dim = torch.gather(
                int_target_all, 3, gather_index_int
            ).squeeze(
                -1
            )  # [B,Nt,D]
            int_target_values = chosen_int_target_per_dim.sum(dim=-1)  # [B,Nt]
            r_int_only = (
                r_int if isinstance(r_int, torch.Tensor) else torch.zeros_like(b_r)
            )
            int_target_values = (
                r_int_only.unsqueeze(1)
                + (1 - b_term).unsqueeze(1) * self.gamma * int_target_values
            )

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

        # Store last auxiliary losses for logging (optional)
        self.last_losses = {
            "extrinsic": float(loss.item()),
            "intrinsic": float(int_loss.item()),
            "rnd": float(rnd_loss.item()),
            "avg_r_int": (
                float(r_int.mean().item())
                if isinstance(r_int, torch.Tensor)
                else (r_int if isinstance(r_int, (int, float)) else 0.0)
            ),
            "entropy_reg": entropy_val,
            "abs_r_ext": float(b_r.abs().mean().item()),
            "abs_r_int": (
                float(r_int.abs().mean().item())
                if isinstance(r_int, torch.Tensor)
                else r_int
            ),
        }

        # Inline TensorBoard logging if writer attached
        if self.tb_writer is not None:
            try:
                for k, v in self.last_losses.items():
                    if isinstance(v, (int, float)):
                        self.tb_writer.add_scalar(
                            f"{self.tb_prefix}/{k}", float(v), step
                        )
            except Exception:
                pass

        return loss.item()

    @torch.no_grad()
    def update_running_stats(self, next_obs: torch.Tensor):
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

    def sample_action(
        self,
        obs: torch.Tensor,
        eps: float,
        step: int,
        n_steps: int = 100000,
        min_eps=0.01,
    ):
        """Return vector of length D with selected bin indices for each action dimension."""
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
            eps_curr = 1 - step / n_steps
            if (not (self.soft or self.munchausen)) and random.random() < eps_curr:
                return torch.randint(
                    0, self.n_action_bins, (self.n_action_dims,)
                ).tolist()
            if self.soft or self.munchausen:
                # Sample per dimension from softmax policy
                actions = []
                for d in range(self.n_action_dims):
                    logits_d = q_comb[0, d] / self.tau
                    a_d = (
                        torch.distributions.Categorical(logits=logits_d).sample().item()
                    )
                    actions.append(a_d)
                return actions
            # Greedy per dimension
            return torch.argmax(q_comb.squeeze(0), dim=-1).tolist()


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
        l_clip: float = -1.0,
        soft: bool = False,
        munchausen: bool = False,
        Thompson: bool = False,
        dueling: bool = False,
        Beta: float = 0.0,
        delayed: bool = True,
        ent_reg_coef: float = 0.0,
        # Intrinsic/RND configs
        rnd_output_dim: int = 128,
        rnd_lr: float = 1e-3,
        intrinsic_lr: float = 1e-3,
    ):
        self.ext_online = EV_Q_Network(
            input_dim,
            n_action_dims,
            n_action_bins,
            hidden_layer_sizes=hidden_layer_sizes,
            dueling=dueling,
        ).float()
        self.ext_target = EV_Q_Network(
            input_dim,
            n_action_dims,
            n_action_bins,
            hidden_layer_sizes=hidden_layer_sizes,
            dueling=dueling,
        ).float()
        self.int_online = EV_Q_Network(
            input_dim,
            n_action_dims,
            n_action_bins,
            hidden_layer_sizes=hidden_layer_sizes,
            dueling=dueling,
        ).float()
        self.int_target = EV_Q_Network(
            input_dim,
            n_action_dims,
            n_action_bins,
            hidden_layer_sizes=hidden_layer_sizes,
            dueling=dueling,
        ).float()

        self.alpha = alpha
        self.tau = tau
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
                tp.data.mul_(1.0 - self.tau).add_(op.data, alpha=self.tau)
            for tp, op in zip(
                self.int_target.parameters(), self.int_online.parameters()
            ):
                tp.data.mul_(1.0 - self.tau).add_(op.data, alpha=self.tau)

    @torch.no_grad()
    def _soft_policy(self, q_values: torch.Tensor):
        pi = torch.softmax(q_values / self.tau, dim=-1)
        logpi = torch.log_softmax(q_values / self.tau, dim=-1)
        ent = -(pi * logpi).sum(dim=-1)  # [B]
        return pi, logpi, ent

    def update(self, obs, a, r, next_obs, term, batch_size, step=0):
        idx = torch.randint(low=0, high=step, size=(batch_size,))
        b_obs = obs[idx]
        b_r = r[idx]
        b_next_obs = next_obs[idx]
        b_term = term[idx]
        b_actions = a[idx]

        # RND intrinsic reward
        with torch.no_grad():
            norm_next_obs = self.obs_rms.normalize(b_next_obs.to(dtype=torch.float64))
        rnd_errors = self.rnd(norm_next_obs.to(dtype=torch.float32))
        rnd_loss = rnd_errors.mean()
        self.rnd_optim.zero_grad()
        rnd_loss.backward()
        self.rnd_optim.step()

        r_int = 0
        if self.Beta > 0.0:
            with torch.no_grad():
                norm_int = self.int_rms.normalize(
                    rnd_errors.detach().to(dtype=torch.float64)
                )
                r_int = norm_int.to(dtype=torch.float32)

        b_r_total = b_r + self.Beta * r_int

        # Current and next Q-values
        q_now = self.ext_online(b_obs)  # [B,D,Bins]
        q_next_online = self.ext_online(b_next_obs)  # [B,D,Bins]
        q_next_target = (self.ext_target if self.delayed_target else self.ext_online)(
            b_next_obs
        )  # [B,D,Bins]

        # Munchausen augmentation
        r_aug = b_r_total
        if self.munchausen:
            # Treat each dimension independently; average log-probs of selected bins
            _, logpi_hist, _ = self._soft_policy(
                q_now.view(q_now.shape[0], -1)
            )  # flatten for reuse (approximation)
            # Simpler: compute per-dim logpi directly
            logpi_dims = torch.log_softmax(q_now / self.tau, dim=-1)  # [B,D,Bins]
            if b_actions.ndim == 1:
                b_act_view = b_actions.view(-1, 1)
            else:
                b_act_view = b_actions  # [B,D]
            gather_idx = b_act_view.unsqueeze(-1)
            selected_logpi = torch.gather(logpi_dims, -1, gather_idx).squeeze(
                -1
            )  # [B,D]
            logpi_a = torch.clamp(selected_logpi.mean(dim=-1), min=self.l_clip)
            r_aug = r_aug + self.alpha * self.tau * logpi_a

        # Compute next-state value
        if self.soft or self.munchausen:
            # Per-dimension soft backup
            pi_next = torch.softmax(q_next_online / self.tau, dim=-1)  # [B,D,Bins]
            v_next_per_dim = (pi_next * q_next_target).sum(dim=-1)  # [B,D]
            ent_next_per_dim = -(
                pi_next * torch.log_softmax(q_next_online / self.tau, dim=-1)
            ).sum(dim=-1)
            v_next = (v_next_per_dim + ent_next_per_dim).sum(
                dim=-1
            )  # aggregate over dims
        else:
            next_bins = torch.argmax(q_next_online, dim=-1)  # [B,D]
            gather_idx = next_bins.unsqueeze(-1)
            chosen_vals = torch.gather(q_next_target, -1, gather_idx).squeeze(
                -1
            )  # [B,D]
            v_next = chosen_vals.sum(dim=-1)

        y = r_aug + (1 - b_term) * (self.gamma * v_next)

        # Extrinsic loss
        # Gather chosen bins per dimension and aggregate
        if b_actions.ndim == 1:
            b_act_view = b_actions.view(-1, 1)
        else:
            b_act_view = b_actions  # [B,D]
        gather_idx = b_act_view.unsqueeze(-1)
        chosen_per_dim = torch.gather(q_now, -1, gather_idx).squeeze(-1)  # [B,D]
        q_selected = chosen_per_dim.sum(dim=-1)  # joint value as sum
        extrinsic_loss = torch.nn.functional.smooth_l1_loss(q_selected, y)

        # Entropy regularization
        entropy_val = 0.0
        if self.ent_reg_coef > 0.0:
            pi_dims = torch.softmax(q_now / self.tau, dim=-1)
            logpi_dims = torch.log_softmax(q_now / self.tau, dim=-1)
            entropy_dims = -(pi_dims * logpi_dims).sum(dim=-1).mean()
            entropy_val = float(entropy_dims.item())
            extrinsic_loss = extrinsic_loss - self.ent_reg_coef * entropy_dims

        self.optim.zero_grad()
        extrinsic_loss.backward()
        self.optim.step()

        # Intrinsic Q update
        int_q_now = self.int_online(b_obs)  # [B,D,Bins]
        int_q_next_online = self.int_online(b_next_obs)
        int_q_next_target = (
            self.int_target if self.delayed_target else self.int_online
        )(b_next_obs)

        r_int_only = r_int if isinstance(r_int, torch.Tensor) else torch.zeros_like(b_r)
        if self.soft or self.munchausen:
            pi_next_i, logpi_next_i, ent_next_i = self._soft_policy(int_q_next_online)
            v_next_i = (pi_next_i * int_q_next_target).sum(dim=1) + ent_next_i
        else:
            next_bins_int = torch.argmax(int_q_next_online, dim=-1)  # [B,D]
            gather_idx_int = next_bins_int.unsqueeze(-1)
            v_next_i = (
                torch.gather(int_q_next_target, -1, gather_idx_int)
                .squeeze(-1)
                .sum(dim=-1)
            )
        y_i = r_int_only + (1 - b_term) * (self.gamma * v_next_i)
        gather_idx_intr = b_act_view.unsqueeze(-1)
        chosen_int_per_dim = torch.gather(int_q_now, -1, gather_idx_intr).squeeze(
            -1
        )  # [B,D]
        int_q_sel = chosen_int_per_dim.sum(dim=-1)
        intrinsic_loss = torch.nn.functional.smooth_l1_loss(int_q_sel, y_i)

        self.int_optim.zero_grad()
        intrinsic_loss.backward()
        self.int_optim.step()

        if isinstance(r_int, torch.Tensor):
            r_int_log = float(r_int.mean().item())
        else:
            r_int_log = 0.0
        self.last_losses = {
            "extrinsic": float(extrinsic_loss.item()),
            "intrinsic": float(intrinsic_loss.item()),
            "rnd": float(rnd_loss.item()),
            "avg_r_int": r_int_log,
            "entropy_reg": entropy_val,
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
    def update_running_stats(self, next_obs: torch.Tensor):
        x64 = next_obs.to(dtype=torch.float64, device=self.obs_rms.mean.device)
        self.obs_rms.update(x64)
        norm_x64 = self.obs_rms.normalize(x64)
        if norm_x64.ndim == 1:
            norm_x64 = norm_x64.unsqueeze(0)
        rnd_err = self.rnd(norm_x64.to(dtype=torch.float32)).squeeze()
        self.int_rms.update(rnd_err.to(dtype=torch.float64))

    def sample_action(
        self,
        obs: torch.Tensor,
        eps: float,
        step: int,
        n_steps: int = 100000,
        min_ent=0.01,
    ):
        with torch.no_grad():
            q_ext = self.ext_online(obs.unsqueeze(0) if obs.ndim == 1 else obs).squeeze(
                0
            )  # [D,Bins]
            if self.Beta > 0.0:
                q_int = self.int_online(
                    obs.unsqueeze(0) if obs.ndim == 1 else obs
                ).squeeze(0)
                q_comb = q_ext + self.Beta * q_int
            else:
                q_comb = q_ext
            eps_curr = 1 - step / n_steps
            if (not (self.soft or self.munchausen)) and random.random() < eps_curr:
                return torch.randint(
                    0, self.n_action_bins, (self.n_action_dims,)
                ).tolist()
            if self.soft or self.munchausen:
                actions = []
                for d in range(self.n_action_dims):
                    logits_d = q_comb[d] / self.tau
                    a_d = (
                        torch.distributions.Categorical(logits=logits_d).sample().item()
                    )
                    actions.append(a_d)
                return actions
            if self.Thompson:
                g = -torch.log(-torch.log(torch.rand_like(q_comb)))
                q_comb = q_comb + g
            return torch.argmax(q_comb, dim=-1).tolist()
