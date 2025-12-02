import torch
import torch.nn as nn
import random
from typing import Optional
from torch.utils.tensorboard import SummaryWriter
from RandomDistilation import RNDModel, RunningMeanStd
from PopArtLayer import PopArtLayer
from PopArtDuelingLayer import PopArtDuelingHead
from PopArtIQNLayer import PopArtIQNLayer
from PopArtDuelingIQNLayer import PopArtDuelingIQNLayer


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
        popart: bool = False,
    ):
        super().__init__()
        self.dueling = dueling
        self.popart = popart
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
            if self.popart:
                self.output_layer = PopArtDuelingHead(last_hidden, self.n_actions_total)
            else:
                self.value_layer = nn.Linear(last_hidden, 1)
                self.advantage_layer = nn.Linear(last_hidden, self.n_actions_total)
        else:
            if self.popart:
                self.output_layer = PopArtLayer(last_hidden, self.n_actions_total)
            else:
                self.out_layer = nn.Linear(last_hidden, self.n_actions_total)

    def forward(self, x: torch.Tensor, normalized: bool = False) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = self.relu(layer(h))

        if self.popart:
            out = self.output_layer(h, normalized=normalized)
            if h.ndim == 1:
                return out.view(self.n_action_dims, self.n_action_bins)
            return out.view(h.shape[0], self.n_action_dims, self.n_action_bins)

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
        popart: bool = False,
    ):
        super().__init__()
        self.dueling = dueling
        self.popart = popart
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
            if self.popart:
                self.output_layer = PopArtDuelingIQNLayer(
                    last_hidden, n_action_dims, n_action_bins
                )
            else:
                self.value_head = nn.Linear(last_hidden, 1)
                self.adv_head = nn.Linear(last_hidden, self.n_actions_total)
        else:
            if self.popart:
                self.output_layer = PopArtIQNLayer(last_hidden, self.n_actions_total)
            else:
                self.out_head = nn.Linear(last_hidden, self.n_actions_total)

    def _phi(self, tau: torch.Tensor) -> torch.Tensor:
        i = torch.arange(self.n_cosines, device=tau.device, dtype=tau.dtype).view(
            1, 1, -1
        )
        cosines = torch.cos(i * torch.pi * tau.unsqueeze(-1))
        emb = self.relu(self.cosine_layer(cosines))  # [B,N,H]
        return emb

    def forward(
        self, x: torch.Tensor, tau: torch.Tensor, normalized: bool = False
    ) -> torch.Tensor:
        h = x
        for layer in self.base_layers:
            h = self.relu(layer(h))  # [B,H]
        h = h.unsqueeze(1)  # [B,1,H]
        tau_emb = self._phi(tau)  # [B,N,H]
        h = h * tau_emb  # [B,N,H]

        if self.popart:
            out = self.output_layer(h, normalized=normalized)
            if self.dueling:
                return out
            else:
                return out.view(
                    out.shape[0], out.shape[1], self.n_action_dims, self.n_action_bins
                )

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
        lr: float = 1e-3,
        gamma: float = 0.99,
        alpha: float = 0.9,
        tau: float = 0.03,
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
    ):
        self.popart = popart
        self.online = IQN_Network(
            input_dim,
            n_action_dims,
            n_action_bins,
            hidden_layer_sizes=hidden_layer_sizes,
            dueling=dueling,
            popart=popart,
        ).float()
        self.target = IQN_Network(
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
        self.alpha = alpha
        self.tau = tau
        self.l_clip = l_clip
        self.soft = soft
        self.munchausen = munchausen
        self.Thompson = Thompson
        self.target.requires_grad_(False)
        self.target.load_state_dict(self.online.state_dict())
        self.int_target.requires_grad_(False)
        self.int_target.load_state_dict(self.int_online.state_dict())

        self.optim = torch.optim.Adam(self.online.parameters(), lr=lr)
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
            for tp, op in zip(self.target.parameters(), self.online.parameters()):
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
                r_int = r_int.clamp(-5.0, 5.0)

        # Combine rewards for extrinsic head

        b_r_total = b_r + self.Beta * r_int

        # 2) IQN forward passes
        device = b_obs.device
        taus = self._sample_taus(batch_size, self.n_quantiles, device)

        # A. Evaluation / Target Calculation (No Grad)
        with torch.no_grad():
            # Get unnormalized Qs for action selection / Munchausen
            # Note: We use the same taus for consistency, though resampling is also valid IQN
            q_eval = self.online(b_obs, taus, normalized=False)  # [B,N,D,Bins]

            target_taus = self._sample_taus(batch_size, self.n_target_quantiles, device)
            target_quantiles_all = (
                self.target if self.delayed_target else self.online
            )(
                b_next_obs, target_taus, normalized=False
            )  # [B,Nt,D,Bins]
            online_next_quantiles = self.online(
                b_next_obs, taus, normalized=False
            )  # [B,N,D,Bins]
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
            logpi_a = 0
            if self.munchausen:
                ev_hist = q_eval.mean(dim=1)  # [B,D,Bins]
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
                logpi_a = torch.clamp(selected_logpi, min=self.l_clip).mean(
                    dim=-1
                )  # average over dims
                r_aug = r_aug + self.alpha * self.tau * logpi_a

            target_values = r_aug.unsqueeze(1) + (1 - b_term).unsqueeze(
                1
            ) * self.gamma * (
                mixed_target + ent_bonus.unsqueeze(1)
            )  # [B,Nt]

            if self.popart:
                self.online.output_layer.update_stats(target_values)
                target_values = self.online.output_layer.normalize(target_values)

        # B. Training Forward Pass (Grad)
        # Now we run forward pass with updated weights
        quantiles = self.online(b_obs, taus, normalized=self.popart)  # [B,N,D,Bins]

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
            # If popart is on, ev_all is normalized. We should unnormalize for policy entropy?
            if self.popart:
                # Reconstruct unnormalized Qs for entropy calculation
                sigma = self.online.output_layer.sigma
                mu = self.online.output_layer.mu
                ev_all = ev_all * sigma + mu

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

        m_r = 0
        if self.munchausen:
            m_r = (self.tau * logpi_a).mean().item()
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
            "munchausen_r": float(m_r),
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
        r = random.random()
        if r < min_eps or ((not (self.soft or self.munchausen)) and r < eps):
            return torch.randint(0, self.n_action_bins, (self.n_action_dims,)).tolist()
        with torch.no_grad():
            obs_b = obs.unsqueeze(0) if obs.ndim == 1 else obs
            taus = self._sample_taus(obs_b.shape[0], self.n_quantiles, obs_b.device)
            ext_q = self.online(obs_b, taus).mean(dim=1)  # [B,D,Bins]
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
        popart: bool = False,
        Beta: float = 0.0,
        delayed: bool = True,
        ent_reg_coef: float = 0.0,
        # Intrinsic/RND configs
        rnd_output_dim: int = 128,
        rnd_lr: float = 1e-3,
        intrinsic_lr: float = 1e-3,
    ):
        self.popart = popart
        self.online = EV_Q_Network(
            input_dim,
            n_action_dims,
            n_action_bins,
            hidden_layer_sizes=hidden_layer_sizes,
            dueling=dueling,
            popart=popart,
        ).float()
        self.target = EV_Q_Network(
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

        self.target.requires_grad_(False)
        self.target.load_state_dict(self.online.state_dict())
        self.int_target.requires_grad_(False)
        self.int_target.load_state_dict(self.int_online.state_dict())

        self.optim = torch.optim.Adam(self.online.parameters(), lr=lr)
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
            for tp, op in zip(self.target.parameters(), self.online.parameters()):
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

        # Get Batch items
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
        b_r_total = b_r + self.Beta * r_int

        # A. Evaluation / Target Calculation (No Grad)
        with torch.no_grad():
            # Get unnormalized Qs for action selection / Munchausen
            q_eval = self.online(b_obs, normalized=False)  # [B,D,Bins]

            if b_actions.ndim == 1:
                b_act_view = b_actions.view(-1, 1)
            else:
                b_act_view = b_actions  # [B,D]
            action_idx_now = b_act_view.unsqueeze(-1)

            q_next = (self.target if self.delayed_target else self.online)(
                b_next_obs, normalized=False
            )  # [B,D,Bins]

            if self.munchausen:
                # Calculate logpi from q_eval (unnormalized)
                logpi_eval = torch.log_softmax(q_eval / self.tau, dim=-1)
                logpi_eval_selected = torch.gather(
                    logpi_eval, -1, action_idx_now
                ).squeeze(
                    -1
                )  # [B,D]
                r_kl = torch.clamp(logpi_eval_selected, min=self.l_clip)
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

            if self.popart:
                self.online.output_layer.update_stats(td_target)
                td_target = self.online.output_layer.normalize(td_target)

        # B. Training Forward Pass (Grad)
        q_now = self.online(b_obs, normalized=self.popart)  # [B,D,Bins]

        entropy_loss = 0
        if self.ent_reg_coef > 0.0:
            # Unnormalize if needed for entropy
            q_for_ent = q_now
            if self.popart:
                sigma = self.online.output_layer.sigma
                mu = self.online.output_layer.mu
                q_for_ent = q_now * sigma + mu

            logpi_now = torch.log_softmax(q_for_ent / self.tau, dim=-1)
            pi_now = torch.exp(logpi_now)
            entropy_loss = -torch.sum(pi_now * logpi_now, dim=-1).mean()

        q_selected = torch.gather(q_now, -1, action_idx_now).squeeze(
            -1
        )  # joint value as sum
        # if multiple actions sum for vdn
        if q_selected.ndim > 1:
            q_selected = q_selected.sum(-1)

        extrinsic_loss = torch.nn.functional.smooth_l1_loss(q_selected, td_target)
        extrinsic_loss = extrinsic_loss + self.alpha * entropy_loss
        self.optim.zero_grad()
        extrinsic_loss.backward()
        self.optim.step()

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
            q_ext = self.online(obs.unsqueeze(0) if obs.ndim == 1 else obs).squeeze(
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
            if self.soft or self.munchausen:
                actions = []
                for d in range(self.n_action_dims):
                    logits_d = q_comb[d] / self.tau
                    a_d = (
                        torch.distributions.Categorical(logits=logits_d).sample().item()
                    )
                    actions.append(a_d)
                return actions
            if random.random() < eps_curr:
                return torch.randint(
                    0, self.n_action_bins, (self.n_action_dims,)
                ).tolist()
            return torch.argmax(q_comb, dim=-1).tolist()


if __name__ == "__main__":
    import itertools

    print("Starting Integration Tests for RainbowDQN and EVRainbowDQN...")

    # Hyperparameters
    OBS_DIM = 16
    ACTION_BINS = 4
    HIDDEN_LAYER_SIZES = [128, 128]

    # Grid Search Parameters
    SOFT_AND_MUNCHAUSEN = [[True, True], [False, False]]
    DUELING = [True, False]
    POPART = [True, False]
    BETA = [0.0, 0.2]
    D_ACTION_DIMS = [1, 3]

    # Data Generation
    buffer_size = 100
    batch_size = 32
    torch.manual_seed(42)
    # torch.autograd.set_detect_anomaly(True) # Optional for debugging

    single_obs = torch.rand(size=[OBS_DIM])

    # Create "Replay Buffer"
    buffer_obs = torch.rand(size=[buffer_size, OBS_DIM])
    buffer_next_obs = torch.rand(size=[buffer_size, OBS_DIM])
    buffer_rewards = torch.randn(size=[buffer_size]) * 5.0 + 2.0
    buffer_terminated = torch.randint(0, 2, [buffer_size])

    # Actions for D=1 and D=3
    buffer_actions_1 = torch.randint(0, ACTION_BINS, [buffer_size])
    buffer_actions_3 = torch.randint(0, ACTION_BINS, [buffer_size, 3])

    # Iterate over models
    models_to_test = [RainbowDQN, EVRainbowDQN]

    total_tests = (
        len(models_to_test)
        * len(SOFT_AND_MUNCHAUSEN)
        * len(DUELING)
        * len(POPART)
        * len(BETA)
        * len(D_ACTION_DIMS)
    )
    current_test = 0
    n_extrinsic_tests = 0
    n_extrinsic_run = 0
    n_extrinsic_passed = 0

    n_rnd_tests = 0
    n_rnd_run = 0
    n_rnd_pass = 0
    for ModelClass in models_to_test:
        print(f"\n{'='*20} Testing {ModelClass.__name__} {'='*20}")

        # Grid Search
        combinations = itertools.product(
            SOFT_AND_MUNCHAUSEN, DUELING, POPART, BETA, D_ACTION_DIMS
        )

        for sm, dueling, popart, beta, d_dim in combinations:
            current_test += 1
            soft, munchausen = sm

            print(
                f"\nTest {current_test}/{total_tests}: Soft={soft}, Munch={munchausen}, Dueling={dueling}, PopArt={popart}, Beta={beta}, D_Dim={d_dim}"
            )

            # Initialize Model
            try:
                agent = ModelClass(
                    input_dim=OBS_DIM,
                    n_action_dims=d_dim,
                    n_action_bins=ACTION_BINS,
                    hidden_layer_sizes=HIDDEN_LAYER_SIZES,
                    soft=soft,
                    munchausen=munchausen,
                    dueling=dueling,
                    popart=popart,
                    Beta=beta,
                    rnd_lr=1e-2,  # Higher LR to see RND convergence faster
                    intrinsic_lr=1e-3,
                )
                # Pre-warm running stats to avoid non-stationary input distribution for RND
                # This prevents the RND loss from spiking due to shifting normalization statistics
                agent.update_running_stats(buffer_obs)
            except Exception as e:
                print(f"FAILED to initialize model: {e}")
                raise e

            # 1. Test sample_action
            try:
                action = agent.sample_action(single_obs, eps=0.1, step=0)
                assert (
                    len(action) == d_dim
                ), f"Sampled action dim mismatch. Expected {d_dim}, got {len(action)}"
            except Exception as e:
                print(f"FAILED in sample_action: {e}")
                raise e

            # 2. Test update loop (10 steps)
            actions = buffer_actions_1 if d_dim == 1 else buffer_actions_3

            initial_rnd_loss = None
            initial_sigma = None

            # Get initial sigma if popart
            if popart:
                if hasattr(agent.online, "output_layer"):
                    layer = agent.online.output_layer
                    initial_sigma = layer.sigma.mean().item()
            n_extrinsic_run += 1
            if beta > 0:
                n_rnd_run += 1
            try:
                initial_ext_loss = 0
                loss = torch.zeros(1)
                for step in range(50):
                    # Pass step=buffer_size to allow sampling from the full buffer
                    loss = agent.update(
                        buffer_obs,
                        actions,
                        buffer_rewards,
                        buffer_next_obs,
                        buffer_terminated,
                        batch_size,
                        step=buffer_size,
                    )
                    if step == 0:
                        print(f"Initial Extrinsic loss: {loss:.4f}")
                    # Update running stats (usually done in runner, but needed for RND/PopArt to see shifts)
                    # We use a random batch from the buffer for this simulation
                    idx = torch.randint(0, buffer_size, (batch_size,))
                    agent.update_running_stats(buffer_next_obs[idx])

                    if step == 0:
                        initial_rnd_loss = agent.last_losses.get("rnd", 0.0)
                        initial_ext_loss = loss
                # Post-loop checks
                print(f"  Final Extrinsic Loss: {loss:.4f}")
                if loss < initial_ext_loss:
                    n_extrinsic_passed += 1
                # Check RND
                if beta > 0:
                    final_rnd = agent.last_losses.get("rnd", 0.0)
                    print(f"  RND Loss: {initial_rnd_loss:.4f} -> {final_rnd:.4f}")
                    if initial_rnd_loss > final_rnd:
                        n_rnd_pass += 1
                # Check PopArt
                if popart:
                    layer = agent.online.output_layer
                    final_sigma = layer.sigma.mean().item()
                    print(f"  PopArt Sigma: {initial_sigma:.4f} -> {final_sigma:.4f}")

                    # Check if sigma updated (it starts at 1.0)
                    if abs(final_sigma - 1.0) < 1e-6:
                        print("  WARNING: PopArt sigma did not change from 1.0!")
                    else:
                        print("  PopArt sigma updated successfully.")

            except Exception as e:
                print(f"FAILED in update loop: {e}")
                raise e
    print(
        f"Extrinsic experiments tried: {n_extrinsic_run} run: {n_extrinsic_run} passed: {n_extrinsic_passed}"
    )
    print(f"RND experiments tried: {n_rnd_tests} run: {n_rnd_run} passed: {n_rnd_pass}")
