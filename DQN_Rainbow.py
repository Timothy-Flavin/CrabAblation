import torch
import torch.nn as nn
import random
from RandomDistilation import RNDModel, RunningMeanStd


class EV_Q_Network(nn.Module):
    """Single-valued (non-distributional) Q network with optional dueling."""

    def __init__(
        self,
        input_dim,
        n_actions,
        hidden_layer_sizes=[128, 128],
        dueling: bool = False,
    ):
        super().__init__()
        self.dueling = dueling
        layers = []
        layers.append(nn.Linear(input_dim, hidden_layer_sizes[0]))
        for li in range(len(hidden_layer_sizes) - 1):
            layers.append(nn.Linear(hidden_layer_sizes[li], hidden_layer_sizes[li + 1]))
        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()
        last_hidden = hidden_layer_sizes[-1]
        self.n_actions = n_actions
        if self.dueling:
            self.value_layer = nn.Linear(last_hidden, 1)
            self.advantage_layer = nn.Linear(last_hidden, n_actions)
        else:
            self.out_layer = nn.Linear(last_hidden, n_actions)

    def forward(self, x: torch.Tensor):
        h = x
        for layer in self.layers:
            h = self.relu(layer(h))
        if not self.dueling:
            return self.out_layer(h)
        if h.ndim == 1:
            v = self.value_layer(h).view(1)  # [] -> [1]
            a = self.advantage_layer(h).view(self.n_actions)
            a_mean = a.mean()
            q = a - a_mean + v
            return q
        else:
            bsz = h.shape[0]
            v = self.value_layer(h).view(bsz, 1)
            a = self.advantage_layer(h).view(bsz, self.n_actions)
            a_mean = a.mean(dim=1, keepdim=True)
            q = a - a_mean + v
            return q

    # Keep API parity with distributional network
    def expected_value(self, x: torch.Tensor, probs: bool = False):
        # For single-valued Q, the 'expected value' is the Q itself
        return x


class Q_Network(nn.Module):
    def __init__(
        self,
        input_dim,
        n_actions,
        hidden_layer_sizes=[128, 128],
        n_atoms=101,
        zmin=-50,
        zmax=50,
        dueling: bool = False,
    ):
        super().__init__()
        self.dueling = dueling
        layers = []
        layers.append(nn.Linear(input_dim, hidden_layer_sizes[0]))
        for li in range(len(hidden_layer_sizes) - 1):
            layers.append(nn.Linear(hidden_layer_sizes[li], hidden_layer_sizes[li + 1]))
        self.layers = nn.ModuleList(layers)
        last_hidden = hidden_layer_sizes[-1]
        if self.dueling:
            # Value stream outputs per-atom logits [N]
            self.value_layer = nn.Linear(last_hidden, n_atoms)
            # Advantage stream outputs per-action per-atom logits [A*N]
            self.advantage_layer = nn.Linear(last_hidden, n_actions * n_atoms)
        else:
            self.out_layer = nn.Linear(last_hidden, n_actions * n_atoms)
        self.relu = nn.ReLU()
        self.zmax = zmax
        self.zmin = zmin
        # Register the support as a buffer so it moves with the module/device
        self.register_buffer("atoms", torch.linspace(zmin, zmax, steps=n_atoms))
        self.n_actions = n_actions
        self.n_atoms = n_atoms

    def forward(self, x):
        # Base MLP
        h = x
        for layer in self.layers:
            h = self.relu(layer(h))
        if not self.dueling:
            shape = []
            if h.ndim > 1:
                shape.append(h.shape[0])
            shape = shape + [self.n_actions, self.n_atoms]
            out = self.out_layer(h).view(shape)
            return out
        # Dueling: compute per-atom value and advantages, then combine
        if h.ndim == 1:
            # Single sample path
            v = self.value_layer(h).view(1, self.n_atoms)  # [1, N]
            a = self.advantage_layer(h).view(self.n_actions, self.n_atoms)  # [A, N]
            a_mean = a.mean(dim=0, keepdim=True)  # [1, N]
            q = a - a_mean + v  # broadcast v to [A, N]
            return q
        else:
            # Batched path
            bsz = h.shape[0]
            v = self.value_layer(h).view(bsz, 1, self.n_atoms)  # [B,1,N]
            a = self.advantage_layer(h).view(
                bsz, self.n_actions, self.n_atoms
            )  # [B,A,N]
            a_mean = a.mean(dim=1, keepdim=True)  # [B,1,N]
            q = a - a_mean + v  # [B,A,N]
            return q

    def expected_value(self, x: torch.Tensor, probs: bool = False):
        assert (
            x.shape[-1] == self.n_atoms
        ), f"x with shape: {x.shape} does not match self.n_atoms {self.n_atoms} at dim -1"
        view_dim = [1] * (x.ndim - 1)
        view_dim.append(self.n_atoms)
        atom_view = self.atoms.view(view_dim)  # type: ignore
        if probs:
            return (x * atom_view).sum(-1)
        return (torch.softmax(x, dim=-1) * atom_view).sum(-1)


class RainbowDQN:
    """Maintains online (current) and target Q_Networks and training logic."""

    def __init__(
        self,
        input_dim,
        n_actions,
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
        self.online = Q_Network(
            input_dim,
            n_actions,
            hidden_layer_sizes=hidden_layer_sizes,
            n_atoms=n_atoms,
            zmin=zmin,
            zmax=zmax,
            dueling=dueling,
        ).float()
        self.target = Q_Network(
            input_dim,
            n_actions,
            hidden_layer_sizes=hidden_layer_sizes,
            n_atoms=n_atoms,
            zmin=zmin,
            zmax=zmax,
            dueling=dueling,
        ).float()
        # Intrinsic Q networks (separate head trained on intrinsic rewards only)
        self.int_online = Q_Network(
            input_dim,
            n_actions,
            hidden_layer_sizes=hidden_layer_sizes,
            n_atoms=n_atoms,
            zmin=zmin,
            zmax=zmax,
            dueling=dueling,
        ).float()
        self.int_target = Q_Network(
            input_dim,
            n_actions,
            hidden_layer_sizes=hidden_layer_sizes,
            n_atoms=n_atoms,
            zmin=zmin,
            zmax=zmax,
            dueling=dueling,
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
        self.n_atoms = n_atoms
        self.zmin = zmin
        self.zmax = zmax
        self.n_actions = n_actions
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

    @torch.no_grad()
    def _project_next_distribution(
        self,
        b_r,
        b_term,
        b_next_obs,
        hist_logits=None,
        hist_actions=None,
        net_online: Q_Network = None,  # type: ignore
        net_target: Q_Network = None,  # type: ignore
    ):
        # Select which networks to use (extrinsic by default)
        if net_online is None:
            net_online = self.online
        if net_target is None:
            net_target = self.target if self.delayed_target else net_online
        # Prepare atom view
        atom_view_dim = [1] * (b_next_obs.ndim - 1)
        atom_view_dim.append(self.n_atoms)
        expanded_atoms = net_online.atoms.view(atom_view_dim)  # type: ignore

        # Double DQN-style: policy from online expected values, evaluation by target
        online_next_logits = net_online(b_next_obs)
        next_ev_online = net_online.expected_value(online_next_logits, probs=False)
        eval_logits = net_target(b_next_obs)
        next_probs = torch.softmax(eval_logits, dim=-1)  # [B, A, N]

        if self.soft or self.munchausen:
            # Soft policy over actions using expected values
            pi_next = torch.softmax(next_ev_online / self.tau, dim=-1)  # [B, A]
            # Mixture distribution across actions
            next_value_probs = (pi_next.unsqueeze(-1) * next_probs).sum(dim=1)  # [B, N]
            # Munchausen entropy term for next state: -tau * sum_a pi(a) log pi(a)
            logpi_next = torch.log_softmax(next_ev_online / self.tau, dim=-1)
            ent_bonus = -self.tau * (pi_next * logpi_next).sum(dim=-1)  # [B]
        else:
            # Greedy backup as fallback
            next_actions = (
                torch.argmax(next_ev_online, dim=-1, keepdim=True)
                .unsqueeze(-1)
                .expand((b_next_obs.shape[0], 1, self.n_atoms))
            )
            next_value_probs = torch.gather(
                next_probs, dim=1, index=next_actions
            ).squeeze()  # [B, N]
            ent_bonus = torch.zeros(b_next_obs.shape[0], device=b_next_obs.device)

        # Munchausen reward augmentation on current transition
        r_aug = b_r
        if self.munchausen and (hist_logits is not None) and (hist_actions is not None):
            # Compute policy log-prob for taken actions at (s_t, a_t)
            ev_hist = net_online.expected_value(hist_logits, probs=False)  # [B, A]
            logpi_hist = torch.log_softmax(ev_hist / self.tau, dim=-1)  # [B, A]
            logpi_a = torch.gather(
                logpi_hist, dim=1, index=hist_actions.view(-1, 1)
            ).squeeze(1)
            logpi_a = torch.clamp(logpi_a, min=self.l_clip)
            r_aug = r_aug + self.alpha * self.tau * logpi_a

        # Bellman update on the support (C51 projection pre-step)
        ra = r_aug
        if isinstance(ra, torch.Tensor):
            ra = r_aug.unsqueeze(-1)
        shifted_atoms = ra + (1 - b_term.unsqueeze(-1)) * (self.gamma * expanded_atoms)
        if self.munchausen or self.soft:
            # Add soft-entropy correction as a constant shift
            shifted_atoms = shifted_atoms + (1 - b_term.unsqueeze(-1)) * (
                self.gamma * ent_bonus.unsqueeze(-1)
            )

        # Project to support indices
        float_index = (
            (torch.clamp(shifted_atoms, self.zmin, self.zmax) - self.zmin)
            / (self.zmax - self.zmin)
            * (self.n_atoms - 1)
        )
        lower_idx = torch.floor(float_index).clamp(0, self.n_atoms - 1)
        upper_idx = torch.ceil(float_index).clamp(0, self.n_atoms - 1)
        lower_weight = upper_idx - float_index
        upper_weight = 1.0 - lower_weight

        target_dist = torch.zeros_like(shifted_atoms)
        target_dist.scatter_add_(-1, lower_idx.long(), next_value_probs * lower_weight)
        target_dist.scatter_add_(-1, upper_idx.long(), next_value_probs * upper_weight)
        return target_dist

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

        # Combine rewards for extrinsic head

        b_r_total = b_r + self.Beta * r_int

        # 2) Update extrinsic Q network on combined rewards
        logits = self.online(b_obs)
        with torch.no_grad():
            target_dist = self._project_next_distribution(
                b_r_total,
                b_term,
                b_next_obs,
                hist_logits=logits,
                hist_actions=b_actions,
                net_online=self.online,
                net_target=self.target,
            )

        # Current logits for chosen actions
        actions_viewed = b_actions.view(batch_size, 1, 1).expand(
            (batch_size, 1, self.n_atoms)
        )
        selected_logits = torch.gather(logits, dim=1, index=actions_viewed).squeeze()

        # Cross-entropy between projected target distribution and current predicted logits
        current_l_probs = torch.log_softmax(selected_logits, dim=-1)
        loss = -torch.sum(target_dist * current_l_probs, dim=-1).mean()

        # Optional entropy regularization for current policy over actions at b_obs
        entropy_val = 0.0
        if self.ent_reg_coef > 0.0:
            # Policy over actions from expected values
            ev_all = self.online.expected_value(logits, probs=False)  # [B, A]
            pi = torch.softmax(ev_all / self.tau, dim=-1)
            logpi = torch.log_softmax(ev_all / self.tau, dim=-1)
            entropy = -(pi * logpi).sum(dim=-1).mean()
            entropy_val = float(entropy.item())
            loss = loss - self.ent_reg_coef * entropy

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # 3) Update intrinsic Q network on intrinsic rewards only
        int_logits = self.int_online(b_obs)
        with torch.no_grad():
            int_target_dist = self._project_next_distribution(
                r_int,
                b_term,
                b_next_obs,
                hist_logits=int_logits,
                hist_actions=b_actions,
                net_online=self.int_online,
                net_target=self.int_target,
            )

        int_selected_logits = torch.gather(
            int_logits, dim=1, index=actions_viewed
        ).squeeze()
        int_current_l_probs = torch.log_softmax(int_selected_logits, dim=-1)
        int_loss = -torch.sum(int_target_dist * int_current_l_probs, dim=-1).mean()

        self.int_optim.zero_grad()
        int_loss.backward()
        self.int_optim.step()

        if isinstance(r_int, torch.Tensor):
            r_int = float(r_int.mean().item())
        # Store last auxiliary losses for logging (optional)
        self.last_losses = {
            "extrinsic": float(loss.item()),
            "intrinsic": float(int_loss.item()),
            "rnd": float(rnd_loss.item()),
            "avg_r_int": r_int,
            "entropy_reg": entropy_val,
        }

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
        self, obs: torch.Tensor, eps: float, step: int, n_steps: int = 100000
    ):
        # linear annealing from 1.0 -> 0.0 across n_steps
        if self.Thompson:
            with torch.no_grad():
                logits_ext = self.online(obs)  # [A, N] or [N] per action when no batch
                logits_int = self.int_online(obs)
                probs_ext = torch.softmax(logits_ext, dim=-1)
                probs_int = torch.softmax(logits_int, dim=-1)
                atoms = self.online.atoms  # [N]

                sampled_idx_ext = torch.distributions.Categorical(
                    probs=probs_ext
                ).sample()

                sampled_values_int = 0.0
                if self.Beta > 0.0:
                    sampled_idx_int = torch.distributions.Categorical(
                        probs=probs_int
                    ).sample()
                    sampled_values_int = atoms[sampled_idx_int]  # type: ignore

                sampled_values_ext = atoms[sampled_idx_ext]  # type: ignore
                sampled_values = sampled_values_ext + self.Beta * sampled_values_int
                action = torch.argmax(sampled_values).item()
        elif self.soft or self.munchausen:
            if random.random() <= 0.05:
                return random.randint(0, self.n_actions - 1)
            logits_ext = self.online(obs)
            ev_ext = self.online.expected_value(logits_ext)
            ev_int = 0.0
            if self.Beta > 0.0:
                logits_int = self.int_online(obs)
                ev_int = self.int_online.expected_value(logits_int)
            ev_comb = ev_ext + self.Beta * ev_int
            action = (
                torch.distributions.Categorical(logits=ev_comb / self.tau)
                .sample()
                .item()
            )
        else:
            eps = 1 - step / n_steps
            if random.random() < eps:
                action = random.randint(0, self.n_actions - 1)
            else:
                with torch.no_grad():
                    logits_ext = self.online(obs)
                    ev_ext = self.online.expected_value(logits_ext)
                    ev_int = 0.0
                    if self.Beta > 0.0:
                        logits_int = self.int_online(obs)
                        ev_int = self.int_online.expected_value(logits_int)
                    ev_comb = ev_ext + self.Beta * ev_int
                    action = torch.argmax(ev_comb, dim=-1).item()
        return action


class EVRainbowDQN:
    """Non-distributional counterpart to RainbowDQN with optional dueling and all five pillars."""

    def __init__(
        self,
        input_dim,
        n_actions,
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
        self.online = EV_Q_Network(
            input_dim,
            n_actions,
            hidden_layer_sizes=hidden_layer_sizes,
            dueling=dueling,
        ).float()
        self.target = EV_Q_Network(
            input_dim,
            n_actions,
            hidden_layer_sizes=hidden_layer_sizes,
            dueling=dueling,
        ).float()
        self.int_online = EV_Q_Network(
            input_dim,
            n_actions,
            hidden_layer_sizes=hidden_layer_sizes,
            dueling=dueling,
        ).float()
        self.int_target = EV_Q_Network(
            input_dim,
            n_actions,
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
        self.n_actions = n_actions

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
        q_now = self.online(b_obs)  # [B, A]
        q_next_online = self.online(b_next_obs)
        q_next_target = (self.target if self.delayed_target else self.online)(
            b_next_obs
        )

        # Munchausen augmentation
        r_aug = b_r_total
        if self.munchausen:
            ev_hist = q_now  # [B, A]
            _, logpi_hist, _ = self._soft_policy(ev_hist)
            logpi_a = torch.gather(
                logpi_hist, dim=1, index=b_actions.view(-1, 1)
            ).squeeze(1)
            logpi_a = torch.clamp(logpi_a, min=self.l_clip)
            r_aug = r_aug + self.alpha * self.tau * logpi_a

        # Compute next-state value
        if self.soft or self.munchausen:
            pi_next, logpi_next, ent_next = self._soft_policy(q_next_online)
            v_next = (pi_next * q_next_target).sum(dim=1)
            v_next = v_next + ent_next  # add entropy bonus
        else:
            next_actions = torch.argmax(q_next_online, dim=-1, keepdim=True)
            v_next = torch.gather(q_next_target, dim=1, index=next_actions).squeeze(1)

        y = r_aug + (1 - b_term) * (self.gamma * v_next)

        # Extrinsic loss
        q_selected = torch.gather(q_now, dim=1, index=b_actions.view(-1, 1)).squeeze(1)
        extrinsic_loss = torch.nn.functional.smooth_l1_loss(q_selected, y)

        # Entropy regularization
        entropy_val = 0.0
        if self.ent_reg_coef > 0.0:
            pi, logpi, _ = self._soft_policy(q_now)
            entropy = -(pi * logpi).sum(dim=-1).mean()
            entropy_val = float(entropy.item())
            extrinsic_loss = extrinsic_loss - self.ent_reg_coef * entropy

        self.optim.zero_grad()
        extrinsic_loss.backward()
        self.optim.step()

        # Intrinsic Q update
        int_q_now = self.int_online(b_obs)
        int_q_next_online = self.int_online(b_next_obs)
        int_q_next_target = (
            self.int_target if self.delayed_target else self.int_online
        )(b_next_obs)

        r_int_only = r_int if isinstance(r_int, torch.Tensor) else torch.zeros_like(b_r)
        if self.soft or self.munchausen:
            pi_next_i, logpi_next_i, ent_next_i = self._soft_policy(int_q_next_online)
            v_next_i = (pi_next_i * int_q_next_target).sum(dim=1) + ent_next_i
        else:
            next_a_i = torch.argmax(int_q_next_online, dim=-1, keepdim=True)
            v_next_i = torch.gather(int_q_next_target, dim=1, index=next_a_i).squeeze(1)
        y_i = r_int_only + (1 - b_term) * (self.gamma * v_next_i)
        int_q_sel = torch.gather(int_q_now, 1, index=b_actions.view(-1, 1)).squeeze(1)
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
        self, obs: torch.Tensor, eps: float, step: int, n_steps: int = 100000
    ):
        if self.soft or self.munchausen:
            q_ext = self.online(obs)
            q_int = self.int_online(obs) if self.Beta > 0.0 else 0.0
            q_comb = q_ext + self.Beta * q_int
            return (
                torch.distributions.Categorical(logits=q_comb / self.tau)
                .sample()
                .item()
            )
        # Epsilon greedy (with optional Thompson via Gumbel noise)
        eps = 1 - step / n_steps
        if torch.rand(()) < eps:
            return int(torch.randint(0, self.n_actions, ()).item())
        with torch.no_grad():
            q_ext = self.online(obs)
            q_int = self.int_online(obs) if self.Beta > 0.0 else 0.0
            q_comb = q_ext + self.Beta * q_int
            if self.Thompson:
                # Approximate posterior sampling by adding Gumbel noise
                g = -torch.log(-torch.log(torch.rand_like(q_comb)))
                q_comb = q_comb + g
            return torch.argmax(q_comb, dim=-1).item()
