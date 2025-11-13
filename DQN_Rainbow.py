import torch
import torch.nn as nn
import gymnasium as gym
import random
import matplotlib.pyplot as plt


class Q_Network(nn.Module):
    def __init__(
        self,
        input_dim,
        n_actions,
        hidden_layer_sizes=[128, 128],
        n_atoms=101,
        zmin=-50,
        zmax=50,
    ):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_layer_sizes[0]))
        for li in range(len(hidden_layer_sizes) - 1):
            layers.append(nn.Linear(hidden_layer_sizes[li], hidden_layer_sizes[li + 1]))
        self.out_layer = nn.Linear(hidden_layer_sizes[-1], n_actions * n_atoms)
        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()
        self.zmax = zmax
        self.zmin = zmin
        # Register the support as a buffer so it moves with the module/device
        self.register_buffer("atoms", torch.linspace(zmin, zmax, steps=n_atoms))
        self.n_actions = n_actions
        self.n_atoms = n_atoms

    def forward(self, x):
        shape = []
        if x.ndim > 1:
            shape.append(x.shape[0])
        shape = shape + [self.n_actions, self.n_atoms]
        for layer in self.layers:
            x = self.relu(layer(x))
        x = self.out_layer(x).view(shape)
        return x

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
    ):
        self.online = Q_Network(
            input_dim,
            n_actions,
            hidden_layer_sizes=hidden_layer_sizes,
            n_atoms=n_atoms,
            zmin=zmin,
            zmax=zmax,
        ).float()
        self.target = Q_Network(
            input_dim,
            n_actions,
            hidden_layer_sizes=hidden_layer_sizes,
            n_atoms=n_atoms,
            zmin=zmin,
            zmax=zmax,
        ).float()
        self.alpha = alpha
        self.tau = tau
        self.l_clip = l_clip
        self.soft = soft
        self.munchausen = munchausen
        self.Thompson = Thompson
        self.target.requires_grad_(False)
        self.target.load_state_dict(self.online.state_dict())

        self.optim = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.gamma = gamma
        self.n_atoms = n_atoms
        self.zmin = zmin
        self.zmax = zmax
        self.n_actions = n_actions

    def update_target(self):
        """Polyak averaging: target = (1 - tau) * target + tau * online."""
        with torch.no_grad():
            for tp, op in zip(self.target.parameters(), self.online.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(op.data, alpha=self.tau)

    @torch.no_grad()
    def _project_next_distribution(
        self, b_r, b_term, b_next_obs, hist_logits=None, hist_actions=None
    ):
        # Prepare atom view
        atom_view_dim = [1] * (b_next_obs.ndim - 1)
        atom_view_dim.append(self.n_atoms)
        expanded_atoms = self.online.atoms.view(atom_view_dim)  # type: ignore

        # Double DQN-style: policy from online expected values, evaluation by target
        online_next_logits = self.online(b_next_obs)
        next_ev_online = self.online.expected_value(online_next_logits, probs=False)
        eval_logits = self.target(b_next_obs)
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
            ev_hist = self.online.expected_value(hist_logits, probs=False)  # [B, A]
            logpi_hist = torch.log_softmax(ev_hist / self.tau, dim=-1)  # [B, A]
            logpi_a = torch.gather(
                logpi_hist, dim=1, index=hist_actions.view(-1, 1)
            ).squeeze(1)
            logpi_a = torch.clamp(logpi_a, min=self.l_clip)
            r_aug = r_aug + self.alpha * self.tau * logpi_a

        # Bellman update on the support (C51 projection pre-step)
        shifted_atoms = r_aug.unsqueeze(-1) + (1 - b_term.unsqueeze(-1)) * (
            self.gamma * expanded_atoms
        )
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
        logits = self.online(b_obs)
        with torch.no_grad():
            target_dist = self._project_next_distribution(
                b_r, b_term, b_next_obs, hist_logits=logits, hist_actions=b_actions
            )

        # Current logits for chosen actions
        actions_viewed = b_actions.view(batch_size, 1, 1).expand(
            (batch_size, 1, self.n_atoms)
        )
        selected_logits = torch.gather(logits, dim=1, index=actions_viewed).squeeze()

        # Cross-entropy between projected target distribution and current predicted logits
        current_l_probs = torch.log_softmax(selected_logits, dim=-1)
        loss = -torch.sum(target_dist * current_l_probs, dim=-1).mean()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item()

    def sample_action(
        self, obs: torch.Tensor, eps: float, step: int, n_steps: int = 100000
    ):
        # linear annealing from 1.0 -> 0.0 across n_steps
        if self.Thompson:
            with torch.no_grad():
                logits = self.online(obs)  # [A, N]
                probs = torch.softmax(logits, dim=-1)  # [A, N]
                # Sample one atom per action
                sampled_indices = torch.distributions.Categorical(
                    probs=probs
                ).sample()  # [A]
                # Map indices to support values
                atoms = self.online.atoms  # [N]
                sampled_values = atoms[sampled_indices]  # type: ignore
                action = torch.argmax(sampled_values).item()
        elif self.soft or self.munchausen:
            logits = self.online(obs)
            ev = self.online.expected_value(logits)
            action = (
                torch.distributions.Categorical(logits=ev / self.tau).sample().item()
            )
        else:
            eps = 1 - step / n_steps
            if random.random() < eps:
                action = random.randint(0, self.n_actions - 1)
            else:
                with torch.no_grad():
                    logits = self.online(obs)
                    ev = self.online.expected_value(logits)
                    action = torch.argmax(ev, dim=-1).item()
        return action
