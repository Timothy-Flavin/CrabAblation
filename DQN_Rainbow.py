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
        atom_view = self.atoms.view(view_dim)
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
        l: float = -1.0,
        soft: bool = False,
        munchausen: bool = False,
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
        self.l = l
        self.soft = soft
        self.munchausen = munchausen
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
    def _project_next_distribution(self, b_r, b_term, b_next_obs, hist_logits=None):
        # Prepare atom view
        atom_view_dim = [1] * (b_next_obs.ndim - 1)
        atom_view_dim.append(self.n_atoms)
        expanded_atoms = self.online.atoms.view(atom_view_dim)

        # Double DQN action selection via online, evaluation via target
        online_next_logits = self.online(b_next_obs)
        next_ev_online = self.online.expected_value(online_next_logits, probs=False)
        next_actions = (
            torch.argmax(next_ev_online, dim=-1, keepdim=True)
            .unsqueeze(-1)
            .expand((b_next_obs.shape[0], 1, self.n_atoms))
        )

        eval_logits = self.target(b_next_obs)
        next_probs = torch.softmax(eval_logits, dim=-1)
        max_value_probs = torch.gather(next_probs, dim=1, index=next_actions).squeeze()

        # Bellman update on the support (C51 projection pre-step)
        shifted_atoms = (
            b_r.unsqueeze(-1) + (1 - b_term.unsqueeze(-1)) * self.gamma * expanded_atoms
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
        target_dist.scatter_add_(-1, lower_idx.long(), max_value_probs * lower_weight)
        target_dist.scatter_add_(-1, upper_idx.long(), max_value_probs * upper_weight)
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
                b_r, b_term, b_next_obs, logits
            )

        # Current logits for chosen actions
        selected_logits = torch.gather(
            logits, dim=1, index=b_actions.view(-1, 1, 1).expand(-1, 1, self.n_atoms)
        ).squeeze()

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

        if self.soft or self.munchausen:
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


if __name__ == "__main__":

    def eval(agent: RainbowDQN):
        lenv = gym.make("CartPole-v1")
        obs, info = lenv.reset()
        done = False
        reward = 0.0
        while not done:
            with torch.no_grad():
                logits = agent.online(torch.from_numpy(obs))
                ev = agent.online.expected_value(logits)
                action = torch.argmax(ev, dim=-1).item()
            obs, r, term, trunc, info = lenv.step(action)
            reward += float(r)
            done = term or trunc
        return reward

    env = gym.make("CartPole-v1")
    obs, info = env.reset()
    assert env.observation_space.shape is not None
    obs_dim = env.observation_space.shape[0]
    assert obs_dim is not None

    dqn = RainbowDQN(obs_dim, 2, zmin=0, zmax=200, n_atoms=51)

    n_steps = 100000
    rhist = []
    smooth_rhist = []
    lhist = []
    eval_hist = []
    r_ep = 0.0
    smooth_r = 0.0
    ep = 0
    blen = 10000
    update_every = 4

    # Memory buffer
    buff_actions = torch.zeros((blen,), dtype=torch.long)
    buff_obs = torch.zeros(size=(blen, int(obs_dim)), dtype=torch.float32)
    buff_next_obs = torch.zeros(size=(blen, int(obs_dim)), dtype=torch.float32)
    buff_term = torch.zeros(size=(blen,), dtype=torch.float32)
    buff_r = torch.zeros(size=(blen,), dtype=torch.float32)

    for i in range(n_steps):
        eps_current = 1 - i / n_steps
        action = dqn.sample_action(
            torch.from_numpy(obs), eps=eps_current, step=i, n_steps=n_steps
        )
        next_obs, r, term, trunc, info = env.step(action)

        # Save transition to memory buffer
        buff_actions[i % blen] = action
        buff_obs[i % blen] = torch.from_numpy(obs)
        buff_next_obs[i % blen] = torch.from_numpy(next_obs)
        buff_term[i % blen] = term
        buff_r[i % blen] = float(r)

        if i > 512:
            lhist.append(
                dqn.update(
                    buff_obs,
                    buff_actions,
                    buff_r,
                    buff_next_obs,
                    buff_term,
                    batch_size=64,
                    step=min(i, blen),
                )
            )
            if i % update_every == 0:
                dqn.update_target()

        if i % 1000 == 0:
            evalr = 0.0
            for k in range(5):
                evalr += eval(dqn)
            print(f"eval mean: {evalr/5}")
            eval_hist.append(evalr / 5)

        r_ep += float(r)

        if term or trunc:
            next_obs, info = env.reset()
            rhist.append(r_ep)
            if len(rhist) < 20:
                print(f"reward for episode: {ep}: {r_ep}")
                smooth_r = sum(rhist) / len(rhist)
            else:
                smooth_r = 0.05 * rhist[-1] + 0.95 * smooth_r
                print(
                    f"smooth reward for episode: {ep}: {smooth_r} at eps: {eps_current}"
                )
            smooth_rhist.append(smooth_r)
            r_ep = 0.0
            ep += 1

        obs = next_obs

    plt.plot(rhist)
    plt.plot(smooth_rhist)
    plt.legend(["R hist", "Smooth R hist"])
    plt.xlabel("Episode")
    plt.ylabel("Training Episode Reward")
    plt.grid()
    plt.show()

    plt.plot(eval_hist)
    plt.grid()
    plt.title("eval scores")
    plt.show()
