# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl_buffers import ReplayBuffer
from agent import Agent
from RainbowNetworks import EV_Q_Network, IQN_Network
from runner_utilities import make_env_thunk


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    hidden_layer_sizes: tuple[int, int] = (256, 256)
    """two hidden layer sizes for actor/critic MLPs"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    entropy_coef_zero: bool = False
    """if True, force entropy coefficient alpha=0 and disable autotune"""
    distributional: bool = False
    """if True, use IQN critics; else use expected-value critics"""
    dueling: bool = False
    """if True, enable dueling critic heads"""
    popart: bool = False
    """if True, use PopArt-scaled critic outputs"""
    delayed_critics: bool = True
    """if True, use delayed target critics; else use online critics as targets"""
    n_quantiles: int = 32
    """number of quantiles for IQN critics"""
    n_target_quantiles: int = 32
    """number of target quantiles for IQN critics"""
    random_start_steps: int = 0
    """max random steps sampled in [0, n] after each reset to desynchronize parallel envs"""

LOG_STD_MAX = 2
LOG_STD_MIN = -5


def hidden_layer_sizes_for_env_id(env_id: str) -> tuple[int, int]:
    env = env_id.lower()
    if "mujoco" in env or "cheetah" in env or "hopper" in env or "walker" in env or "ant" in env or "humanoid" in env:
        return (128, 128)
    if "cartpole" in env:
        return (32, 32)
    return (128, 128)


class Actor(nn.Module):
    def __init__(self, env, hidden_layer_sizes: tuple[int, int] = (256, 256)):
        super().__init__()
        obs_dim = int(np.array(env.single_observation_space.shape).prod())
        act_dim = int(np.prod(env.single_action_space.shape))
        hidden1, hidden2 = int(hidden_layer_sizes[0]), int(hidden_layer_sizes[1])
        self.fc1 = nn.Linear(obs_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc_mean = nn.Linear(hidden2, act_dim)
        self.fc_logstd = nn.Linear(hidden2, act_dim)
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class SACAgent(Agent):
    def __init__(
        self,
        envs,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_lr: float = 3e-4,
        q_lr: float = 1e-3,
        policy_frequency: int = 2,
        target_network_frequency: int = 1,
        alpha: float = 0.2,
        autotune: bool = True,
        entropy_coef_zero: bool = False,
        distributional: bool = False,
        dueling: bool = False,
        popart: bool = False,
        delayed_critics: bool = True,
        hidden_layer_sizes: tuple[int, int] = (128, 128),
        n_quantiles: int = 32,
        n_target_quantiles: int = 32,
    ):
        super().__init__()
        self.gamma = gamma
        self.tau = tau
        self.policy_frequency = policy_frequency
        self.target_network_frequency = target_network_frequency
        self.autotune = autotune and (not entropy_coef_zero)
        self.entropy_coef_zero = entropy_coef_zero
        self.distributional = distributional
        self.dueling = dueling
        self.popart = popart
        self.delayed_critics = delayed_critics
        hidden1, hidden2 = int(hidden_layer_sizes[0]), int(hidden_layer_sizes[1])
        self.n_quantiles = n_quantiles
        self.n_target_quantiles = n_target_quantiles

        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        critic_input_dim = obs_dim + act_dim

        self.actor = Actor(envs, hidden_layer_sizes=(hidden1, hidden2))

        CriticClass = IQN_Network if self.distributional else EV_Q_Network
        critic_kwargs = dict(
            input_dim=critic_input_dim,
            n_action_dims=1,
            n_action_bins=1,
            hidden_layer_sizes=[hidden1, hidden2],
            dueling=self.dueling,
            popart=self.popart,
        )
        self.qf1 = CriticClass(**critic_kwargs)
        self.qf2 = CriticClass(**critic_kwargs)
        self.qf1_target = CriticClass(**critic_kwargs)
        self.qf2_target = CriticClass(**critic_kwargs)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        self.q_optimizer = optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=q_lr
        )
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=policy_lr)

        if self.autotune:
            self.target_entropy = -torch.prod(
                torch.Tensor(envs.single_action_space.shape)
            ).item()
            self.log_alpha = nn.Parameter(torch.zeros(1))
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=q_lr)
        else:
            self.target_entropy = None
            self.log_alpha = None
            self.a_optimizer = None
            self.alpha = 0.0 if self.entropy_coef_zero else alpha

        self.step = 0

    def to(self, device):
        device = torch.device(device)
        self.actor.to(device)
        self.qf1.to(device)
        self.qf2.to(device)
        self.qf1_target.to(device)
        self.qf2_target.to(device)
        if self.autotune and self.log_alpha is not None:
            self.log_alpha = nn.Parameter(self.log_alpha.detach().to(device))
            self.target_entropy = torch.tensor(
                float(self.target_entropy), dtype=torch.float32, device=device
            )
            self.alpha = self.log_alpha.exp().item()
            q_lr = self.q_optimizer.param_groups[0]["lr"]
            self.a_optimizer = optim.Adam([self.log_alpha], lr=q_lr)
        return self

    @torch.no_grad()
    def sample_action(self, obs, deterministic: bool = False):
        if isinstance(obs, np.ndarray):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.actor.fc1.weight.device)
        else:
            obs_t = obs.to(device=self.actor.fc1.weight.device, dtype=torch.float32)

        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)
            single = True
        else:
            single = False

        if deterministic:
            _, _, action_mean = self.actor.get_action(obs_t)
            action = action_mean
        else:
            action, _, _ = self.actor.get_action(obs_t)

        action_np = action.detach().cpu().numpy()
        if single:
            return action_np[0]
        return action_np

    def update_target(self):
        if not self.delayed_critics:
            return
        for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def _sample_taus(self, batch_size: int, n: int, device: torch.device):
        return torch.rand(batch_size, n, device=device)

    def _quantile_huber_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        taus: torch.Tensor,
        kappa: float = 1.0,
    ) -> torch.Tensor:
        td = target.unsqueeze(1) - pred.unsqueeze(2)
        abs_td = torch.abs(td)
        huber = torch.where(
            abs_td <= kappa, 0.5 * td.pow(2), kappa * (abs_td - 0.5 * kappa)
        )
        I = (td < 0).float()
        taus_expanded = taus.unsqueeze(2)
        return (torch.abs(taus_expanded - I) * huber).mean()

    def _critic_input(self, obs: torch.Tensor, act: torch.Tensor):
        return torch.cat([obs, act.to(dtype=obs.dtype)], dim=1)

    def _critic_scalar_value(self, critic, critic_input: torch.Tensor, normalized: bool = False):
        q = critic(critic_input, normalized=normalized)
        return q.view(-1)

    def _critic_quantiles(self, critic, critic_input: torch.Tensor, taus: torch.Tensor, normalized: bool = False):
        q = critic(critic_input, taus, normalized=normalized)
        return q.view(q.shape[0], q.shape[1])

    def update(self, data, global_step: int):
        target_qf1 = self.qf1_target if self.delayed_critics else self.qf1
        target_qf2 = self.qf2_target if self.delayed_critics else self.qf2

        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.actor.get_action(
                data.next_observations
            )
            next_critic_input = self._critic_input(
                data.next_observations, next_state_actions
            )

            if self.distributional:
                target_taus = self._sample_taus(
                    next_critic_input.shape[0],
                    self.n_target_quantiles,
                    next_critic_input.device,
                )
                qf1_next_quantiles = self._critic_quantiles(
                    target_qf1, next_critic_input, target_taus, normalized=False
                )
                qf2_next_quantiles = self._critic_quantiles(
                    target_qf2, next_critic_input, target_taus, normalized=False
                )
                min_qf_next_quantiles = torch.min(qf1_next_quantiles, qf2_next_quantiles)
                next_q_quantiles = (
                    data.rewards.flatten().unsqueeze(1)
                    + (1 - data.dones.flatten()).unsqueeze(1)
                    * self.gamma
                    * (min_qf_next_quantiles - self.alpha * next_state_log_pi)
                )
            else:
                qf1_next_target = self._critic_scalar_value(
                    target_qf1, next_critic_input, normalized=False
                )
                qf2_next_target = self._critic_scalar_value(
                    target_qf2, next_critic_input, normalized=False
                )
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.gamma * (
                    min_qf_next_target - self.alpha * next_state_log_pi.view(-1)
                )

        critic_input = self._critic_input(data.observations, data.actions)
        if self.distributional:
            taus = self._sample_taus(
                critic_input.shape[0], self.n_quantiles, critic_input.device
            )
            if self.popart:
                self.qf1.output_layer.update_stats(next_q_quantiles)
                self.qf2.output_layer.update_stats(next_q_quantiles)
                target_quantiles_1 = self.qf1.output_layer.normalize(next_q_quantiles)
                target_quantiles_2 = self.qf2.output_layer.normalize(next_q_quantiles)
                qf1_quantiles = self._critic_quantiles(
                    self.qf1, critic_input, taus, normalized=True
                )
                qf2_quantiles = self._critic_quantiles(
                    self.qf2, critic_input, taus, normalized=True
                )
            else:
                target_quantiles_1 = next_q_quantiles
                target_quantiles_2 = next_q_quantiles
                qf1_quantiles = self._critic_quantiles(
                    self.qf1, critic_input, taus, normalized=False
                )
                qf2_quantiles = self._critic_quantiles(
                    self.qf2, critic_input, taus, normalized=False
                )
            qf1_loss = self._quantile_huber_loss(
                qf1_quantiles, target_quantiles_1.detach(), taus.detach()
            )
            qf2_loss = self._quantile_huber_loss(
                qf2_quantiles, target_quantiles_2.detach(), taus.detach()
            )
            qf1_a_values = qf1_quantiles.mean(dim=1)
            qf2_a_values = qf2_quantiles.mean(dim=1)
        else:
            if self.popart:
                target_for_stats = next_q_value.unsqueeze(1)
                self.qf1.output_layer.update_stats(target_for_stats)
                self.qf2.output_layer.update_stats(target_for_stats)
                next_q_value_norm_1 = self.qf1.output_layer.normalize(target_for_stats).view(-1)
                next_q_value_norm_2 = self.qf2.output_layer.normalize(target_for_stats).view(-1)
                qf1_a_values = self._critic_scalar_value(
                    self.qf1, critic_input, normalized=True
                )
                qf2_a_values = self._critic_scalar_value(
                    self.qf2, critic_input, normalized=True
                )
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value_norm_1)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value_norm_2)
            else:
                qf1_a_values = self._critic_scalar_value(
                    self.qf1, critic_input, normalized=False
                )
                qf2_a_values = self._critic_scalar_value(
                    self.qf2, critic_input, normalized=False
                )
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)

        qf_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        actor_loss = None
        alpha_loss = None

        if global_step % self.policy_frequency == 0:
            for _ in range(self.policy_frequency):
                pi, log_pi, _ = self.actor.get_action(data.observations)
                pi_critic_input = self._critic_input(data.observations, pi)
                if self.distributional:
                    pi_taus = self._sample_taus(
                        pi_critic_input.shape[0], self.n_quantiles, pi_critic_input.device
                    )
                    qf1_pi = self._critic_quantiles(
                        self.qf1, pi_critic_input, pi_taus, normalized=False
                    ).mean(dim=1, keepdim=True)
                    qf2_pi = self._critic_quantiles(
                        self.qf2, pi_critic_input, pi_taus, normalized=False
                    ).mean(dim=1, keepdim=True)
                else:
                    qf1_pi = self._critic_scalar_value(
                        self.qf1, pi_critic_input, normalized=False
                    ).unsqueeze(1)
                    qf2_pi = self._critic_scalar_value(
                        self.qf2, pi_critic_input, normalized=False
                    ).unsqueeze(1)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                if self.autotune and self.log_alpha is not None and self.a_optimizer is not None:
                    with torch.no_grad():
                        _, log_pi, _ = self.actor.get_action(data.observations)
                    alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

                    self.a_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.a_optimizer.step()
                    self.alpha = self.log_alpha.exp().item()

        if global_step % self.target_network_frequency == 0:
            self.update_target()

        self.step = global_step
        self.last_losses = {
            "qf1_values": float(qf1_a_values.mean().item()),
            "qf2_values": float(qf2_a_values.mean().item()),
            "qf1_loss": float(qf1_loss.item()),
            "qf2_loss": float(qf2_loss.item()),
            "qf_loss": float((qf_loss / 2.0).item()),
            "actor_loss": float(actor_loss.item()) if actor_loss is not None else 0.0,
            "alpha": float(self.alpha),
            "alpha_loss": float(alpha_loss.item()) if alpha_loss is not None else 0.0,
            "distributional": float(self.distributional),
            "popart": float(self.popart),
            "dueling": float(self.dueling),
            "delayed_critics": float(self.delayed_critics),
        }

        if self.tb_writer is not None and global_step % 100 == 0:
            for k, v in self.last_losses.items():
                self.tb_writer.add_scalar(f"{self.tb_prefix}/losses/{k}", v, global_step)

        return float(qf_loss.item())


if __name__ == "__main__":

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env_thunk(
                fully_obs=False,
                env_name="mujoco",
                env_id=args.env_id,
                seed=args.seed + i,
                idx=i,
                capture_video=args.capture_video,
                run_name=run_name,
                random_start_steps=args.random_start_steps,
            )
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    hidden_layer_sizes = hidden_layer_sizes_for_env_id(args.env_id)
    agent = SACAgent(
        envs,
        gamma=args.gamma,
        tau=args.tau,
        policy_lr=args.policy_lr,
        q_lr=args.q_lr,
        policy_frequency=args.policy_frequency,
        target_network_frequency=args.target_network_frequency,
        alpha=args.alpha,
        autotune=args.autotune,
        entropy_coef_zero=args.entropy_coef_zero,
        distributional=args.distributional,
        dueling=args.dueling,
        popart=args.popart,
        delayed_critics=args.delayed_critics,
        hidden_layer_sizes=hidden_layer_sizes,
        n_quantiles=args.n_quantiles,
        n_target_quantiles=args.n_target_quantiles,
    ).to(device)
    agent.attach_tensorboard(writer, prefix="agent")

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    env_steps = 0
    updates_performed = 0
    while env_steps < args.total_timesteps:
        # ALGO LOGIC: put action logic here
        if env_steps < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions = agent.sample_action(obs)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    print(f"global_step={env_steps}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], env_steps)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], env_steps)
                    break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        env_steps += args.num_envs

        # ALGO LOGIC: training.
        if env_steps > args.learning_starts:
            target_updates = env_steps // 8
            while updates_performed < target_updates:
                if rb.size() < args.batch_size:
                    break
                data = rb.sample(args.batch_size)
                agent.update(data, env_steps)
                updates_performed += 1

            if env_steps % 100 == 0:
                writer.add_scalar(
                    "losses/qf1_values", agent.last_losses.get("qf1_values", 0.0), env_steps
                )
                writer.add_scalar(
                    "losses/qf2_values", agent.last_losses.get("qf2_values", 0.0), env_steps
                )
                writer.add_scalar(
                    "losses/qf1_loss", agent.last_losses.get("qf1_loss", 0.0), env_steps
                )
                writer.add_scalar(
                    "losses/qf2_loss", agent.last_losses.get("qf2_loss", 0.0), env_steps
                )
                writer.add_scalar(
                    "losses/qf_loss", agent.last_losses.get("qf_loss", 0.0), env_steps
                )
                writer.add_scalar(
                    "losses/actor_loss", agent.last_losses.get("actor_loss", 0.0), env_steps
                )
                writer.add_scalar("losses/alpha", agent.last_losses.get("alpha", 0.0), env_steps)
                print("SPS:", int(env_steps / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(env_steps / (time.time() - start_time)),
                    env_steps,
                )
                if args.autotune:
                    writer.add_scalar(
                        "losses/alpha_loss", agent.last_losses.get("alpha_loss", 0.0), env_steps
                    )

    envs.close()
    writer.close()