# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
import copy
from dataclasses import dataclass
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from learning_algorithms.MixedObservationEncoder import infer_encoder_out_dim
from learning_algorithms.cleanrl_buffers import ReplayBuffer
from learning_algorithms.agent import Agent
from learning_algorithms.RainbowNetworks import EV_Q_Network, IQN_Network
from learning_algorithms.RandomDistilation import RNDModel, RunningMeanStd
from environment_utils import make_env_thunk
import sys
import platform

if sys.platform == "darwin" and platform.machine() == "x86_64":
    # This makes all @torch.compile calls return the original function
    import torch._dynamo

    torch._dynamo.config.disable = True


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
    wandb_entity: str | None = None
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
    learning_starts: int = int(1e3)
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
    alpha: float = 0.01
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    entropy_coef_zero: bool = False
    """if True, force entropy coefficient alpha=0 and disable autotune"""
    distributional: bool = False
    """if True, use IQN critics; else use expected-value critics"""
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


@torch.compile
def old_quantile_huber_loss(pred, target, taus, kappa=1.0):
    td = target.unsqueeze(1) - pred.unsqueeze(2)
    abs_td = torch.abs(td)
    huber = torch.where(
        abs_td <= kappa, 0.5 * td.pow(2), kappa * (abs_td - 0.5 * kappa)
    )
    I_ = (td < 0).float()
    return (torch.abs(taus.unsqueeze(2) - I_) * huber).mean()


def fused_quantile_huber_loss(pred, target, taus, kappa=1.0):
    # F.smooth_l1_loss is exactly the Huber loss, natively fused in CUDA
    huber = F.smooth_l1_loss(
        pred.unsqueeze(2), target.unsqueeze(1), reduction="none", beta=kappa
    )

    # We still need the TD sign for the quantile weighting
    td = target.unsqueeze(1) - pred.unsqueeze(2)
    I_ = (td < 0).float()

    return (torch.abs(taus.unsqueeze(2) - I_) * huber).mean()


class ObsActEncoder(nn.Module):
    """Encode observation slice, then concatenate untouched action slice."""

    def __init__(self, obs_encoder: nn.Module, obs_dim: int, act_dim: int):
        super().__init__()
        self.obs_encoder = obs_encoder
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.output_dim = (
            infer_encoder_out_dim(obs_encoder, self.obs_dim) + self.act_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        obs = x[..., : self.obs_dim]
        act = x[..., self.obs_dim : self.obs_dim + self.act_dim]
        obs_features = self.obs_encoder(obs)
        return torch.cat([obs_features, act], dim=-1)


def hidden_layer_sizes_for_env_id(env_id: str) -> tuple[int, int]:
    env = env_id.lower()
    if (
        "mujoco" in env
        or "cheetah" in env
        or "hopper" in env
        or "walker" in env
        or "ant" in env
        or "humanoid" in env
    ):
        return (128, 128)
    if "cartpole" in env:
        return (32, 32)
    return (128, 128)


class Actor(nn.Module):
    def __init__(
        self,
        env,
        hidden_layer_sizes: tuple[int, int] = (256, 256),
        encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        obs_dim = int(np.array(env.single_observation_space.shape).prod())
        act_dim = int(np.prod(env.single_action_space.shape))
        hidden1, hidden2 = int(hidden_layer_sizes[0]), int(hidden_layer_sizes[1])
        if encoder is None:
            self.encoder = nn.Sequential(
                nn.Linear(obs_dim, hidden1),
                nn.ReLU(),
                nn.Linear(hidden1, hidden2),
                nn.ReLU(),
            )
            head_in_dim = hidden2
        else:
            self.encoder = encoder
            head_in_dim = infer_encoder_out_dim(encoder, obs_dim)

        self.fc_mean = nn.Linear(head_in_dim, act_dim)
        self.fc_logstd = nn.Linear(head_in_dim, act_dim)
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
        features = self.encoder(x)
        mean = self.fc_mean(features)
        log_std = self.fc_logstd(features)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return mean, log_std

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias  # type:ignore
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(
            self.action_scale * (1 - y_t.pow(2)) + 1e-6  # type:ignore
        )
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias  # type:ignore
        return action, log_prob, mean


class BaseSAC(Agent):
    """
    Abstract Base class for Soft Actor-Critic.
    Defines the standard update loops, buffer management, RND, and Munchausen logic.
    Subclasses must implement critic initialization and target/loss logic.
    """

    distributional = False

    def __init__(
        self,
        envs,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_lr: float = 3e-4,
        q_lr: float = 1e-3,
        policy_frequency: int = 2,
        target_network_frequency: int = 1,
        alpha: float = 0.01,
        autotune: bool = True,
        entropy_coef_zero: bool = False,
        delayed_critics: bool = True,
        hidden_layer_sizes: tuple[int, int] = (128, 128),
        n_quantiles: int = 32,
        n_target_quantiles: int = 32,
        encoder_factory: Optional[Callable[[], nn.Module]] = None,
        munchausen: bool = True,
        beta_rnd: float = 0.01,
        munchausen_constant: float = 0.1,
        l_clip: float = -10.0,
        rnd_output_dim: int = 128,
        rnd_lr: float = 1e-3,
        beta_half_life_steps: Optional[int] = None,
        buffer_size: int = int(1e5),
        device: str = "cpu",
        buffer_device: str = "cpu",
        min_std: float = 0.01,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.buffer_device = torch.device(buffer_device)
        if not torch.cuda.is_available():
            self.device = "cpu"
            self.buffer_device = "cpu"
        self.update_steps = 0

        self.gamma = gamma
        self.tau = tau
        self.policy_frequency = policy_frequency
        self.target_network_frequency = target_network_frequency
        self.autotune = autotune and (not entropy_coef_zero)
        self.entropy_coef_zero = entropy_coef_zero
        self.delayed_critics = delayed_critics
        hidden1, hidden2 = int(hidden_layer_sizes[0]), int(hidden_layer_sizes[1])
        self.n_quantiles = n_quantiles
        self.n_target_quantiles = n_target_quantiles

        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        critic_input_dim = obs_dim + act_dim

        self.buffer = ReplayBuffer(
            buffer_size,
            observation_space=envs.single_observation_space,
            action_space=envs.single_action_space,
            device=self.buffer_device,
            n_envs=envs.num_envs,
            handle_timeout_termination=True,
            optimize_memory_usage=False,
        )

        actor_encoder = encoder_factory() if encoder_factory is not None else None
        self.actor = Actor(
            envs,
            hidden_layer_sizes=(hidden1, hidden2),
            encoder=actor_encoder,
        )

        critic_kwargs = {
            "input_dim": critic_input_dim,
            "n_action_dims": 1,
            "n_action_bins": 1,
            "hidden_layer_sizes": [hidden1, hidden2],
            "dueling": False,
            "popart": True,
            "min_std": min_std,
        }

        def _critic_encoder_kwargs():
            if encoder_factory is None:
                return {}
            obs_encoder = encoder_factory()
            obs_act_encoder = ObsActEncoder(
                obs_encoder, obs_dim=obs_dim, act_dim=act_dim
            )
            return {
                "encoder": obs_act_encoder,
                "encoder_out_dim": int(obs_act_encoder.output_dim),
            }

        qf1_kwargs = dict(critic_kwargs)
        qf1_kwargs.update(_critic_encoder_kwargs())
        qf2_kwargs = dict(critic_kwargs)
        qf2_kwargs.update(_critic_encoder_kwargs())
        qf1_target_kwargs = dict(critic_kwargs)
        qf1_target_kwargs.update(_critic_encoder_kwargs())
        qf2_target_kwargs = dict(critic_kwargs)
        qf2_target_kwargs.update(_critic_encoder_kwargs())

        # Subclasses handle the specific instantiations
        self._init_critics(qf1_kwargs, qf2_kwargs, qf1_target_kwargs, qf2_target_kwargs)

        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.qf1_target_int.load_state_dict(self.qf1_int.state_dict())
        self.qf2_target_int.load_state_dict(self.qf2_int.state_dict())

        # self.q_optimizer = optim.Adam(
        #     list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=q_lr
        # )
        # self.q_int_optimizer = optim.Adam(
        #     list(self.qf1_int.parameters()) + list(self.qf2_int.parameters()), lr=q_lr
        # )
        self.q_total_optimizer = optim.Adam(
            list(self.qf1.parameters())
            + list(self.qf2.parameters())
            + list(self.qf1_int.parameters())
            + list(self.qf2_int.parameters()),
            lr=q_lr,
        )

        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=policy_lr)

        if self.autotune:
            self.target_entropy = -torch.prod(
                torch.Tensor(envs.single_action_space.shape)
            ).item()
            self.log_alpha = nn.Parameter(torch.zeros(1) - 4)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=q_lr)
        else:
            self.target_entropy = None
            self.log_alpha = None
            self.a_optimizer = None
            self.alpha = 0.0 if self.entropy_coef_zero else alpha

        self.step = 0
        self.timing = {}

        # Munchausen KL penalty
        self.munchausen = munchausen
        self.munchausen_constant = munchausen_constant
        self.l_clip = l_clip

        # RND intrinsic reward
        self.Beta = beta_rnd
        self.start_Beta = beta_rnd
        self.beta_half_life_steps = beta_half_life_steps
        self.rnd = RNDModel(obs_dim, rnd_output_dim).float()
        self.rnd_optim = optim.Adam(self.rnd.predictor.parameters(), lr=rnd_lr)
        self.obs_rms = RunningMeanStd(shape=(obs_dim,))

    def _init_critics(
        self, qf1_kwargs, qf2_kwargs, qf1_target_kwargs, qf2_target_kwargs
    ):
        raise NotImplementedError

    def _compute_targets(
        self,
        next_critic_input,
        augmented_rewards,
        int_r,
        terminations,
        next_state_log_pi,
        current_sigma,
    ):
        raise NotImplementedError

    def _compute_critic_losses(self, critic_input, next_q, next_q_int):
        raise NotImplementedError

    def _get_actor_q_values(self, pi_critic_input):
        raise NotImplementedError

    def to(self, device):
        device = torch.device(device)
        self.device = device

        if hasattr(self, "buffer") and self.buffer is not None:
            self.buffer.device = device
            self.buffer_device = device

        self.actor.to(device)
        self.qf1.to(device)
        self.qf2.to(device)
        self.qf1_target.to(device)
        self.qf2_target.to(device)
        self.qf1_int.to(device)
        self.qf2_int.to(device)
        self.qf1_target_int.to(device)
        self.qf2_target_int.to(device)

        if self.autotune and self.log_alpha is not None:
            self.log_alpha = nn.Parameter(self.log_alpha.detach().to(device))
            self.target_entropy = torch.tensor(
                float(self.target_entropy),  # type:ignore
                dtype=torch.float32,
                device=device,
            )
            self.alpha = self.log_alpha.exp().item()
            q_lr = self.q_total_optimizer.param_groups[0]["lr"]
            self.a_optimizer = optim.Adam([self.log_alpha], lr=q_lr)

        policy_lr = self.actor_optimizer.param_groups[0]["lr"]
        q_lr = self.q_total_optimizer.param_groups[0]["lr"]
        self.q_total_optimizer = optim.Adam(
            list(self.qf1.parameters())
            + list(self.qf2.parameters())
            + list(self.qf1_int.parameters())
            + list(self.qf2_int.parameters()),
            lr=q_lr,
        )
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=policy_lr)

        self.rnd.to(device)
        self.obs_rms.to(device)
        rnd_lr = self.rnd_optim.param_groups[0]["lr"]
        self.rnd_optim = optim.Adam(self.rnd.predictor.parameters(), lr=rnd_lr)
        return self

    def buffer_to(self, device):
        self.buffer_device = torch.device(device)
        if hasattr(self, "buffer") and self.buffer is not None:
            self.buffer.device = self.buffer_device
        if hasattr(self, "obs_rms") and hasattr(self.obs_rms, "to"):
            self.obs_rms.to(self.buffer_device)

    @torch.no_grad()
    def sample_action(self, obs, deterministic: bool = False):
        t0 = time.time()
        actor_device = self.actor.device
        if isinstance(obs, np.ndarray):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=actor_device)
        else:
            obs_t = obs.to(device=actor_device, dtype=torch.float32)

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
        self.timing["action sampling"] = self.timing.get("action sampling", 0.0) + (
            time.time() - t0
        )
        if single:
            return action_np[0]
        return action_np

    def observe(self, obs, action, reward, next_obs, terminated, truncated, info=None):
        t0 = time.time()
        if self.Beta > 0.0 or getattr(self, "always_update_rnd", False):
            if isinstance(next_obs, np.ndarray):
                b_next_obs = torch.as_tensor(
                    next_obs, dtype=torch.float32, device=self.buffer_device
                )
            else:
                b_next_obs = (
                    next_obs.clone()
                    .detach()
                    .to(device=self.buffer_device, dtype=torch.float32)
                )

            if b_next_obs.ndim == len(self.buffer.obs_shape):
                b_next_obs = b_next_obs.unsqueeze(0)

            self.obs_rms.update(
                b_next_obs.to(dtype=torch.float32, device=self.obs_rms.mean.device)
            )

        self.buffer.add(obs, next_obs, action, reward, terminated, truncated)
        self.timing["observe"] = self.timing.get("observe", 0.0) + (time.time() - t0)

    def _critic_input(self, obs: torch.Tensor, act: torch.Tensor):
        return torch.cat([obs, act.to(dtype=obs.dtype)], dim=1)

    @torch.no_grad()
    def update_target(self):
        if not self.delayed_critics:
            return

        for online, target in [
            (self.qf1, self.qf1_target),
            (self.qf2, self.qf2_target),
            (self.qf1_int, self.qf1_target_int),
            (self.qf2_int, self.qf2_target_int),
        ]:
            # Multiplies target weights by (1 - tau) in-place
            torch._foreach_mul_(list(target.parameters()), 1.0 - self.tau)
            # Adds online weights * tau in-place
            torch._foreach_add_(
                list(target.parameters()), list(online.parameters()), alpha=self.tau
            )

            target_buffers = list(target.buffers())
            online_buffers = list(online.buffers())
            if len(target_buffers) > 0:
                torch._foreach_mul_(target_buffers, 1.0 - self.tau)
                torch._foreach_add_(target_buffers, online_buffers, alpha=self.tau)

    def update(self, batch_size: int, global_step: int):
        data = self.buffer.sample(batch_size)
        self.update_steps += 1
        if self.beta_half_life_steps is not None and self.beta_half_life_steps > 0:
            self.Beta = self.start_Beta * (
                0.5 ** (global_step / self.beta_half_life_steps)
            )

        augmented_rewards = data.rewards.flatten()
        terminations = data.terminations.flatten()
        rnd_loss_val = 0.0

        t0 = time.time()
        if self.Beta > 0.0 or getattr(self, "always_update_rnd", False):
            with torch.no_grad():
                norm_next_obs = self.obs_rms.normalize(
                    data.next_observations.to(dtype=torch.float32)
                ).to(dtype=torch.float32)
            rnd_errors = self.rnd(norm_next_obs)
            rnd_loss = rnd_errors.mean()
            self.rnd_optim.zero_grad()
            rnd_loss.backward()
            self.rnd_optim.step()

            with torch.no_grad():
                int_r = torch.clamp(rnd_errors.detach(), -5.0, 5.0).squeeze()
        else:
            int_r = torch.zeros(
                data.rewards.shape[0], dtype=torch.float32, device=data.rewards.device
            )
        self.timing["updating the random network distilation (rnd)"] = self.timing.get(
            "updating the random network distilation (rnd)", 0.0
        ) + (time.time() - t0)
        current_sigma = self.qf1.output_layer.sigma.detach()

        t0 = time.time()
        m_r_val = 0.0
        if self.munchausen:
            with torch.no_grad():
                mean, log_std = self.actor(data.observations)
                std = log_std.exp()
                normal = torch.distributions.Normal(mean, std)
                a_norm = (
                    data.actions - self.actor.action_bias
                ) / self.actor.action_scale
                a_norm = torch.clamp(a_norm, -0.9999, 0.9999)
                x_pretanh = torch.atanh(a_norm)
                log_prob = normal.log_prob(x_pretanh)
                log_prob -= torch.log(
                    self.actor.action_scale * (1 - a_norm.pow(2)) + 1e-6
                )
                log_pi_replay = log_prob.sum(1)
                m_r = (
                    current_sigma
                    * self.alpha
                    * self.munchausen_constant
                    * torch.clamp(log_pi_replay, min=self.l_clip)
                )
            augmented_rewards = augmented_rewards + m_r

        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.actor.get_action(
                data.next_observations
            )
            next_critic_input = self._critic_input(
                data.next_observations, next_state_actions
            )

            next_q, next_q_int = self._compute_targets(
                next_critic_input,
                augmented_rewards,
                int_r,
                terminations,
                next_state_log_pi,
                current_sigma,
            )

            self._update_popart(next_q, next_q_int)
        self.timing["getting q ext q int and targets for q ext and q int"] = (
            self.timing.get("getting q ext q int and targets for q ext and q int", 0.0)
            + (time.time() - t0)
        )

        t0 = time.time()
        critic_input = self._critic_input(data.observations, data.actions)

        qf1_loss, qf2_loss, qf1_int_loss, qf2_int_loss, critic_logs = (
            self._compute_critic_losses(critic_input, next_q, next_q_int)
        )
        self.timing["_compute_critic_losses"] = self.timing.get(
            "_compute_critic_losses", 0.0
        ) + (time.time() - t0)

        t0 = time.time()
        q_loss = qf1_loss + qf2_loss + qf1_int_loss + qf2_int_loss
        self.q_total_optimizer.zero_grad()
        q_loss.backward()
        self.q_total_optimizer.step()
        self.timing["critic backprop"] = self.timing.get("critic backprop", 0.0) + (
            time.time() - t0
        )

        actor_loss = None
        alpha_loss = None
        min_qf_pi = None
        min_qf_pi_int = None

        t0 = time.time()
        if self.update_steps % self.policy_frequency == 0:

            # Freeze critics to avoid unnecessary gradient computation during actor update
            self.qf1.requires_grad_(False)
            self.qf2.requires_grad_(False)
            self.qf1_int.requires_grad_(False)
            self.qf2_int.requires_grad_(False)

            t1 = time.time()
            pi, log_pi, _ = self.actor.get_action(data.observations)
            pi_critic_input = self._critic_input(data.observations, pi)

            min_qf_pi, min_qf_pi_int = self._get_actor_q_values(pi_critic_input)

            actor_loss = (
                (self.alpha * log_pi)
                - ((1.0 - self.Beta) * min_qf_pi + self.Beta * min_qf_pi_int)
            ).mean()
            self.timing["actor forward and loss"] = self.timing.get(
                "actor forward and loss", 0.0
            ) + (time.time() - t1)

            t1 = time.time()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.timing["actor backprop"] = self.timing.get("actor backprop", 0.0) + (
                time.time() - t1
            )

            # Unfreeze critics
            self.qf1.requires_grad_(True)
            self.qf2.requires_grad_(True)
            self.qf1_int.requires_grad_(True)
            self.qf2_int.requires_grad_(True)

            t1 = time.time()
            if (
                self.autotune
                and self.log_alpha is not None
                and self.a_optimizer is not None
            ):
                with torch.no_grad():
                    _, log_pi, _ = self.actor.get_action(data.observations)
                alpha_loss = (
                    -self.log_alpha.exp() * (log_pi + self.target_entropy)
                ).mean()  # type:ignore

                self.a_optimizer.zero_grad()
                alpha_loss.backward()
                self.a_optimizer.step()
                self.alpha = self.log_alpha.exp().item()
            self.timing["alpha loss and backprop"] = self.timing.get(
                "alpha loss and backprop", 0.0
            ) + (time.time() - t1)

        self.timing["final loss calculation and backward()"] = self.timing.get(
            "final loss calculation and backward()", 0.0
        ) + (time.time() - t0)

        t0 = time.time()
        if self.update_steps % self.target_network_frequency == 0:
            self.update_target()
        self.timing["update target"] = self.timing.get("update target", 0.0) + (
            time.time() - t0
        )

        if self.update_steps % 100 == 0:
            t0 = time.time()
            self.step = global_step
            if self.Beta > 0.0:
                rnd_loss_val = float(rnd_loss.item())
            if self.munchausen:
                m_r_val = float(m_r.mean().item())
            self.last_losses = {
                "min_qf_pi": (
                    float(min_qf_pi.mean().item()) if min_qf_pi is not None else 0.0
                ),
                "min_qf_pi_int": (
                    float(min_qf_pi_int.mean().item())
                    if min_qf_pi_int is not None
                    else 0.0
                ),
                "qf1_values": float(critic_logs["qf1_values"].mean().item()),
                "qf2_values": float(critic_logs["qf2_values"].mean().item()),
                "qf1_loss": float(qf1_loss.item()),
                "qf2_loss": float(qf2_loss.item()),
                "qf_loss": float((q_loss / 2.0).item()),
                "actor_loss": (
                    float(actor_loss.item()) if actor_loss is not None else 0.0
                ),
                "alpha": float(self.alpha),
                "alpha_loss": (
                    float(alpha_loss.item()) if alpha_loss is not None else 0.0
                ),
                "distributional": float(self.distributional),
                "delayed_critics": float(self.delayed_critics),
                "rnd_loss": rnd_loss_val,
                "munchausen_r": m_r_val,
                "Beta": float(self.Beta),
                "nextq": float(next_q.mean().item()),
                "nextintq": float(next_q_int.mean().item()),
            }
            self.timing["last_losses"] = self.timing.get("last_losses", 0.0) + (
                time.time() - t0
            )
            # print(self.last_losses)
            # print(f"{self.timing}")
        return float(q_loss.item())


class DistSAC(BaseSAC):
    distributional = True

    def _init_critics(
        self, qf1_kwargs, qf2_kwargs, qf1_target_kwargs, qf2_target_kwargs
    ):
        self.qf1 = IQN_Network(**qf1_kwargs)
        self.qf2 = IQN_Network(**qf2_kwargs)
        self.qf1_target = IQN_Network(**qf1_target_kwargs)
        self.qf2_target = IQN_Network(**qf2_target_kwargs)

        self.qf1_int = IQN_Network(**qf1_kwargs)
        self.qf2_int = IQN_Network(**qf2_kwargs)
        self.qf1_target_int = IQN_Network(**qf1_target_kwargs)
        self.qf2_target_int = IQN_Network(**qf2_target_kwargs)

    def _sample_taus(self, batch_size: int, n: int, device: torch.device):
        return torch.rand(batch_size, n, device=device)

    def _critic_quantiles(
        self,
        critic,
        critic_input: torch.Tensor,
        taus: torch.Tensor,
        normalized: bool = False,
    ):
        q = critic(critic_input, taus, normalized=normalized)
        return q.view(q.shape[0], q.shape[1])

    def _compute_targets(
        self,
        next_critic_input,
        augmented_rewards,
        int_r,
        terminations,
        next_state_log_pi,
        current_sigma,
    ):
        batch_size = next_critic_input.shape[0]
        assert augmented_rewards.shape == (
            batch_size,
        ), f"Expected augmented_rewards shape ({batch_size},), got {augmented_rewards.shape}"
        assert int_r.shape == (
            batch_size,
        ), f"Expected int_r shape ({batch_size},), got {int_r.shape}"
        assert terminations.shape == (
            batch_size,
        ), f"Expected terminations shape ({batch_size},), got {terminations.shape}"

        target_qf1 = self.qf1_target if self.delayed_critics else self.qf1
        target_qf2 = self.qf2_target if self.delayed_critics else self.qf2
        target_qf1_int = self.qf1_target_int if self.delayed_critics else self.qf1_int
        target_qf2_int = self.qf2_target_int if self.delayed_critics else self.qf2_int

        target_taus = self._sample_taus(
            next_critic_input.shape[0], self.n_target_quantiles, self.actor.device
        )

        qf1_next = self._critic_quantiles(
            target_qf1, next_critic_input, target_taus, normalized=False
        )
        qf2_next = self._critic_quantiles(
            target_qf2, next_critic_input, target_taus, normalized=False
        )

        # Maintain quantiles, compute element-wise min across the ensemble
        min_qf_next = torch.min(qf1_next, qf2_next)

        aug_r = augmented_rewards.unsqueeze(1)
        dones_mask = (1 - terminations).unsqueeze(1)
        assert next_state_log_pi.shape == (
            batch_size,
            1,
        ), f"Expected next_state_log_pi shape ({batch_size}, 1), got {next_state_log_pi.shape}"

        # next_state_log_pi properly broadcasts against min_qf_next (batch, target_quantiles)
        next_q = aug_r + dones_mask * self.gamma * (
            min_qf_next - current_sigma * self.alpha * next_state_log_pi
        )

        qf1_next_int = self._critic_quantiles(
            target_qf1_int, next_critic_input, target_taus, normalized=False
        )
        qf2_next_int = self._critic_quantiles(
            target_qf2_int, next_critic_input, target_taus, normalized=False
        )
        min_qf_next_int = torch.min(qf1_next_int, qf2_next_int)

        int_r_expanded = int_r.unsqueeze(1) if int_r.ndim == 1 else int_r
        if int_r_expanded.shape[1] != 1:
            int_r_expanded = int_r_expanded.unsqueeze(1)
        next_q_int = int_r_expanded + self.gamma * min_qf_next_int

        return next_q, next_q_int

    def _compute_critic_losses(
        self, critic_input, next_q_quantiles, next_q_quantiles_int
    ):
        taus = self._sample_taus(
            critic_input.shape[0], self.n_quantiles, critic_input.device
        )

        target_1 = self.qf1.output_layer.normalize(next_q_quantiles)
        target_2 = self.qf2.output_layer.normalize(next_q_quantiles)
        target_1_int = self.qf1_int.output_layer.normalize(next_q_quantiles_int)
        target_2_int = self.qf2_int.output_layer.normalize(next_q_quantiles_int)

        qf1_q = self._critic_quantiles(self.qf1, critic_input, taus, normalized=True)
        qf2_q = self._critic_quantiles(self.qf2, critic_input, taus, normalized=True)
        qf1_q_int = self._critic_quantiles(
            self.qf1_int, critic_input, taus, normalized=True
        )
        qf2_q_int = self._critic_quantiles(
            self.qf2_int, critic_input, taus, normalized=True
        )

        qf1_loss = fused_quantile_huber_loss(qf1_q, target_1.detach(), taus.detach())
        qf2_loss = fused_quantile_huber_loss(qf2_q, target_2.detach(), taus.detach())
        qf1_int_loss = fused_quantile_huber_loss(
            qf1_q_int, target_1_int.detach(), taus.detach()
        )
        qf2_int_loss = fused_quantile_huber_loss(
            qf2_q_int, target_2_int.detach(), taus.detach()
        )

        logs = {"qf1_values": qf1_q.mean(dim=1), "qf2_values": qf2_q.mean(dim=1)}
        return qf1_loss, qf2_loss, qf1_int_loss, qf2_int_loss, logs

    def _get_actor_q_values(self, pi_critic_input):
        taus = self._sample_taus(
            pi_critic_input.shape[0], self.n_quantiles, pi_critic_input.device
        )

        qf1_pi = self._critic_quantiles(
            self.qf1, pi_critic_input, taus, normalized=True
        ).mean(dim=1, keepdim=True)
        qf2_pi = self._critic_quantiles(
            self.qf2, pi_critic_input, taus, normalized=True
        ).mean(dim=1, keepdim=True)

        qf1_pi_int = self._critic_quantiles(
            self.qf1_int, pi_critic_input, taus, normalized=True
        ).mean(dim=1, keepdim=True)
        qf2_pi_int = self._critic_quantiles(
            self.qf2_int, pi_critic_input, taus, normalized=True
        ).mean(dim=1, keepdim=True)

        return torch.min(qf1_pi, qf2_pi), torch.min(qf1_pi_int, qf2_pi_int)

    def _update_popart(self, q, qi):
        # Apply PopArt updates immediately
        qm = q.detach().mean(-1)
        qim = qi.detach().mean(-1)
        self.qf1.output_layer.update_stats(qm)
        self.qf2.output_layer.update_stats(qm)
        self.qf1_int.output_layer.update_stats(qim)
        self.qf2_int.output_layer.update_stats(qim)


class EVSAC(BaseSAC):
    distributional = False

    def _init_critics(
        self, qf1_kwargs, qf2_kwargs, qf1_target_kwargs, qf2_target_kwargs
    ):
        self.qf1 = EV_Q_Network(**qf1_kwargs)
        self.qf2 = EV_Q_Network(**qf2_kwargs)
        self.qf1_target = EV_Q_Network(**qf1_target_kwargs)
        self.qf2_target = EV_Q_Network(**qf2_target_kwargs)

        self.qf1_int = EV_Q_Network(**qf1_kwargs)
        self.qf2_int = EV_Q_Network(**qf2_kwargs)
        self.qf1_target_int = EV_Q_Network(**qf1_target_kwargs)
        self.qf2_target_int = EV_Q_Network(**qf2_target_kwargs)

    def _critic_scalar_value(
        self, critic, critic_input: torch.Tensor, normalized: bool = False
    ):
        q = critic(critic_input, normalized=normalized)
        return q.view(-1)

    def _compute_targets(
        self,
        next_critic_input,
        augmented_rewards,
        int_r,
        terminations,
        next_state_log_pi,
        current_sigma,
    ):
        batch_size = next_critic_input.shape[0]
        assert augmented_rewards.shape == (
            batch_size,
        ), f"Expected augmented_rewards shape ({batch_size},), got {augmented_rewards.shape}"
        assert int_r.shape == (
            batch_size,
        ), f"Expected int_r shape ({batch_size},), got {int_r.shape}"
        assert terminations.shape == (
            batch_size,
        ), f"Expected terminations shape ({batch_size},), got {terminations.shape}"

        target_qf1 = self.qf1_target if self.delayed_critics else self.qf1
        target_qf2 = self.qf2_target if self.delayed_critics else self.qf2
        target_qf1_int = self.qf1_target_int if self.delayed_critics else self.qf1_int
        target_qf2_int = self.qf2_target_int if self.delayed_critics else self.qf2_int

        qf1_next = self._critic_scalar_value(
            target_qf1, next_critic_input, normalized=False
        )
        qf2_next = self._critic_scalar_value(
            target_qf2, next_critic_input, normalized=False
        )
        assert qf1_next.shape == (
            batch_size,
        ), f"Expected qf1_next shape ({batch_size},), got {qf1_next.shape}"
        min_qf_next = torch.min(qf1_next, qf2_next)

        next_state_log_pi = next_state_log_pi.view(-1)
        assert next_state_log_pi.shape == (
            batch_size,
        ), f"Expected next_state_log_pi shape ({batch_size},), got {next_state_log_pi.shape}"
        next_q = augmented_rewards + (1 - terminations) * self.gamma * (
            min_qf_next - current_sigma * self.alpha * next_state_log_pi
        )

        qf1_next_int = self._critic_scalar_value(
            target_qf1_int, next_critic_input, normalized=False
        )
        qf2_next_int = self._critic_scalar_value(
            target_qf2_int, next_critic_input, normalized=False
        )
        assert qf1_next_int.shape == (
            batch_size,
        ), f"Expected qf1_next_int shape ({batch_size},), got {qf1_next_int.shape}"
        min_qf_next_int = torch.min(qf1_next_int, qf2_next_int)

        next_q_int = int_r + self.gamma * min_qf_next_int
        return next_q, next_q_int

    def _compute_critic_losses(self, critic_input, next_q, next_q_int):
        target_1 = self.qf1.output_layer.normalize(next_q.unsqueeze(1)).view(-1)
        target_2 = self.qf2.output_layer.normalize(next_q.unsqueeze(1)).view(-1)
        target_1_int = self.qf1_int.output_layer.normalize(
            next_q_int.unsqueeze(1)
        ).view(-1)
        target_2_int = self.qf2_int.output_layer.normalize(
            next_q_int.unsqueeze(1)
        ).view(-1)

        qf1_v = self._critic_scalar_value(self.qf1, critic_input, normalized=True)
        qf2_v = self._critic_scalar_value(self.qf2, critic_input, normalized=True)
        qf1_v_int = self._critic_scalar_value(
            self.qf1_int, critic_input, normalized=True
        )
        qf2_v_int = self._critic_scalar_value(
            self.qf2_int, critic_input, normalized=True
        )

        qf1_loss = F.mse_loss(qf1_v, target_1.detach())
        qf2_loss = F.mse_loss(qf2_v, target_2.detach())
        qf1_int_loss = F.mse_loss(qf1_v_int, target_1_int.detach())
        qf2_int_loss = F.mse_loss(qf2_v_int, target_2_int.detach())

        logs = {"qf1_values": qf1_v, "qf2_values": qf2_v}
        return qf1_loss, qf2_loss, qf1_int_loss, qf2_int_loss, logs

    def _get_actor_q_values(self, pi_critic_input):
        qf1_pi = self._critic_scalar_value(
            self.qf1, pi_critic_input, normalized=True
        ).unsqueeze(1)
        qf2_pi = self._critic_scalar_value(
            self.qf2, pi_critic_input, normalized=True
        ).unsqueeze(1)

        qf1_pi_int = self._critic_scalar_value(
            self.qf1_int, pi_critic_input, normalized=True
        ).unsqueeze(1)
        qf2_pi_int = self._critic_scalar_value(
            self.qf2_int, pi_critic_input, normalized=True
        ).unsqueeze(1)

        return torch.min(qf1_pi, qf2_pi), torch.min(qf1_pi_int, qf2_pi_int)

    def _update_popart(self, q, qi):
        # Apply PopArt updates immediately
        self.qf1.output_layer.update_stats(q)
        self.qf2.output_layer.update_stats(q)
        self.qf1_int.output_layer.update_stats(qi)
        self.qf2_int.output_layer.update_stats(qi)


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"

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
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    hidden_layer_sizes = hidden_layer_sizes_for_env_id(args.env_id)
    envs.single_observation_space.dtype = np.float32  # type:ignore

    AgentClass = DistSAC if args.distributional else EVSAC
    agent = AgentClass(
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
        delayed_critics=args.delayed_critics,
        hidden_layer_sizes=hidden_layer_sizes,
        n_quantiles=args.n_quantiles,
        n_target_quantiles=args.n_target_quantiles,
        buffer_size=args.buffer_size,
        device=device,
        buffer_device=device,
    ).to(device)

    agent.attach_tensorboard(writer, prefix="agent")

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    env_steps = 0
    updates_performed = 0
    while env_steps < args.total_timesteps:
        # ALGO LOGIC: put action logic here
        if env_steps < args.learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            actions = agent.sample_action(obs)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    print(
                        f"global_step={env_steps}, episodic_return={info['episode']['r']}"
                    )
                    writer.add_scalar(
                        "charts/episodic_return", info["episode"]["r"], env_steps
                    )
                    writer.add_scalar(
                        "charts/episodic_length", info["episode"]["l"], env_steps
                    )
                    break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        if "final_observation" in infos:
            for idx, trunc in enumerate(truncations):
                if trunc and infos["_final_observation"][idx]:
                    real_next_obs[idx] = infos["final_observation"][idx]

        # Use unified observer
        agent.observe(
            obs, actions, rewards, real_next_obs, terminations, truncations, infos
        )

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        env_steps += args.num_envs

        # ALGO LOGIC: training.
        if env_steps > args.learning_starts:
            target_updates = env_steps // 8
            while updates_performed < target_updates:
                if agent.buffer.size() < args.batch_size:
                    break
                agent.update(args.batch_size, env_steps)
                updates_performed += 1

            if env_steps % 100 == 0:
                writer.add_scalar(
                    "losses/qf1_values",
                    agent.last_losses.get("qf1_values", 0.0),
                    env_steps,
                )
                writer.add_scalar(
                    "losses/qf2_values",
                    agent.last_losses.get("qf2_values", 0.0),
                    env_steps,
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
                    "losses/actor_loss",
                    agent.last_losses.get("actor_loss", 0.0),
                    env_steps,
                )
                writer.add_scalar(
                    "losses/alpha", agent.last_losses.get("alpha", 0.0), env_steps
                )
                print("SPS:", int(env_steps / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(env_steps / (time.time() - start_time)),
                    env_steps,
                )
                if args.autotune:
                    writer.add_scalar(
                        "losses/alpha_loss",
                        agent.last_losses.get("alpha_loss", 0.0),
                        env_steps,
                    )

    envs.close()
    writer.close()
