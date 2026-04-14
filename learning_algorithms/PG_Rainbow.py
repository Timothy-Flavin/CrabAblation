# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from learning_algorithms.MixedObservationEncoder import infer_encoder_out_dim
from learning_algorithms.RainbowNetworks import IQN_Network
from learning_algorithms.PopArtLayer import PopArtLayer
from learning_algorithms.RandomDistilation import RNDModel, RunningMeanStd
from learning_algorithms.agent import Agent
import tyro


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
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    hidden_layer_sizes: tuple[int, int] = (64, 64)
    """two hidden layer sizes for actor/critic MLPs"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 256
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: None | float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOAgent(Agent):
    def __init__(
        self,
        envs,
        learning_rate: float = 2.5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        clip_vloss: bool = True,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        norm_adv: bool = True,
        target_kl: None | float = None,
        update_epochs: int = 4,
        num_minibatches: int = 4,
        num_envs: int = 4,
        num_steps: int = 128,
        hidden_layer_sizes: tuple[int, int] = (64, 64),
        anneal_lr=False,
        distributional: bool = False,
        popart: bool = True,
        Beta: float = 0.0,
        beta_half_life_steps=None,
        rnd_output_dim: int = 128,
        rnd_lr: float = 1e-3,
        intrinsic_lr: float = 2.5e-4,
        use_gae: bool = True,
        encoder_factory: Optional[Callable[[], nn.Module]] = None,
    ):
        super().__init__()
        self.anneal_lr = anneal_lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.use_gae = use_gae
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.norm_adv = norm_adv
        self.target_kl = target_kl
        self.update_epochs = update_epochs
        self.num_minibatches = num_minibatches

        self.distributional = distributional
        self.popart = popart
        self.Beta = Beta
        self.start_Beta = Beta
        self.beta_half_life_steps = beta_half_life_steps

        self.obs_shape = envs.single_observation_space.shape
        self.action_shape = envs.single_action_space.shape
        if isinstance(envs.single_action_space, gym.spaces.Box):
            self.n_action_dims = envs.single_action_space.shape[0]
            self.n_action_bins = 3
        else:
            self.n_action_dims = 1
            self.n_action_bins = envs.single_action_space.n

        self.batch_size = int(num_envs * num_steps)
        self.minibatch_size = int(self.batch_size // self.num_minibatches)
        self.num_steps = num_steps
        hidden1, hidden2 = int(hidden_layer_sizes[0]), int(hidden_layer_sizes[1])

        input_dim = np.array(self.obs_shape).prod()
        if encoder_factory is None:
            self.actor = nn.Sequential(
                layer_init(nn.Linear(input_dim, hidden1)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden1, hidden2)),
                nn.Tanh(),
                layer_init(
                    nn.Linear(hidden2, self.n_action_dims * self.n_action_bins),
                    std=0.01,
                ),
            )
        else:
            actor_encoder = encoder_factory()
            actor_out_dim = infer_encoder_out_dim(actor_encoder, int(input_dim))
            self.actor = nn.Sequential(
                actor_encoder,
                layer_init(
                    nn.Linear(actor_out_dim, self.n_action_dims * self.n_action_bins),
                    std=0.01,
                ),
            )

        if self.distributional:
            ext_critic_kwargs = {}
            int_critic_kwargs = {}
            if encoder_factory is not None:
                ext_encoder = encoder_factory()
                int_encoder = encoder_factory()
                ext_critic_kwargs = {
                    "encoder": ext_encoder,
                    "encoder_out_dim": infer_encoder_out_dim(
                        ext_encoder, int(input_dim)
                    ),
                }
                int_critic_kwargs = {
                    "encoder": int_encoder,
                    "encoder_out_dim": infer_encoder_out_dim(
                        int_encoder, int(input_dim)
                    ),
                }

            self.ext_critic = IQN_Network(
                input_dim=input_dim,
                n_action_dims=1,
                n_action_bins=1,
                hidden_layer_sizes=[hidden1, hidden2],
                dueling=False,
                popart=self.popart,
                **ext_critic_kwargs,
            )
            self.int_critic = IQN_Network(
                input_dim=input_dim,
                n_action_dims=1,
                n_action_bins=1,
                hidden_layer_sizes=[hidden1, hidden2],
                dueling=False,
                popart=self.popart,
                **int_critic_kwargs,
            )
        else:
            if encoder_factory is None:
                self.ext_critic_base = nn.Sequential(
                    layer_init(nn.Linear(input_dim, hidden1)),
                    nn.Tanh(),
                    layer_init(nn.Linear(hidden1, hidden2)),
                    nn.Tanh(),
                )
                self.int_critic_base = nn.Sequential(
                    layer_init(nn.Linear(input_dim, hidden1)),
                    nn.Tanh(),
                    layer_init(nn.Linear(hidden1, hidden2)),
                    nn.Tanh(),
                )
            else:
                ext_encoder = encoder_factory()
                int_encoder = encoder_factory()
                ext_out_dim = infer_encoder_out_dim(ext_encoder, int(input_dim))
                int_out_dim = infer_encoder_out_dim(int_encoder, int(input_dim))
                self.ext_critic_base = ext_encoder
                self.int_critic_base = int_encoder

            if self.popart:
                self.ext_critic_head = PopArtLayer(hidden2 if encoder_factory is None else ext_out_dim, 1)
                self.int_critic_head = PopArtLayer(hidden2 if encoder_factory is None else int_out_dim, 1)
            else:
                self.ext_critic_head = layer_init(nn.Linear(hidden2 if encoder_factory is None else ext_out_dim, 1), std=1.0)
                self.int_critic_head = layer_init(nn.Linear(hidden2 if encoder_factory is None else int_out_dim, 1), std=1.0)

            # Helper functions to act like a single module for your get_actions method
            def ext_critic_forward(x, normalized=False):
                base_out = self.ext_critic_base(x)
                if self.popart:
                    return self.ext_critic_head(base_out, normalized=normalized)
                return self.ext_critic_head(base_out)

            def int_critic_forward(x, normalized=False):
                base_out = self.int_critic_base(x)
                if self.popart:
                    return self.int_critic_head(base_out, normalized=normalized)
                return self.int_critic_head(base_out)

            self.ext_critic = ext_critic_forward
            self.int_critic = int_critic_forward

        self.rnd = RNDModel(input_dim, rnd_output_dim)
        self.obs_rms = RunningMeanStd(shape=self.obs_shape)
        self.int_rms = RunningMeanStd(shape=())
        self.ext_rms = RunningMeanStd(shape=())

        if self.distributional:
            self.optimizer = optim.Adam(
                list(self.actor.parameters()) + list(self.ext_critic.parameters()),
                lr=learning_rate, eps=1e-5,
            )
            self.int_optim = optim.Adam(
                self.int_critic.parameters(), lr=intrinsic_lr, eps=1e-5
            )
        else:
            self.optimizer = optim.Adam(
                list(self.actor.parameters()) + list(self.ext_critic_base.parameters()) + list(self.ext_critic_head.parameters()),
                lr=learning_rate, eps=1e-5,
            )
            self.int_optim = optim.Adam(
                list(self.int_critic_base.parameters()) + list(self.int_critic_head.parameters()), 
                lr=intrinsic_lr, eps=1e-5
            )
        self.rnd_optim = optim.Adam(self.rnd.predictor.parameters(), lr=rnd_lr)

        self.step = 0

    def to(self, device):
        self.actor.to(device)
        if self.distributional:
            self.ext_critic.to(device)
            self.int_critic.to(device)
        else:
            self.ext_critic_base.to(device)
            self.ext_critic_head.to(device)
            self.int_critic_base.to(device)
            self.int_critic_head.to(device)
        self.rnd.to(device)
        self.obs_rms.to(device)
        self.int_rms.to(device)
        self.ext_rms.to(device)
        return self

    def get_action_and_values(self, obs, action=None):
        logits = self.actor(obs)
        if self.n_action_dims > 1:
            logits = logits.view(-1, self.n_action_dims, self.n_action_bins)
            probs = Categorical(logits=logits)
            if action is None:
                action = probs.sample()
            log_prob = probs.log_prob(action).sum(dim=-1)
            entropy = probs.entropy().sum(dim=-1)
        else:
            probs = Categorical(logits=logits)
            if action is None:
                action = probs.sample()
            log_prob = probs.log_prob(action)
            entropy = probs.entropy()

        if self.distributional:
            taus = torch.rand(obs.shape[0], self.n_quantiles, device=obs.device)
            ext_v = self.ext_critic(obs, taus).mean(dim=1).view(-1, 1).squeeze(-1)
            int_v = self.int_critic(obs, taus).mean(dim=1).view(-1, 1).squeeze(-1)
        else:
            ext_v = self.ext_critic(obs).squeeze(-1)
            int_v = self.int_critic(obs).squeeze(-1)

        return action, log_prob, entropy, ext_v, int_v

    def sample_action(self, obs):
        with torch.no_grad():
            action, logprob, _, ext_v, int_v = self.get_action_and_values(obs)
        return action, logprob, ext_v, int_v

    @torch.no_grad()
    def update_running_stats(self, next_obs, r=None):
        # Flatten observations across the batch
        obs_flat = next_obs.view(-1, *self.obs_shape)
        x64 = obs_flat.to(dtype=torch.float64, device=self.obs_rms.mean.device)
        self.obs_rms.update(x64)

        norm_x64 = self.obs_rms.normalize(x64)
        rnd_err = self.rnd(norm_x64.to(dtype=torch.float32)).squeeze().detach()
        self.int_rms.update(rnd_err.to(dtype=torch.float64))
        if r is not None:
            self.ext_rms.update(r.reshape(-1).to(dtype=torch.float64))

    def _quantile_huber_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        taus: torch.Tensor,
        kappa: float = 1.0,
    ) -> torch.Tensor:
        # pred: [B, N], target: [B] or [B, 1]
        if target.dim() == 1:
            target = target.unsqueeze(1)  # [B, 1]

        td = target - pred  # [B, 1] - [B, N] -> [B, N]
        abs_td = torch.abs(td)
        huber = torch.where(
            abs_td <= kappa, 0.5 * td.pow(2), kappa * (abs_td - 0.5 * kappa)
        )
        I_ = (td < 0).float()
        loss = (torch.abs(taus - I_) * huber).sum(dim=1).mean()
        return loss

    def update(
        self,
        obs,
        actions,
        logprobs,
        rewards,
        dones,
        ext_values,
        int_values,
        next_obs,
        next_done,
    ):
        device = obs.device
        if self.anneal_lr:
            frac = 1.0 - (self.step - 1.0) / (
                self.update_epochs * 1000
            )  # Simple fallback decay
            lrnow = frac * self.optimizer.param_groups[0]["lr"]
            self.optimizer.param_groups[0]["lr"] = lrnow
            self.int_optim.param_groups[0]["lr"] = lrnow

        if self.beta_half_life_steps is not None and self.beta_half_life_steps > 0:
            self.Beta = self.start_Beta * (
                0.5 ** (self.step / self.beta_half_life_steps)
            )

        # 1. Update RND running stats and compute intrinsic rewards
        next_obs_batch = torch.cat([obs[1:], next_obs.unsqueeze(0)], dim=0)
        flat_next_obs = next_obs_batch.reshape(-1, *self.obs_shape)

        self.update_running_stats(flat_next_obs, rewards)

        # Normalize observations for RND (without tracking gradients for the normalizer)
        with torch.no_grad():
            norm_next_obs = self.obs_rms.normalize(flat_next_obs.to(torch.float64)).to(
                torch.float32
            )

        # Get RND errors with gradients enabled for training the predictor
        rnd_errors = self.rnd(norm_next_obs)

        with torch.no_grad():
            int_rewards_flat = rnd_errors.detach()
            norm_int_rewards_flat = self.int_rms.scale(
                int_rewards_flat.to(torch.float64)
            ).to(torch.float32)
            int_rewards = norm_int_rewards_flat.view(
                self.num_steps, self.batch_size // self.num_steps
            )

        # Train RND Predictor
        rnd_loss = rnd_errors.mean()
        self.rnd_optim.zero_grad()
        rnd_loss.backward()
        self.rnd_optim.step()

        # bootstrap value if not done
        with torch.no_grad():
            if self.distributional:
                taus_next = torch.rand(
                    next_obs.shape[0], self.n_quantiles, device=device
                )
                next_ext_value = (
                    self.ext_critic(next_obs, taus_next).mean(dim=1).view(1, -1)
                )
                next_int_value = (
                    self.int_critic(next_obs, taus_next).mean(dim=1).view(1, -1)
                )
            else:
                next_ext_value = self.ext_critic(next_obs).view(1, -1)
                next_int_value = self.int_critic(next_obs).view(1, -1)

            ext_advantages = torch.zeros_like(rewards).to(device)
            int_advantages = torch.zeros_like(rewards).to(device)
            lastgaelam_ext = 0
            lastgaelam_int = 0

            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    next_ext_values = next_ext_value
                    next_int_values = next_int_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    next_ext_values = ext_values[t + 1]
                    next_int_values = int_values[t + 1]

                delta_ext = (
                    rewards[t]
                    + self.gamma * next_ext_values * nextnonterminal
                    - ext_values[t]
                )

                delta_int = (
                    int_rewards[t]
                    + self.gamma * next_int_values * nextnonterminal
                    - int_values[t]
                )

                if self.use_gae:
                    ext_advantages[t] = lastgaelam_ext = (
                        delta_ext
                        + self.gamma
                        * self.gae_lambda
                        * nextnonterminal
                        * lastgaelam_ext
                    )
                    int_advantages[t] = lastgaelam_int = (
                        delta_int
                        + self.gamma
                        * self.gae_lambda
                        * nextnonterminal
                        * lastgaelam_int
                    )
                else:
                    ext_advantages[t] = lastgaelam_ext = (
                        delta_ext + self.gamma * nextnonterminal * lastgaelam_ext
                    )
                    int_advantages[t] = lastgaelam_int = (
                        delta_int + self.gamma * nextnonterminal * lastgaelam_int
                    )

            ext_returns = ext_advantages + ext_values
            int_returns = int_advantages + int_values

            # Combine advantages
            combined_advantages = ext_advantages + self.Beta * int_advantages

        # flatten the batch
        b_obs = obs.reshape((-1,) + self.obs_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + self.action_shape)
        b_combined_advantages = combined_advantages.reshape(-1)
        b_ext_returns = ext_returns.reshape(-1)
        b_int_returns = int_returns.reshape(-1)
        b_ext_values = ext_values.reshape(-1)
        b_int_values = int_values.reshape(-1)

        b_inds = np.arange(self.batch_size)
        clipfracs = []

        pg_loss_total = torch.tensor(0.0)
        v_loss_ext_total = torch.tensor(0.0)
        v_loss_int_total = torch.tensor(0.0)
        entropy_loss_total = torch.tensor(0.0)
        old_approx_kl = torch.tensor(0.0)
        approx_kl = torch.tensor(0.0)

        for epoch in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, new_ext_value, new_int_value = (
                    self.get_action_and_values(
                        b_obs[mb_inds], b_actions.long()[mb_inds]
                    )
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_combined_advantages[mb_inds]
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std(unbiased=False) + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.clip_coef, 1 + self.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (Extrinsic)
                if self.distributional:
                    # IQN PopArt updates
                    if self.popart:
                        self.ext_critic.output_layer.update_stats(b_ext_returns[mb_inds])
                        norm_ext_returns = self.ext_critic.output_layer.normalize(b_ext_returns[mb_inds])
                    else:
                        norm_ext_returns = b_ext_returns[mb_inds]

                    ext_taus = torch.rand(len(mb_inds), self.n_quantiles, device=device)
                    ext_quantiles_norm = self.ext_critic(b_obs[mb_inds], ext_taus, normalized=self.popart).view(len(mb_inds), self.n_quantiles)
                    v_loss_ext = self._quantile_huber_loss(ext_quantiles_norm, norm_ext_returns, ext_taus)
                else:
                    if self.popart:
                        # 1. Update PopArt statistics with raw targets
                        self.ext_critic_head.update_stats(b_ext_returns[mb_inds].unsqueeze(1))
                        # 2. Normalize the targets using the updated stats
                        norm_ext_returns = self.ext_critic_head.normalize(b_ext_returns[mb_inds].unsqueeze(1)).view(-1)
                        # 3. Get new normalized predictions
                        norm_ext_value = self.ext_critic(b_obs[mb_inds], normalized=True).view(-1)
                    else:
                        norm_ext_returns = b_ext_returns[mb_inds]
                        norm_ext_value = new_ext_value.view(-1)

                    if self.clip_vloss:
                        if self.popart:
                            # 4. Normalize the historical rollout values using CURRENT stats for accurate clipping
                            norm_old_ext_values = self.ext_critic_head.normalize(b_ext_values[mb_inds].unsqueeze(1)).view(-1)
                        else:
                            norm_old_ext_values = b_ext_values[mb_inds]

                        v_clipped = norm_old_ext_values + torch.clamp(
                            norm_ext_value - norm_old_ext_values,
                            -self.clip_coef,
                            self.clip_coef,
                        )
                        v_loss_unclipped = (norm_ext_value - norm_ext_returns) ** 2
                        v_loss_clipped = (v_clipped - norm_ext_returns) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss_ext = 0.5 * v_loss_max.mean()
                    else:
                        v_loss_ext = 0.5 * ((norm_ext_value - norm_ext_returns) ** 2).mean()

                # Value loss (Intrinsic)
                if self.distributional:
                    # IQN PopArt updates
                    if self.popart:
                        self.int_critic.output_layer.update_stats(b_int_returns[mb_inds])
                        norm_int_returns = self.int_critic.output_layer.normalize(b_int_returns[mb_inds])
                    else:
                        norm_int_returns = b_int_returns[mb_inds]

                    int_taus = torch.rand(len(mb_inds), self.n_quantiles, device=device)
                    int_quantiles_norm = self.int_critic(b_obs[mb_inds], int_taus, normalized=self.popart).view(len(mb_inds), self.n_quantiles)
                    v_loss_int = self._quantile_huber_loss(int_quantiles_norm, norm_int_returns, int_taus)
                else:
                    if self.popart:
                        # 1. Update PopArt statistics with raw targets
                        self.int_critic_head.update_stats(b_int_returns[mb_inds].unsqueeze(1))
                        # 2. Normalize the targets using the updated stats
                        norm_int_returns = self.int_critic_head.normalize(b_int_returns[mb_inds].unsqueeze(1)).view(-1)
                        # 3. Get new normalized predictions
                        norm_int_value = self.int_critic(b_obs[mb_inds], normalized=True).view(-1)
                    else:
                        norm_int_returns = b_int_returns[mb_inds]
                        norm_int_value = new_int_value.view(-1)

                    if self.clip_vloss:
                        if self.popart:
                            # 4. Normalize the historical rollout values using CURRENT stats for accurate clipping
                            norm_old_int_values = self.int_critic_head.normalize(b_int_values[mb_inds].unsqueeze(1)).view(-1)
                        else:
                            norm_old_int_values = b_int_values[mb_inds]

                        v_clipped_int = norm_old_int_values + torch.clamp(
                            norm_int_value - norm_old_int_values,
                            -self.clip_coef,
                            self.clip_coef,
                        )
                        v_loss_int_unclipped = (norm_int_value - norm_int_returns) ** 2
                        v_loss_int_clipped = (v_clipped_int - norm_int_returns) ** 2
                        v_loss_int_max = torch.max(v_loss_int_unclipped, v_loss_int_clipped)
                        v_loss_int = 0.5 * v_loss_int_max.mean()
                    else:
                        v_loss_int = 0.5 * ((norm_int_value - norm_int_returns) ** 2).mean()

                entropy_loss = entropy.mean()

                loss = (
                    pg_loss - self.ent_coef * entropy_loss + v_loss_ext * self.vf_coef
                )

                self.optimizer.zero_grad()
                loss.backward()
                if self.distributional:
                    ext_params = list(self.ext_critic.parameters())
                else:
                    ext_params = list(self.ext_critic_base.parameters()) + list(self.ext_critic_head.parameters())
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + ext_params,
                    self.max_grad_norm,
                )
                self.optimizer.step()

                self.int_optim.zero_grad()
                v_loss_int.backward()
                if self.distributional:
                    int_params = list(self.int_critic.parameters())
                else:
                    int_params = list(self.int_critic_base.parameters()) + list(self.int_critic_head.parameters())
                nn.utils.clip_grad_norm_(
                    int_params, self.max_grad_norm
                )
                self.int_optim.step()

                pg_loss_total = pg_loss_total + pg_loss
                v_loss_ext_total = v_loss_ext_total + v_loss_ext
                v_loss_int_total = v_loss_int_total + v_loss_int
                entropy_loss_total = entropy_loss_total + entropy_loss

            if self.target_kl is not None and approx_kl > self.target_kl:
                break

        y_pred, y_true = b_ext_values.cpu().numpy(), b_ext_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        PG_loss = pg_loss_total.item()
        V_loss = v_loss_ext_total.item()
        V_int_loss = v_loss_int_total.item()
        Entropy = entropy_loss_total.item()

        self.last_losses = {
            "policy_loss": PG_loss,
            "value_loss": V_loss,
            "int_value_loss": V_int_loss,
            "rnd_loss": rnd_loss.item(),
            "entropy": Entropy,
            "old_approx_kl": old_approx_kl.item(),
            "approx_kl": approx_kl.item(),
            "clipfrac": np.mean(clipfracs) if clipfracs else 0.0,
            "explained_variance": explained_var,
            "Beta": float(self.Beta),
        }

        self.step += 1

        if self.tb_writer is not None:
            self.tb_writer.add_scalar(
                f"{self.tb_prefix}/losses/value_loss", V_loss, self.step
            )
            self.tb_writer.add_scalar(
                f"{self.tb_prefix}/losses/int_value_loss", V_int_loss, self.step
            )
            self.tb_writer.add_scalar(
                f"{self.tb_prefix}/losses/policy_loss", PG_loss, self.step
            )
            self.tb_writer.add_scalar(
                f"{self.tb_prefix}/losses/rnd_loss", rnd_loss.item(), self.step
            )
            self.tb_writer.add_scalar(
                f"{self.tb_prefix}/losses/entropy", Entropy, self.step
            )
            self.tb_writer.add_scalar(
                f"{self.tb_prefix}/losses/old_approx_kl",
                old_approx_kl.item(),
                self.step,
            )
            self.tb_writer.add_scalar(
                f"{self.tb_prefix}/losses/approx_kl", approx_kl.item(), self.step
            )
            self.tb_writer.add_scalar(
                f"{self.tb_prefix}/losses/clipfrac",
                np.mean(clipfracs) if clipfracs else 0.0,
                self.step,
            )
            self.tb_writer.add_scalar(
                f"{self.tb_prefix}/losses/explained_variance", explained_var, self.step
            )
            self.tb_writer.add_scalar(
                f"{self.tb_prefix}/charts/Beta", float(self.Beta), self.step
            )

        return pg_loss_total.item()


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
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

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, i, args.capture_video, run_name)
            for i in range(args.num_envs)
        ],
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    agent = PPOAgent(
        envs,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        clip_vloss=args.clip_vloss,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        norm_adv=args.norm_adv,
        target_kl=args.target_kl,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        hidden_layer_sizes=args.hidden_layer_sizes,
        distributional=False,  # Set to True to use IQN
        Beta=0.01,  # Set > 0 to use RND intrinsic reward
        beta_half_life_steps=10000,
    ).to(device)
    # If keeping tensorboard inside agent (optional):
    # agent.attach_tensorboard(writer)

    # ALGO Logic: Storage setup
    assert (
        envs.single_observation_space is not None
        and envs.single_observation_space.shape is not None
    )
    assert (
        envs.single_action_space is not None
        and envs.single_action_space.shape is not None
    )
    obs = torch.zeros(
        [args.num_steps, args.num_envs] + list(envs.single_observation_space.shape)
    ).to(device)
    actions = torch.zeros(
        [args.num_steps, args.num_envs] + list(envs.single_action_space.shape)
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    ext_values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    int_values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    env_episode_rewards = torch.zeros(args.num_envs).to(device)
    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            action, logprob, ext_v, int_v = agent.sample_action(next_obs)
            ext_values[step] = ext_v
            int_values[step] = int_v
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            env_episode_rewards += rewards[step]
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                next_done
            ).to(device)

            for ne in range(args.num_envs):
                if terminations[ne] or truncations[ne]:
                    print(f"episode reward: {env_episode_rewards[ne]}")
                    env_episode_rewards[ne] = 0

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(
                            f"global_step={global_step}, episodic_return={info['episode']['r']}"
                        )
                        writer.add_scalar(
                            "charts/episodic_return", info["episode"]["r"], global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step
                        )

        # Perform PPO update
        agent.update(
            obs,
            actions,
            logprobs,
            rewards,
            dones,
            ext_values,
            int_values,
            next_obs,
            next_done,
        )

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", agent.optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar(
            "losses/value_loss", agent.last_losses.get("value_loss", 0), global_step
        )
        writer.add_scalar(
            "losses/policy_loss", agent.last_losses.get("policy_loss", 0), global_step
        )
        writer.add_scalar(
            "losses/entropy", agent.last_losses.get("entropy", 0), global_step
        )
        writer.add_scalar(
            "losses/old_approx_kl",
            agent.last_losses.get("old_approx_kl", 0),
            global_step,
        )
        writer.add_scalar(
            "losses/approx_kl", agent.last_losses.get("approx_kl", 0), global_step
        )
        writer.add_scalar(
            "losses/clipfrac", agent.last_losses.get("clipfrac", 0), global_step
        )
        writer.add_scalar(
            "losses/explained_variance",
            agent.last_losses.get("explained_variance", 0),
            global_step,
        )
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )

    envs.close()
    writer.close()
