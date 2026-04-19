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


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from torch.distributions.categorical import Categorical
from typing import Callable, Optional

# Assuming layer_init, PopArtLayer, RNDModel, RunningMeanStd, infer_encoder_out_dim, IQN_Network, Agent are imported

class BasePPOAgent(Agent):
    """
    Master Base Class containing all shared PPO, RND, and GAE logic.
    Subclasses only need to implement network building and value-loss specifics.
    """
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
        popart: bool = True,
        Beta: float = 0.0,
        beta_half_life_steps=None,
        rnd_output_dim: int = 128,
        rnd_lr: float = 1e-3,
        intrinsic_lr: float = 2.5e-4,
        use_gae: bool = True,
        encoder_factory: Optional[Callable[[], nn.Module]] = None,
        device = "cpu",
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
        self.timing = {}
        self.device = device
        if not torch.cuda.is_available():
            self.device = "cpu"

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

        self.num_envs = num_envs
        self.batch_size = int(num_envs * num_steps)
        self.minibatch_size = int(self.batch_size // self.num_minibatches)
        self.num_steps = num_steps
        self.input_dim = np.array(self.obs_shape).prod()

        # 1. Setup Shared Actor
        self._build_actor(encoder_factory, hidden_layer_sizes)

        # 2. Setup Specific Critics (Implemented by Subclass)
        self._build_critics(encoder_factory, hidden_layer_sizes)

        # 3. Setup RND and RMS (Shared)
        self.rnd = RNDModel(self.input_dim, rnd_output_dim)
        self.obs_rms = RunningMeanStd(shape=self.obs_shape)
        self.int_rms = RunningMeanStd(shape=())
        self.ext_rms = RunningMeanStd(shape=())

        # 4. Setup Optimizers
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + self._get_ext_critic_params(),
            lr=learning_rate, eps=1e-5,
        )
        self.int_optim = optim.Adam(
            self._get_int_critic_params(), 
            lr=intrinsic_lr, eps=1e-5
        )
        self.rnd_optim = optim.Adam(self.rnd.predictor.parameters(), lr=rnd_lr)
        self.step = 0
        self.step_idx = 0
        self._init_buffers()

    def _build_actor(self, encoder_factory, hidden_layer_sizes):
        hidden1, hidden2 = int(hidden_layer_sizes[0]), int(hidden_layer_sizes[1])
        if encoder_factory is None:
            self.actor = nn.Sequential(
                layer_init(nn.Linear(self.input_dim, hidden1)), nn.Tanh(),
                layer_init(nn.Linear(hidden1, hidden2)), nn.Tanh(),
                layer_init(nn.Linear(hidden2, self.n_action_dims * self.n_action_bins), std=0.01),
            )
        else:
            actor_encoder = encoder_factory()
            actor_out_dim = infer_encoder_out_dim(actor_encoder, int(self.input_dim))
            self.actor = nn.Sequential(
                actor_encoder,
                layer_init(nn.Linear(actor_out_dim, self.n_action_dims * self.n_action_bins), std=0.01),
            )

    def _init_buffers(self):
        """Initializes internal memory buffers on the correct device"""
        self.agent_obs = torch.zeros((self.num_steps, self.num_envs) + self.obs_shape, device="cpu")
        self.agent_actions = torch.zeros((self.num_steps, self.num_envs) + self.action_shape, device="cpu")
        self.agent_logprobs = torch.zeros((self.num_steps, self.num_envs), device="cpu")
        self.agent_rewards = torch.zeros((self.num_steps, self.num_envs), device="cpu")
        self.agent_ext_values = torch.zeros((self.num_steps, self.num_envs), device="cpu")
        self.agent_int_values = torch.zeros((self.num_steps, self.num_envs), device="cpu")
        self.agent_terminations = torch.zeros((self.num_steps, self.num_envs), device="cpu")
        self.agent_truncations = torch.zeros((self.num_steps, self.num_envs), device="cpu")

        trunc_obs_list = []
        trunc_indices = []

        self.last_next_obs = torch.zeros((self.num_envs,) + self.obs_shape, device="cpu")
        self.last_next_term = torch.zeros((self.num_envs,), device="cpu")
        self.last_next_trunc = torch.zeros((self.num_envs,), device="cpu")

    def to(self, device):
        self.device = device
        self.actor.to(device)
        self.rnd.to(device)
        self.obs_rms.to(device)
        self._critics_to(device)
        self._init_buffers()
        return self

    def _get_action(self, obs, entropy=False):
        logits = self.actor(obs)
        if self.n_action_dims > 1:
            logits = logits.view(-1, self.n_action_dims, self.n_action_bins)
            probs = Categorical(logits=logits)
        else:
            probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        log_prob = probs.log_prob(action).sum(dim=-1) if self.n_action_dims > 1 else probs.log_prob(action)
        entropy_calc=None
        if entropy:
            entropy_calc = probs.entropy().sum(dim=-1) if self.n_action_dims > 1 else probs.entropy()
        return action, log_prob, entropy_calc

    # Getting action and values for training with gradient's attached
    def get_action_and_values(self, obs, action=None):
        action, logprob, entropy = self._get_action(entropy=True)
        ext_v, int_v = self._get_values(obs)
        return action, log_prob, entropy, ext_v, int_v

    # Grad free just action for buffer
    @torch.no_grad
    def sample_action(self, obs):
        action, logprob, _ = self._get_action(entropy=False)
        return action, logprob

    @torch.no_grad()
    def update_running_stats(self, next_obs, r=None):
        obs_flat = next_obs.view(-1, *self.obs_shape)
        self.obs_rms.update(obs_flat)
    
    def update(self, global_step=None):
        if self.step_idx < self.num_steps:
            return None # Buffer not yet full
        device = self.device
        
        if self.anneal_lr:
            frac = 1.0 - (self.step - 1.0) / (self.update_epochs * 1000)
            lrnow = frac * self.optimizer.param_groups[0]["lr"]
            self.optimizer.param_groups[0]["lr"] = lrnow
            self.int_optim.param_groups[0]["lr"] = lrnow

        # Use global_step for beta decay if provided
        if self.beta_half_life_steps is not None and self.beta_half_life_steps > 0 and global_step is not None:
            self.Beta = self.start_Beta * (0.5 ** (global_step / self.beta_half_life_steps))
        elif self.beta_half_life_steps is not None and self.beta_half_life_steps > 0:
            self.Beta = self.start_Beta * (0.5 ** (self.step / self.beta_half_life_steps))

        # # --- Shared RND Processing ---
        # next_obs_batch = torch.cat([self.agent_obs[1:], self.last_next_obs.unsqueeze(0)], dim=0)
        # flat_next_obs = next_obs_batch.reshape(-1, *self.obs_shape)

        # --- Shared GAE Calculation ---
        with torch.no_grad():
            # 1. Build the mathematically perfect 'next_obs' tensor
            # This handles the T-1 shift and patches the truncated states
            true_next_obs = torch.zeros_like(self.agent_obs)
            true_next_obs[:-1] = self.agent_obs[1:]
            true_next_obs[-1] = self.last_next_obs
            
            if len(self.trunc_obs_list) > 0:
                t_idx, env_idx = zip(*self.trunc_indices)
                true_next_obs[t_idx, env_idx] = torch.stack(self.trunc_obs_list).to(true_next_obs.device)

            # 2. Shared RND Processing
            flat_true_next_obs = true_next_obs.view(-1, *self.obs_shape)
            self.obs_rms.update(flat_true_next_obs)
            norm_next_obs = self.obs_rms.normalize(flat_true_next_obs).to(torch.float32)
            rnd_errors = self.rnd(norm_next_obs)
            int_rewards = rnd_errors.view(self.num_steps, self.num_envs).detach()

            # 3. Massive Batched Value Forward Passes
            flat_obs = self.agent_obs.view(-1, *self.obs_shape)
            
            # Values for current states
            ext_values_flat, int_values_flat = self._get_values(flat_obs)
            self.agent_ext_values = ext_values_flat.view(self.num_steps, self.num_envs)
            self.agent_int_values = int_values_flat.view(self.num_steps, self.num_envs)
            
            # Values for next states (Bootstraps)
            next_ext_values_flat, next_int_values_flat = self._get_values(flat_true_next_obs)
            bootstrap_ext_values = next_ext_values_flat.view(self.num_steps, self.num_envs)
            bootstrap_int_values = next_int_values_flat.view(self.num_steps, self.num_envs)

            # 4. Shared GAE Calculation (Now completely branchless)
            ext_advantages = torch.zeros_like(self.agent_rewards).to(device)
            int_advantages = torch.zeros_like(self.agent_rewards).to(device)
            lastgaelam_ext, lastgaelam_int = 0, 0

            
            for t in reversed(range(self.num_steps)):
                # Masking logic
                next_is_term = self.agent_terminations[t]
                next_is_trunc = self.agent_truncations[t]
                
                nextnonterminal_value = 1.0 - next_is_term
                nextnonterminal_gae = 1.0 - torch.clamp(next_is_term + next_is_trunc, 0.0, 1.0)
                
                # Delta calculations utilizing our clean bootstrap tensors
                delta_ext = self.agent_rewards[t] + self.gamma * bootstrap_ext_values[t] * nextnonterminal_value - self.agent_ext_values[t]
                delta_int = int_rewards[t] + self.gamma * bootstrap_int_values[t] - self.agent_int_values[t]

                if self.use_gae:
                    ext_advantages[t] = lastgaelam_ext = delta_ext + self.gamma * self.gae_lambda * nextnonterminal_gae * lastgaelam_ext
                    int_advantages[t] = lastgaelam_int = delta_int + self.gamma * self.gae_lambda * lastgaelam_int # Add * nextnonterminal_gae here if intrinsic shouldn't leak across boundaries
                else:
                    ext_advantages[t] = lastgaelam_ext = delta_ext + self.gamma * nextnonterminal_gae * lastgaelam_ext
                    int_advantages[t] = lastgaelam_int = delta_int + self.gamma * lastgaelam_int

            # Reset the truncation tracking lists for the next rollout
            self.trunc_obs_list.clear()
            self.trunc_indices.clear()
            assert ext_advantages.shape == ext_values.shape, f"Shape mismatch: {ext_advantages.shape}, {ext_values.shape}"
            assert int_advantages.shape == int_values.shape, f"Shape mismatch: {int_advantages.shape}, {int_values.shape}"
            ext_returns = ext_advantages + self.agent_ext_values
            int_returns = int_advantages + self.agent_int_values

            sigma_ext, sigma_int = self._get_advantages_scaling()
            assert ext_advantages.shape == int_advantages.shape, f"Shape mismatch: {ext_advantages.shape}, {int_advantages.shape}"
            combined_advantages = (ext_advantages / sigma_ext) + self.Beta * (int_advantages / sigma_int)

        # Batch Flattening
        b_obs = self.agent_obs.reshape((-1,) + self.obs_shape)
        b_logprobs = self.agent_logprobs.reshape(-1)
        b_actions = self.agent_actions.reshape((-1,) + self.action_shape)
        b_combined_advantages = combined_advantages.reshape(-1)
        b_ext_advantages = ext_advantages.reshape(-1)
        b_int_advantages = int_advantages.reshape(-1)
        b_ext_values = self.agent_ext_values.reshape(-1)
        b_int_values = self.agent_int_values.reshape(-1)
        b_inds = np.arange(self.batch_size)
        b_obs_next = true_next_obs.reshape((-1,) + self.obs_shape)


        clipfracs = []
        pg_loss_total, v_loss_ext_total, v_loss_int_total, entropy_loss_total = 0.0, 0.0, 0.0, 0.0
        approx_kl, old_approx_kl = 0.0, 0.0

        with torch.no_grad():
            self._update_popart_stats(b_ext_returns, b_int_returns)

        # --- Shared Epoch Loop ---
        for epoch in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                # RND Predictor Update
                with torch.no_grad():
                    norm_mb_obs = self.obs_rms.normalize(b_obs[mb_inds]).float()
                mb_rnd_errors = self.rnd(norm_mb_obs)
                rnd_loss = mb_rnd_errors.mean()
                self.rnd_optim.zero_grad()
                rnd_loss.backward()
                self.rnd_optim.step()

                _, newlogprob, entropy, new_ext_value, new_int_value = self.get_action_and_values(
                    b_obs[mb_inds], b_actions.long()[mb_inds] if self.n_action_dims == 1 else b_actions[mb_inds]
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                mb_advantages = b_combined_advantages[mb_inds]
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std(unbiased=False) + 1e-8)

                assert mb_advantages.shape == ratio.shape, f"Shape mismatch: mb_advantages {mb_advantages.shape}, ratio {ratio.shape}"
                # Policy Loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                entropy_loss = entropy.mean()
                # Value Loss (Delegated to Subclass)
                v_loss_ext, v_loss_int = self._compute_value_losses(
                    b_obs, b_obs_next, mb_inds, b_ext_returns, b_int_returns, 
                    new_ext_value, new_int_value, b_ext_values, b_int_values, device
                )

                loss = pg_loss - self.ent_coef * entropy_loss + v_loss_ext * self.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + self._get_ext_critic_params(), self.max_grad_norm
                )
                self.optimizer.step()

                self.int_optim.zero_grad()
                v_loss_int.backward()
                nn.utils.clip_grad_norm_(self._get_int_critic_params(), self.max_grad_norm)
                self.int_optim.step()

                pg_loss_total += pg_loss.item()
                v_loss_ext_total += v_loss_ext.item()
                v_loss_int_total += v_loss_int.item()
                entropy_loss_total += entropy_loss.item()

            if self.target_kl is not None and approx_kl > self.target_kl:
                break

        y_pred, y_true = b_ext_values.cpu().numpy(), b_ext_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        self.last_losses = {
            "policy_loss": pg_loss_total,
            "value_loss": v_loss_ext_total,
            "int_value_loss": v_loss_int_total,
            "rnd_loss": rnd_loss.item(),
            "entropy": entropy_loss_total,
            "old_approx_kl": old_approx_kl.item() if isinstance(old_approx_kl, torch.Tensor) else old_approx_kl,
            "approx_kl": approx_kl.item() if isinstance(approx_kl, torch.Tensor) else approx_kl,
            "clipfrac": np.mean(clipfracs) if clipfracs else 0.0,
            "explained_variance": explained_var,
            "Beta": float(self.Beta),
        }
        self.step += 1
        # Reset the buffer index directly here
        self.step_idx = 0
        return pg_loss_total

    def observe(self, obs, action, logprob, reward, next_obs, term, trunc, infos):
        """Stores a transition in the rollout buffer."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        action_t = torch.as_tensor(action, device=self.device)
        logprob_t = torch.as_tensor(logprob, device=self.device)
        reward_t = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        term_t = torch.as_tensor(term, dtype=torch.float32, device=self.device)
        trunc_t = torch.as_tensor(trunc, dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)

        self.agent_obs[self.step_idx] = obs_t
        self.agent_actions[self.step_idx] = action_t
        self.agent_logprobs[self.step_idx] = logprob_t
        self.agent_rewards[self.step_idx] = reward_t
        self.agent_terminations[self.step_idx] = term_t
        self.agent_truncations[self.step_idx] = trunc_t

        # Extract actual observations for truncated states cleanly
        if trunc.any() and "final_observation" in infos:
            for env_idx, is_trunc in enumerate(trunc):
                if is_trunc:
                    true_obs = infos["final_observation"][env_idx]
                    self.trunc_obs_list.append(torch.as_tensor(true_obs, dtype=torch.float32, device=self.device))
                    self.trunc_indices.append((self.step_idx, env_idx))

        self.last_next_obs = next_obs_t
        self.last_next_term = term_t
        self.last_next_trunc = trunc_t

        self.step_idx += 1

    # =======================================================
    # Abstract Methods for Subclasses
    # =======================================================
    def _build_critics(self, encoder_factory, hidden_layer_sizes): raise NotImplementedError
    def _critics_to(self, device): raise NotImplementedError
    def _get_ext_critic_params(self): raise NotImplementedError
    def _get_int_critic_params(self): raise NotImplementedError
    def _get_raw_values(self, obs): raise NotImplementedError
    def _values_ev_from_raw(self, obs): raise NotImplementedError
    def _get_advantages_scaling(self): raise NotImplementedError
    def _update_popart_stats(self, b_ext_returns, b_int_returns): raise NotImplementedError
    def _compute_value_losses(self, b_obs, mb_inds, b_ext_returns, b_int_returns, new_ext_value, new_int_value, b_ext_values, b_int_values, device): raise NotImplementedError


class StandardPPOAgent(BasePPOAgent):
    def _build_critics(self, encoder_factory, hidden_layer_sizes):
        hidden1, hidden2 = int(hidden_layer_sizes[0]), int(hidden_layer_sizes[1])
        if encoder_factory is None:
            self.ext_critic_base = nn.Sequential(
                layer_init(nn.Linear(self.input_dim, hidden1)), nn.Tanh(),
                layer_init(nn.Linear(hidden1, hidden2)), nn.Tanh()
            )
            self.int_critic_base = nn.Sequential(
                layer_init(nn.Linear(self.input_dim, hidden1)), nn.Tanh(),
                layer_init(nn.Linear(hidden1, hidden2)), nn.Tanh()
            )
            critic_hidden, int_critic_hidden = hidden2, hidden2
        else:
            self.ext_critic_base, self.int_critic_base = encoder_factory(), encoder_factory()
            critic_hidden = infer_encoder_out_dim(self.ext_critic_base, int(self.input_dim))
            int_critic_hidden = infer_encoder_out_dim(self.int_critic_base, int(self.input_dim))

        if self.popart:
            self.ext_critic_head = PopArtLayer(critic_hidden, 1)
            self.int_critic_head = PopArtLayer(int_critic_hidden, 1)
        else:
            self.ext_critic_head = layer_init(nn.Linear(critic_hidden, 1), std=1.0)
            self.int_critic_head = layer_init(nn.Linear(int_critic_hidden, 1), std=1.0)

        self.ext_critic = lambda x, normalized=False: self.ext_critic_head(self.ext_critic_base(x), normalized=normalized) if self.popart else self.ext_critic_head(self.ext_critic_base(x))
        self.int_critic = lambda x, normalized=False: self.int_critic_head(self.int_critic_base(x), normalized=normalized) if self.popart else self.int_critic_head(self.int_critic_base(x))

    def _critics_to(self, device):
        self.ext_critic_base.to(device); self.ext_critic_head.to(device)
        self.int_critic_base.to(device); self.int_critic_head.to(device)

    def _get_ext_critic_params(self): return list(self.ext_critic_base.parameters()) + list(self.ext_critic_head.parameters())
    def _get_int_critic_params(self): return list(self.int_critic_base.parameters()) + list(self.int_critic_head.parameters())
    def _get_values(self,obs,norm=False)
        return self.ext_critic(obs,norm=norm).squeeze(-1), self.int_critic(obs,norm=norm).squeeze(-1)
    def _get_raw_values(self,obs, norm=False):
        return self.ext_critic(obs, norm=norm).squeeze(-1), self.int_critic(obs, norm=norm).squeeze(-1)
    def _values_ev_from_raw(self,values):
        return values
    def _get_advantages_scaling(self):
        return (self.ext_critic_head.sigma.detach() if self.popart else 1.0), (self.int_critic_head.sigma.detach() if self.popart else 1.0)
    def _update_popart_stats(self, b_ext_returns, b_int_returns):
        if self.popart:
            self.ext_critic_head.update_stats(b_ext_returns.unsqueeze(1))
            self.int_critic_head.update_stats(b_int_returns.unsqueeze(1))

    def _compute_value_losses(self, b_obs, mb_inds, b_ext_returns, b_int_returns, new_ext_value, new_int_value, b_ext_values, b_int_values, device):
        # Extrinsic
        norm_ext_returns = self.ext_critic_head.normalize(b_ext_returns[mb_inds].unsqueeze(1)).view(-1) if self.popart else b_ext_returns[mb_inds]
        norm_ext_value = self.ext_critic(b_obs[mb_inds], normalized=True).view(-1) if self.popart else new_ext_value.view(-1)
        
        assert norm_ext_value.shape == norm_ext_returns.shape, f"Shape mismatch: norm_ext_value {norm_ext_value.shape}, norm_ext_returns {norm_ext_returns.shape}"

        if self.clip_vloss:
            norm_old_ext_values = self.ext_critic_head.normalize(b_ext_values[mb_inds].unsqueeze(1)).view(-1) if self.popart else b_ext_values[mb_inds]
            v_clipped = norm_old_ext_values + torch.clamp(norm_ext_value - norm_old_ext_values, -self.clip_coef, self.clip_coef)
            v_loss_ext = 0.5 * torch.max((norm_ext_value - norm_ext_returns)**2, (v_clipped - norm_ext_returns)**2).mean()
        else:
            v_loss_ext = 0.5 * ((norm_ext_value - norm_ext_returns)**2).mean()

        # Intrinsic
        norm_int_returns = self.int_critic_head.normalize(b_int_returns[mb_inds].unsqueeze(1)).view(-1) if self.popart else b_int_returns[mb_inds]
        norm_int_value = self.int_critic(b_obs[mb_inds], normalized=True).view(-1) if self.popart else new_int_value.view(-1)

        assert norm_int_value.shape == norm_int_returns.shape, f"Shape mismatch: norm_int_value {norm_int_value.shape}, norm_int_returns {norm_int_returns.shape}"

        if self.clip_vloss:
            norm_old_int_values = self.int_critic_head.normalize(b_int_values[mb_inds].unsqueeze(1)).view(-1) if self.popart else b_int_values[mb_inds]
            v_clipped_int = norm_old_int_values + torch.clamp(norm_int_value - norm_old_int_values, -self.clip_coef, self.clip_coef)
            v_loss_int = 0.5 * torch.max((norm_int_value - norm_int_returns)**2, (v_clipped_int - norm_int_returns)**2).mean()
        else:
            v_loss_int = 0.5 * ((norm_int_value - norm_int_returns)**2).mean()

        return v_loss_ext, v_loss_int


class DistributionalPPOAgent(BasePPOAgent):
    def __init__(self, *args, **kwargs):
        self.n_quantiles = 32
        super().__init__(*args, **kwargs)

    def _build_critics(self, encoder_factory, hidden_layer_sizes):
        hidden1, hidden2 = int(hidden_layer_sizes[0]), int(hidden_layer_sizes[1])
        ext_kwargs, int_kwargs = {}, {}
        if encoder_factory is not None:
            ext_enc, int_enc = encoder_factory(), encoder_factory()
            ext_kwargs = {"encoder": ext_enc, "encoder_out_dim": infer_encoder_out_dim(ext_enc, int(self.input_dim))}
            int_kwargs = {"encoder": int_enc, "encoder_out_dim": infer_encoder_out_dim(int_enc, int(self.input_dim))}

        self.ext_critic = IQN_Network(input_dim=self.input_dim, n_action_dims=1, n_action_bins=1, hidden_layer_sizes=[hidden1, hidden2], dueling=False, popart=self.popart, **ext_kwargs)
        self.int_critic = IQN_Network(input_dim=self.input_dim, n_action_dims=1, n_action_bins=1, hidden_layer_sizes=[hidden1, hidden2], dueling=False, popart=self.popart, min_std=0.01, **int_kwargs)

    def _critics_to(self, device):
        self.ext_critic.to(device)
        self.int_critic.to(device)

    def _get_ext_critic_params(self): return list(self.ext_critic.parameters())
    def _get_int_critic_params(self): return list(self.int_critic.parameters())
    def _get_values(self,obs, norm=False)
        taus = torch.rand(obs.shape[0], self.n_quantiles, device=obs.device)
        return self.ext_critic(obs, taus, norm=norm).mean(-1).view(-1), self.int_critic(obs, taus, norm=norm).mean(-1).view(-1)
    def _get_raw_values(self,obs, norm=False)
        taus = torch.rand(obs.shape[0], self.n_quantiles, device=obs.device)
        return self.ext_critic(obs, taus, norm=norm), self.int_critic(obs, taus, norm=norm)
    def _values_ev_from_raw(self, values)
        return values.mean(dim=1).view(-1)
    
    def _get_advantages_scaling(self):
        return (self.ext_critic.output_layer.sigma.detach() if self.popart else 1.0), (self.int_critic.output_layer.sigma.detach() if self.popart else 1.0)
    def _update_popart_stats(self, b_ext_returns, b_int_returns):
        if self.popart:
            self.ext_critic.output_layer.update_stats(b_ext_returns)
            self.int_critic.output_layer.update_stats(b_int_returns)

    def _quantile_huber_loss(self, pred: torch.Tensor, target: torch.Tensor, taus: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
        if target.dim() == 1: target = target.unsqueeze(1)
        td = target - pred
        abs_td = torch.abs(td)
        huber = torch.where(abs_td <= kappa, 0.5 * td.pow(2), kappa * (abs_td - 0.5 * kappa))
        loss = (torch.abs(taus - (td < 0).float()) * huber).sum(dim=1).mean()
        return loss

    def _compute_value_losses(self, b_obs, mb_inds, b_ext_returns, b_int_returns, new_ext_value, new_int_value, b_ext_values, b_int_values, device):
        # Calculate scalar advantages for the shift
        mb_ext_advantages = b_ext_returns[mb_inds] - b_ext_values[mb_inds]
        mb_int_advantages = b_int_returns[mb_inds] - b_int_values[mb_inds]

        # --- Extrinsic Critic ---
        ext_taus = torch.rand(len(mb_inds), self.n_quantiles, device=device)
        with torch.no_grad():
            old_ext_quantiles = self.ext_critic(b_obs[mb_inds], ext_taus, normalized=False).view(len(mb_inds), self.n_quantiles)
            ext_targets = old_ext_quantiles + mb_ext_advantages.unsqueeze(-1)
        
        norm_ext_targets = self.ext_critic.output_layer.normalize(ext_targets) if self.popart else ext_targets
        ext_quantiles_norm = self.ext_critic(b_obs[mb_inds], ext_taus, normalized=self.popart).view(len(mb_inds), self.n_quantiles)
        v_loss_ext = self._quantile_huber_loss(ext_quantiles_norm, norm_ext_targets, ext_taus)

        # --- Intrinsic Critic ---
        int_taus = torch.rand(len(mb_inds), self.n_quantiles, device=device)
        with torch.no_grad():
            old_int_quantiles = self.int_critic(b_obs[mb_inds], int_taus, normalized=False).view(len(mb_inds), self.n_quantiles)
            int_targets = old_int_quantiles + mb_int_advantages.unsqueeze(-1)
        
        norm_int_targets = self.int_critic.output_layer.normalize(int_targets) if self.popart else int_targets
        int_quantiles_norm = self.int_critic(b_obs[mb_inds], int_taus, normalized=self.popart).view(len(mb_inds), self.n_quantiles)
        v_loss_int = self._quantile_huber_loss(int_quantiles_norm, norm_int_targets, int_taus)

        return v_loss_ext, v_loss_int