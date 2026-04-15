# Copyright notice
#
# This file contains code adapted from stable-baselines3
# (https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py)
# licensed under the MIT License.

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any, NamedTuple

import numpy as np
import torch as th
from gymnasium import spaces

try:
    import psutil
except ImportError:
    psutil = None

__all__ = [
    "BaseBuffer",
    "RolloutBuffer",
    "ReplayBuffer",
    "RolloutBufferSamples",
    "ReplayBufferSamples",
]


class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor


def get_action_dim(action_space: spaces.Space) -> int:
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


def get_obs_shape(
    observation_space: spaces.Space,
) -> tuple[int, ...] | dict[str, tuple[int, ...]]:
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}
    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")


def get_device(device: th.device | str = "auto") -> th.device:
    if device == "auto":
        device = "cuda" if th.cuda.is_available() else "cpu"
    device = th.device(device)
    if device.type == "cuda" and not th.cuda.is_available():
        return th.device("cpu")
    return device


class BaseBuffer(ABC):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space = None,
        action_space: spaces.Space = None,
        device: th.device | str = "auto",
        n_envs: int = 1,
        obs_shape: tuple[int, ...] = None,
        action_dim: int = None,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.device = get_device(device)
        self.n_envs = n_envs

        # Use custom dim if provided, else parse from space
        if obs_shape is not None:
            self.obs_shape = obs_shape
        else:
            self.obs_shape = get_obs_shape(observation_space)

        if action_dim is not None:
            self.action_dim = action_dim
        else:
            self.action_dim = get_action_dim(action_space)

        self.pos = 0
        self.full = False

    @staticmethod
    def swap_and_flatten(tensor: th.Tensor) -> th.Tensor:
        shape = tensor.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return tensor.transpose(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        return self.buffer_size if self.full else self.pos

    @abstractmethod
    def add(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int):
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = th.randint(0, upper_bound, (batch_size,))
        return self._get_samples(batch_inds)

    @abstractmethod
    def _get_samples(self, batch_inds: th.Tensor) -> ReplayBufferSamples | RolloutBufferSamples:
        raise NotImplementedError()

    def to_torch(self, tensor: th.Tensor) -> th.Tensor:
        return tensor.to(self.device, non_blocking=True)


class ReplayBuffer(BaseBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space = None,
        action_space: spaces.Space = None,
        device: th.device | str = "auto",
        n_envs: int = 1,
        obs_shape: tuple[int, ...] = None,
        action_dim: int = None,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(
            buffer_size, observation_space, action_space, device, n_envs, 
            obs_shape=obs_shape, action_dim=action_dim
        )
        
        self.buffer_size = max(buffer_size // n_envs, 1)
        self.optimize_memory_usage = optimize_memory_usage
        self.handle_timeout_termination = handle_timeout_termination

        # Initialize Pinned Tensors
        self.observations = th.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=th.float32).pin_memory()
        if not optimize_memory_usage:
            self.next_observations = th.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=th.float32).pin_memory()

        self.actions = th.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=th.float32).pin_memory()
        self.rewards = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32).pin_memory()
        self.dones = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32).pin_memory()
        self.timeouts = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32).pin_memory()

    def add(
        self,
        obs: np.ndarray | th.Tensor,
        next_obs: np.ndarray | th.Tensor,
        action: np.ndarray | th.Tensor,
        reward: np.ndarray | th.Tensor,
        done: np.ndarray | th.Tensor,
        infos: list[dict[str, Any]],
    ) -> None:
        self.observations[self.pos].copy_(th.as_tensor(obs).reshape((self.n_envs, *self.obs_shape)))
        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size].copy_(th.as_tensor(next_obs).reshape((self.n_envs, *self.obs_shape)))
        else:
            self.next_observations[self.pos].copy_(th.as_tensor(next_obs).reshape((self.n_envs, *self.obs_shape)))

        self.actions[self.pos].copy_(th.as_tensor(action).reshape((self.n_envs, self.action_dim)))
        self.rewards[self.pos].copy_(th.as_tensor(reward))
        self.dones[self.pos].copy_(th.as_tensor(done))

        if self.handle_timeout_termination:
            timeout_vals = th.tensor([info.get("TimeLimit.truncated", False) for info in infos], dtype=th.float32)
            self.timeouts[self.pos].copy_(timeout_vals)

        self.pos = (self.pos + 1) % self.buffer_size
        if self.pos == 0:
            self.full = True

    def _get_samples(self, batch_inds: th.Tensor) -> ReplayBufferSamples:
        env_indices = th.randint(0, self.n_envs, (len(batch_inds),))
        obs = self.observations[batch_inds, env_indices]
        actions = self.actions[batch_inds, env_indices]
        
        if self.optimize_memory_usage:
            next_obs = self.observations[(batch_inds + 1) % self.buffer_size, env_indices]
        else:
            next_obs = self.next_observations[batch_inds, env_indices]

        rewards = self.rewards[batch_inds, env_indices].reshape(-1, 1)
        dones = (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1)

        return ReplayBufferSamples(
            self.to_torch(obs), self.to_torch(actions), self.to_torch(next_obs),
            self.to_torch(dones), self.to_torch(rewards)
        )


class RolloutBuffer(BaseBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space = None,
        action_space: spaces.Space = None,
        device: th.device | str = "auto",
        n_envs: int = 1,
        obs_shape: tuple[int, ...] = None,
        action_dim: int = None,
        gae_lambda: float = 1.0,
        gamma: float = 0.99,
    ):
        super().__init__(
            buffer_size, observation_space, action_space, device, n_envs,
            obs_shape=obs_shape, action_dim=action_dim
        )
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = th.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=th.float32).pin_memory()
        self.actions = th.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=th.float32).pin_memory()
        self.rewards = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32).pin_memory()
        self.returns = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32).pin_memory()
        self.episode_starts = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32).pin_memory()
        self.values = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32).pin_memory()
        self.log_probs = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32).pin_memory()
        self.advantages = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32).pin_memory()
        self.generator_ready = False
        super().reset()

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        last_values = last_values.flatten()
        last_gae_lam = 0
        dones_t = th.as_tensor(dones, dtype=th.float32)

        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones_t
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        
        self.returns.copy_(self.advantages + self.values)

    def add(self, obs, action, reward, episode_start, value, log_prob) -> None:
        self.observations[self.pos].copy_(th.as_tensor(obs).reshape((self.n_envs, *self.obs_shape)))
        self.actions[self.pos].copy_(th.as_tensor(action).reshape((self.n_envs, self.action_dim)))
        self.rewards[self.pos].copy_(th.as_tensor(reward))
        self.episode_starts[self.pos].copy_(th.as_tensor(episode_start))
        self.values[self.pos].copy_(value.flatten())
        self.log_probs[self.pos].copy_(log_prob.reshape(self.n_envs))
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: int | None = None) -> Generator[RolloutBufferSamples]:
        assert self.full
        indices = th.randperm(self.buffer_size * self.n_envs)
        
        if not self.generator_ready:
            for tensor_name in ["observations", "actions", "values", "log_probs", "advantages", "returns"]:
                self.__dict__[tensor_name] = self.swap_and_flatten(self.__dict__[tensor_name])
            self.generator_ready = True

        batch_size = batch_size or (self.buffer_size * self.n_envs)
        for start_idx in range(0, self.buffer_size * self.n_envs, batch_size):
            yield self._get_samples(indices[start_idx : start_idx + batch_size])

    def _get_samples(self, batch_inds: th.Tensor) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds], self.actions[batch_inds], self.values[batch_inds],
            self.log_probs[batch_inds], self.advantages[batch_inds], self.returns[batch_inds],
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))