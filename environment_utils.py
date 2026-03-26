from __future__ import annotations

import gymnasium as gym
import numpy as np

try:
    import minigrid  # noqa: F401
except Exception:
    minigrid = None


def bins_to_continuous(action_bins):
    """Map per-dimension discrete bin indices to continuous actions in [-1, 1]."""
    return np.array([b - 1.0 for b in action_bins], dtype=np.float32)


def get_env_benchmark_spec(env_name: str):
    """Return shared benchmark model/action configuration by environment family."""
    if env_name == "mujoco":
        return {
            "n_action_dims": 6,
            "n_action_bins": 3,
            "hidden_layer_sizes": [128, 128],
        }
    if env_name == "cartpole":
        return {
            "n_action_dims": 1,
            "n_action_bins": 2,
            "hidden_layer_sizes": [32, 32],
        }
    return {
        "n_action_dims": 1,
        "n_action_bins": 7,
        "hidden_layer_sizes": [128, 128],
    }


class obs_transformer:
    def __init__(self):
        self.image_flat_size = 7 * 7 * 2
        self.direction_size = 4
        self.frame_size = self.image_flat_size + self.direction_size
        self.last_obs = np.zeros(self.frame_size)

    def transform(self, obs):
        img = obs["image"][:, :, 0]
        direction = obs["direction"]

        one_hot_img = np.zeros((7, 7, 2), dtype=np.float32)
        one_hot_img[:, :, 0] = img == 2
        one_hot_img[:, :, 1] = img == 8

        one_hot_dir = np.zeros(4, dtype=np.float32)
        one_hot_dir[direction] = 1.0

        current_obs = one_hot_img.flatten()
        current_full = np.concatenate([current_obs, one_hot_dir])
        transformed_obs = np.concatenate([current_full, self.last_obs])

        self.last_obs = current_full
        return transformed_obs

    def reset(self):
        self.last_obs = np.zeros(self.frame_size)
        return np.concatenate([self.last_obs, self.last_obs])


class FastObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.transformer = obs_transformer()
        dummy_obs = self.transformer.reset()
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(len(dummy_obs),),
            dtype=np.float32,
        )

    def observation(self, observation):
        return self.transformer.transform(observation)

    def reset(self, **kwargs):
        self.transformer.reset()
        return super().reset(**kwargs)


class RandomStartWrapper(gym.Wrapper):
    def __init__(self, env, max_random_start_steps=0, rng_seed=None):
        super().__init__(env)
        self.max_random_start_steps = int(max(0, max_random_start_steps))
        self._rng = np.random.default_rng(rng_seed)
        self._random_start_applied = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.max_random_start_steps <= 0 or self._random_start_applied:
            return obs, info

        n_steps = int(self._rng.integers(0, self.max_random_start_steps + 1))
        for _ in range(n_steps):
            a = self.env.action_space.sample()
            obs, _, term, trunc, info = self.env.step(a)
            if term or trunc:
                obs, info = self.env.reset()

        self._random_start_applied = True
        info = dict(info) if isinstance(info, dict) else {}
        info["random_start_steps"] = n_steps
        return obs, info


def make_env_thunk(
    fully_obs,
    env_name,
    env_id=None,
    seed=None,
    idx=0,
    capture_video=False,
    run_name=None,
    random_start_steps=0,
):
    """Return a thunk that builds one environment with common wrappers."""

    def thunk():
        if env_name == "minigrid":
            if minigrid is None:
                import minigrid as _minigrid  # noqa: F401
            resolved_env_id = env_id if env_id is not None else "MiniGrid-FourRooms-v0"
            env = gym.make(resolved_env_id)
            env = FastObsWrapper(env)
        elif env_name == "cartpole":
            resolved_env_id = env_id if env_id is not None else "CartPole-v1"
            env = gym.make(resolved_env_id)
        elif env_name == "mujoco":
            resolved_env_id = env_id if env_id is not None else "HalfCheetah-v5"
            if capture_video and idx == 0 and run_name is not None:
                env = gym.make(resolved_env_id, render_mode="rgb_array")
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            else:
                env = gym.make(resolved_env_id)
        else:
            raise ValueError(f"Unsupported env_name: {env_name}")

        env = gym.wrappers.RecordEpisodeStatistics(env)
        if seed is not None:
            env.action_space.seed(int(seed))

        if random_start_steps and random_start_steps > 0:
            rng_seed = None if seed is None else int(seed) + int(idx)
            env = RandomStartWrapper(
                env,
                max_random_start_steps=random_start_steps,
                rng_seed=rng_seed,
            )

        return env

    return thunk


def _proxy_action_space(action_space):
    """Return a Box proxy action space for algorithms expecting continuous actions."""
    if isinstance(action_space, gym.spaces.Box):
        return action_space
    if isinstance(action_space, gym.spaces.Discrete):
        return gym.spaces.Box(
            low=np.zeros((1,), dtype=np.float32),
            high=np.ones((1,), dtype=np.float32),
            dtype=np.float32,
        )
    if isinstance(action_space, gym.spaces.MultiDiscrete):
        n_dims = int(len(action_space.nvec))
        return gym.spaces.Box(
            low=np.zeros((n_dims,), dtype=np.float32),
            high=np.ones((n_dims,), dtype=np.float32),
            dtype=np.float32,
        )
    raise NotImplementedError(f"Unsupported action space: {action_space}")


def _continuous_to_env_action(actions_cont, env_action_space):
    """Map continuous proxy actions to the environment's native action space."""
    actions_cont = np.asarray(actions_cont, dtype=np.float32)

    if isinstance(env_action_space, gym.spaces.Box):
        return np.clip(actions_cont, env_action_space.low, env_action_space.high)

    if actions_cont.ndim == 1:
        actions_cont = actions_cont.reshape(-1, 1)

    if isinstance(env_action_space, gym.spaces.Discrete):
        x = np.clip(actions_cont[:, 0], 0.0, 1.0 - 1e-8)
        bins = (x * env_action_space.n).astype(np.int64)
        return np.clip(bins, 0, env_action_space.n - 1)

    if isinstance(env_action_space, gym.spaces.MultiDiscrete):
        out = np.zeros((actions_cont.shape[0], len(env_action_space.nvec)), dtype=np.int64)
        for dim, n in enumerate(env_action_space.nvec):
            x = np.clip(actions_cont[:, dim], 0.0, 1.0 - 1e-8)
            out[:, dim] = np.clip((x * int(n)).astype(np.int64), 0, int(n) - 1)
        return out

    raise NotImplementedError(f"Unsupported env action space: {env_action_space}")
