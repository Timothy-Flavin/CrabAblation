from __future__ import annotations

import importlib
import gymnasium as gym
import numpy as np

try:
    import minigrid  # noqa: F401
except Exception:
    minigrid = None


def get_env_benchmark_spec(env_name: str):
    """Return shared benchmark model/action configuration by environment family."""
    if env_name == "hide-and-seek-engine":
        return {
            "n_action_dims": 1,
            "n_action_bins": 36,
            "hidden_layer_sizes": [256, 256],
        }
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


def _extract_mixed_observation_space_parts(observation_space):
    if (
        isinstance(observation_space, gym.spaces.Tuple)
        and len(observation_space.spaces) >= 2
    ):
        first = observation_space.spaces[0]
        second = observation_space.spaces[1]
        if isinstance(first, gym.spaces.Box) and isinstance(second, gym.spaces.Box):
            return first, second

    if isinstance(observation_space, gym.spaces.Dict):
        if (
            "spatial" in observation_space.spaces
            and "internal" in observation_space.spaces
        ):
            spatial = observation_space.spaces["spatial"]
            vector = observation_space.spaces["internal"]
            if isinstance(spatial, gym.spaces.Box) and isinstance(
                vector, gym.spaces.Box
            ):
                return spatial, vector

        if (
            "spatial" in observation_space.spaces
            and "vector" in observation_space.spaces
        ):
            spatial = observation_space.spaces["spatial"]
            vector = observation_space.spaces["vector"]
            if isinstance(spatial, gym.spaces.Box) and isinstance(
                vector, gym.spaces.Box
            ):
                return spatial, vector

        box_items = [
            (k, v)
            for k, v in observation_space.spaces.items()
            if isinstance(v, gym.spaces.Box)
        ]
        if len(box_items) >= 2:
            box_items.sort(key=lambda kv: len(kv[1].shape), reverse=True)
            return box_items[0][1], box_items[1][1]

    raise ValueError(
        "Expected mixed observation space with two Box branches (spatial + vector)."
    )


def extract_mixed_observation_shapes(observation_space):
    spatial_space, vector_space = _extract_mixed_observation_space_parts(
        observation_space
    )
    spatial_shape = tuple(int(v) for v in spatial_space.shape)
    vector_dim = int(np.prod(vector_space.shape))
    return spatial_shape, vector_dim


def _extract_mixed_observation_parts(obs):
    if isinstance(obs, (tuple, list)) and len(obs) >= 2:
        return obs[0], obs[1]
    if isinstance(obs, dict):
        if "spatial" in obs and "internal" in obs:
            return obs["spatial"], obs["internal"]
        if "spatial" in obs and "vector" in obs:
            return obs["spatial"], obs["vector"]
        # Common fallback key names
        spatial = None
        vector = None
        for key in ("image", "board", "grid", "spatial"):
            if key in obs:
                spatial = obs[key]
                break
        for key in ("internal", "vector", "state", "features", "agent_state"):
            if key in obs:
                vector = obs[key]
                break
        if spatial is not None and vector is not None:
            return spatial, vector
    raise ValueError("Unsupported mixed observation payload format")


def flatten_mixed_observation(obs, spatial_shape, vector_dim: int):
    spatial_raw, vector_raw = _extract_mixed_observation_parts(obs)
    spatial = np.asarray(spatial_raw, dtype=np.float32).reshape(-1)
    vector = np.asarray(vector_raw, dtype=np.float32).reshape(-1)
    expected_spatial = int(np.prod(spatial_shape))
    expected_vector = int(vector_dim)
    if spatial.size != expected_spatial:
        raise ValueError(
            f"Expected spatial size {expected_spatial}, got {spatial.size}"
        )
    if vector.size != expected_vector:
        raise ValueError(f"Expected vector size {expected_vector}, got {vector.size}")
    return np.concatenate([spatial, vector], axis=0).astype(np.float32)


def _mixed_action_space_parts(action_space):
    if isinstance(action_space, gym.spaces.Tuple) and len(action_space.spaces) >= 2:
        box_space = action_space.spaces[0]
        discrete_space = action_space.spaces[1]
        if isinstance(box_space, gym.spaces.Box) and isinstance(
            discrete_space, gym.spaces.Discrete
        ):
            return box_space, discrete_space, ("tuple", None, None)

    if isinstance(action_space, gym.spaces.Dict):
        if "move" in action_space.spaces and "radio" in action_space.spaces:
            move_space = action_space.spaces["move"]
            radio_space = action_space.spaces["radio"]
            if isinstance(move_space, gym.spaces.Box) and isinstance(
                radio_space, gym.spaces.Discrete
            ):
                return move_space, radio_space, ("dict", "move", "radio")

        box_items = [
            (k, v)
            for k, v in action_space.spaces.items()
            if isinstance(v, gym.spaces.Box)
        ]
        discrete_items = [
            (k, v)
            for k, v in action_space.spaces.items()
            if isinstance(v, gym.spaces.Discrete)
        ]
        if box_items and discrete_items:
            move_key, move_space = box_items[0]
            radio_key, radio_space = discrete_items[0]
            return move_space, radio_space, ("dict", move_key, radio_key)

    raise ValueError(
        "Expected hybrid action space (Tuple(Box, Discrete) or Dict(move, radio))"
    )


def _format_mixed_action(move_values, radio_value: int, structure):
    kind, move_key, radio_key = structure
    if kind == "tuple":
        return (move_values, int(radio_value))
    if kind == "dict":
        assert move_key is not None and radio_key is not None
        return {move_key: move_values, radio_key: int(radio_value)}
    raise ValueError(f"Unsupported mixed action structure: {kind}")


def map_continuous_to_mixed_action(flat_action, action_space):
    box_space, discrete_space, structure = _mixed_action_space_parts(action_space)
    flat_action = np.asarray(flat_action, dtype=np.float32).reshape(-1)
    box_dim = int(np.prod(box_space.shape))
    discrete_dim = int(discrete_space.n)

    if flat_action.size < box_dim:
        raise ValueError(f"Expected at least {box_dim} values, got {flat_action.size}")

    box_values = flat_action[:box_dim].reshape(box_space.shape)
    box_values = np.clip(box_values, box_space.low, box_space.high).astype(np.float32)

    discrete_slice = flat_action[box_dim : box_dim + discrete_dim]
    if discrete_slice.size == discrete_dim:
        discrete_value = int(np.argmax(discrete_slice))
    elif discrete_slice.size == 1:
        x = float(np.clip(discrete_slice[0], -1.0, 1.0))
        scaled = (x + 1.0) * 0.5 * float(discrete_space.n)
        discrete_value = int(np.floor(scaled))
    else:
        discrete_value = 0

    discrete_value = int(np.clip(discrete_value, 0, discrete_space.n - 1))
    return _format_mixed_action(box_values, discrete_value, structure)


def mixed_discrete_action_size(action_space, bins_per_dim: int = 3) -> int:
    box_space, discrete_space, _ = _mixed_action_space_parts(action_space)
    box_dim = int(np.prod(box_space.shape))
    return int((bins_per_dim**box_dim) * int(discrete_space.n))


def map_discrete_to_mixed_action(
    action_index: int, action_space, bins_per_dim: int = 3
):
    box_space, discrete_space, structure = _mixed_action_space_parts(action_space)
    total_actions = mixed_discrete_action_size(action_space, bins_per_dim=bins_per_dim)
    idx = int(np.clip(int(action_index), 0, total_actions - 1))

    discrete_value = idx % int(discrete_space.n)
    box_code = idx // int(discrete_space.n)

    box_dim = int(np.prod(box_space.shape))
    digits = []
    for _ in range(box_dim):
        digits.append(box_code % bins_per_dim)
        box_code //= bins_per_dim

    digit_arr = np.asarray(digits, dtype=np.float32)
    if bins_per_dim == 1:
        ratio = np.zeros_like(digit_arr)
    else:
        ratio = digit_arr / float(bins_per_dim - 1)

    low = np.asarray(box_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(box_space.high, dtype=np.float32).reshape(-1)
    box_values = (
        (low + ratio * (high - low)).reshape(box_space.shape).astype(np.float32)
    )
    return _format_mixed_action(box_values, int(discrete_value), structure)


class ContinuousMixedActionWrapper(gym.ActionWrapper):
    """Accept flat continuous action and map it to hybrid move/radio action."""

    def __init__(self, env):
        super().__init__(env)
        box_space, discrete_space, _ = _mixed_action_space_parts(env.action_space)
        low = np.concatenate(
            [
                np.asarray(box_space.low, dtype=np.float32).reshape(-1),
                -np.ones((discrete_space.n,), dtype=np.float32),
            ]
        )
        high = np.concatenate(
            [
                np.asarray(box_space.high, dtype=np.float32).reshape(-1),
                np.ones((discrete_space.n,), dtype=np.float32),
            ]
        )
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def action(self, action):
        return map_continuous_to_mixed_action(action, self.env.action_space)


class DiscreteMixedActionWrapper(gym.ActionWrapper):
    """Accept integer action and map it to hybrid move/radio action."""

    def __init__(self, env, bins_per_dim: int = 3):
        super().__init__(env)
        self.bins_per_dim = int(max(1, bins_per_dim))
        n_actions = mixed_discrete_action_size(
            env.action_space, bins_per_dim=self.bins_per_dim
        )
        self.action_space = gym.spaces.Discrete(n_actions)

    def action(self, action):
        return map_discrete_to_mixed_action(
            int(action), self.env.action_space, bins_per_dim=self.bins_per_dim
        )


def _normalize_parallel_reset_output(reset_out):
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        return reset_out[0], reset_out[1]
    return reset_out, {}


class PettingZooParallelVecAdapter:
    """Adapter exposing a PettingZoo parallel env with a vec-env-like API."""

    def __init__(
        self, parallel_env, action_mode: str = "discrete", bins_per_dim: int = 3
    ):
        self.env = parallel_env
        self.agent_ids = list(getattr(self.env, "possible_agents", []))
        if not self.agent_ids:
            self.agent_ids = list(getattr(self.env, "agents", []))
        if not self.agent_ids:
            raise ValueError(
                "Parallel environment must expose possible_agents or agents"
            )

        self.num_envs = len(self.agent_ids)
        self.action_mode = action_mode
        self.bins_per_dim = int(max(1, bins_per_dim))

        first_agent = self.agent_ids[0]
        self.raw_observation_space = self.env.observation_space(first_agent)
        self.raw_action_space = self.env.action_space(first_agent)

        self.spatial_shape, self.vector_dim = extract_mixed_observation_shapes(
            self.raw_observation_space
        )
        self.flat_obs_dim = int(np.prod(self.spatial_shape)) + int(self.vector_dim)
        self.single_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.flat_obs_dim,),
            dtype=np.float32,
        )

        if self.action_mode == "continuous":
            box_space, discrete_space, _ = _mixed_action_space_parts(
                self.raw_action_space
            )
            low = np.concatenate(
                [
                    np.asarray(box_space.low, dtype=np.float32).reshape(-1),
                    -np.ones((discrete_space.n,), dtype=np.float32),
                ]
            )
            high = np.concatenate(
                [
                    np.asarray(box_space.high, dtype=np.float32).reshape(-1),
                    np.ones((discrete_space.n,), dtype=np.float32),
                ]
            )
            self.single_action_space = gym.spaces.Box(
                low=low,
                high=high,
                dtype=np.float32,
            )
            self._raw_action_transform = ActionTransformHandler(
                "hide-and-seek-engine",
                "sac",
                self.raw_action_space,
                bins_per_dim=self.bins_per_dim,
                batched=False,
            )
        elif self.action_mode == "discrete":
            n_actions = mixed_discrete_action_size(
                self.raw_action_space, bins_per_dim=self.bins_per_dim
            )
            self.single_action_space = gym.spaces.Discrete(n_actions)
            self._raw_action_transform = ActionTransformHandler(
                "hide-and-seek-engine",
                "dqn",
                self.raw_action_space,
                bins_per_dim=self.bins_per_dim,
                batched=False,
            )
        else:
            raise ValueError(f"Unsupported action_mode: {self.action_mode}")

        self._zero_obs = np.zeros((self.flat_obs_dim,), dtype=np.float32)

    def _flatten_obs_for_agent(self, obs_dict, agent_id):
        raw_obs = obs_dict.get(agent_id)
        if raw_obs is None:
            return self._zero_obs.copy()
        return flatten_mixed_observation(raw_obs, self.spatial_shape, self.vector_dim)

    def reset(self, **kwargs):
        obs_dict, info_dict = _normalize_parallel_reset_output(self.env.reset(**kwargs))
        obs_batch = np.stack(
            [self._flatten_obs_for_agent(obs_dict, aid) for aid in self.agent_ids],
            axis=0,
        )
        return obs_batch, info_dict

    def step(self, actions):
        actions_arr = np.asarray(actions)
        if actions_arr.ndim == 0:
            actions_arr = actions_arr.reshape(1)
        action_dict = {}
        for i, aid in enumerate(self.agent_ids):
            action_dict[aid] = self._raw_action_transform.transform_action(
                actions_arr[i]
            )

        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self.env.step(
            action_dict
        )

        obs_batch = np.stack(
            [self._flatten_obs_for_agent(obs_dict, aid) for aid in self.agent_ids],
            axis=0,
        )
        rewards = np.asarray([float(rew_dict.get(aid, 0.0)) for aid in self.agent_ids])
        terminations = np.asarray(
            [bool(term_dict.get(aid, False)) for aid in self.agent_ids], dtype=bool
        )
        truncations = np.asarray(
            [bool(trunc_dict.get(aid, False)) for aid in self.agent_ids], dtype=bool
        )

        if bool(np.all(np.logical_or(terminations, truncations))):
            final_observation = obs_batch.copy()
            reset_obs, reset_info = self.reset()
            infos = {
                "final_observation": final_observation,
                "final_info": info_dict,
                "reset_info": reset_info,
            }
            return reset_obs, rewards, terminations, truncations, infos

        return obs_batch, rewards, terminations, truncations, info_dict

    def close(self):
        self.env.close()


def _load_hide_and_seek_parallel_env():
    import os
    from hide_and_seek_engine.env_wrapper import SARParallelPettingZooEnv

    level_dir = "./test_level"

    return SARParallelPettingZooEnv(
        map_png=os.path.join(level_dir, "level.png"),
        tiles_json=os.path.join(level_dir, "tiles.json"),
        agents_json=os.path.join(level_dir, "agents.json"),
        survivors_json=os.path.join(level_dir, "survivors.json"),
    )


def make_hide_and_seek_vec_env(action_mode: str = "discrete", bins_per_dim: int = 3):
    parallel_env = _load_hide_and_seek_parallel_env()
    return PettingZooParallelVecAdapter(
        parallel_env,
        action_mode=action_mode,
        bins_per_dim=bins_per_dim,
    )


SUPPORTED_ENVIRONMENTS = (
    "minigrid",
    "mujoco",
    "cartpole",
    "hide-and-seek-engine",
)
SUPPORTED_ALGOS = ("dqn", "ppo", "sac")


class ActionTransformHandler:
    """Explicit env+algo action transformation with fail-fast dispatch."""

    def __init__(
        self,
        env_name: str,
        algo: str,
        action_space,
        *,
        bins_per_dim: int = 3,
        discrete_bins: int = 3,
        batched: bool = True,
    ):
        self.env_name = str(env_name)
        self.algo = str(algo)
        self.action_space = action_space
        self.bins_per_dim = int(max(1, bins_per_dim))
        self.discrete_bins = int(max(1, discrete_bins))
        self.batched = bool(batched)

        if self.env_name not in SUPPORTED_ENVIRONMENTS:
            raise ValueError(
                f"Unsupported env_name '{self.env_name}'. Supported: {SUPPORTED_ENVIRONMENTS}"
            )
        if self.algo not in SUPPORTED_ALGOS:
            raise ValueError(
                f"Unsupported algo '{self.algo}'. Supported: {SUPPORTED_ALGOS}"
            )

        self.transform_action = self._assign_transform()

    def _assign_transform(self):
        if self.env_name in ("cartpole", "minigrid"):
            if self.algo in ("dqn", "ppo"):
                return self._dummy
            if self.algo == "sac":
                return self._continuous_to_discrete_classic

        if self.env_name == "mujoco":
            if self.algo in ("dqn", "ppo"):
                return self._discrete_to_continuous_mujoco
            if self.algo == "sac":
                return self._continuous_passthrough

        if self.env_name == "hide-and-seek-engine":
            if self.algo in ("dqn", "ppo"):
                if isinstance(self.action_space, gym.spaces.Discrete):
                    # Vec adapter consumes integer ids and does hybrid conversion internally.
                    return self._dummy
                return self._discrete_to_hybrid_hide_and_seek
            if self.algo == "sac":
                if isinstance(self.action_space, gym.spaces.Box):
                    # Vec adapter consumes continuous vectors and does hybrid conversion internally.
                    return self._continuous_passthrough
                return self._continuous_to_hybrid_hide_and_seek

        raise ValueError(
            f"Unsupported env/algo combination: env='{self.env_name}', algo='{self.algo}'"
        )

    def _dummy(self, action):
        arr = np.asarray(action)
        if self.batched:
            if arr.ndim > 1 and arr.shape[-1] == 1:
                arr = arr[..., 0]
            return arr.astype(np.int64) if np.issubdtype(arr.dtype, np.integer) else arr
        if arr.ndim == 0:
            return arr.item()
        if arr.size == 1:
            return arr.reshape(-1)[0].item()
        return arr

    def _continuous_passthrough(self, continuous_action):
        arr = np.asarray(continuous_action, dtype=np.float32)
        if isinstance(self.action_space, gym.spaces.Box):
            arr = np.clip(arr, self.action_space.low, self.action_space.high)
        if self.batched:
            return arr
        if arr.ndim == 0:
            return float(arr.item())
        return arr

    def _continuous_to_discrete_classic(self, continuous_action):
        arr = np.asarray(continuous_action, dtype=np.float32)

        if isinstance(self.action_space, gym.spaces.Discrete):
            if self.batched:
                if arr.ndim == 0:
                    arr = arr.reshape(1, 1)
                elif arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                x = np.clip(arr[:, 0], 0.0, 1.0 - 1e-8)
                bins = (x * self.action_space.n).astype(np.int64)
                return np.clip(bins, 0, self.action_space.n - 1)

            x = float(np.clip(np.asarray(arr).reshape(-1)[0], 0.0, 1.0 - 1e-8))
            return int(
                np.clip(int(x * self.action_space.n), 0, self.action_space.n - 1)
            )

        if isinstance(self.action_space, gym.spaces.MultiDiscrete):
            nvec = np.asarray(self.action_space.nvec, dtype=np.int64)
            if self.batched:
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                out = np.zeros((arr.shape[0], len(nvec)), dtype=np.int64)
                for dim, n in enumerate(nvec):
                    x = np.clip(arr[:, dim], 0.0, 1.0 - 1e-8)
                    out[:, dim] = np.clip((x * n).astype(np.int64), 0, n - 1)
                return out

            arr1 = np.asarray(arr).reshape(-1)
            out = np.zeros((len(nvec),), dtype=np.int64)
            for dim, n in enumerate(nvec):
                x = float(np.clip(arr1[dim], 0.0, 1.0 - 1e-8))
                out[dim] = int(np.clip(int(x * n), 0, n - 1))
            return out

        raise ValueError(
            f"SAC discrete transform expects Discrete/MultiDiscrete action space, got {type(self.action_space)}"
        )

    def _discrete_to_continuous_mujoco(self, discrete_action):
        arr = np.asarray(discrete_action, dtype=np.float32)
        if self.batched:
            if arr.ndim == 1:
                if isinstance(self.action_space, gym.spaces.Box):
                    act_dim = int(np.prod(self.action_space.shape))
                    if arr.shape[0] == act_dim:
                        arr = arr.reshape(1, act_dim)
                    else:
                        arr = arr.reshape(-1, 1)
                else:
                    arr = arr.reshape(-1, 1)
            if arr.ndim > 2:
                arr = arr.reshape(arr.shape[0], -1)
        else:
            arr = arr.reshape(-1)

        if self.discrete_bins <= 1:
            cont = np.zeros_like(arr, dtype=np.float32)
        else:
            cont = 2.0 * (arr / float(self.discrete_bins - 1)) - 1.0

        cont = np.clip(cont, -1.0, 1.0)
        if isinstance(self.action_space, gym.spaces.Box):
            cont = np.clip(cont, self.action_space.low, self.action_space.high)

        if self.batched:
            return cont.astype(np.float32)
        return cont.astype(np.float32)

    def _discrete_to_hybrid_hide_and_seek(self, discrete_action):
        if self.batched:
            arr = np.asarray(discrete_action)
            if arr.ndim > 1 and arr.shape[-1] == 1:
                arr = arr[..., 0]
            return [
                map_discrete_to_mixed_action(
                    int(a),
                    self.action_space,
                    bins_per_dim=self.bins_per_dim,
                )
                for a in arr.reshape(-1)
            ]

        return map_discrete_to_mixed_action(
            int(np.asarray(discrete_action).reshape(-1)[0]),
            self.action_space,
            bins_per_dim=self.bins_per_dim,
        )

    def _continuous_to_hybrid_hide_and_seek(self, continuous_action):
        if self.batched:
            arr = np.asarray(continuous_action, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return [map_continuous_to_mixed_action(a, self.action_space) for a in arr]

        return map_continuous_to_mixed_action(
            np.asarray(continuous_action, dtype=np.float32),
            self.action_space,
        )


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
