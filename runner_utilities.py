import os
import json
import time
import numpy as np
import gymnasium as gym
import torch

try:
    import minigrid  # noqa: F401
except Exception:
    minigrid = None

def get_device_name():
    try:
        with open("device_name.txt", "r") as f:
            return f.read().strip()
    except Exception:
        try:
            return os.getlogin()
        except Exception:
            import getpass
            return getpass.getuser()

# Map per-dimension discrete bin indices to continuous actions (-1, 0, 1)
def bins_to_continuous(action_bins):
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
        "n_action_bins": 3,
        "hidden_layer_sizes": [128, 128],
    }

class obs_transformer:
    def __init__(self):
        # 7x7 image, 3 channels (one-hot for IDs 1, 2, 8)
        self.image_flat_size = 7 * 7 * 2
        # Direction is one-hot encoded (4 values)
        self.direction_size = 4
        self.frame_size = self.image_flat_size + self.direction_size

        # Last obs includes image and direction
        self.last_obs = np.zeros(self.frame_size)

    def transform(self, obs):
        # Extract object ID channel
        img = obs["image"][:, :, 0]
        direction = obs["direction"]

        # One-hot encode IDs: 1=Empty, 2=Wall, 8=Goal
        one_hot_img = np.zeros((7, 7, 2), dtype=np.float32)
        # one_hot_img[:, :, 0] = img == 1
        one_hot_img[:, :, 0] = img == 2
        one_hot_img[:, :, 1] = img == 8

        # One-hot encode direction
        one_hot_dir = np.zeros(4, dtype=np.float32)
        one_hot_dir[direction] = 1.0

        current_obs = one_hot_img.flatten()

        # Combine current image and direction
        current_full = np.concatenate([current_obs, one_hot_dir])

        # Stack current and last frame
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
            low=0, high=1, shape=(len(dummy_obs),), dtype=np.float32
        )

    def observation(self, obs):
        return self.transformer.transform(obs)

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
    def thunk():
        if env_name == "minigrid":
            # Ensure MiniGrid environments are registered with Gymnasium.
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


def get_benchmark_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def resolve_torch_device(requested_device: str | None = None):
    if requested_device is None:
        requested_device = "cpu"
    dev = str(requested_device).strip()
    if dev.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(dev)


def benchmark_updates_generic(
    agent,
    device,
    batch_sizes,
    iters,
    make_batch_fn,
    update_fn,
    warmup_iters=3,
    header="Benchmarking Updates",
):
    print(f"\n--- {header} on {device.upper()} ---")
    agent.to(torch.device(device))

    for bs in batch_sizes:
        batch = make_batch_fn(bs, device)

        for _ in range(warmup_iters):
            update_fn(batch, 0)

        if device == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()
        for i in range(iters):
            update_fn(batch, i)

        if device == "cuda":
            torch.cuda.synchronize()

        end_time = time.time()
        avg_time = (end_time - start_time) / iters
        print(
            f"Batch Size {bs:4d}: {avg_time*1000:.2f} ms / update | Updates/sec: {1.0/avg_time:.2f}"
        )


def benchmark_action_sampling_generic(
    agent,
    obs_dim,
    device,
    batch_sizes,
    iters,
    sample_fn,
    warmup_iters=5,
    header="Benchmarking Action Sampling",
):
    print(f"\n--- {header} on {device.upper()} ---")
    agent.to(torch.device(device))

    for bs in batch_sizes:
        obs = torch.randn((bs, obs_dim), device=device)

        for _ in range(warmup_iters):
            sample_fn(obs)

        if device == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()
        for _ in range(iters):
            sample_fn(obs)

        if device == "cuda":
            torch.cuda.synchronize()

        end_time = time.time()
        avg_time = (end_time - start_time) / iters
        print(
            f"Batch Size (Num Envs) {bs:4d}: {avg_time*1000:.2f} ms / sample | Batched Samples/sec: {1.0/avg_time:.2f}"
        )


def save_grid_search_results(args, algo_name, best_results, all_results):
    best_filename, all_filename = get_grid_search_paths(args, algo_name)

    with open(best_filename, "w") as f:
        json.dump(best_results, f, indent=4)

    with open(all_filename, "w") as f:
        json.dump(all_results, f, indent=4)

    print(
        f"\nGrid search complete. Saved best configs to {best_filename} and all configs to {all_filename}"
    )


def get_grid_search_paths(args, algo_name):
    if hasattr(args, "device_name") and args.device_name is not None:
        file_prefix = args.device_name
    else:
        file_prefix = get_device_name()

    os.makedirs(f"time_files/{file_prefix}", exist_ok=True)
    best_filename = f"time_files/{file_prefix}/{args.env_name}_{algo_name}_best.json"
    all_filename = f"time_files/{file_prefix}/{args.env_name}_{algo_name}_all.json"
    return best_filename, all_filename


def load_grid_search_results(args, algo_name):
    best_filename, all_filename = get_grid_search_paths(args, algo_name)
    best_results = {}
    all_results = {}

    if os.path.exists(best_filename):
        try:
            with open(best_filename, "r") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    best_results = loaded
        except Exception:
            pass

    if os.path.exists(all_filename):
        try:
            with open(all_filename, "r") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    all_results = loaded
        except Exception:
            pass

    return best_results, all_results


def plot_results(results, args, model_name):
    import matplotlib.pyplot as plt

    runner_name = args.env_name
    results_dir = os.path.join("results", model_name, runner_name)
    os.makedirs(results_dir, exist_ok=True)

    np.save(
        os.path.join(results_dir, f"train_scores_{args.run}_{args.ablation}.npy"),
        results["rhist"],
    )
    np.save(
        os.path.join(results_dir, f"eval_scores_{args.run}_{args.ablation}.npy"),
        results["eval_hist"],
    )
    np.save(
        os.path.join(results_dir, f"loss_hist_{args.run}_{args.ablation}.npy"),
        results["lhist"],
    )
    np.save(
        os.path.join(
            results_dir, f"smooth_train_scores_{args.run}_{args.ablation}.npy"
        ),
        results["smooth_rhist"],
    )

    plt.plot(results["rhist"])
    plt.plot(results["smooth_rhist"])
    plt.legend(["R hist", "Smooth R hist"])
    plt.xlabel("Episode")
    plt.ylabel("Training Episode Reward")
    plt.grid()
    plt.title(f"Training rewards, run {args.run} ablated: {args.ablation}")
    plt.savefig(os.path.join(results_dir, f"train_scores_{args.run}_{args.ablation}"))
    plt.close()
    
    plt.plot(results["eval_hist"])
    plt.grid()
    plt.title(f"eval scores, run {args.run} ablated: {args.ablation}")
    plt.savefig(os.path.join(results_dir, f"eval_scores_{args.run}_{args.ablation}"))
    plt.close()

    # Save total wall clock training time
    train_time_seconds = results["train_time"]
    np.save(
        os.path.join(results_dir, f"train_time_{args.run}_{args.ablation}.npy"),
        train_time_seconds,
    )
    print(f"Training wall clock time: {train_time_seconds:.2f} seconds")