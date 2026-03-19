import os
import time
import argparse
import torch
import gymnasium as gym
import numpy as np
from minigrid.wrappers import FlatObsWrapper, OneHotPartialObsWrapper, FullyObsWrapper
from DQN_Rainbow import RainbowDQN, EVRainbowDQN


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

        # One-hot encode IDs: 2=Wall, 8=Goal
        one_hot_img = np.zeros((7, 7, 2), dtype=np.float32)
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


def make_env_thunk(env_name, fully_obs=False):
    def thunk():
        if env_name == "cartpole":
            env = gym.make("CartPole-v1")
        elif env_name == "mujoco":
            env = gym.make("HalfCheetah-v5")
        else:
            env = gym.make("MiniGrid-FourRooms-v0")
            env = FastObsWrapper(env)
        return env

    return thunk


def setup_config(args, obs_dim):
    cfg = {
        "munchausen": True,
        "soft": True,
        "Beta": 0.5 if args.ablation != 4 else 0.05,
        "dueling": True,
        "distributional": True,
        "ent_reg_coef": 0.01,
        "delayed": True,
        "tau": 0.03,
        "alpha": 0.9,
    }

    if args.ablation == 1:
        cfg["munchausen"] = False
        cfg["soft"] = False
    elif args.ablation == 2:
        cfg["ent_reg_coef"] = 0.0
    elif args.ablation == 3:
        cfg["Beta"] = 0.0
    elif args.ablation == 4:
        cfg["distributional"] = False
        cfg["dueling"] = False
        cfg["ent_reg_coef"] = 0.01
    elif args.ablation == 5:
        cfg["delayed"] = False

    if args.env_name == "mujoco":
        n_action_dims = 6
        n_action_bins = 3
        hidden_layer_sizes = [256, 256]
    elif args.env_name == "cartpole":
        n_action_dims = 1
        n_action_bins = 2
        hidden_layer_sizes = [64, 64]
    else:
        n_action_dims = 1
        n_action_bins = 7
        hidden_layer_sizes = [128, 128]

    AgentClass = RainbowDQN if cfg["distributional"] else EVRainbowDQN
    common_kwargs = dict(
        soft=cfg["soft"],
        munchausen=cfg["munchausen"],
        Thompson=False,
        dueling=cfg["dueling"],
        Beta=cfg["Beta"],
        ent_reg_coef=cfg["ent_reg_coef"],
        delayed=cfg["delayed"],
        tau=cfg["tau"],
        alpha=cfg["alpha"],
    )

    if AgentClass is RainbowDQN:
        dqn = AgentClass(
            obs_dim,
            n_action_dims,
            n_action_bins,
            hidden_layer_sizes=hidden_layer_sizes,
            Beta_half_life_steps=50000,
            norm_obs=False,
            burn_in_updates=10,  # low for benchmark
            **common_kwargs,
        )
    else:
        dqn = AgentClass(
            obs_dim,
            n_action_dims,
            n_action_bins,
            hidden_layer_sizes=hidden_layer_sizes,
            norm_obs=False,
            burn_in_updates=10,
            **common_kwargs,
        )
    return dqn


def benchmark_updates(
    dqn, obs_dim, args, device="cpu", batch_sizes=[64, 256, 1024], iters=50
):
    print(f"\n--- Benchmarking Updates on {device.upper()} ---")
    dqn.to(torch.device(device))

    if args.env_name == "mujoco":
        n_action_dims = 6
        n_action_bins = 3
    elif args.env_name == "cartpole":
        n_action_dims = 1
        n_action_bins = 2
    else:
        n_action_dims = 1
        n_action_bins = 7

    for bs in batch_sizes:
        # Create dummy buffer
        obs = torch.randn((bs, obs_dim), device=device)
        next_obs = torch.randn((bs, obs_dim), device=device)
        if n_action_dims == 1:
            actions = torch.randint(0, n_action_bins, (bs,), device=device)
        else:
            actions = torch.randint(
                0, n_action_bins, (bs, n_action_dims), device=device
            )
        rewards = torch.randn((bs,), device=device)
        terms = torch.zeros((bs,), device=device)

        # Warmup
        for _ in range(3):
            dqn.update(obs, actions, rewards, next_obs, terms, batch_size=bs, step=bs)

        if device == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()
        for i in range(iters):
            dqn.update(obs, actions, rewards, next_obs, terms, batch_size=bs, step=bs)

        if device == "cuda":
            torch.cuda.synchronize()

        end_time = time.time()
        avg_time = (end_time - start_time) / iters
        print(
            f"Batch Size {bs:4d}: {avg_time*1000:.2f} ms / update | Updates/sec: {1.0/avg_time:.2f}"
        )


def benchmark_action_sampling(
    dqn, obs_dim, device="cpu", batch_sizes=[1, 4, 16, 64], iters=200
):
    print(f"\n--- Benchmarking Action Sampling on {device.upper()} ---")
    dqn.to(torch.device(device))

    for bs in batch_sizes:
        obs = torch.randn((bs, obs_dim), device=device)

        # Warmup
        for _ in range(5):
            dqn.sample_action(obs, eps=0.1, step=0)

        if device == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()
        for _ in range(iters):
            dqn.sample_action(obs, eps=0.1, step=0)

        if device == "cuda":
            torch.cuda.synchronize()

        end_time = time.time()
        avg_time = (end_time - start_time) / iters
        print(
            f"Batch Size (Num Envs) {bs:4d}: {avg_time*1000:.2f} ms / sample | Batched Samples/sec: {1.0/avg_time:.2f}"
        )


def bins_to_continuous(action_bins):
    return np.array([b - 1.0 for b in action_bins], dtype=np.float32)


def benchmark_env_rollouts(args, dqn, obs_dim, total_steps=1000, batch_size=64):
    print(f"\n--- Benchmarking Environment Rollouts ---")
    print(f"Num Parallel Envs: {args.num_envs}, Device: {args.device}")
    device = torch.device(args.device)
    dqn.to(device)

    # Init VecEnv
    print("Initializing Vector Environment...")
    env_fns = [
        make_env_thunk(args.env_name, args.fully_obs) for _ in range(args.num_envs)
    ]
    vec_env = gym.vector.AsyncVectorEnv(env_fns)

    obs, info = vec_env.reset()

    if args.env_name == "mujoco":
        n_action_dims = 6
    elif args.env_name == "cartpole":
        n_action_dims = 1
    else:
        n_action_dims = 1

    print(f"Starting {total_steps} parallel steps...")
    start_time = time.time()

    buffer_size = 512
    obs_buffer = torch.zeros((buffer_size, obs_dim), device=device)
    next_obs_buffer = torch.zeros((buffer_size, obs_dim), device=device)
    if n_action_dims == 1:
        actions_buffer = torch.zeros((buffer_size,), dtype=torch.long, device=device)
    else:
        actions_buffer = torch.zeros(
            (buffer_size, n_action_dims), dtype=torch.long, device=device
        )
    rewards_buffer = torch.zeros((buffer_size,), device=device)
    terms_buffer = torch.zeros((buffer_size,), device=device)
    buffer_ptr = 0
    buffer_full = False

    steps_taken = 0
    steps_since_update = 0
    updates_performed = 0
    while steps_taken < total_steps:
        # 1. CPU -> GPU for obs
        tobs = torch.from_numpy(obs).to(device).float()

        # 2. Network forward pass
        actions = dqn.sample_action(tobs, eps=0.1, step=steps_taken)

        if args.env_name == "mujoco":
            step_actions = np.array([bins_to_continuous(a) for a in actions])
        else:
            if isinstance(actions, list) and isinstance(actions[0], list):
                actions = [
                    a[0] for a in actions
                ]  # Flatten if list of lists (for dim=1)
            step_actions = np.array(actions)

        actions = np.array(actions)

        # 4. Env step (multiprocessing over CPU)
        next_obs, r, term, trunc, info = vec_env.step(step_actions)

        tabs = torch.tensor(actions, device=device)
        tr = torch.tensor(r, dtype=torch.float32, device=device)
        tterms = torch.tensor(term, dtype=torch.float32, device=device)
        tnext_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)

        for i in range(args.num_envs):
            idx = (buffer_ptr + i) % buffer_size
            obs_buffer[idx] = tobs[i]
            actions_buffer[idx] = tabs[i]
            rewards_buffer[idx] = tr[i]
            next_obs_buffer[idx] = tnext_obs[i]
            terms_buffer[idx] = tterms[i]

        buffer_ptr = (buffer_ptr + args.num_envs) % buffer_size
        if buffer_ptr < args.num_envs:
            buffer_full = True

        steps_taken += args.num_envs
        steps_since_update += args.num_envs

        while steps_since_update >= 8:
            if buffer_full or buffer_ptr >= batch_size:
                updates_performed += 1
                max_idx = buffer_size if buffer_full else buffer_ptr
                dqn.update(
                    obs_buffer[:max_idx],
                    actions_buffer[:max_idx],
                    rewards_buffer[:max_idx],
                    next_obs_buffer[:max_idx],
                    terms_buffer[:max_idx],
                    batch_size=batch_size,
                    step=max_idx,
                )
            steps_since_update -= 8

        obs = next_obs

    end_time = time.time()
    duration = end_time - start_time
    sps = steps_taken / duration
    print(f"Total time for {steps_taken} frame steps: {duration:.2f}s")
    print(f"Real Steps/sec: {sps:.2f}")
    print(f"Updates/sec: {updates_performed/duration}")

    vec_env.close()
    return sps, updates_performed / duration


def run_grid_search(args, obs_dim, fully_obs=False, total_steps=2000):
    import json
    import os

    print("\n=== Starting Grid Search for Best Parameters ===")

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    num_envs_list = [1, 4, 8, 12, 16]
    best_results = {}
    all_results = {}
    args.fully_obs = fully_obs

    for ablation in range(6):
        print(f"\n--- Grid Search: Ablation {ablation} ---")
        args.ablation = ablation

        # Instantiate DQN
        dqn = setup_config(args, obs_dim)

        best_sps = 0.0
        best_config = None
        all_results[f"ablation_{ablation}"] = []

        for dev in devices:
            for num_envs in num_envs_list:
                args.device = dev
                args.num_envs = num_envs

                try:
                    sps, ups = benchmark_env_rollouts(
                        args, dqn, obs_dim, total_steps=total_steps, batch_size=64
                    )

                    current_config = {
                        "device": dev,
                        "num_envs": num_envs,
                        "steps_per_sec": sps,
                        "updates_per_sec": ups,
                    }
                    all_results[f"ablation_{ablation}"].append(current_config)

                    if sps > best_sps:
                        best_sps = sps
                        best_config = current_config
                except Exception as e:
                    import traceback

                    print(f"Error for device={dev}, num_envs={num_envs}: {e}")
                    traceback.print_exc()

        best_results[f"ablation_{ablation}"] = best_config
        print(f"Best for Ablation {ablation}: {best_config}")

    if hasattr(args, "grid_name") and args.grid_name is not None:
        file_prefix = args.grid_name
    else:
        try:
            file_prefix = os.getlogin()
        except Exception:
            import getpass

            file_prefix = getpass.getuser()

    best_filename = f"{file_prefix}_{args.env_name}_best.json"
    all_filename = f"{file_prefix}_{args.env_name}_all.json"

    with open(best_filename, "w") as f:
        json.dump(best_results, f, indent=4)

    with open(all_filename, "w") as f:
        json.dump(all_results, f, indent=4)

    print(
        f"\nGrid search complete. Saved best configs to {best_filename} and all configs to {all_filename}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel Environment Benchmarking")
    parser.add_argument(
        "--env_name",
        type=str,
        default="minigrid",
        choices=["cartpole", "minigrid", "mujoco"],
        help="Environment to use",
    )
    parser.add_argument(
        "--grid_search",
        action="store_true",
        help="Run parameter grid search and save json",
    )
    parser.add_argument(
        "--grid_name",
        type=str,
        default=None,
        help="Custom prefix for the grid search json output file (default: system username)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Main device",
    )
    parser.add_argument("--ablation", type=int, default=0, choices=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--fully_obs", action="store_true")
    parser.add_argument(
        "--num_envs", type=int, default=8, help="Number of parallel environments to run"
    )
    args = parser.parse_args()

    # Get obs dim from dummy env
    env = make_env_thunk(args.env_name, args.fully_obs)()
    obs, _ = env.reset()
    obs_dim = int(np.prod(np.asarray(obs).shape))
    env.close()

    print("=== Configuration ===")
    print(f"Environment: {args.env_name}")
    print(f"Ablation Level: {args.ablation}")
    print(f"Obs Dim: {obs_dim}")
    print("=====================")

    if args.grid_search:
        run_grid_search(args, obs_dim, fully_obs=args.fully_obs, total_steps=5000)
    else:
        dqn = setup_config(args, obs_dim)

        # 1. Benchmark Updates
        benchmark_updates(dqn, obs_dim, args, device="cpu", batch_sizes=[64, 256, 1024])
        if torch.cuda.is_available():
            benchmark_updates(
                dqn, obs_dim, args, device="cuda", batch_sizes=[64, 256, 1024]
            )

        # 2. Benchmark Action Sampling
        benchmark_action_sampling(
            dqn, obs_dim, device="cpu", batch_sizes=[1, 4, 16, 64, 256]
        )
        if torch.cuda.is_available():
            benchmark_action_sampling(
                dqn, obs_dim, device="cuda", batch_sizes=[1, 4, 16, 64, 256]
            )

        # 3. Benchmark Parallel Env Execution
        args.device = "cpu"
        for i in [1, 4, 8]:
            args.num_envs = i
            benchmark_env_rollouts(args, dqn, obs_dim, total_steps=5000)
        if torch.cuda.is_available():
            args.device = "cuda"
            for i in [1, 4, 8, 12, 16]:
                args.num_envs = i
                benchmark_env_rollouts(args, dqn, obs_dim, total_steps=5000)
