import os
import time
import argparse
import torch
import gymnasium as gym
import numpy as np

from DQN_Rainbow import RainbowDQN, EVRainbowDQN
from runner_utilities import (
    make_env_thunk,
    bins_to_continuous,
    benchmark_updates_generic,
    benchmark_action_sampling_generic,
    get_benchmark_devices,
    save_grid_search_results,
    load_grid_search_results,
    resolve_torch_device,
)


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
    if args.env_name == "mujoco":
        n_action_dims = 6
        n_action_bins = 3
    elif args.env_name == "cartpole":
        n_action_dims = 1
        n_action_bins = 2
    else:
        n_action_dims = 1
        n_action_bins = 7

    def make_batch(bs, dev):
        obs = torch.randn((bs, obs_dim), device=dev)
        next_obs = torch.randn((bs, obs_dim), device=dev)
        if n_action_dims == 1:
            actions = torch.randint(0, n_action_bins, (bs,), device=dev)
        else:
            actions = torch.randint(
                0, n_action_bins, (bs, n_action_dims), device=dev
            )
        rewards = torch.randn((bs,), device=dev)
        terms = torch.zeros((bs,), device=dev)
        return {
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "next_obs": next_obs,
            "terms": terms,
            "bs": bs,
        }

    def run_update(batch, _):
        dqn.update(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["next_obs"],
            batch["terms"],
            batch_size=batch["bs"],
            step=batch["bs"],
        )

    benchmark_updates_generic(
        dqn,
        device=device,
        batch_sizes=batch_sizes,
        iters=iters,
        make_batch_fn=make_batch,
        update_fn=run_update,
    )


def benchmark_action_sampling(
    dqn, obs_dim, device="cpu", batch_sizes=[1, 4, 16, 64], iters=200
):
    benchmark_action_sampling_generic(
        dqn,
        obs_dim=obs_dim,
        device=device,
        batch_sizes=batch_sizes,
        iters=iters,
        sample_fn=lambda obs: dqn.sample_action(obs, eps=0.1, step=0),
    )


def benchmark_env_rollouts(args, dqn, obs_dim, total_steps=1000, batch_size=64):
    print(f"\n--- Benchmarking Environment Rollouts ---")
    print(f"Num Parallel Envs: {args.num_envs}, Device: {args.device}")
    device = resolve_torch_device(args.device)
    dqn.to(device)

    # Init VecEnv
    print("Initializing Vector Environment...")
    env_fns = [
        make_env_thunk(args.fully_obs, args.env_name) for _ in range(args.num_envs)
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
    print("\n=== Starting Grid Search for Best Parameters ===")
    devices = get_benchmark_devices()

    num_envs_list = [1, 4, 8, 12, 16]
    if args.replace_existing:
        best_results = {}
        all_results = {}
    else:
        best_results, all_results = load_grid_search_results(args, "dqn")
    args.fully_obs = fully_obs

    for ablation in range(6):
        print(f"\n--- Grid Search: Ablation {ablation} ---")
        args.ablation = ablation

        # Instantiate DQN
        dqn = setup_config(args, obs_dim)

        best_sps = 0.0
        best_config = None
        all_results.setdefault(f"ablation_{ablation}", [])
        existing_trials = {
            (entry.get("device"), entry.get("num_envs"))
            for entry in all_results[f"ablation_{ablation}"]
            if isinstance(entry, dict)
        }

        for dev in devices:
            for num_envs in num_envs_list:
                args.device = dev
                args.num_envs = num_envs

                if (not args.replace_existing) and ((dev, num_envs) in existing_trials):
                    print(
                        f"Skipping existing trial: ablation={ablation}, device={dev}, num_envs={num_envs}"
                    )
                    continue

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
                    if args.replace_existing and ((dev, num_envs) in existing_trials):
                        for idx, entry in enumerate(all_results[f"ablation_{ablation}"]):
                            if (
                                isinstance(entry, dict)
                                and entry.get("device") == dev
                                and entry.get("num_envs") == num_envs
                            ):
                                all_results[f"ablation_{ablation}"][idx] = current_config
                                break
                    else:
                        all_results[f"ablation_{ablation}"].append(current_config)
                        existing_trials.add((dev, num_envs))

                    ablation_entries = [
                        e
                        for e in all_results[f"ablation_{ablation}"]
                        if isinstance(e, dict) and "steps_per_sec" in e
                    ]
                    if ablation_entries:
                        best_results[f"ablation_{ablation}"] = max(
                            ablation_entries, key=lambda e: e["steps_per_sec"]
                        )

                    save_grid_search_results(args, "dqn", best_results, all_results)

                    if sps > best_sps:
                        best_sps = sps
                        best_config = current_config
                except Exception as e:
                    import traceback

                    print(f"Error for device={dev}, num_envs={num_envs}: {e}")
                    traceback.print_exc()

        best_results[f"ablation_{ablation}"] = best_config
        print(f"Best for Ablation {ablation}: {best_config}")

    save_grid_search_results(args, "dqn", best_results, all_results)


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
    from runner_utilities import get_device_name
    parser.add_argument(
        "--device_name",
        type=str,
        default=get_device_name(),
        help="Device name for storing time_files",
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
    parser.add_argument(
        "--replace_existing",
        action="store_true",
        default=False,
        help="Re-run and overwrite existing trials in json files",
    )
    args = parser.parse_args()

    # Get obs dim from dummy env
    env = make_env_thunk(args.fully_obs, args.env_name)()
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
