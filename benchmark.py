from __future__ import annotations

import argparse
import traceback 
import warnings
import gc
import gymnasium as gym
import numpy as np
import torch

# Safely import psutil for RAM prediction
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not installed. RAM prediction disabled. Run 'pip install psutil' to enable.")

from environment_utils import get_env_benchmark_spec
from runner import (
    build_agent,
    create_vec_env,
    rollout_offline_rl,
    rollout_online_rl,
    get_parser,      # Inherited natively from runner
    process_args     # Inherited natively from runner
)
from runner_utils import (
    benchmark_action_sampling_generic,
    benchmark_updates_generic,
    get_device_name,
    load_grid_search_results,
    resolve_torch_device,
    save_grid_search_results,
)

warnings.filterwarnings("ignore")

def get_args():
    # 1. Inherit all standard hyperparameters natively from runner
    parser = get_parser()
    
    # 2. Add purely benchmark-specific flags
    parser.add_argument("--grid_search", action="store_true", help="Run grid search")
    parser.add_argument(
        "--search_devices",
        type=str,
        nargs="+",
        default=["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"],
        help="List of PyTorch devices to test",
    )
    parser.add_argument("--replace_existing", action="store_true", default=False)
    parser.add_argument("--benchmark_steps", type=int, default=1000000)

    # Force burn-in and learning starts to 0 for benchmarks so they immediately test update speed
    # We also prevent loading best.json parameters globally so we don't accidentally set num_envs=128
    parser.set_defaults(
        rnd_burn_in=0, 
        learning_starts=0, 
        max_wall_time=15.0,
        skip_best_params=True
    )

    # Use parse_known_args to prevent crashes if your wrapper script passes outdated flags
    args, _ = parser.parse_known_args()
    
    args.is_benchmark = True
    args = process_args(args)

    return args


def run_grid_search(args, total_steps=2000):
    devices = args.search_devices
    num_envs_list = [1, 4, 8, 12, 16, 64, 128]

    if args.replace_existing:
        best_results = {}
        all_results = {}
    else:
        best_results, all_results = load_grid_search_results(args, args.algo)

    for ablation in range(6):
        print(f"\n--- Grid Search: Ablation {ablation} ---")
        args.ablation = ablation

        best_sps = 0.0
        best_config = None
        all_results.setdefault(f"ablation_{ablation}", [])
        existing_trials = {
            (entry.get("device"), entry.get("num_envs"))
            for entry in all_results[f"ablation_{ablation}"]
            if isinstance(entry, dict)
        }

        for dev in devices:
            ram_measurements = [] # Track tuples of (num_envs, rss_memory_bytes)

            for num_envs in num_envs_list:
                args.device = dev
                args.num_envs = num_envs

                if (not args.replace_existing) and ((dev, num_envs) in existing_trials):
                    print(f"Skipping existing trial: ablation={ablation}, device={dev}, num_envs={num_envs}")
                    continue

                # --- RAM Prediction Logic ---
                if HAS_PSUTIL and len(ram_measurements) >= 2:
                    x = np.array([m[0] for m in ram_measurements])
                    y = np.array([m[1] for m in ram_measurements])
                    slope, intercept = np.polyfit(x, y, 1)
                    predicted_ram = (slope * num_envs) + intercept
                    
                    sys_mem = psutil.virtual_memory()
                    safe_limit = sys_mem.total * 0.85 # Cut off if approaching 85% of total system memory
                    
                    if predicted_ram > safe_limit:
                        print(f"⚠️ Skipping {num_envs}+ envs: Predicted RAM usage ({predicted_ram / 1024**3:.2f} GB) exceeds safe threshold.")
                        break # Exits the num_envs loop cleanly, moving to next device/ablation

                try:
                    vec_env = create_vec_env(args, num_envs=num_envs)
                    
                    # Measure process memory right after environment instantiation
                    if HAS_PSUTIL:
                        current_ram = psutil.Process().memory_info().rss
                        ram_measurements.append((num_envs, current_ram))

                    device = resolve_torch_device(dev)
                    agent, _ = build_agent(args, vec_env, device)
                    actual_num_envs = int(getattr(vec_env, "num_envs", num_envs))

                    if args.algo == "ppo":
                        results = rollout_online_rl(
                            vec_env, agent, args, device,
                            max_wall_time_seconds=args.max_wall_time,
                            total_steps_override=total_steps,
                        )
                    else:
                        results = rollout_offline_rl(
                            vec_env, agent, args, device,
                            max_wall_time_seconds=args.max_wall_time,
                            total_steps_override=total_steps,
                        )

                    current_config = {
                        "device": dev,
                        "num_envs": actual_num_envs,
                        "steps_per_sec": float(results.get("steps_per_sec", 0.0)),
                        "updates_per_sec": float(results.get("updates_per_sec", 0.0)),
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
                        e for e in all_results[f"ablation_{ablation}"]
                        if isinstance(e, dict) and "steps_per_sec" in e
                    ]
                    if ablation_entries:
                        best_results[f"ablation_{ablation}"] = max(
                            ablation_entries, key=lambda e: e["steps_per_sec"],
                        )

                    save_grid_search_results(args, args.algo, best_results, all_results)

                    if current_config["steps_per_sec"] > best_sps:
                        best_sps = current_config["steps_per_sec"]
                        best_config = current_config

                except Exception as e:
                    print(f"Failed on ablation={ablation}, envs={num_envs}, dev={dev}. Skipping.")
                    traceback.print_exc()
                    # Do NOT call exit() here so the benchmark loop continues
                finally:
                    if 'vec_env' in locals():
                        vec_env.close()
                        del vec_env
                    gc.collect() # Force GC to clear process memory for accurate subsequent RAM readings

        if best_config is None and all_results.get(f"ablation_{ablation}"):
            best_config = max(
                (e for e in all_results[f"ablation_{ablation}"] if isinstance(e, dict) and "steps_per_sec" in e),
                key=lambda e: e["steps_per_sec"],
                default=None
            )
        best_results[f"ablation_{ablation}"] = best_config

    save_grid_search_results(args, args.algo, best_results, all_results)
    import json
    print(json.dumps({"best_results": best_results, "all_results": all_results}, indent=4))


def benchmark_updates(
    agent, args, obs_dim, action_dim, vec_env=None, device="cpu", batch_sizes=None, iters=50
):
    import time
    if args.algo in ["sac", "dqn"]:
        agent.to(device)
        print(f"\n--- Benchmarking Offline Env Rollout for {args.algo.upper()} ---")
        t_start = time.time()
        from runner import rollout_offline_rl
        print("Using rollout_offline_rl with true env dynamics.")
        rollout_offline_rl(
            vec_env, 
            agent, 
            args, 
            device, 
            max_wall_time_seconds=None, 
            total_steps_override=getattr(args, 'benchmark_steps', iters * 5)
        )
        t_end = time.time()
        print(f"Offline RL Update Benchmark for {args.algo.upper()} complete. Real Training Time: {(t_end - t_start):.3f}s.")
        return

def benchmark_action_sampling(
    agent, args, obs_dim, device="cpu", batch_sizes=None, iters=200
):
    if batch_sizes is None:
        batch_sizes = [1, 4, 16, 64, 256]

    if args.algo == "dqn":
        def sample_fn(obs):
            return agent.sample_action(obs, eps=0.1, step=0)

    elif args.algo == "ppo":
        def sample_fn(obs):
            return agent.sample_action(obs)[0]

    else:
        def sample_fn(obs):
            return agent.sample_action(obs, deterministic=False)

    benchmark_action_sampling_generic(
        agent,
        obs_dim=obs_dim,
        device=device,
        batch_sizes=batch_sizes,
        iters=iters,
        sample_fn=sample_fn,
    )


def _action_dim_for_space(space):
    if isinstance(space, gym.spaces.Box):
        return int(np.prod(space.shape))
    if isinstance(space, gym.spaces.Discrete):
        return 1
    if isinstance(space, gym.spaces.MultiDiscrete):
        return int(len(space.nvec))
    return 1


def main():
    args = get_args()

    # FORCE num_envs to 1 for the probe environment to ensure we never OOM just grabbing the shape
    probe_env = create_vec_env(args, num_envs=1)
    obs_shape = probe_env.single_observation_space.shape
    if obs_shape is None:
        raise ValueError("Environment observation space shape is undefined")
    obs_dim = int(np.prod(obs_shape))
    action_dim = _action_dim_for_space(probe_env.single_action_space)
    probe_env.close()

    print("=== Configuration ===")
    print(f"Algorithm: {args.algo}")
    print(f"Environment: {args.env_name}")
    print(f"Ablation Level: {args.ablation}")
    print(f"Obs Dim: {obs_dim}")
    print(f"Action Dim: {action_dim}")
    print("=====================")

    if args.grid_search:
        run_grid_search(args, total_steps=args.benchmark_steps)
        return

    # Create the standard vec_env only if not running the grid search block
    vec_env = create_vec_env(args, num_envs=max(1, args.num_envs))
    args._vec_env = vec_env
    try:
        device = resolve_torch_device(args.device)
        agent, _ = build_agent(args, vec_env, device)

        benchmark_updates(
            agent, args, obs_dim, action_dim, vec_env, device="cpu", batch_sizes=[64, 256]
        )
        if torch.cuda.is_available():
            benchmark_updates(
                agent, args, obs_dim, action_dim, vec_env, device="cuda", batch_sizes=[64, 256]
            )

        benchmark_action_sampling(
            agent, args, obs_dim, device="cpu", batch_sizes=[1, 4, 16, 64, 256]
        )
        if torch.cuda.is_available():
            benchmark_action_sampling(
                agent,
                args,
                obs_dim,
                device="cuda",
                batch_sizes=[1, 4, 16, 64, 256],
            )

        for dev in args.search_devices:
            args.device = dev
            for n_envs in [1, 4, 8]:
                args.num_envs = n_envs
                rollout_env = create_vec_env(args, num_envs=n_envs)
                try:
                    rollout_device = resolve_torch_device(dev)
                    rollout_agent, _ = build_agent(args, rollout_env, rollout_device)
                    if args.algo == "ppo":
                        results = rollout_online_rl(
                            rollout_env,
                            rollout_agent,
                            args,
                            rollout_device,
                            max_wall_time_seconds=args.max_wall_time,
                            total_steps_override=args.benchmark_steps,
                        )
                    else:
                        results = rollout_offline_rl(
                            rollout_env,
                            rollout_agent,
                            args,
                            rollout_device,
                            max_wall_time_seconds=args.max_wall_time,
                            total_steps_override=args.benchmark_steps,
                        )
                    print(
                        f"Env Rollout | algo={args.algo} | device={dev} | num_envs={n_envs} | "
                        f"SPS={results['steps_per_sec']:.2f} | UPS={results['updates_per_sec']:.2f}"
                    )
                finally:
                    rollout_env.close()
    finally:
        if hasattr(args, "_vec_env"):
            delattr(args, "_vec_env")
        vec_env.close()


if __name__ == "__main__":
    main()