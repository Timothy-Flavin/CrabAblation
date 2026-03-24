import argparse
import json
import os
import time

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym

from PG_Rainbow import PPOAgent
from runner_utilities import (
    obs_transformer,
    FastObsWrapper,
    make_env_thunk,
    bins_to_continuous,
)
from pg_runner import train_pg


def run_grid_search(args, fully_obs=False, total_steps=2000):
    print("\n=== Starting Policy Gradient Grid Search for Best Parameters ===")

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    num_envs_list = [1, 4, 8, 12, 16]
    best_results = {}
    all_results = {}
    args.fully_obs = fully_obs

    # Defaults
    cfg = {
        "clip_coef": 0.2,
        "ent_coef": 0.01,
        "Beta": 0.01,
        "distributional": True,
        "use_gae": True,
    }

    for ablation in range(6):
        print(f"\n--- Grid Search: Ablation {ablation} ---")
        args.ablation = ablation

        cfg_abl = cfg.copy()
        if args.ablation == 1:
            cfg_abl["clip_coef"] = 100.0  # Effectively disables KL/surrogate clip
        elif args.ablation == 2:
            cfg_abl["ent_coef"] = 0.0
        elif args.ablation == 3:
            cfg_abl["Beta"] = 0.0
        elif args.ablation == 4:
            cfg_abl["distributional"] = False
        elif args.ablation == 5:
            # Placeholder for replacing GAE with Monte Carlo
            cfg_abl["use_gae"] = False

        best_sps = 0.0
        best_config = None
        all_results[f"ablation_{ablation}"] = []

        for dev in devices:
            for num_envs in num_envs_list:
                args.device = dev
                args.num_envs = num_envs
                args.num_steps = 128

                # Assign explicitly here for the modified train_pg to know how long to run
                args.total_steps = total_steps

                device = torch.device(dev)

                env_fns = [
                    make_env_thunk(args.fully_obs, args.env_name)
                    for _ in range(args.num_envs)
                ]
                vec_env = gym.vector.SyncVectorEnv(env_fns)
                try:
                    agent = PPOAgent(
                        vec_env,
                        clip_coef=cfg_abl["clip_coef"],
                        ent_coef=cfg_abl["ent_coef"],
                        distributional=cfg_abl["distributional"],
                        Beta=cfg_abl["Beta"],
                        num_envs=args.num_envs,
                        num_steps=args.num_steps,
                        use_gae=cfg_abl["use_gae"],
                    ).to(device)

                    results = train_pg(vec_env, agent, cfg_abl, args, device)
                    train_time = results["train_time"]
                    # Total steps inside train_pg uses:
                    # num_iterations = (total_steps // args.num_envs) // args.num_steps
                    # So precise steps:
                    actual_steps_run = (
                        max(1, (total_steps // args.num_envs) // args.num_steps)
                        * args.num_steps
                        * args.num_envs
                    )
                    sps = actual_steps_run / train_time

                    current_config = {
                        "device": dev,
                        "num_envs": num_envs,
                        "steps_per_sec": sps,
                    }
                    all_results[f"ablation_{ablation}"].append(current_config)

                    if sps > best_sps:
                        best_sps = sps
                        best_config = current_config
                    print(
                        f"Env: {args.env_name} | Ablation: {ablation} | Device: {dev} | Num Envs: {num_envs} | SPS: {sps:.2f}"
                    )

                except Exception as e:
                    import traceback

                    print(f"Error for device={dev}, num_envs={num_envs}: {e}")
                    traceback.print_exc()
                finally:
                    vec_env.close()

        best_results[f"ablation_{ablation}"] = best_config
        print(f"Best for Ablation {ablation}: {best_config}")

    if hasattr(args, "device_name") and args.device_name is not None:
        file_prefix = args.device_name
    else:
        from runner_utilities import get_device_name
        file_prefix = get_device_name()

    os.makedirs(f"time_files/{file_prefix}", exist_ok=True)
    best_filename = f"time_files/{file_prefix}/{args.env_name}_ppo_best.json"
    all_filename = f"time_files/{file_prefix}/{args.env_name}_ppo_all.json"

    with open(best_filename, "w") as f:
        json.dump(best_results, f, indent=4)

    with open(all_filename, "w") as f:
        json.dump(all_results, f, indent=4)

    print(
        f"\nGrid search complete. Saved best configs to {best_filename} and all configs to {all_filename}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parallel Environment Benchmarking PPO"
    )
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
    # run as a single test if left without grid_search flag
    parser.add_argument("--run", type=int, default=1)
    args = parser.parse_args()

    print("=== Configuration ===")
    print(f"Environment: {args.env_name}")
    print(f"Ablation Level: {args.ablation}")
    print("=====================")

    if args.grid_search:
        # PPO requires batch sizes of num_envs * num_steps.
        run_grid_search(args, fully_obs=args.fully_obs, total_steps=5000)
    else:
        print("To run grid search add the --grid_search flag.")
