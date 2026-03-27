#!/usr/bin/env python3
"""Aggregate and plot reward statistics across ablation runs.

Updated for file structure: results/algorithm/env/...
Saves to: results/{algorithm}_{env}_{xaxis}.png
"""

import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import matplotlib.pyplot as plt

# Mapping for filenames and legends
ABLATION_MAP = {
    0: "None",
    1: "KL_Penalty",
    2: "Magnet_Reg",
    3: "Optimism",
    4: "Dist-RL",
    5: "Delayed",
}


def ema(values: np.ndarray, weight: float) -> np.ndarray:
    if values.size == 0:
        return values
    out = np.empty_like(values, dtype=np.float64)
    out[0] = values[0]
    w = weight
    one_minus = 1.0 - w
    for i in range(1, len(values)):
        out[i] = w * out[i - 1] + one_minus * values[i]
    return out


def load_run_arrays(
    env_dir: Path, run: int, ablation: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Loads arrays from the specific algorithm/env directory."""
    train_path = env_dir / f"train_scores_{run}_{ablation}.npy"
    eval_path = env_dir / f"eval_scores_{run}_{ablation}.npy"
    if not train_path.exists() or not eval_path.exists():
        raise FileNotFoundError(f"Missing files in {env_dir}")
    return np.load(train_path), np.load(eval_path)


def aggregate_runs(
    runs_data: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not runs_data:
        return np.array([]), np.array([]), np.array([])
    min_len = min(arr.size for arr in runs_data)
    stacked = np.vstack([arr[:min_len] for arr in runs_data])
    return stacked.mean(axis=0), stacked.min(axis=0), stacked.max(axis=0)


def collect_for_algo(
    algo_env_dir: Path,
    runs: List[int],
    weight: float,
    xaxis: str,
    max_steps: Optional[int],
) -> Dict[int, Dict[str, Any]]:
    """Collects stats for a specific algorithm + environment pair."""
    ablation_stats: Dict[int, Dict[str, Any]] = {}

    for ablation in range(6):
        train_runs, eval_runs, train_times = [], [], []
        all_ok = True

        for run in runs:
            try:
                train_arr, eval_arr = load_run_arrays(algo_env_dir, run, ablation)
                train_runs.append(ema(train_arr.astype(np.float64), weight))
                eval_runs.append(ema(eval_arr.astype(np.float64), weight))

                if xaxis == "time":
                    time_path = algo_env_dir / f"train_time_{run}_{ablation}.npy"
                    train_times.append(float(np.load(time_path)))
            except (FileNotFoundError, ValueError):
                all_ok = False
                break

        if not all_ok or not train_runs:
            continue

        t_mean, t_min, t_max = aggregate_runs(train_runs)
        e_mean, e_min, e_max = aggregate_runs(eval_runs)

        # X-Axis Logic
        if xaxis == "episodes":
            x_train, x_eval, x_label = (
                np.arange(t_mean.size),
                np.arange(e_mean.size),
                "Episode",
            )
        elif xaxis == "steps":
            if not max_steps:
                raise ValueError("--xaxis steps requires --max_steps")
            x_train = np.linspace(0, max_steps, num=t_mean.size)
            x_eval = np.linspace(0, max_steps, num=e_mean.size)
            x_label = "Steps"
        else:  # time
            max_t = max(train_times) if train_times else 1.0
            x_train = np.linspace(0, max_t, num=t_mean.size)
            x_eval = np.linspace(0, max_t, num=e_mean.size)
            x_label = "Time (s)"

        ablation_stats[ablation] = {
            "train_mean": t_mean,
            "train_min": t_min,
            "train_max": t_max,
            "eval_mean": e_mean,
            "eval_min": e_min,
            "eval_max": e_max,
            "x_train": x_train,
            "x_eval": x_eval,
            "x_label": x_label,
        }
    return ablation_stats


def plot_algo_stats(
    stats_dict: Dict[int, Dict[str, Any]], algo: str, env: str, output_path: Path
):
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")

    for i, (ablation, stats) in enumerate(sorted(stats_dict.items())):
        color = cmap(i % 10)
        lbl = ABLATION_MAP.get(ablation, f"Abl {ablation}")

        # Plot Train
        plt.plot(
            stats["x_train"],
            stats["train_mean"],
            color=color,
            label=f"Train: {lbl}",
            lw=2,
        )
        plt.fill_between(
            stats["x_train"],
            stats["train_min"],
            stats["train_max"],
            color=color,
            alpha=0.1,
        )

        # Plot Eval
        if stats["eval_mean"].size > 0:
            plt.plot(
                stats["x_eval"],
                stats["eval_mean"],
                color=color,
                linestyle="--",
                alpha=0.7,
            )

    plt.title(f"Algorithm: {algo} | Env: {env}")
    plt.xlabel(next(iter(stats_dict.values()))["x_label"])
    plt.ylabel("Reward")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--runs", type=int, nargs="*", default=[1, 2, 3])
    parser.add_argument("--weight", type=float, default=0.95)
    parser.add_argument(
        "--xaxis", type=str, default="episodes", choices=["episodes", "steps", "time"]
    )
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    results_root = Path("results")
    if not results_root.exists():
        print("Error: 'results/' directory not found.")
        return

    # Iterate through each algorithm folder
    found_any = False
    for algo_path in results_root.iterdir():
        if not algo_path.is_dir():
            continue

        algo_name = algo_path.name
        env_dir = algo_path / args.env

        if not env_dir.exists():
            continue

        print(f"Processing algorithm: {algo_name}...")
        stats = collect_for_algo(
            env_dir, args.runs, args.weight, args.xaxis, args.max_steps
        )

        if stats:
            found_any = True
            # Save at top level: results/ALGO_ENV_XAXIS.png
            out_file = results_root / f"{algo_name}_{args.env}_{args.xaxis}.png"
            plot_algo_stats(stats, algo_name, args.env, out_file)
            print(f"  -> Saved to {out_file}")

    if not found_any:
        print(f"No data found for environment '{args.env}' in any algorithm folder.")


if __name__ == "__main__":
    main()
