#!/usr/bin/env python3
"""Aggregate and plot reward statistics across ablation runs.

Usage:
  python graph.py --env cartpole --runs 1 2 3 --output aggregated_rewards.png

For the specified environment name (folder under results/), this script:
  - Loads train_scores_{run}_{ablation}.npy and eval_scores_{run}_{ablation}.npy
        for runs provided (default 1 2 3) and ablations 0..5.
  - Applies an exponential moving average (EMA) with weight 0.95 to each run
        (smooth_t = weight * smooth_{t-1} + (1-weight) * x_t).
  - Truncates all runs for an ablation to the minimum common length so episode
        indices align.
  - Computes per-episode mean, min, max of the smoothed rewards across runs.
  - Plots mean (solid for train, dashed for eval) and a shaded min-max band
        for each ablation.
  - Saves the figure under results/<env>/<output> (default aggregated_rewards.png).

If some files are missing for an ablation, that ablation is skipped with a warning.
"""

import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt


def ema(values: np.ndarray, weight: float) -> np.ndarray:
    """Compute exponential moving average of a 1D array with given weight.

    weight: previous contribution weight (e.g. 0.95)."""
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
    """Load train and eval arrays for a given run and ablation. Returns (train, eval).
    Raises FileNotFoundError if either is missing."""
    train_path = env_dir / f"train_scores_{run}_{ablation}.npy"
    eval_path = env_dir / f"eval_scores_{run}_{ablation}.npy"
    if not train_path.exists() or not eval_path.exists():
        missing = []
        if not train_path.exists():
            missing.append(str(train_path))
        if not eval_path.exists():
            missing.append(str(eval_path))
        raise FileNotFoundError("Missing files: " + ", ".join(missing))
    return np.load(train_path), np.load(eval_path)


def aggregate_runs(
    runs_data: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Given list of 1D arrays (already smoothed), truncate to min length and
    return (mean, min, max) arrays across runs per time index."""
    if len(runs_data) == 0:
        return np.array([]), np.array([]), np.array([])
    min_len = min(arr.size for arr in runs_data)
    if min_len == 0:
        return np.array([]), np.array([]), np.array([])
    stacked = np.vstack([arr[:min_len] for arr in runs_data])
    mean = stacked.mean(axis=0)
    minv = stacked.min(axis=0)
    maxv = stacked.max(axis=0)
    return mean, minv, maxv


def collect(
    env: str, runs: List[int], weight: float
) -> Dict[int, Dict[str, np.ndarray]]:
    """Collect aggregated statistics for each ablation.
    Returns dict: ablation -> { 'train_mean': ..., 'train_min': ..., 'train_max': ..., 'eval_mean': ..., ... }
    Skips ablations missing any run data."""
    env_dir = Path("results") / env
    if not env_dir.exists():
        raise FileNotFoundError(f"Environment directory not found: {env_dir}")

    ablation_stats: Dict[int, Dict[str, np.ndarray]] = {}
    for ablation in range(6):  # 0..5 inclusive
        train_runs = []
        eval_runs = []
        all_ok = True
        for run in runs:
            try:
                train_arr, eval_arr = load_run_arrays(env_dir, run, ablation)
            except FileNotFoundError as e:
                print(f"[warn] Skipping ablation {ablation}: {e}")
                all_ok = False
                break
            train_runs.append(ema(train_arr.astype(np.float64), weight))
            eval_runs.append(ema(eval_arr.astype(np.float64), weight))
        if not all_ok or len(train_runs) == 0:
            continue
        t_mean, t_min, t_max = aggregate_runs(train_runs)
        e_mean, e_min, e_max = aggregate_runs(eval_runs)
        ablation_stats[ablation] = {
            "train_mean": t_mean,
            "train_min": t_min,
            "train_max": t_max,
            "eval_mean": e_mean,
            "eval_min": e_min,
            "eval_max": e_max,
        }
    return ablation_stats


def pick_colors(n: int) -> List[str]:
    """Return a list of RGBA hex strings (matplotlib handles both)."""
    cmap = plt.get_cmap("tab10")
    colors = []
    for i in range(n):
        rgba = cmap(i % 10)
        # Convert to hex for explicit string typing
        r, g, b, a = rgba
        colors.append("#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255)))
    return colors


def plot_stats(
    ablation_stats: Dict[int, Dict[str, np.ndarray]], env: str, output: Path
):
    plt.figure(figsize=(12, 7))
    colors = pick_colors(6)
    for ablation, stats in sorted(ablation_stats.items()):
        color = colors[ablation % len(colors)]
        t_mean = stats["train_mean"]
        t_min = stats["train_min"]
        t_max = stats["train_max"]
        e_mean = stats["eval_mean"]
        e_min = stats["eval_min"]
        e_max = stats["eval_max"]
        if t_mean.size == 0:
            continue
        x_train = np.arange(t_mean.size)
        plt.plot(
            x_train, t_mean, color=color, linewidth=2, label=f"Train Abl {ablation}"
        )
        plt.fill_between(x_train, t_min, t_max, color=color, alpha=0.15)
        if e_mean.size > 0:
            x_eval = np.arange(e_mean.size)
            # Match eval length to train for visual alignment if different
            if e_mean.size != t_mean.size:
                min_len = min(e_mean.size, t_mean.size)
                x_eval = x_eval[:min_len]
                e_mean = e_mean[:min_len]
                e_min = e_min[:min_len]
                e_max = e_max[:min_len]
            plt.plot(
                x_eval,
                e_mean,
                color=color,
                linestyle="--",
                linewidth=1.5,
                label=f"Eval Abl {ablation}",
            )
            plt.fill_between(x_eval, e_min, e_max, color=color, alpha=0.08)

    plt.title(f"{env} Reward Aggregates Across Ablations (EMA weight=0.95)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output)
    print(f"Saved figure to {output}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate and plot reward stats across ablations"
    )
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        help="Environment folder name under results/, e.g. cartpole",
    )
    parser.add_argument(
        "--runs",
        type=int,
        nargs="*",
        default=[1, 2, 3],
        help="List of run IDs to aggregate",
    )
    parser.add_argument(
        "--weight", type=float, default=0.95, help="EMA weight (previous value weight)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="aggregated_rewards.png",
        help="Output figure filename",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    stats = collect(args.env, args.runs, args.weight)
    if not stats:
        print("No ablation stats collected. Check file availability.")
        return
    env_dir = Path("results") / args.env
    out_path = env_dir / args.output
    plot_stats(stats, args.env, out_path)


if __name__ == "__main__":
    main()
