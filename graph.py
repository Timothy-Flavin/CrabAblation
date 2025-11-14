#!/usr/bin/env python3
"""Aggregate and plot reward statistics across ablation runs.

Usage examples:
    python graph.py --env cartpole --runs 1 2 3 --output aggregated_rewards.png
    python graph.py --env cartpole --xaxis steps --max_steps 50000
    python graph.py --env cartpole --xaxis time

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
    - Supports x-axis modes:
             episodes (default): raw episode index
             steps: linear scaling 0..max_steps (requires --max_steps)
             time: linear scaling 0..max wall clock training time across runs (requires train_time files)
    - Saves the figure under results/<env>/<output> (default aggregated_rewards.png).

If some files are missing for an ablation, that ablation is skipped with a warning.
"""

import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

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
    env: str, runs: List[int], weight: float, xaxis: str, max_steps: Optional[int]
) -> Dict[int, Dict[str, Any]]:
    """Collect aggregated statistics for each ablation.
    Returns dict: ablation -> stats dict including reward aggregates and x-axis arrays.
    Skips ablations missing any run data. Raises ValueError if xaxis requirements unmet.
    """
    env_dir = Path("results") / env
    if not env_dir.exists():
        raise FileNotFoundError(f"Environment directory not found: {env_dir}")

    ablation_stats: Dict[int, Dict[str, Any]] = {}
    for ablation in range(6):  # 0..5 inclusive
        train_runs: List[np.ndarray] = []
        eval_runs: List[np.ndarray] = []
        train_times: List[float] = []
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
            if xaxis == "time":
                time_path = env_dir / f"train_time_{run}_{ablation}.npy"
                if not time_path.exists():
                    raise ValueError(f"Requested time x-axis but missing {time_path}")
                train_times.append(float(np.load(time_path)))
        if not all_ok or len(train_runs) == 0:
            continue
        t_mean, t_min, t_max = aggregate_runs(train_runs)
        e_mean, e_min, e_max = aggregate_runs(eval_runs)
        # Build x-axis arrays
        if xaxis == "episodes":
            x_train = np.arange(t_mean.size)
            x_eval = np.arange(e_mean.size)
            x_label = "Episode"
        elif xaxis == "steps":
            if not max_steps:
                raise ValueError("--xaxis steps requires --max_steps")
            x_train = np.linspace(0, max_steps, num=t_mean.size, endpoint=True)
            x_eval = np.linspace(0, max_steps, num=e_mean.size, endpoint=True)
            x_label = "Steps"
        elif xaxis == "time":
            max_time = max(train_times) if train_times else 0.0
            x_train = np.linspace(0, max_time, num=t_mean.size, endpoint=True)
            x_eval = np.linspace(0, max_time, num=e_mean.size, endpoint=True)
            x_label = "Time (s)"
        else:
            raise ValueError(f"Unknown xaxis mode: {xaxis}")
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


def plot_stats(ablation_stats: Dict[int, Dict[str, Any]], env: str, output: Path):
    # (0) No Ablation
    # (1) Mirror Descent / Bregman Proximal Optimization
    # (2) Magnet Policy Regularization
    # (3) Optimism in the face of Uncertainty
    # (4) Dual/Dueling/Distributional Value Estimates
    # (5) Delayed / Two-Timescale Optimization
    ablation_map = {
        0: "None",
        1: "KL Penalty",
        2: "Magnet Reg",
        3: "Optimism",
        4: "Dist-RL",
        5: "Delayed",
    }
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
        x_train = stats["x_train"]
        x_eval = stats["x_eval"]
        if t_mean.size == 0:
            continue
        plt.plot(
            x_train,
            t_mean,
            color=color,
            linewidth=2,
            label=f"Train Abl {ablation_map[ablation]}",
        )
        plt.fill_between(x_train, t_min, t_max, color=color, alpha=0.15)
        if e_mean.size > 0:
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
                label=f"Eval Abl {ablation_map[ablation]}",
            )
            plt.fill_between(x_eval, e_min, e_max, color=color, alpha=0.08)
    plt.title(f"{env} Reward Aggregates Across Ablations (EMA weight=0.95)")
    if ablation_stats:
        any_key = next(iter(ablation_stats))
        plt.xlabel(ablation_stats[any_key]["x_label"])
    else:
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
    parser.add_argument(
        "--xaxis",
        type=str,
        default="episodes",
        choices=["episodes", "steps", "time"],
        help="X-axis mode: episodes (default), steps (requires --max_steps), time (requires train_time files)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum steps for scaling when --xaxis steps is used",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    stats = collect(args.env, args.runs, args.weight, args.xaxis, args.max_steps)
    if not stats:
        print("No ablation stats collected. Check file availability.")
        return
    env_dir = Path("results") / args.env
    # Derive filename with x-axis suffix for disambiguation
    axis_suffix_map = {"episodes": "episode", "steps": "steps", "time": "walltime"}
    suffix = axis_suffix_map.get(args.xaxis, args.xaxis)
    base, ext = os.path.splitext(args.output)
    if not ext:
        ext = ".png"
    out_path = env_dir / f"{base}_{suffix}{ext}"
    plot_stats(stats, args.env, out_path)


if __name__ == "__main__":
    main()
