#!/usr/bin/env python3
"""Aggregate ablation performance across environments and algorithms.

Produces two figures:

1. ``aggregate_per_algo.png`` -- one subplot per algorithm (DQN, SAC, PPO).
   For every (algorithm, environment) the score is min-max normalized so that
   environment spans 0..1, and the step axis is normalized so training
   progress runs 0.0 -> 1.0. Curves are then averaged across environments to
   give a single per-ablation aggregate per algorithm.

2. ``aggregate_combined.png`` -- a single plot that folds all three
   algorithms together into one combined per-ablation curve.

Train scores are drawn as solid lines, eval scores as dotted lines.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from graph import ema, ABLATION_MAP


def _load_norm_progress_curve(
    env_dir: Path,
    run: int,
    ablation: int,
    kind: str,
    weight: float,
    grid: np.ndarray,
) -> Optional[np.ndarray]:
    """Load one run, smooth it, and interpolate onto a [0,1] progress grid.

    ``kind`` is "train" or "eval". The step axis is normalized by the run's
    total training steps so train and eval share the same progress scale.
    Returns ``None`` if files are missing.
    """
    scores_path = env_dir / f"{kind}_scores_{run}_{ablation}.npy"
    steps_path = env_dir / f"{kind}_steps_{run}_{ablation}.npy"
    train_steps_path = env_dir / f"train_steps_{run}_{ablation}.npy"
    if not scores_path.exists() or not steps_path.exists():
        return None
    if not train_steps_path.exists():
        return None

    scores = np.load(scores_path).astype(np.float64)
    steps = np.load(steps_path).astype(np.float64)
    total_steps = float(np.load(train_steps_path)[-1])
    if scores.size == 0 or steps.size == 0 or total_steps <= 0:
        return None

    n = min(scores.size, steps.size)
    scores = ema(scores[:n].astype(np.float32), weight).astype(np.float64)
    x = np.clip(steps[:n] / total_steps, 0.0, 1.0)
    # np.interp needs strictly increasing x; cumulative steps are monotonic.
    return np.interp(grid, x, scores)


def collect_normalized(
    results_root: Path,
    algos: List[str],
    envs: List[str],
    runs: List[int],
    ablations: List[int],
    train_weight: float,
    eval_weight: float,
    grid_size: int,
) -> Tuple[np.ndarray, Dict]:
    """Build per-(algo, env, ablation) stacks normalized per (algo, env).

    Returns the shared progress grid and a nested dict::

        data[algo][env][ablation] = {"train": (R, G), "eval": (R, G)}

    where each entry is a stack of runs already min-max normalized using the
    combined train+eval extremes of that (algo, env).
    """
    grid = np.linspace(0.0, 1.0, num=grid_size)
    data: Dict = {}

    for algo in algos:
        data[algo] = {}
        for env in envs:
            env_dir = results_root / algo / env
            if not env_dir.exists():
                continue

            raw: Dict[int, Dict[str, List[np.ndarray]]] = {}
            for ablation in ablations:
                train_curves, eval_curves = [], []
                for run in runs:
                    tc = _load_norm_progress_curve(
                        env_dir, run, ablation, "train", train_weight, grid
                    )
                    ec = _load_norm_progress_curve(
                        env_dir, run, ablation, "eval", eval_weight, grid
                    )
                    if tc is not None:
                        train_curves.append(tc)
                    if ec is not None:
                        eval_curves.append(ec)
                if train_curves or eval_curves:
                    raw[ablation] = {"train": train_curves, "eval": eval_curves}

            if not raw:
                continue

            # Min-max scale shared across all ablations/runs of this (algo, env),
            # using both train and eval values so the two are comparable.
            all_vals = []
            for ab in raw.values():
                all_vals.extend(ab["train"])
                all_vals.extend(ab["eval"])
            stacked_all = np.concatenate([a.ravel() for a in all_vals])
            vmin = float(stacked_all.min())
            vmax = float(stacked_all.max())
            denom = (vmax - vmin) if vmax > vmin else 1.0

            data[algo][env] = {}
            for ablation, ab in raw.items():
                entry = {}
                for kind in ("train", "eval"):
                    if ab[kind]:
                        stk = (np.vstack(ab[kind]) - vmin) / denom
                    else:
                        stk = np.empty((0, grid_size))
                    entry[kind] = stk
                data[algo][env][ablation] = entry

    return grid, data


def _aggregate_over_envs(
    data: Dict, algo: str, ablation: int, kind: str
) -> Optional[np.ndarray]:
    """Stack all (env, run) normalized curves for one algo+ablation+kind."""
    parts = []
    for env in data[algo]:
        entry = data[algo][env].get(ablation)
        if entry is not None and entry[kind].shape[0] > 0:
            parts.append(entry[kind])
    if not parts:
        return None
    return np.vstack(parts)


def _aggregate_over_all(
    data: Dict, ablation: int, kind: str
) -> Optional[np.ndarray]:
    """Stack all (algo, env, run) normalized curves for one ablation+kind."""
    parts = []
    for algo in data:
        s = _aggregate_over_envs(data, algo, ablation, kind)
        if s is not None:
            parts.append(s)
    if not parts:
        return None
    return np.vstack(parts)


def _plot_ablation(ax, grid, stack, color, label):
    """Plot the mean curve with a +/- standard-error-of-the-mean band."""
    if stack is None or stack.shape[0] == 0:
        return
    n = stack.shape[0]
    mean = stack.mean(axis=0)
    sem = stack.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    ax.plot(grid, mean, color=color, lw=2, label=label)
    ax.fill_between(grid, mean - sem, mean + sem, color=color, alpha=0.15)


def plot_per_algo(grid, data, ablations, out_path: Path):
    algos = list(data.keys())
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, len(algos), figsize=(6 * len(algos), 5),
                             sharey=True)
    if len(algos) == 1:
        axes = [axes]

    for ax, algo in zip(axes, algos):
        for i, ablation in enumerate(ablations):
            color = cmap(i % 10)
            lbl = ABLATION_MAP.get(ablation, f"Abl {ablation}")
            train = _aggregate_over_envs(data, algo, ablation, "train")
            _plot_ablation(ax, grid, train, color, lbl)
        ax.set_title(algo.upper())
        ax.set_xlabel("Training progress (normalized)")
        ax.grid(True, alpha=0.2)
        ax.set_xlim(0, 1)
    axes[0].set_ylabel("Normalized score (min-max per env)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               ncol=min(len(labels), 7), bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Per-algorithm ablation performance "
                 "(mean +/- SEM over envs x seeds)", y=1.12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  -> Saved {out_path}")


def plot_combined(grid, data, ablations, out_path: Path):
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, ablation in enumerate(ablations):
        color = cmap(i % 10)
        lbl = ABLATION_MAP.get(ablation, f"Abl {ablation}")
        train = _aggregate_over_all(data, ablation, "train")
        _plot_ablation(ax, grid, train, color, lbl)
    ax.set_title("Combined ablation performance across DQN + SAC + PPO\n"
                 "(mean +/- SEM over algos x envs x seeds)")
    ax.set_xlabel("Training progress (normalized)")
    ax.set_ylabel("Normalized score (min-max per env)")
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, 1)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  -> Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", type=str, default="all_results")
    parser.add_argument("--algos", nargs="+", default=["dqn", "sac", "ppo"])
    parser.add_argument("--envs", nargs="+",
                        default=["cartpole", "minigrid", "mujoco"])
    parser.add_argument("--runs", type=int, nargs="+",
                        default=list(range(6, 16)))
    parser.add_argument("--ablations", type=int, nargs="+",
                        default=list(range(7)))
    parser.add_argument("--train_weight", type=float, default=0.95)
    parser.add_argument("--eval_weight", type=float, default=0.5)
    parser.add_argument("--grid_size", type=int, default=500)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    print(f"Aggregating algos={args.algos} envs={args.envs} runs={args.runs}")
    grid, data = collect_normalized(
        results_root, args.algos, args.envs, args.runs, args.ablations,
        args.train_weight, args.eval_weight, args.grid_size,
    )

    plot_per_algo(grid, data, args.ablations,
                  results_root / "aggregate_per_algo.png")
    plot_combined(grid, data, args.ablations,
                  results_root / "aggregate_combined.png")


if __name__ == "__main__":
    main()
