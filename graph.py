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
import yaml

# Mapping for filenames and legends
ABLATION_MAP = {
    0: "None",
    1: "KL_Penalty",
    2: "Magnet_Reg",
    3: "Optimism",
    4: "Dist-RL",
    5: "Delayed",
    6: "Original",
}


def ema(values: np.ndarray, weight: float) -> np.ndarray:
    if values.size == 0:
        return values
    out = np.empty_like(values, dtype=np.float32)
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


def load_run_step_axes(
    env_dir: Path,
    run: int,
    ablation: int,
    train_arr: np.ndarray,
    eval_size: int,
    eval_every_episodes: int = 25,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (train_steps, eval_steps) for one run.

    Prefers saved per-episode step counts; falls back to cumsum of the
    reward array (correct when reward == episode length, e.g. CartPole)
    and to the eval-every-N-episodes schedule for eval_steps.
    """
    train_steps_path = env_dir / f"train_steps_{run}_{ablation}.npy"
    eval_steps_path = env_dir / f"eval_steps_{run}_{ablation}.npy"

    if train_steps_path.exists():
        train_steps = np.load(train_steps_path).astype(np.float64)
    else:
        train_steps = np.cumsum(train_arr.astype(np.float64))

    if eval_steps_path.exists():
        eval_steps = np.load(eval_steps_path).astype(np.float64)
        if eval_steps.size < eval_size:
            eval_size = eval_steps.size
        eval_steps = eval_steps[:eval_size]
    elif eval_size > 0 and train_steps.size > 0:
        idxs = np.clip(
            np.arange(1, eval_size + 1) * eval_every_episodes - 1,
            0,
            train_steps.size - 1,
        )
        eval_steps = train_steps[idxs]
    else:
        eval_steps = np.array([], dtype=np.float64)

    return train_steps, eval_steps


def _iqm_stack(stacked: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_runs = stacked.shape[0]
    if num_runs <= 2:
        iqm = stacked.mean(axis=0)
    else:
        sorted_stacked = np.sort(stacked, axis=0)
        lower_idx = int(np.floor(num_runs * 0.25))
        upper_idx = int(np.ceil(num_runs * 0.75))
        trimmed = sorted_stacked[lower_idx:upper_idx, :]
        iqm = trimmed.mean(axis=0)
    return iqm, stacked.min(axis=0), stacked.max(axis=0)


def aggregate_runs(
    runs_data: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not runs_data:
        return np.array([]), np.array([]), np.array([])
    min_len = min(arr.size for arr in runs_data)
    stacked = np.vstack([arr[:min_len] for arr in runs_data])
    return _iqm_stack(stacked)


def aggregate_runs_on_grid(
    runs_data: List[np.ndarray],
    x_arrays: List[np.ndarray],
    grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate each run onto a shared x grid before stacking."""
    if not runs_data:
        return np.array([]), np.array([]), np.array([])
    interped = []
    for y, x in zip(runs_data, x_arrays):
        if y.size == 0 or x.size == 0:
            continue
        n = min(y.size, x.size)
        interped.append(np.interp(grid, x[:n], y[:n]))
    if not interped:
        return np.array([]), np.array([]), np.array([])
    stacked = np.vstack(interped)
    return _iqm_stack(stacked)


def collect_for_algo(
    algo_env_dir: Path,
    runs: List[int],
    weight: float,
    xaxis: str,
    max_steps: Optional[int],
    grid_size: int = 1000,
    eval_weight: Optional[float] = None,
) -> Dict[int, Dict[str, Any]]:
    """Collects stats for a specific algorithm + environment pair."""
    ablation_stats: Dict[int, Dict[str, Any]] = {}
    eval_w = weight if eval_weight is None else eval_weight

    for ablation in range(7):
        train_runs, eval_runs, train_times = [], [], []
        train_step_axes, eval_step_axes = [], []
        missing_runs: List[int] = []

        for run in runs:
            try:
                train_arr, eval_arr = load_run_arrays(algo_env_dir, run, ablation)
                train_smoothed = ema(train_arr.astype(np.float32), weight)
                eval_smoothed = ema(eval_arr.astype(np.float32), eval_w)
                train_runs.append(train_smoothed)
                eval_runs.append(eval_smoothed)

                if xaxis == "steps":
                    train_steps, eval_steps = load_run_step_axes(
                        algo_env_dir, run, ablation, train_arr, eval_smoothed.size
                    )
                    train_step_axes.append(train_steps)
                    eval_step_axes.append(eval_steps)
                elif xaxis == "time":
                    time_path = algo_env_dir / f"train_time_{run}_{ablation}.npy"
                    train_times.append(float(np.load(time_path)))
            except (FileNotFoundError, ValueError):
                missing_runs.append(run)
                continue

        if not train_runs:
            if missing_runs:
                print(
                    f"  [skip] ablation {ablation}: no runs available "
                    f"(missing runs: {missing_runs})"
                )
            continue
        if missing_runs:
            print(
                f"  [warn] ablation {ablation}: using {len(train_runs)} runs, "
                f"missing {missing_runs}"
            )

        # X-Axis Logic
        if xaxis == "episodes":
            t_mean, t_min, t_max = aggregate_runs(train_runs)
            e_mean, e_min, e_max = aggregate_runs(eval_runs)
            x_train = np.arange(t_mean.size)
            x_eval = np.linspace(0, t_mean.size, num=e_mean.size)
            x_label = "Episode"
        elif xaxis == "steps":
            if not max_steps:
                raise ValueError("--xaxis steps requires --max_steps")
            # Cap the grid at the smallest step count any run actually reached,
            # so we never extrapolate. Bounded above by max_steps.
            max_reached = min(
                float(ts[-1]) if ts.size > 0 else 0.0 for ts in train_step_axes
            )
            grid_end = min(float(max_steps), max_reached) if max_reached > 0 else float(max_steps)
            x_train = np.linspace(0.0, grid_end, num=grid_size)
            t_mean, t_min, t_max = aggregate_runs_on_grid(
                train_runs, train_step_axes, x_train
            )

            eval_pairs = [
                (e, x) for e, x in zip(eval_runs, eval_step_axes) if e.size > 0 and x.size > 0
            ]
            if eval_pairs:
                eval_runs_f = [e for e, _ in eval_pairs]
                eval_axes_f = [x for _, x in eval_pairs]
                eval_max_reached = min(float(x[-1]) for x in eval_axes_f)
                eval_grid_end = min(grid_end, eval_max_reached) if eval_max_reached > 0 else grid_end
                x_eval = np.linspace(0.0, eval_grid_end, num=grid_size)
                e_mean, e_min, e_max = aggregate_runs_on_grid(
                    eval_runs_f, eval_axes_f, x_eval
                )
            else:
                x_eval = np.array([])
                e_mean = e_min = e_max = np.array([])
            x_label = "Steps"
        else:  # time
            t_mean, t_min, t_max = aggregate_runs(train_runs)
            e_mean, e_min, e_max = aggregate_runs(eval_runs)
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
    parser.add_argument("--eval_weight", type=float, default=0.5,
                        help="EMA weight for eval curves (default 0.5; lighter than train because eval is sparse)")
    parser.add_argument(
        "--xaxis", type=str, default="episodes", choices=["episodes", "steps", "time"]
    )
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument(
        "--yaml", type=str, default="env_config.yaml", help="Path to env_config.yaml"
    )
    args = parser.parse_args()

    # Load YAML config for per-algo max_steps
    with open(args.yaml, "r") as f:
        env_config = yaml.safe_load(f)

    results_root = Path("all_results")
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

        # Always get max_steps from env_config.yaml if xaxis==steps, unless --max_steps is explicitly provided
        if args.xaxis == "steps":
            if args.max_steps is not None:
                max_steps = args.max_steps
            else:
                env_cfg = env_config.get(args.env, {})
                max_steps_cfg = env_cfg.get("max_steps", 1000000)
                if isinstance(max_steps_cfg, dict):
                    max_steps = int(max_steps_cfg.get(algo_name, 1000000))
                else:
                    max_steps = int(max_steps_cfg)
        else:
            max_steps = args.max_steps

        print(f"Processing algorithm: {algo_name}...")
        stats = collect_for_algo(
            env_dir, args.runs, args.weight, args.xaxis, max_steps,
            eval_weight=args.eval_weight,
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
