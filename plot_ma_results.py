import os
import re
import numpy as np
import matplotlib.pyplot as plt
import argparse


def _detect_ablations(all_files, prefix):
    """Pull ablation indices from files named '<prefix><i>.npy' (old, no-seed scheme)
    or '<prefix><i>_seed<s>.npy' (current scheme). Robust to either."""
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)(?:_seed\d+)?\.npy$")
    found = set()
    for f in all_files:
        m = pat.match(f)
        if m:
            found.add(int(m.group(1)))
    return sorted(found)


def _iqm(seed_data):
    """Interquartile mean across seeds (axis=0): mean of values in [Q1, Q3]."""
    arr = np.array(seed_data)
    q1 = np.percentile(arr, 25, axis=0)
    q3 = np.percentile(arr, 75, axis=0)
    masked = np.where((arr >= q1) & (arr <= q3), arr, np.nan)
    return np.nanmean(masked, axis=0)


def plot_ma_results(algo, env_name, base_results_dir):
    results_dir = os.path.join(base_results_dir, algo, env_name)
    if not os.path.exists(results_dir):
        print(f"Directory {results_dir} not found.")
        return

    # Automatically detect ablations and seeds
    all_files = os.listdir(results_dir)
    ablations = _detect_ablations(all_files, "exploitability_")
    if not ablations:
        # Try another prefix
        ablations = _detect_ablations(all_files, "train_scores_player_0_")

    if not ablations:
        print(f"No ablation data found in {results_dir}")
        return

    num_ablations = len(ablations)
    colors = plt.cm.tab10(np.linspace(0, 1, max(7, num_ablations)))

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        f"Multi-Agent Results: {algo.upper()} on {env_name.upper()} (IQM across seeds)"
    )

    # 1. Exploitability
    ax = axs[0, 0]
    ax.set_title("Exploitability (Lower is Better)")
    ax.set_xlabel("Evaluation Step")
    ax.set_ylabel("Exploitability")

    for i in ablations:
        seed_data = []
        for seed in range(10):  # Check up to 10 seeds
            path = os.path.join(results_dir, f"exploitability_{i}_seed{seed}.npy")
            if os.path.exists(path):
                try:
                    data = np.load(path)
                    if data.ndim == 0:
                        data = np.array([data.item()])
                    seed_data.append(data)
                except Exception:
                    pass
        if seed_data:
            min_len = min(len(s) for s in seed_data)
            seed_data = [s[:min_len] for s in seed_data]
            median_data = _iqm(seed_data)
            ax.plot(
                median_data,
                label=f"Ablation {i}",
                color=colors[i] if i < len(colors) else None,
            )
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Train Scores (Player 0)
    ax = axs[0, 1]
    ax.set_title("Train Scores (Player 0)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")

    for i in ablations:
        seed_data = []
        for seed in range(10):
            path = os.path.join(
                results_dir, f"train_scores_player_0_{i}_seed{seed}.npy"
            )
            if os.path.exists(path):
                try:
                    data = np.load(path)
                    seed_data.append(data)
                except Exception:
                    pass
        if seed_data:
            min_len = min(len(s) for s in seed_data)
            seed_data = [s[:min_len] for s in seed_data]
            median_data = _iqm(seed_data)
            # Smooth data
            window = max(1, len(median_data) // 50)
            smoothed = np.convolve(median_data, np.ones(window) / window, mode="valid")
            ax.plot(
                smoothed,
                label=f"Ablation {i}",
                color=colors[i] if i < len(colors) else None,
                alpha=0.8,
            )
    ax.legend(fontsize="small", ncol=2)
    ax.grid(True, alpha=0.3)

    # 3. vs Random (Player 0)
    ax = axs[1, 0]
    ax.set_title("vs Random (Player 0)")
    ax.set_xlabel("Evaluation Step")
    ax.set_ylabel("Average Reward")

    for i in ablations:
        seed_data = []
        for seed in range(10):
            path = os.path.join(
                results_dir, f"evaluate_vs_random_p0_{i}_seed{seed}.npy"
            )
            if os.path.exists(path):
                try:
                    data = np.load(path)
                    if data.ndim == 0:
                        data = np.array([data.item()])
                    seed_data.append(data)
                except Exception:
                    pass
        if seed_data:
            min_len = min(len(s) for s in seed_data)
            seed_data = [s[:min_len] for s in seed_data]
            median_data = _iqm(seed_data)
            ax.plot(
                median_data,
                label=f"Ablation {i}",
                color=colors[i] if i < len(colors) else None,
            )
    ax.legend(fontsize="small", ncol=2)
    ax.grid(True, alpha=0.3)

    # 4. Action Distribution (rps) or vs Random Player 1 (tictactoe/leduc)
    ax = axs[1, 1]

    if "rps" in env_name.lower():
        ax.set_title("Action Probs (P0, Best Ablation)")
        ax.set_xlabel("Logged Step")
        ax.set_ylabel("Probability")

        rep_ablation = ablations[-1] if ablations else 0
        path = os.path.join(results_dir, f"action_dist_p0_{rep_ablation}_seed0.npy")
        if os.path.exists(path):
            try:
                data = np.load(path, allow_pickle=True)
                if data.dtype == object:
                    data = np.array(list(data))
                data = data.squeeze()
                if data.ndim == 2:
                    n_cols = data.shape[1]
                    labels = (
                        ["Rock", "Paper", "Scissors"]
                        if n_cols == 3
                        else [f"Act {j}" for j in range(n_cols)]
                    )
                    for j in range(min(n_cols, 10)):
                        label = labels[j] if j < len(labels) else f"Act {j}"
                        ax.plot(data[:, j], label=label)
                    ax.set_title(f"Action Probs (P0, Ablation {rep_ablation})")
                    ax.legend(fontsize="small", ncol=2)
                else:
                    ax.text(
                        0.5,
                        0.5,
                        f"Unexpected data shape: {data.shape}",
                        ha="center",
                        va="center",
                    )
            except Exception as e:
                ax.text(
                    0.5, 0.5, f"Error plotting dist: {str(e)}", ha="center", va="center"
                )
        else:
            ax.text(0.5, 0.5, "No distribution data found", ha="center", va="center")
    else:
        ax.set_title("vs Random (Player 1)")
        ax.set_xlabel("Evaluation Step")
        ax.set_ylabel("Average Reward")

        for i in ablations:
            seed_data = []
            for seed in range(10):
                path = os.path.join(
                    results_dir, f"evaluate_vs_random_p1_{i}_seed{seed}.npy"
                )
                if os.path.exists(path):
                    try:
                        data = np.load(path)
                        if data.ndim == 0:
                            data = np.array([data.item()])
                        seed_data.append(data)
                    except Exception:
                        pass
            if seed_data:
                min_len = min(len(s) for s in seed_data)
                seed_data = [s[:min_len] for s in seed_data]
                median_data = _iqm(seed_data)
                ax.plot(
                    median_data,
                    label=f"Ablation {i}",
                    color=colors[i] if i < len(colors) else None,
                )
        ax.legend(fontsize="small", ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(results_dir, "summary_plot.png")
    plt.savefig(save_path)
    # Also save a copy in the base directory with a descriptive name
    global_save_path = os.path.join(
        base_results_dir, f"{algo}_{env_name}_ma_summary.png"
    )
    plt.savefig(global_save_path)
    plt.close()
    print(f"Plots saved to {save_path} and {global_save_path}")


def _load_iqm_for_metric(results_dir, algo, env, metric_prefix, n_seeds=10):
    """Return {ablation_idx: IQM array} for a given metric prefix."""
    env_dir = os.path.join(results_dir, algo, env)
    if not os.path.exists(env_dir):
        return {}
    all_files = os.listdir(env_dir)
    ablations = _detect_ablations(all_files, metric_prefix)
    result = {}
    for i in ablations:
        seed_data = []
        for seed in range(n_seeds):
            path = os.path.join(env_dir, f"{metric_prefix}{i}_seed{seed}.npy")
            if os.path.exists(path):
                try:
                    data = np.load(path)
                    if data.ndim == 0:
                        data = np.array([data.item()])
                    seed_data.append(data)
                except Exception:
                    pass
        if seed_data:
            min_len = min(len(s) for s in seed_data)
            seed_data = [s[:min_len] for s in seed_data]
            result[i] = _iqm(seed_data)
    return result


def _minmax_normalize(curves_dict):
    """Normalize all ablation curves to [0, 1] using a single global min/max."""
    if not curves_dict:
        return {}
    all_vals = np.concatenate(list(curves_dict.values()))
    v_min, v_max = float(np.min(all_vals)), float(np.max(all_vals))
    if v_max == v_min:
        return {k: np.zeros_like(v) for k, v in curves_dict.items()}
    return {k: (v - v_min) / (v_max - v_min) for k, v in curves_dict.items()}


def _interp_unit_x(curve, n_points):
    """Interpolate curve onto n_points uniformly spaced in [0, 1]."""
    x_old = np.linspace(0, 1, len(curve))
    x_new = np.linspace(0, 1, n_points)
    return np.interp(x_new, x_old, curve)


def plot_aggregate_results(algos, envs, base_results_dir, n_points=200):
    """
    1. Per-algo aggregate plots: per-ablation curves averaged over all envs.
    2. Final ablation plot: per-ablation curves averaged over all algos and envs.

    Y-axis: min-max normalized per (env, metric) to [0, 1].
    X-axis: rescaled to [0, 1] (training progress) via linear interpolation.
    Metrics: exploitability and score vs random (player 0).
    """
    METRIC_PREFIXES = ["exploitability_", "evaluate_vs_random_p0_"]
    METRIC_LABELS = [
        "Normalized Exploitability (lower = better)",
        "Normalized Score vs Random P0 (higher = better)",
    ]
    x_grid = np.linspace(0, 1, n_points)

    # all_data[algo][env][prefix] = {ablation_idx: interp_array on [0,1]}
    all_data = {algo: {env: {} for env in envs} for algo in algos}

    for algo in algos:
        for env in envs:
            for prefix in METRIC_PREFIXES:
                raw = _load_iqm_for_metric(base_results_dir, algo, env, prefix)
                if not raw:
                    continue
                normalized = _minmax_normalize(raw)
                all_data[algo][env][prefix] = {
                    i: _interp_unit_x(v, n_points) for i, v in normalized.items()
                }

    # Per-algo aggregate plots
    for algo in algos:
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"Aggregate: {algo.upper()} – mean over {len(envs)} envs ({', '.join(envs)})"
        )
        for ax, prefix, label in zip(axs, METRIC_PREFIXES, METRIC_LABELS):
            ax.set_title(label)
            ax.set_xlabel("Training Progress")
            ax.set_ylabel("Normalized Score [0–1]")
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(0, 1)

            all_ablations = set()
            for env in envs:
                if prefix in all_data[algo][env]:
                    all_ablations.update(all_data[algo][env][prefix].keys())

            if not all_ablations:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                continue

            colors = plt.cm.tab10(np.linspace(0, 1, max(7, len(all_ablations))))
            for i in sorted(all_ablations):
                env_curves = [
                    all_data[algo][env][prefix][i]
                    for env in envs
                    if prefix in all_data[algo][env] and i in all_data[algo][env][prefix]
                ]
                if env_curves:
                    mean_curve = np.mean(env_curves, axis=0)
                    ax.plot(x_grid, mean_curve, label=f"Ablation {i}",
                            color=colors[i % len(colors)])
            ax.legend(fontsize="small")
            ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = os.path.join(base_results_dir, f"{algo}_aggregate.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Per-algo aggregate plot saved to {save_path}")

    # Final ablation plot: average over all algos and envs
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Final Ablation Study – mean over {len(algos)} algos × {len(envs)} envs"
    )
    for ax, prefix, label in zip(axs, METRIC_PREFIXES, METRIC_LABELS):
        ax.set_title(label)
        ax.set_xlabel("Training Progress")
        ax.set_ylabel("Normalized Score [0–1]")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, 1)

        all_ablations = set()
        for algo in algos:
            for env in envs:
                if prefix in all_data[algo][env]:
                    all_ablations.update(all_data[algo][env][prefix].keys())

        if not all_ablations:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue

        colors = plt.cm.tab10(np.linspace(0, 1, max(7, len(all_ablations))))
        for i in sorted(all_ablations):
            all_curves = []
            for algo in algos:
                for env in envs:
                    if prefix in all_data[algo][env] and i in all_data[algo][env][prefix]:
                        all_curves.append(all_data[algo][env][prefix][i])
            if all_curves:
                mean_curve = np.mean(all_curves, axis=0)
                ax.plot(x_grid, mean_curve, label=f"Ablation {i}",
                        color=colors[i % len(colors)])
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(base_results_dir, "final_ablation_aggregate.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Final aggregate plot saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="single",
                        choices=["single", "aggregate"],
                        help="'single': one algo/env plot; 'aggregate': cross-env/algo summary")
    # Single-run args
    parser.add_argument("--algo", type=str, default="dqn")
    parser.add_argument("--env_name", type=str, default="rps")
    # Aggregate args
    parser.add_argument("--algos", type=str, nargs="+", default=["dqn", "ppo", "sac"])
    parser.add_argument("--envs", type=str, nargs="+",
                        default=["rps", "leduc", "tictactoe"])
    parser.add_argument("--results_dir", type=str, default="all_results")
    args = parser.parse_args()

    if args.mode == "aggregate":
        plot_aggregate_results(args.algos, args.envs, args.results_dir)
    else:
        plot_ma_results(args.algo, args.env_name, args.results_dir)
