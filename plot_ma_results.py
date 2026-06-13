import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot_ma_results(algo, env_name, base_results_dir):
    results_dir = os.path.join(base_results_dir, algo, env_name)
    if not os.path.exists(results_dir):
        print(f"Directory {results_dir} not found.")
        return

    # Automatically detect ablations and seeds
    all_files = os.listdir(results_dir)
    ablations = sorted(list(set([int(f.split('_')[-2]) for f in all_files if f.startswith('exploitability_') and f.endswith('.npy')])))
    if not ablations:
        # Try another prefix
        ablations = sorted(list(set([int(f.split('_')[-2]) for f in all_files if f.startswith('train_scores_player_0_') and f.endswith('.npy')])))
    
    if not ablations:
        print(f"No ablation data found in {results_dir}")
        return

    num_ablations = len(ablations)
    colors = plt.cm.tab10(np.linspace(0, 1, max(7, num_ablations)))

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Multi-Agent Results: {algo.upper()} on {env_name.upper()}")

    # 1. Exploitability
    ax = axs[0, 0]
    ax.set_title("Exploitability (Lower is Better)")
    ax.set_xlabel("Evaluation Step")
    ax.set_ylabel("Exploitability")
    
    for i in ablations:
        seed_data = []
        for seed in range(10): # Check up to 10 seeds
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
            median_data = np.median(np.array(seed_data), axis=0)
            ax.plot(median_data, label=f"Ablation {i}", color=colors[i] if i < len(colors) else None)
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
            path = os.path.join(results_dir, f"train_scores_player_0_{i}_seed{seed}.npy")
            if os.path.exists(path):
                try:
                    data = np.load(path)
                    seed_data.append(data)
                except Exception:
                    pass
        if seed_data:
            min_len = min(len(s) for s in seed_data)
            seed_data = [s[:min_len] for s in seed_data]
            median_data = np.median(np.array(seed_data), axis=0)
            # Smooth data
            window = max(1, len(median_data) // 50)
            smoothed = np.convolve(median_data, np.ones(window)/window, mode='valid')
            ax.plot(smoothed, label=f"Ablation {i}", color=colors[i] if i < len(colors) else None, alpha=0.8)
    ax.grid(True, alpha=0.3)

    # 3. vs Random (Player 0)
    ax = axs[1, 0]
    ax.set_title("vs Random (Player 0)")
    ax.set_xlabel("Evaluation Step")
    ax.set_ylabel("Average Reward")
    
    for i in ablations:
        seed_data = []
        for seed in range(10):
            path = os.path.join(results_dir, f"evaluate_vs_random_p0_{i}_seed{seed}.npy")
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
            median_data = np.median(np.array(seed_data), axis=0)
            ax.plot(median_data, label=f"Ablation {i}", color=colors[i] if i < len(colors) else None)
    ax.grid(True, alpha=0.3)

    # 4. Action Distribution
    ax = axs[1, 1]
    ax.set_title("Action Probs (P0, Best Ablation)")
    ax.set_xlabel("Logged Step")
    ax.set_ylabel("Probability")
    
    # Try to find a representative ablation (highest index usually has most features enabled)
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
                if n_cols == 3 and "rps" in env_name.lower():
                    labels = ["Rock", "Paper", "Scissors"]
                elif n_cols == 9 and "tictactoe" in env_name.lower():
                    labels = [f"Pos {j}" for j in range(n_cols)]
                else:
                    labels = [f"Act {j}" for j in range(n_cols)]
                
                for j in range(min(n_cols, 10)):
                    label = labels[j] if j < len(labels) else f"Act {j}"
                    ax.plot(data[:, j], label=label)
                ax.set_title(f"Action Probs (P0, Ablation {rep_ablation})")
                ax.legend(fontsize='small', ncol=2)
            else:
                ax.text(0.5, 0.5, f"Unexpected data shape: {data.shape}", ha='center', va='center')
        except Exception as e:
            ax.text(0.5, 0.5, f"Error plotting dist: {str(e)}", ha='center', va='center')
    else:
        ax.text(0.5, 0.5, "No distribution data found", ha='center', va='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(results_dir, "summary_plot.png")
    plt.savefig(save_path)
    # Also save a copy in the base directory with a descriptive name
    global_save_path = os.path.join(base_results_dir, f"{algo}_{env_name}_ma_summary.png")
    plt.savefig(global_save_path)
    plt.close()
    print(f"Plots saved to {save_path} and {global_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="dqn")
    parser.add_argument("--env_name", type=str, default="rps")
    parser.add_argument("--results_dir", type=str, default="all_results")
    args = parser.parse_args()
    plot_ma_results(args.algo, args.env_name, args.results_dir)
