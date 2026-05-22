import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot_ma_results(algo, env_name):
    results_dir = f"all_results/{algo}/{env_name}"
    if not os.path.exists(results_dir):
        print(f"Directory {results_dir} not found.")
        return

    ablations = range(7)
    colors = plt.cm.tab10(np.linspace(0, 1, 7))

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Multi-Agent Results: {algo.upper()} on {env_name.upper()}")

    # 1. Exploitability
    ax = axs[0, 0]
    ax.set_title("Exploitability (Lower is Better)")
    ax.set_xlabel("Evaluation Step")
    ax.set_ylabel("Exploitability")
    
    for i in ablations:
        path = os.path.join(results_dir, f"exploitability_{i}.npy")
        if os.path.exists(path):
            data = np.load(path)
            ax.plot(data, label=f"Ablation {i}", color=colors[i])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Train Scores (Player 0)
    ax = axs[0, 1]
    ax.set_title("Train Scores (Player 0)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    
    for i in ablations:
        path = os.path.join(results_dir, f"train_scores_player_0_{i}.npy")
        if os.path.exists(path):
            data = np.load(path)
            # Smooth data
            window = max(1, len(data) // 50)
            smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
            ax.plot(smoothed, label=f"Ablation {i}", color=colors[i], alpha=0.8)
    ax.grid(True, alpha=0.3)

    # 3. vs Random (Player 0)
    ax = axs[1, 0]
    ax.set_title("vs Random (Player 0)")
    ax.set_xlabel("Evaluation Step")
    ax.set_ylabel("Average Reward")
    
    for i in ablations:
        path = os.path.join(results_dir, f"evaluate_vs_random_p0_{i}.npy")
        if os.path.exists(path):
            data = np.load(path)
            ax.plot(data, label=f"Ablation {i}", color=colors[i])
    ax.grid(True, alpha=0.3)

    # 4. Action Distribution (for RPS or Tic-Tac-Toe if applicable)
    # For RPS, we can plot the strategy of Player 0 for one ablation (e.g. 0 or 6)
    ax = axs[1, 1]
    ax.set_title("Action Probs (P0, Ablation 0/6)")
    ax.set_xlabel("Logged Step")
    ax.set_ylabel("Probability")
    
    # Try to find a representative ablation for strategy plotting
    rep_ablation = 6 if os.path.exists(os.path.join(results_dir, f"action_dist_p0_6.npy")) else 0
    path = os.path.join(results_dir, f"action_dist_p0_{rep_ablation}.npy")
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        if len(data.shape) == 2 and data.shape[1] == 3: # RPS
            labels = ["Rock", "Paper", "Scissors"]
            for j in range(3):
                ax.plot(data[:, j], label=labels[j])
            ax.set_title(f"Action Probs (P0, Ablation {rep_ablation})")
            ax.legend()
        elif len(data.shape) == 2: # Discrete actions
            for j in range(min(data.shape[1], 9)):
                ax.plot(data[:, j], label=f"Act {j}")
            ax.legend()
    else:
        ax.text(0.5, 0.5, "No distribution data found", ha='center', va='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = f"{results_dir}/summary_plot.png"
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="dqn")
    parser.add_argument("--env_name", type=str, default="rps")
    args = parser.parse_args()
    plot_ma_results(args.algo, args.env_name)
