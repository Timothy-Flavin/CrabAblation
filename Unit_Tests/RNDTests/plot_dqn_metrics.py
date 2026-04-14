import sys
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from learning_algorithms.DQN_Rainbow import RainbowDQN
from Unit_Tests.RNDTests.test_rnd_integration import NChainEnv, smooth_ema

def train_dqn_metrics(use_rnd=False):
    env = NChainEnv(n=8)
    agent = RainbowDQN(
        input_dim=8, 
        n_action_dims=1, 
        n_action_bins=2, 
        Beta=10.0 if use_rnd else 0.0,
        lr=1e-3,
        ext_r_clamp=10.0,
        burn_in_updates=0
    )
    
    metrics = {
        'returns': [],
        'extrinsic_loss': [],
        'intrinsic_loss': [],
        'rnd_loss': [],
        'beta': [],
        'epsilon': [],
        'q_ext_mean': [],
        'q_int_mean': [],
        'avg_r_int': []
    }
    
    global_step = 0
    for episode in range(1000):
        import sys; sys.stdout.write(f"\repisode {episode}")
        obs, _ = env.reset()
        episode_return = 0
        done = False
        while not done:
            global_step += 1
            
            with torch.no_grad():
                action_tensor = agent.sample_action(
                    torch.tensor(obs, dtype=torch.float32), 
                    eps=0.5-episode/2000, 
                    step=global_step, 
                    n_steps=1000
                )
                action = int(action_tensor[0])
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.update_running_stats(
                torch.tensor(next_obs, dtype=torch.float64), 
                torch.tensor([reward], dtype=torch.float64)
            )
            
            agent.update(
                torch.tensor(obs).unsqueeze(0), 
                torch.tensor([[action]]), 
                torch.tensor([reward]), 
                torch.tensor(next_obs).unsqueeze(0), 
                torch.tensor([terminated], dtype=torch.float32), 
                batch_size=1,
                step=1
            )
            agent.update_target()
            
            if hasattr(agent, 'last_losses') and agent.last_losses is not None:
                metrics['extrinsic_loss'].append(agent.last_losses.get('extrinsic', 0.0))
                metrics['intrinsic_loss'].append(agent.last_losses.get('intrinsic', 0.0))
                metrics['rnd_loss'].append(agent.last_losses.get('rnd', 0.0))
                metrics['beta'].append(agent.last_losses.get('Beta', 0.0))
                metrics['epsilon'].append(agent.last_losses.get('last_eps', 0.0))
                metrics['q_ext_mean'].append(agent.last_losses.get('Q_ext_mean', 0.0))
                metrics['q_int_mean'].append(agent.last_losses.get('Q_int_mean', 0.0))
                metrics['avg_r_int'].append(agent.last_losses.get('avg_r_int', 0.0))
                
            obs = next_obs
            episode_return += reward
            
        metrics['returns'].append(episode_return)
        
    return metrics

def run_multiple_seeds_metrics(n_seeds=20):
    all_metrics = {'rnd': {}, 'no_rnd': {}}
    
    for use_rnd, key in [(True, 'rnd'), (False, 'no_rnd')]:
        seed_metrics = []
        for seed in range(n_seeds):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            print(f"Running {key} seed {seed}...")
            seed_metrics.append(train_dqn_metrics(use_rnd=use_rnd))
            
        # Aggregate
        keys = seed_metrics[0].keys()
        agg = {}
        for k in keys:
            min_len = min(len(m[k]) for m in seed_metrics)
            arr = np.array([m[k][:min_len] for m in seed_metrics])
            agg[k] = {
                'mean': np.mean(arr, axis=0),
                'sem': np.std(arr, axis=0) / np.sqrt(n_seeds) if n_seeds > 1 else np.zeros_like(np.mean(arr, axis=0))
            }
        all_metrics[key] = agg
        
    return all_metrics

def plot_metrics(agg_metrics, n_seeds):
    metrics_to_plot = list(agg_metrics['rnd'].keys())
    os.makedirs('Unit_Tests/RNDTests/debug_plots', exist_ok=True)
    
    for k in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        
        for key, label in [('no_rnd', 'No RND'), ('rnd', 'With RND')]:
            mean = agg_metrics[key][k]['mean']
            sem = agg_metrics[key][k]['sem']
            
            smoothed_mean = smooth_ema(mean, weight=0.9)
            smoothed_sem = smooth_ema(sem, weight=0.9)
            x = np.arange(len(mean))
            
            line = plt.plot(x, smoothed_mean, label=label)[0]
            plt.fill_between(
                x, 
                smoothed_mean - smoothed_sem, 
                smoothed_mean + smoothed_sem, 
                color=line.get_color(), 
                alpha=0.2
            )
            
        plt.title(f"Averaged DQN {k} over {n_seeds} seeds")
        plt.xlabel("Step" if k != 'returns' else "Episode")
        plt.ylabel(k.replace('_', ' ').title())
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"Unit_Tests/RNDTests/debug_plots/{k}_over_time.png")
        plt.close()
        print(f"Saved plot for {k}")

if __name__ == "__main__":
    n_seeds = 5
    agg = run_multiple_seeds_metrics(n_seeds=n_seeds)
    plot_metrics(agg, n_seeds)

