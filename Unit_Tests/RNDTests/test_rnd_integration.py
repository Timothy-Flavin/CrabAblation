import sys
import os
import random
import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from learning_algorithms.DQN_Rainbow import EVRainbowDQN
rng = np.random.default_rng()
# Simple Deterministic N-Chain Environment
class NChainEnv(gym.Env):
    def __init__(self, n=10):
        super().__init__()
        self.n = n
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(n,), dtype=np.float32)
        self.state = 0
        self.max_steps = n + 10
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0
        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        obs = (rng.random(size=self.n,dtype=np.float32)-0.5)/5
        obs[self.state] = 1.0
        return obs

    def step(self, action):
        self.steps += 1
        reward = 0.0
        
        if action == 0:  # Forward
            if self.state < self.n - 1:
                self.state += 1
            if self.state == self.n - 1:
                reward = 10.0 # Big reward at the end
        else:  # Backward
            self.state = 0
            reward = 0.01 # Small reward distraction
            
        terminated = (self.state == self.n - 1)
        truncated = (self.steps >= self.max_steps)
        
        return self._get_obs(), reward, terminated, truncated, {}

def train_dqn(use_rnd=False):
    env = NChainEnv(n=7)
    agent = EVRainbowDQN(
        input_dim=7, 
        n_action_dims=1, 
        n_action_bins=2, 
        Beta=1.0 if use_rnd else 0.0,
        lr=1e-3,
        burn_in_updates=20, # Start updating right away
        beta_half_life_steps=2500
    )
    
    returns = []
    global_step = 0
    
    # We will use extremely short horizons and wait for converge
    for episode in range(1000):
        obs, _ = env.reset()
        episode_return = 0
        done = False
        while not done:
            global_step += 1
            
            with torch.no_grad():
                action_tensor = agent.sample_action(
                    torch.tensor(obs, dtype=torch.float32), 
                    eps=0.25-episode/4000, 
                    step=global_step, 
                    n_steps=global_step
                )
                action = int(action_tensor[0])
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Since step logic updates the rms:
            agent.update_running_stats(
                torch.tensor(next_obs, dtype=torch.float64), 
                torch.tensor([reward], dtype=torch.float64)
            )
            
            # Update
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
            
            obs = next_obs
            episode_return += reward
            
        returns.append(episode_return)
        
    print(f"Final list of returns for train_dqn(use_rnd={use_rnd}): {returns[-20:]}")
    return returns

class MockBatch:
    def __init__(self, obs, act, rew, next_obs, dones):
        self.observations = obs
        self.actions = act
        self.rewards = rew
        self.next_observations = next_obs
        self.dones = dones

from learning_algorithms.SAC_Rainbow import SACAgent

class ContinuousNChainEnv(NChainEnv):
    def __init__(self, n=10):
        super().__init__(n)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
    def step(self, action):
        a = 1 if float(action) > 0 else 0
        return super().step(a)

def train_sac(use_rnd=False):
    env = ContinuousNChainEnv(n=7)
    envs = gym.vector.SyncVectorEnv([lambda: env])
    agent = SACAgent(
        envs=envs,
        beta_rnd=1.0 if use_rnd else 0.0,
        policy_lr=1e-3,
        q_lr=1e-3,
        munchausen=False,
        alpha=0.0,
        autotune=False,
        popart=True,
        beta_half_life_steps=2500
    )
    
    returns = []
    global_step = 0
    
    for episode in range(1000):
        obs, _ = env.reset()
        episode_return = 0
        done = False
        while not done:
            global_step += 1
            
            with torch.no_grad():
                # Get action from SAC
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                action_t, _, _ = agent.actor.get_action(obs_t)
                action = action_t.item()
                
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.update_running_stats(
                torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0), 
                torch.tensor([reward], dtype=torch.float32)
            )
            
            # Form dummy batch
            batch = MockBatch(
                obs=torch.tensor(obs, dtype=torch.float32).unsqueeze(0),
                act=torch.tensor([[action]], dtype=torch.float32),
                rew=torch.tensor([reward], dtype=torch.float32),
                next_obs=torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0),
                dones=torch.tensor([terminated], dtype=torch.float32)
            )
            
            agent.update(batch, global_step=global_step)
            
            obs = next_obs
            episode_return += reward
                
        returns.append(episode_return)
        
    return returns

from learning_algorithms.PG_Rainbow import PPOAgent

class MockEnvPPO:
    def __init__(self, obs_shape, act_shape):
        import types
        self.single_observation_space = types.SimpleNamespace(shape=obs_shape)
        self.single_action_space = types.SimpleNamespace(shape=act_shape, n=2)
        self.num_envs = 1

def train_ppo(use_rnd=False):
    env = NChainEnv(n=7)
    envs = MockEnvPPO((7,), ())
    agent = PPOAgent(
        envs=envs,
        Beta=1.0 if use_rnd else 0.0,
        learning_rate=1e-3,
        num_envs=1,
        ent_coef=0.1
    )
    
    returns = []
    
    for episode in range(1000):
        obs, _ = env.reset()
        episode_return = 0
        done = False
        
        obs_b = []
        act_b = []
        logp_b = []
        rew_b = []
        val_b = []
        int_val_b = []
        done_b = []
        
        while not done:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                action_t, logprob, _, ext_v, int_v = agent.get_action_and_values(obs_t)
                action = int(action_t.squeeze())
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.update_running_stats(
                torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0), 
                r=torch.tensor([reward], dtype=torch.float32)
            )
            
            obs_b.append(obs_t)
            act_b.append(action_t)
            logp_b.append(logprob)
            rew_b.append(torch.tensor([reward], dtype=torch.float32))
            val_b.append(ext_v.flatten())
            int_val_b.append(int_v.flatten())
            done_b.append(torch.tensor([terminated], dtype=torch.float32))
            
            obs = next_obs
            episode_return += reward
            
        returns.append(episode_return)
        
        agent.num_steps = len(obs_b)
        agent.batch_size = len(obs_b)
        agent.num_minibatches = 1
        agent.minibatch_size = len(obs_b)
        
        agent.update(
            torch.cat(obs_b).unsqueeze(1),
            torch.cat(act_b).unsqueeze(1),
            torch.cat(logp_b).unsqueeze(1),
            torch.cat(rew_b).unsqueeze(1),
            torch.cat(done_b).unsqueeze(1),
            torch.cat(val_b).unsqueeze(1),
            torch.cat(int_val_b).unsqueeze(1),
            torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0),
            torch.tensor([done], dtype=torch.float32)
        )
        
    return returns

def smooth_ema(scalars, weight=0.9):
    """
    Computes an Exponential Moving Average.
    weight: [0, 1). Higher weight = more smoothing.
    """
    if len(scalars) == 0:
        return np.array(scalars)
        
    last = scalars[0]  # Initialize with the first data point
    smoothed = np.empty_like(scalars, dtype=np.float32)
    
    for i, point in enumerate(scalars):
        smoothed_val = last * weight + (1 - weight) * point
        smoothed[i] = smoothed_val
        last = smoothed_val
        
    return smoothed

def run_multiple_seeds(train_func, use_rnd, n_seeds=5):
    """
    Executes a training function over a specified number of seeds 
    and returns the averaged return array and the standard error of the mean (SEM).
    """
    all_returns = []
    for seed in range(n_seeds):
        # Enforce reproducibility per seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        returns = train_func(use_rnd=use_rnd)
        all_returns.append(returns)
        
    returns_array = np.array(all_returns)
    
    # Calculate the mean and SEM across the seed axis
    mean_returns = np.mean(returns_array, axis=0)
    sem_returns = np.std(returns_array, axis=0) / np.sqrt(n_seeds)
    
    return mean_returns, sem_returns

def plot_with_shaded_error(mean, sem, label, weight=0.9):
    """
    Plots the smoothed mean and a shaded region for the smoothed SEM.
    """
    smoothed_mean = smooth_ema(mean, weight=weight)
    smoothed_sem = smooth_ema(sem, weight=weight)
    episodes = np.arange(len(mean))
    
    # Plot the mean line and capture the color assigned by matplotlib
    line = plt.plot(episodes, smoothed_mean, label=label)[0]
    
    # Fill the region between (mean - sem) and (mean + sem)
    plt.fill_between(
        episodes, 
        smoothed_mean - smoothed_sem, 
        smoothed_mean + smoothed_sem, 
        color=line.get_color(), 
        alpha=0.2
    )

def run_integration(algo="all", n_seeds=10):
    w = 0.9 
    os.makedirs("Unit_Tests/RNDTests", exist_ok=True)
    
    # Initialize a fresh figure to prevent overlapping if called sequentially
    plt.figure()

    if algo in ["dqn", "all"]:
        print(f"Testing DQN without RND over {n_seeds} seeds...")
        no_rnd_dqn_mean, no_rnd_dqn_sem = run_multiple_seeds(train_dqn, use_rnd=False, n_seeds=n_seeds)
        print(f"Testing DQN with RND over {n_seeds} seeds...")
        rnd_dqn_mean, rnd_dqn_sem = run_multiple_seeds(train_dqn, use_rnd=True, n_seeds=n_seeds)
        
        plot_with_shaded_error(no_rnd_dqn_mean, no_rnd_dqn_sem, "DQN No RND", weight=w)
        plot_with_shaded_error(rnd_dqn_mean, rnd_dqn_sem, "DQN With RND", weight=w)

    if algo in ["sac", "all"]:
        print(f"Testing SAC without RND over {n_seeds} seeds...")
        no_rnd_sac_mean, no_rnd_sac_sem = run_multiple_seeds(train_sac, use_rnd=False, n_seeds=n_seeds)
        print(f"Testing SAC with RND over {n_seeds} seeds...")
        rnd_sac_mean, rnd_sac_sem = run_multiple_seeds(train_sac, use_rnd=True, n_seeds=n_seeds)
        
        plot_with_shaded_error(no_rnd_sac_mean, no_rnd_sac_sem, "SAC No RND", weight=w)
        plot_with_shaded_error(rnd_sac_mean, rnd_sac_sem, "SAC With RND", weight=w)

    if algo in ["ppo", "all"]:
        print(f"Testing PPO without RND over {n_seeds} seeds...")
        no_rnd_ppo_mean, no_rnd_ppo_sem = run_multiple_seeds(train_ppo, use_rnd=False, n_seeds=n_seeds)
        print(f"Testing PPO with RND over {n_seeds} seeds...")
        rnd_ppo_mean, rnd_ppo_sem = run_multiple_seeds(train_ppo, use_rnd=True, n_seeds=n_seeds)
        
        plot_with_shaded_error(no_rnd_ppo_mean, no_rnd_ppo_sem, "PPO No RND", weight=w)
        plot_with_shaded_error(rnd_ppo_mean, rnd_ppo_sem, "PPO With RND", weight=w)

    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Average Return")
    plt.title(f"RND Integration Test - {algo.upper()}")
    
    save_path = f"Unit_Tests/RNDTests/nchain_integration_{algo}.png"
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    
    # Close the figure to free memory
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RND Integration Unit Tests")
    parser.add_argument(
        "--algo", 
        type=str, 
        default="all", 
        choices=["dqn", "sac", "ppo", "all"], 
        help="Specify which algorithm to test (dqn, sac, ppo, or all)"
    )
    parser.add_argument(
        "--seeds", 
        type=int, 
        default=20, 
        help="Number of random seeds to average over"
    )
    
    args = parser.parse_args()
    
    run_integration(algo=args.algo, n_seeds=args.seeds)