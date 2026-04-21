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
from learning_algorithms.SAC_Rainbow import EVSAC
from learning_algorithms.PG_Rainbow import StandardPPOAgent

rng = np.random.default_rng()

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
        
        if action == 1:  # Forward
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

class ContinuousNChainEnv(NChainEnv):
    def __init__(self, n=10):
        super().__init__(n)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
    def step(self, action):
        a = 1 if float(np.sum(action)) > 0 else 0
        return super().step(a)

def make_env(is_continuous=False):
    def thunk():
        return ContinuousNChainEnv(n=10) if is_continuous else NChainEnv(n=10)
    return thunk

def train_dqn(use_rnd=False):
    envs = gym.vector.SyncVectorEnv([make_env(False)])
    agent = EVRainbowDQN(
        input_dim=10, n_action_dims=1, n_action_bins=2, n_envs=1,
        Beta=1.0 if use_rnd else 0.0, lr=1e-3, burn_in_updates=20, 
        beta_half_life_steps=2500, buffer_size=10000
    )
    
    returns = []
    global_step = 0
    obs, _ = envs.reset()
    r_ep = 0.0
    episodes = 0
    
    while episodes < 300:
        global_step += 1
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32)
            action = agent.sample_action(obs_t, eps=max(0.01, 0.25 - episodes/1000), step=global_step, n_steps=10000)
        
        action_np = np.array(action, dtype=np.int32)
        next_obs, reward, term, trunc, infos = envs.step(action_np)
        
        agent.observe(obs, action_np, reward, next_obs, term, trunc, infos)
        
        r_ep += reward[0]
        if term[0] or trunc[0]:
            returns.append(r_ep)
            r_ep = 0.0
            episodes += 1
        
        if agent.buffer.pos >= 32 or getattr(agent.buffer, "full", False):
            agent.update(batch_size=32, step=global_step)
            
        obs = next_obs
    return returns

def train_sac(use_rnd=False):
    envs = gym.vector.SyncVectorEnv([make_env(True)])
    agent = EVSAC(
        envs=envs, beta_rnd=1.0 if use_rnd else 0.0,
        policy_lr=1e-3, q_lr=1e-3, munchausen=False, alpha=0.0, autotune=False,
        beta_half_life_steps=2500, buffer_size=10000
    )
    
    returns = []
    global_step = 0
    obs, _ = envs.reset()
    r_ep = 0.0
    episodes = 0
    
    while episodes < 300:
        global_step += 1
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32)
            action_t, _, _ = agent.actor.get_action(obs_t)
            action_np = action_t.cpu().numpy()
        
        next_obs, reward, term, trunc, infos = envs.step(action_np)
        agent.observe(obs, action_np, reward, next_obs, term, trunc, infos)
        
        r_ep += reward[0]
        if term[0] or trunc[0]:
            returns.append(r_ep)
            r_ep = 0.0
            episodes += 1
            
        if agent.buffer.pos >= 64 or getattr(agent.buffer, "full", False):
            agent.update(batch_size=64, global_step=global_step)
            
        obs = next_obs
    return returns

def train_ppo(use_rnd=False):
    envs = gym.vector.SyncVectorEnv([make_env(False)])
    agent = StandardPPOAgent(
        envs=envs, Beta=5.0 if use_rnd else 0.0, learning_rate=1e-3,
        num_envs=1, num_steps=128, num_minibatches=4, ent_coef=0.01,
        beta_half_life_steps=2500
    )
    
    returns = []
    global_step = 0
    obs, _ = envs.reset()
    r_ep = 0.0
    episodes = 0
    
    while episodes < 300:
        global_step += 1
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32)
            action_t, logprob_t, _, _, _ = agent.get_action_and_values(obs_t)
            action_np = action_t.cpu().numpy()
            logprob_np = logprob_t.cpu().numpy()
        
        next_obs, reward, term, trunc, infos = envs.step(action_np)
        agent.observe(obs, action_np, logprob_np, reward, next_obs, term, trunc, infos)
        
        r_ep += reward[0]
        if term[0] or trunc[0]:
            returns.append(r_ep)
            r_ep = 0.0
            episodes += 1
            
        if agent.step_idx >= agent.num_steps:
            agent.update(global_step=global_step)
            
        obs = next_obs
    return returns

def smooth_ema(scalars, weight=0.9):
    if len(scalars) == 0:
        return np.array(scalars)
        
    last = scalars[0] 
    smoothed = np.empty_like(scalars, dtype=np.float32)
    
    for i, point in enumerate(scalars):
        smoothed_val = last * weight + (1 - weight) * point
        smoothed[i] = smoothed_val
        last = smoothed_val
        
    return smoothed

def run_multiple_seeds(train_func, use_rnd, n_seeds=5):
    all_returns = []
    for seed in range(n_seeds):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        returns = train_func(use_rnd=use_rnd)
        all_returns.append(returns)
        
    returns_array = np.array(all_returns)
    mean_returns = np.mean(returns_array, axis=0)
    sem_returns = np.std(returns_array, axis=0) / np.sqrt(n_seeds)
    
    return mean_returns, sem_returns

def plot_with_shaded_error(mean, sem, label, weight=0.9):
    smoothed_mean = smooth_ema(mean, weight=weight)
    smoothed_sem = smooth_ema(sem, weight=weight)
    episodes = np.arange(len(mean))
    
    line = plt.plot(episodes, smoothed_mean, label=label)[0]
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
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RND Integration Unit Tests")
    parser.add_argument(
        "--algo", type=str, default="all", choices=["dqn", "sac", "ppo", "all"], 
        help="Specify which algorithm to test (dqn, sac, ppo, or all)"
    )
    parser.add_argument(
        "--seeds", type=int, default=10, 
        help="Number of random seeds to average over"
    )
    
    args = parser.parse_args()
    run_integration(algo=args.algo, n_seeds=args.seeds)