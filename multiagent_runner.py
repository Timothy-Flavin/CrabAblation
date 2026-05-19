import argparse
import os
import time
import numpy as np
import torch
from pettingzoo.classic import tictactoe_v3, leduc_holdem_v4
from runner import build_agent, process_args, get_parser
from runner_utils import resolve_torch_device
import gymnasium as gym
from types import SimpleNamespace

def get_ma_args():
    parser = get_parser()
    parser.add_argument("--ma_env", type=str, default="tictactoe", choices=["tictactoe", "leduc"])
    parser.add_argument("--total_episodes", type=int, default=10000)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--ent_coef_override", type=float, default=None)
    
    # Override some defaults for MA
    parser.set_defaults(num_envs=1, env_name="tictactoe")
    
    args, _ = parser.parse_known_args()
    # Sync env_name with ma_env for build_agent
    if args.ma_env == "tictactoe":
        args.env_name = "tictactoe"
    else:
        args.env_name = "leduc"
        
    return process_args(args)

class MAAgentWrapper:
    def __init__(self, agent, algo, device, args):
        self.agent = agent
        self.algo = algo
        self.device = device
        self.args = args
        self.action_dist_log = [] # List of policy distributions for Tic-Tac-Toe
        
    def get_action(self, obs, mask, deterministic=False, step=0, total_steps=1000000):
        # We now use the agent's built-in sample_action
        # Prepare inputs
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(mask, dtype=torch.float32, device=self.device)
        
        # Track action distribution for Tic-Tac-Toe ternary plots
        # We'll log the probabilities for the first few turns or fixed states
        # For simplicity, we can log the logits/probs for the current state if it's tictactoe
        if self.args.ma_env == "tictactoe":
            with torch.no_grad():
                probs = self._get_probs(obs_t, mask_t)
                self.action_dist_log.append(probs.cpu().numpy())

        if self.algo == "dqn":
            eps = max(0.5 - 2.0 * (step / total_steps), 0.05)
            # Rainbow DQN sample_action expects (obs, eps, step, ...)
            # We added action_mask support to it.
            action = self.agent.sample_action(
                obs_t, eps=eps, step=step, n_steps=total_steps, action_mask=mask_t
            )
            return action, None
            
        elif self.algo == "ppo":
            # PPO sample_action expects (obs)
            # We added action_mask support to it.
            action, logprob = self.agent.sample_action(obs_t, action_mask=mask_t)
            return action.item(), logprob.item()
            
        elif self.algo == "sac":
            # SAC sample_action expects (obs, deterministic)
            # We added action_mask support to it.
            action = self.agent.sample_action(obs_t, deterministic=deterministic, action_mask=mask_t)
            # SAC returns [act_dim] or [B, act_dim]
            if isinstance(action, np.ndarray):
                return int(action.flatten()[0]), None
            return int(action), None
        
        return None, None

    def _get_probs(self, obs_t, mask_t):
        # Internal helper to extract policy probabilities for tracking
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)
        
        if self.algo == "dqn":
            if hasattr(self.agent, "n_quantiles"):
                # IQN: mean over quantiles
                taus = self.agent._sample_taus(1, self.agent.n_quantiles, self.device)
                q = self.agent.ext_online(obs_t, taus, normalized=True)
                # q shape is [1, 32, 1, 9] for MultiDiscrete or [1, 32, 9] for Discrete
                # We need to mean over quantiles and reduce to [1, n_actions]
                q = q.view(1, self.agent.n_quantiles, -1).mean(dim=1)
            else:
                q = self.agent.ext_online(obs_t, normalized=True)
                q = q.view(1, -1)
            
            q[0][mask_t == 0] = -1e9
            if self.agent.soft or self.agent.munchausen:
                return torch.softmax(q / self.agent.alpha, dim=-1)[0]
            else:
                # Epsilon-greedy approx distribution
                probs = torch.zeros_like(q[0])
                best_act = torch.argmax(q[0])
                eps = self.agent.last_eps
                valid_count = mask_t.sum()
                probs[mask_t == 1] = eps / valid_count
                probs[best_act] += (1.0 - eps)
                return probs
                
        elif self.algo == "ppo":
            logits = self.agent.actor(obs_t)
            logits[0][mask_t == 0] = -1e9
            return torch.softmax(logits, dim=-1)[0]
            
        elif self.algo == "sac":
            # SAC is continuous, distribution tracking is harder. 
            # We'll just return zeros for now or a dummy.
            return torch.zeros_like(mask_t)

    def observe(self, obs, action, reward, next_obs, term, trunc, logprob=None):
        # Flattened obs
        if self.algo == "ppo":
            # PPO observe expects (obs, action, logprob, reward, next_obs, term, trunc, infos)
            # all as batches (num_envs, ...)
            self.agent.observe(
                obs[np.newaxis, ...], 
                np.array([action]), 
                np.array([logprob]), 
                np.array([reward]), 
                next_obs[np.newaxis, ...], 
                np.array([term]), 
                np.array([trunc]), 
                {}
            )
        else:
            # DQN/SAC observe expects (obs, action, reward, next_obs, term, trunc, info)
            # action should be (num_envs, action_dim)
            act_to_store = np.array([[action]]) if self.algo == "dqn" else np.array([[action]]) # SAC might need more but here it's 1D proxy
            self.agent.observe(
                obs[np.newaxis, ...], 
                act_to_store, 
                np.array([reward]), 
                next_obs[np.newaxis, ...], 
                np.array([term]), 
                np.array([trunc]), 
                {}
            )

    def update(self, global_step):
        if self.algo == "ppo":
            return self.agent.update(global_step=global_step)
        else:
            batch_size = getattr(self.args, "dqn_batch_size", 64) if self.algo == "dqn" else getattr(self.args, "batch_size", 64)
            learning_starts = getattr(self.args, "learning_starts", 1000)
            
            # Safety checks for buffer-based agents (DQN, SAC)
            if global_step < learning_starts:
                return None
            if self.agent.buffer.size() < batch_size:
                return None
                
            if self.algo == "dqn":
                return self.agent.update(batch_size=batch_size, step=global_step)
            else:
                return self.agent.update(batch_size=batch_size, global_step=global_step)

def flatten_obs(obs):
    if isinstance(obs, dict):
        return obs["observation"].flatten()
    return obs.flatten()

def train_ma(args):
    device = resolve_torch_device(args.device)
    
    if args.ma_env == "tictactoe":
        env_fn = tictactoe_v3
        obs_dim = 3 * 3 * 2
        n_actions = 9
    else:
        env_fn = leduc_holdem_v4
        obs_dim = 36
        n_actions = 4
        
    mock_env = SimpleNamespace(
        single_observation_space=gym.spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32),
        single_action_space=gym.spaces.Discrete(n_actions),
        num_envs=1
    )
    
    env = env_fn.env()
    
    # Build two agents (self-play)
    agent_raws = []
    for _ in range(len(env.possible_agents)):
        a_raw, _ = build_agent(args, mock_env, device)
        # Apply entropy override if provided (for the unit test)
        if args.ent_coef_override is not None and args.algo == "ppo":
            a_raw.ent_coef = args.ent_coef_override
        agent_raws.append(a_raw)
    
    agents = {
        agent_id: MAAgentWrapper(agent_raws[i], args.algo, device, args)
        for i, agent_id in enumerate(env.possible_agents)
    }
    
    total_steps = 0
    ep_rewards = {agent_id: [] for agent_id in env.possible_agents}
    
    start_time = time.time()
    
    for ep in range(args.total_episodes):
        env.reset()
        
        current_ep_rewards = {agent_id: 0.0 for agent_id in env.possible_agents}
        last_data = {
            agent_id: {"obs": None, "action": None, "logprob": None}
            for agent_id in env.possible_agents
        }
        
        for agent_id in env.agent_iter():
            obs_dict, reward, termination, truncation, info = env.last()
            
            # Accumulate reward for the agent
            current_ep_rewards[agent_id] += reward
            
            # AEC Reward handling: 
            # When it's agent_id's turn, 'reward' is the reward they got 
            # as a result of the PREVIOUS agent's action (or their own previous action).
            # We store the transition for the agent who just acted.
            if last_data[agent_id]["obs"] is not None:
                agents[agent_id].observe(
                    last_data[agent_id]["obs"],
                    last_data[agent_id]["action"],
                    reward,
                    flatten_obs(obs_dict),
                    termination,
                    truncation,
                    logprob=last_data[agent_id]["logprob"]
                )
            
            if termination or truncation:
                action = None
            else:
                obs = flatten_obs(obs_dict)
                mask = obs_dict["action_mask"]
                
                action, logprob = agents[agent_id].get_action(
                    obs, mask, step=total_steps, total_steps=args.total_steps
                )
                
                # Ensure action is a Python scalar for PettingZoo
                if isinstance(action, (np.ndarray, torch.Tensor)):
                    action = int(action.item())
                elif isinstance(action, (list, tuple)):
                    action = int(action[0])
                else:
                    action = int(action)
                
                last_data[agent_id]["obs"] = obs
                last_data[agent_id]["action"] = action
                last_data[agent_id]["logprob"] = logprob
                
            env.step(action)
            total_steps += 1
            
            # Periodic update
            for a in agents.values():
                if args.algo == "ppo":
                    if a.agent.step_idx >= a.agent.num_steps:
                        a.update(total_steps)
                elif total_steps % 4 == 0:
                    a.update(total_steps)
        
        for agent_id in env.possible_agents:
            ep_rewards[agent_id].append(current_ep_rewards[agent_id])
        
        if (ep + 1) % max(1, args.total_episodes // 10) == 0:
            msg = f"Ep {ep+1}/{args.total_episodes} | Steps {total_steps}"
            for agent_id in env.possible_agents:
                avg_r = np.mean(ep_rewards[agent_id][-max(1, args.total_episodes // 10):])
                msg += f" | {agent_id}: {avg_r:.2f}"
            fps = total_steps / (time.time() - start_time)
            msg += f" | FPS {fps:.1f}"
            print(msg)

    env.close()
    return ep_rewards

if __name__ == "__main__":
    args = get_ma_args()
    train_ma(args)
