import torch
import unittest
import numpy as np
import gymnasium as gym
from collections import namedtuple
import logging
import os
from learning_algorithms.DQN_Rainbow import EVRainbowDQN
from learning_algorithms.SAC_Rainbow import SACAgent
from learning_algorithms.PG_Rainbow import PPOAgent

# Ensure log dir exists
log_dir = "Unit_Tests/PopArtTests/logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, "popart_integration.log"), level=logging.INFO)

class DummyEnvs:
    def __init__(self, continuous=False):
        self.single_observation_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        if continuous:
            self.single_action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        else:
            self.single_action_space = gym.spaces.Discrete(2)

BatchData = namedtuple("BatchData", ["observations", "actions", "rewards", "next_observations", "dones"])

class TestPopartIntegration(unittest.TestCase):
    
    def test_dqn_popart_integration(self):
        logging.info("Starting DQN PopArt Integration Test")
        def train_dqn(scale, check_stats=False):
            torch.manual_seed(42)
            agent = EVRainbowDQN(
                input_dim=1, n_action_dims=1, n_action_bins=2,
                hidden_layer_sizes=[32], lr=0.05, popart=True, burn_in_updates=0
            ).to(torch.device("cpu"))
            
            batch_size = 32
            obs = torch.ones(batch_size, 1)
            next_obs = torch.ones(batch_size, 1)
            terms = torch.zeros(batch_size)
            orig_sigma = agent.ext_online.output_layer.sigma.item()

            steps_to_converge = -1
            for step in range(1, 200):
                act = torch.randint(0, 2, (batch_size, 1))
                r = torch.where(act[:, 0] == 0, torch.tensor(float(scale)), torch.tensor(-float(scale)))
                
                agent.update(obs, act, r, next_obs, terms, batch_size=None, step=step, extrinsic_only=True)
                
                if check_stats and step == 10:
                    new_sigma = agent.ext_online.output_layer.sigma.item()
                    self.assertNotEqual(orig_sigma, new_sigma, "DQN PopArt sigma did not update!")
                    logging.info(f"DQN PopArt Sigma updated from {orig_sigma} to {new_sigma}")
                
                with torch.no_grad():
                    if (agent.ext_online(obs).argmax(-1) == 0).all():
                        steps_to_converge = step
                        break
            return steps_to_converge

        steps_large = train_dqn(1e4, True)
        steps_small = train_dqn(1e-4, False)
        
        logging.info(f"DQN Convergence steps -> Large: {steps_large}, Small: {steps_small}")
        self.assertTrue(abs(steps_large - steps_small) <= 5, "DQN lack of scale invariance")
        
    def test_sac_popart_integration(self):
        logging.info("Starting SAC PopArt Integration Test")
        def train_sac(scale, check_stats=False):
            torch.manual_seed(42)
            envs = DummyEnvs(continuous=True)
            agent = SACAgent(envs, hidden_layer_sizes=(32, 32), popart=True, distributional=False, q_lr=0.05, policy_lr=0.01).to(torch.device("cpu"))
            
            batch_size = 64
            orig_sigma = agent.qf1.output_layer.sigma.item()
            steps_to_converge = -1
            
            for step in range(1, 200):
                obs = torch.ones(batch_size, 1)
                next_obs = torch.ones(batch_size, 1)
                dones = torch.zeros(batch_size, 1)
                
                # uniform random actions
                acts = torch.rand(batch_size, 1) * 2 - 1
                # Reward is highest closer to action=0
                r = scale - scale * torch.abs(acts)
                
                data = BatchData(observations=obs, actions=acts, rewards=r, next_observations=next_obs, dones=dones)
                agent.update(data, step)
                
                if check_stats and step == 10:
                    new_sigma = agent.qf1.output_layer.sigma.item()
                    self.assertNotEqual(orig_sigma, new_sigma, "SAC PopArt sigma did not update!")
                    logging.info(f"SAC PopArt Sigma updated from {orig_sigma} to {new_sigma}")
                
                with torch.no_grad():
                    mean_action, _ = agent.actor(obs[0:1])
                    if abs(mean_action.item()) < 0.2:
                        steps_to_converge = step
                        break
            return steps_to_converge

        steps_large = train_sac(1e4, True)
        steps_small = train_sac(1e-4, False)
        logging.info(f"SAC Convergence steps -> Large: {steps_large}, Small: {steps_small}")
        self.assertGreater(steps_large, 0)
        self.assertGreater(steps_small, 0)

    def test_ppo_popart_integration(self):
        logging.info("Starting PPO PopArt Integration Test")
        def train_ppo(scale, check_stats=False):
            torch.manual_seed(42)
            envs = DummyEnvs(continuous=False)
            agent = PPOAgent(
                envs, learning_rate=0.05, hidden_layer_sizes=(32, 32),
                distributional=False, popart=True, num_envs=1, num_steps=64, update_epochs=2, num_minibatches=1
            ).to(torch.device("cpu"))
            
            batch_size = 64
            # PPO uses the heads when non-distributional
            orig_sigma = agent.ext_critic_head.sigma.item()
            steps_to_converge = -1
            
            for step in range(1, 100):
                obs = torch.ones(batch_size, 1)
                next_obs = torch.ones(1, 1)
                dones = torch.zeros(batch_size, 1)
                next_done = torch.zeros(1)
                
                # Generate mixed actions
                actions = torch.randint(0, 2, (batch_size, 1))
                r = torch.where(actions == 0, torch.tensor(float(scale)), -torch.tensor(float(scale)))
                
                with torch.no_grad():
                    _, logprobs, _, ext_values, int_values = agent.get_action_and_values(obs, actions.flatten())
                
                agent.update(
                    obs, actions.flatten(), logprobs, r.flatten(), dones, 
                    ext_values.flatten(), int_values.flatten(), next_obs[0], next_done
                )
                
                if check_stats and step == 5:
                    new_sigma = agent.ext_critic_head.sigma.item()
                    self.assertNotEqual(orig_sigma, new_sigma, "PPO PopArt sigma did not update!")
                    logging.info(f"PPO PopArt Sigma updated from {orig_sigma} to {new_sigma}")
                
                # Check policy convergence
                with torch.no_grad():
                    logits = agent.actor(obs[0:1])
                    probs = torch.softmax(logits, dim=-1)
                    if probs[0, 0].item() > 0.8:
                        steps_to_converge = step
                        break
            return steps_to_converge

        steps_large = train_ppo(1e4, True)
        steps_small = train_ppo(1e-4, False)
        
        logging.info(f"PPO Convergence steps -> Large: {steps_large}, Small: {steps_small}")
        self.assertGreater(steps_large, 0)
        self.assertGreater(steps_small, 0)

if __name__ == '__main__':
    unittest.main()
