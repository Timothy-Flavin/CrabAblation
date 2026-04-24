import sys
import os
import torch
import unittest
import numpy as np
import gymnasium as gym
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from learning_algorithms.DQN_Rainbow import EVRainbowDQN
from learning_algorithms.SAC_Rainbow import EVSAC
from learning_algorithms.PG_Rainbow import StandardPPOAgent

# Ensure log dir exists
log_dir = "Unit_Tests/PopArtTests/logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "popart_integration.log"), level=logging.INFO
)

class DummyEnvs:
    def __init__(self, continuous=False):
        self.single_observation_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        if continuous:
            self.single_action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        else:
            self.single_action_space = gym.spaces.Discrete(10)


class TestPopartIntegration(unittest.TestCase):

    def test_dqn_popart_integration(self):
        logging.info("Starting DQN PopArt Integration Test (25 Seeds)")

        def train_dqn(scale, seed, check_stats=False):
            torch.manual_seed(seed)
            np.random.seed(seed)
            batch_size = 32
            
            agent = EVRainbowDQN(
                input_dim=1,
                n_action_dims=1,
                n_action_bins=10,
                n_envs=batch_size,
                hidden_layer_sizes=[32],
                lr=0.005, # Lowered to ensure a smooth learning curve
                burn_in_updates=0,
                min_std=1e-8,
            ).to(torch.device("cpu"))

            orig_sigma = agent.ext_online.output_layer.sigma.item()

            burn_in_target = torch.full((batch_size, 10), float(scale))
            old_beta = agent.ext_online.output_layer.beta
            agent.ext_online.output_layer.beta = 0.0
            agent.ext_online.output_layer.update_stats(burn_in_target)
            agent.ext_online.output_layer.beta = old_beta

            agent.ext_r_clip = float("inf")

            steps_to_converge = 200
            for step in range(1, 200):
                # 2-State MDP: obs is randomly -1.0 or 1.0
                obs = np.random.choice([-1.0, 1.0], size=(batch_size, 1)).astype(np.float32)
                next_obs = np.random.choice([-1.0, 1.0], size=(batch_size, 1)).astype(np.float32)
                
                # target_act is 0 if obs is -1.0, and 1 if obs is 1.0
                target_act = (obs > 0).astype(np.int64)

                act_tensor = agent.sample_action(torch.tensor(obs), eps=0.2, step=step, n_steps=200)
                act = np.array(act_tensor, dtype=np.int32).reshape(batch_size, 1)
                
                r = np.where(act == target_act, float(scale), -float(scale) / 9).astype(np.float32).flatten()
                
                terms = np.zeros(batch_size, dtype=bool)
                truncs = np.zeros(batch_size, dtype=bool)

                agent.observe(obs, act, r, next_obs, terms, truncs, {})
                agent.update(batch_size=batch_size, step=step)

                if check_stats and step == 10:
                    new_sigma = agent.ext_online.output_layer.sigma.item()
                    self.assertNotEqual(orig_sigma, new_sigma, "DQN PopArt sigma did not update!")

                with torch.no_grad():
                    test_obs = torch.tensor([[-1.0], [1.0]], dtype=torch.float32)
                    test_target = torch.tensor([0, 1])
                    # Win condition requires learning the decision boundary for BOTH states
                    if (agent.ext_online(test_obs).argmax(-1).flatten() == test_target).all():
                        steps_to_converge = step
                        break
            return steps_to_converge

        large_steps = [train_dqn(1e4, s, check_stats=(s == 0)) for s in range(50)]
        small_steps = [train_dqn(1e-4, s, check_stats=False) for s in range(50)]

        large_mean, large_std = np.mean(large_steps), np.std(large_steps)
        small_mean, small_std = np.mean(small_steps), np.std(small_steps)

        logging.info(f"DQN Convergence steps -> Large: {large_mean:.2f} +- {large_std:.2f}, Small: {small_mean:.2f} +- {small_std:.2f}")
        self.assertTrue(abs(large_mean - small_mean) <= 15, f"DQN scale invariance failed: {large_mean} vs {small_mean}")

    def test_sac_popart_integration(self):
        logging.info("Starting SAC PopArt Integration Test (25 Seeds)")
        MAX_STEPS = 200

        def train_sac(scale, seed, check_stats=False):
            torch.manual_seed(seed)
            np.random.seed(seed)
            envs = DummyEnvs(continuous=True)
            batch_size = 64
            envs.num_envs = batch_size
            
            agent = EVSAC(
                envs,
                hidden_layer_sizes=(32, 32),
                q_lr=0.005,
                policy_lr=0.005,
                min_std=1e-8,
                gamma=0.0,
                munchausen=False,
                beta_rnd=0.0,
                entropy_coef_zero=True,
            ).to(torch.device("cpu"))

            orig_sigma = agent.qf1.output_layer.sigma.item()

            acts = torch.rand(batch_size, 1) * 2 - 1
            r = scale - scale * torch.mean(torch.abs(acts), dim=-1, keepdim=True)
            old_beta1, old_beta2 = agent.qf1.output_layer.beta, agent.qf2.output_layer.beta
            agent.qf1.output_layer.beta, agent.qf2.output_layer.beta = 0.0, 0.0
            agent.qf1.output_layer.update_stats(r)
            agent.qf2.output_layer.update_stats(r)
            agent.qf1.output_layer.beta, agent.qf2.output_layer.beta = old_beta1, old_beta2

            sigma_after_burn_in = agent.qf1.output_layer.sigma.item()

            steps_to_converge = MAX_STEPS

            for step in range(1, MAX_STEPS):
                obs = np.random.choice([-1.0, 1.0], size=(batch_size, 1)).astype(np.float32)
                next_obs = np.random.choice([-1.0, 1.0], size=(batch_size, 1)).astype(np.float32)
                
                # Target is to output +0.5 for +1.0 obs, and -0.5 for -1.0 obs
                target_act = (obs * 0.5).astype(np.float32)
                
                acts = agent.sample_action(obs)
                r = scale - scale * np.abs(acts - target_act).flatten()
                
                terms = np.zeros(batch_size, dtype=bool)
                truncs = np.zeros(batch_size, dtype=bool)

                agent.observe(obs, acts, r, next_obs, terms, truncs, {})
                agent.update(batch_size=batch_size, global_step=step)

                if check_stats and step == 10:
                    new_sigma = agent.qf1.output_layer.sigma.item()
                    self.assertNotEqual(orig_sigma, new_sigma, "SAC PopArt sigma did not update!")

                with torch.no_grad():
                    test_obs = torch.tensor([[-1.0], [1.0]], dtype=torch.float32)
                    mean_action, _ = agent.actor(test_obs)
                    # Must correctly output negatives for State A, and positives for State B
                    if mean_action[0, 0].item() < -0.3 and mean_action[1, 0].item() > 0.3:
                        steps_to_converge = step
                        break

            return steps_to_converge, sigma_after_burn_in

        large_results = [train_sac(1e4, s, check_stats=(s == 0)) for s in range(50)]
        small_results = [train_sac(1e-4, s, check_stats=(s == 0)) for s in range(50)]

        large_steps, large_sigmas = [r[0] for r in large_results], [r[1] for r in large_results]
        small_steps, small_sigmas = [r[0] for r in small_results], [r[1] for r in small_results]

        sigma_ratio = np.mean(large_sigmas) / np.mean(small_sigmas)
        logging.info(f"SAC sigma ratio large/small = {sigma_ratio:.3e} (expected ~1e8)")
        self.assertGreater(sigma_ratio, 1e6)

        large_mean, large_std = np.mean(large_steps), np.std(large_steps)
        small_mean, small_std = np.mean(small_steps), np.std(small_steps)

        logging.info(f"SAC Convergence steps -> Large: {large_mean:.2f} +- {large_std:.2f}, Small: {small_mean:.2f} +- {small_std:.2f}")
        self.assertTrue(abs(large_mean - small_mean) <= 15)

    def test_ppo_popart_integration(self):
        logging.info("Starting PPO PopArt Integration Test (25 Seeds)")

        def train_ppo(scale, seed, check_stats=False):
            torch.manual_seed(seed)
            np.random.seed(seed)
            envs = DummyEnvs(continuous=False)
            batch_size = 64
            envs.num_envs = batch_size
            
            agent = StandardPPOAgent(
                envs,
                learning_rate=0.005,
                hidden_layer_sizes=(32, 32),
                popart=True,
                num_envs=batch_size,
                num_steps=1,
                update_epochs=2,
                num_minibatches=1,
                Beta=0.0
            ).to(torch.device("cpu"))

            orig_sigma = agent.ext_critic_head.sigma.item()

            burn_in_target = torch.full((batch_size, 1), float(scale))
            old_beta = agent.ext_critic_head.beta
            agent.ext_critic_head.beta = 0.0
            agent.ext_critic_head.update_stats(burn_in_target)
            agent.ext_critic_head.beta = old_beta

            steps_to_converge = 100

            for step in range(1, 100):
                obs = np.random.choice([-1.0, 1.0], size=(batch_size, 1)).astype(np.float32)
                next_obs = np.random.choice([-1.0, 1.0], size=(batch_size, 1)).astype(np.float32)
                target_act = (obs > 0).astype(np.int32).flatten()

                with torch.no_grad():
                    action_t, logprobs_t, _, _, _ = agent.get_action_and_values(torch.tensor(obs))
                
                actions = action_t.numpy()
                logprobs = logprobs_t.numpy()

                r = np.where(actions == target_act, float(scale), -float(scale) / 9).astype(np.float32)
                
                terms = np.zeros(batch_size, dtype=bool)
                truncs = np.zeros(batch_size, dtype=bool)

                agent.observe(obs, actions, logprobs, r, next_obs, terms, truncs, {})
                agent.update(global_step=step)

                if check_stats and step == 5:
                    new_sigma = agent.ext_critic_head.sigma.item()
                    self.assertNotEqual(orig_sigma, new_sigma, "PPO PopArt sigma did not update!")

                with torch.no_grad():
                    test_obs = torch.tensor([[-1.0], [1.0]], dtype=torch.float32)
                    logits = agent.actor(test_obs)
                    probs = torch.softmax(logits, dim=-1)
                    # Win condition requires 85% confidence on BOTH sides of the decision boundary
                    if probs[0, 0].item() > 0.85 and probs[1, 1].item() > 0.85: 
                        steps_to_converge = step
                        break
            return steps_to_converge

        large_steps = [train_ppo(1e4, s, check_stats=(s == 0)) for s in range(50)]
        small_steps = [train_ppo(1e-4, s, check_stats=False) for s in range(50)]

        large_mean, large_std = np.mean(large_steps), np.std(large_steps)
        small_mean, small_std = np.mean(small_steps), np.std(small_steps)

        logging.info(f"PPO Convergence steps -> Large: {large_mean:.2f} +- {large_std:.2f}, Small: {small_mean:.2f} +- {small_std:.2f}")
        self.assertTrue(abs(large_mean - small_mean) <= 15)

if __name__ == "__main__":
    unittest.main()