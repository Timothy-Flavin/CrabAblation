import torch
import unittest
import numpy as np
import gymnasium as gym
from collections import namedtuple
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from learning_algorithms.DQN_Rainbow import EVRainbowDQN
from learning_algorithms.SAC_Rainbow import SACAgent
from learning_algorithms.PG_Rainbow import PPOAgent

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
            self.single_action_space = gym.spaces.Box(low=-1, high=1, shape=(10,))
        else:
            self.single_action_space = gym.spaces.Discrete(10)


BatchData = namedtuple(
    "BatchData", ["observations", "actions", "rewards", "next_observations", "dones"]
)


class TestPopartIntegration(unittest.TestCase):

    def test_dqn_popart_integration(self):
        logging.info("Starting DQN PopArt Integration Test (25 Seeds)")

        def train_dqn(scale, seed, check_stats=False):
            torch.manual_seed(seed)
            np.random.seed(seed)
            agent = EVRainbowDQN(
                input_dim=1,
                n_action_dims=1,
                n_action_bins=10,
                hidden_layer_sizes=[32],
                lr=0.05,
                burn_in_updates=0,
            ).to(torch.device("cpu"))

            batch_size = 32
            obs = torch.ones(batch_size, 1)
            next_obs = torch.ones(batch_size, 1)
            terms = torch.zeros(batch_size)
            orig_sigma = agent.ext_online.output_layer.sigma.item()

            # For exact invariance without optimizer momentum artifacts, we burn in the target buffer directly
            # using a single step with beta=0.0 to immediately set the exponential moving average.
            burn_in_target = torch.full((batch_size, 10), float(scale))
            old_beta = agent.ext_online.output_layer.beta
            agent.ext_online.output_layer.beta = 0.0
            agent.ext_online.output_layer.update_stats(burn_in_target)
            agent.ext_online.output_layer.beta = old_beta

            # Disable the internal ext_r_clip tracker for this test to ensure pure PopArt functionality
            agent.ext_r_clip = float("inf")

            steps_to_converge = 200
            for step in range(1, 200):
                obs = torch.randn(batch_size, 1)
                next_obs = torch.randn(batch_size, 1)
                target_act = (obs > 0).long()

                act = torch.randint(0, 10, (batch_size, 1))
                r = torch.where(
                    act == target_act,
                    torch.tensor(float(scale)),
                    torch.tensor(-float(scale) / 9),
                )

                agent.update(
                    obs,
                    act,
                    r,
                    next_obs,
                    terms,
                    batch_size=None,
                    step=step,
                    extrinsic_only=True,
                )

                if check_stats and step == 10:
                    new_sigma = agent.ext_online.output_layer.sigma.item()
                    self.assertNotEqual(
                        orig_sigma, new_sigma, "DQN PopArt sigma did not update!"
                    )
                    logging.info(
                        f"DQN PopArt Sigma updated from {orig_sigma} to {new_sigma}"
                    )

                with torch.no_grad():
                    test_obs = torch.tensor([[-1.0], [1.0]])
                    test_target = torch.tensor([0, 1])
                    if (agent.ext_online(test_obs).argmax(-1).flatten() == test_target).all():
                        steps_to_converge = step
                        break
            return steps_to_converge

        large_steps = [train_dqn(1e4, s, check_stats=(s == 0)) for s in range(100)]
        small_steps = [train_dqn(1e-4, s, check_stats=False) for s in range(100)]

        large_mean, large_std = np.mean(large_steps), np.std(large_steps)
        small_mean, small_std = np.mean(small_steps), np.std(small_steps)

        logging.info(
            f"DQN Convergence steps -> Large: {large_mean:.2f} +- {large_std:.2f}, Small: {small_mean:.2f} +- {small_std:.2f}"
        )
        self.assertTrue(
            abs(large_mean - small_mean) <= 60,
            f"DQN scale invariance failed: {large_mean} vs {small_mean}",
        )
        self.assertLess(large_mean, 195)
        self.assertLess(small_mean, 195)

    def test_sac_popart_integration(self):
        logging.info("Starting SAC PopArt Integration Test (25 Seeds)")

        def train_sac(scale, seed, check_stats=False):
            torch.manual_seed(seed)
            np.random.seed(seed)
            envs = DummyEnvs(continuous=True)
            agent = SACAgent(
                envs,
                hidden_layer_sizes=(32, 32),
                popart=True,
                distributional=False,
                q_lr=0.05,
                policy_lr=0.01,
            ).to(torch.device("cpu"))

            batch_size = 64
            orig_sigma = agent.qf1.output_layer.sigma.item()

            # Snap PopArt distribution directly to reality
            acts = torch.rand(batch_size, 10) * 2 - 1
            r = scale - scale * torch.mean(torch.abs(acts), dim=-1, keepdim=True)
            old_beta1 = agent.qf1.output_layer.beta
            old_beta2 = agent.qf2.output_layer.beta
            agent.qf1.output_layer.beta = 0.0
            agent.qf2.output_layer.beta = 0.0
            agent.qf1.output_layer.update_stats(r)
            agent.qf2.output_layer.update_stats(r)
            agent.qf1.output_layer.beta = old_beta1
            agent.qf2.output_layer.beta = old_beta2

            steps_to_converge = 200

            for step in range(1, 200):
                obs = torch.ones(batch_size, 1)
                next_obs = torch.ones(batch_size, 1)
                dones = torch.zeros(batch_size, 1)

                acts = torch.rand(batch_size, 10) * 2 - 1
                r = scale - scale * torch.mean(torch.abs(acts), dim=-1, keepdim=True)

                data = BatchData(
                    observations=obs,
                    actions=acts,
                    rewards=r,
                    next_observations=next_obs,
                    dones=dones,
                )
                agent.update(data, step)

                if check_stats and step == 10:
                    new_sigma = agent.qf1.output_layer.sigma.item()
                    self.assertNotEqual(
                        orig_sigma, new_sigma, "SAC PopArt sigma did not update!"
                    )
                    logging.info(
                        f"SAC PopArt Sigma updated from {orig_sigma} to {new_sigma}"
                    )

                with torch.no_grad():
                    mean_action, _ = agent.actor(obs[0:1])
                    if (mean_action.abs() < 0.2).all():
                        steps_to_converge = step
                        break
            return steps_to_converge

        large_steps = [train_sac(1e4, s, check_stats=(s == 0)) for s in range(100)]
        small_steps = [train_sac(1e-4, s, check_stats=False) for s in range(100)]

        large_mean, large_std = np.mean(large_steps), np.std(large_steps)
        small_mean, small_std = np.mean(small_steps), np.std(small_steps)

        logging.info(
            f"SAC Convergence steps -> Large: {large_mean:.2f} +- {large_std:.2f}, Small: {small_mean:.2f} +- {small_std:.2f}"
        )
        # Ensure both converge within 200 steps on average (so they didn't just timeout constantly)
        self.assertLess(large_mean, 195)
        self.assertLess(small_mean, 195)

    def test_ppo_popart_integration(self):
        logging.info("Starting PPO PopArt Integration Test (25 Seeds)")

        def train_ppo(scale, seed, check_stats=False):
            torch.manual_seed(seed)
            np.random.seed(seed)
            envs = DummyEnvs(continuous=False)
            agent = PPOAgent(
                envs,
                learning_rate=0.05,
                hidden_layer_sizes=(32, 32),
                distributional=False,
                popart=True,
                num_envs=1,
                num_steps=64,
                update_epochs=2,
                num_minibatches=1,
            ).to(torch.device("cpu"))

            batch_size = 64
            orig_sigma = agent.ext_critic_head.sigma.item()

            # Snap PopArt distribution directly to reality
            burn_in_target = torch.full((batch_size, 1), float(scale))
            old_beta = agent.ext_critic_head.beta
            agent.ext_critic_head.beta = 0.0
            agent.ext_critic_head.update_stats(burn_in_target)
            agent.ext_critic_head.beta = old_beta

            steps_to_converge = 100

            for step in range(1, 100):
                obs = torch.ones(batch_size, 1)
                next_obs = torch.ones(1, 1)
                dones = torch.zeros(batch_size, 1)
                next_done = torch.zeros(1)

                actions = torch.randint(0, 10, (batch_size, 1))
                r = torch.where(
                    actions == 0,
                    torch.tensor(float(scale)),
                    -torch.tensor(float(scale) / 9),
                )

                with torch.no_grad():
                    _, logprobs, _, ext_values, int_values = (
                        agent.get_action_and_values(obs, actions.flatten())
                    )

                agent.update(
                    obs,
                    actions.flatten(),
                    logprobs,
                    r.flatten(),
                    dones,
                    ext_values.flatten(),
                    int_values.flatten(),
                    next_obs[0],
                    next_done,
                )

                if check_stats and step == 5:
                    new_sigma = agent.ext_critic_head.sigma.item()
                    self.assertNotEqual(
                        orig_sigma, new_sigma, "PPO PopArt sigma did not update!"
                    )
                    logging.info(
                        f"PPO PopArt Sigma updated from {orig_sigma} to {new_sigma}"
                    )

                with torch.no_grad():
                    logits = agent.actor(obs[0:1])
                    probs = torch.softmax(logits, dim=-1)
                    if probs[0, 0].item() > 0.5:
                        steps_to_converge = step
                        break
            return steps_to_converge

        large_steps = [train_ppo(1e4, s, check_stats=(s == 0)) for s in range(100)]
        small_steps = [train_ppo(1e-4, s, check_stats=False) for s in range(100)]

        large_mean, large_std = np.mean(large_steps), np.std(large_steps)
        small_mean, small_std = np.mean(small_steps), np.std(small_steps)

        logging.info(
            f"PPO Convergence steps -> Large: {large_mean:.2f} +- {large_std:.2f}, Small: {small_mean:.2f} +- {small_std:.2f}"
        )
        self.assertLess(large_mean, 95)
        self.assertLess(small_mean, 95)


if __name__ == "__main__":
    unittest.main()
