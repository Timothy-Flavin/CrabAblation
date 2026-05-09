import sys
import os
import torch
import unittest
import numpy as np
import gymnasium as gym
import logging

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

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

def plot_popart_stats(title, filename, large_mu, large_sigma, small_mu, small_sigma):
    if not MATPLOTLIB_AVAILABLE:
        logging.warning("Matplotlib not available. Skipping plotting.")
        return
    plt.figure(figsize=(12, 5))
    
    # Scale small metrics by 1e8
    small_mu_scaled = [m * 1e8 for m in small_mu]
    small_sigma_scaled = [s * 1e8 for s in small_sigma]
    
    plt.subplot(1, 2, 1)
    plt.plot(large_mu, label='Large Scale (1e4)')
    plt.plot(small_mu_scaled, label='Small Scale (1e-4) * 1e8')
    plt.title(f'{title} - Mu Tracking')
    plt.xlabel('Steps')
    plt.ylabel('Mu')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(large_sigma, label='Large Scale (1e4)')
    plt.plot(small_sigma_scaled, label='Small Scale (1e-4) * 1e8')
    plt.yscale('log')
    plt.title(f'{title} - Sigma Tracking')
    plt.xlabel('Steps')
    plt.ylabel('Sigma')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, filename))
    plt.close()

class DummyEnvs:
    def __init__(self, continuous=False):
        self.single_observation_space = gym.spaces.Box(low=-1, high=1, shape=(5,))
        if continuous:
            self.single_action_space = gym.spaces.Box(low=-1, high=1, shape=(3,))
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
                input_dim=5,
                n_action_dims=1,
                n_action_bins=10,
                n_envs=batch_size,
                hidden_layer_sizes=[32],
                lr=0.005,
                burn_in_updates=100,
                min_std=1e-8,
            ).to(torch.device("cpu"))

            orig_sigma = agent.ext_online.output_layer.sigma.item()
            
            mu_hist = []
            sigma_hist = []

            # Burn in period using random actions
            for step in range(100):
                obs = np.random.uniform(-1, 1, (batch_size, 5)).astype(np.float32)
                next_obs = np.random.uniform(-1, 1, (batch_size, 5)).astype(np.float32)
                act = np.random.randint(0, 10, (batch_size, 1)).astype(np.int32)
                
                r = np.where(
                    act.flatten() == 4,
                    np.random.normal(scale, scale*0.1, size=batch_size),
                    np.random.normal(-scale / 9, scale*0.1, size=batch_size)
                ).astype(np.float32).flatten()
                
                terms = np.zeros(batch_size, dtype=bool)
                truncs = np.zeros(batch_size, dtype=bool)
                agent.observe(obs, act, r, next_obs, terms, truncs, {})
                agent.update(batch_size=batch_size, step=step)
                
                mu_hist.append(agent.ext_online.output_layer.mu.item())
                sigma_hist.append(agent.ext_online.output_layer.sigma.item())

            agent.ext_target.load_state_dict(agent.ext_online.state_dict())
            agent.ext_r_clip = float("inf")

            steps_to_converge = 400
            for step in range(100, 400):
                obs = np.random.uniform(-1, 1, (batch_size, 5)).astype(np.float32)
                next_obs = np.random.uniform(-1, 1, (batch_size, 5)).astype(np.float32)
                act = np.random.randint(0, 10, (batch_size, 1)).astype(np.int32)
                r = np.where(
                    act.flatten() == 4,
                    np.random.normal(scale, scale*0.1, size=batch_size),
                    np.random.normal(-scale / 9, scale*0.1, size=batch_size)
                ).astype(np.float32).flatten()
                
                terms = np.zeros(batch_size, dtype=bool)
                truncs = np.zeros(batch_size, dtype=bool)
                
                agent.observe(obs, act, r, next_obs, terms, truncs, {})
                agent.update(batch_size=batch_size, step=step)
                
                mu_hist.append(agent.ext_online.output_layer.mu.item())
                sigma_hist.append(agent.ext_online.output_layer.sigma.item())

                if check_stats and step == 130:
                    new_sigma = agent.ext_online.output_layer.sigma.item()
                    self.assertNotEqual(
                        orig_sigma, new_sigma, "DQN PopArt sigma did not update!"
                    )
                    logging.info(
                        f"DQN PopArt Sigma updated from {orig_sigma} to {new_sigma}"
                    )

                with torch.no_grad():
                    test_obs = torch.tensor(np.random.uniform(-1, 1, (2, 5)).astype(np.float32))
                    test_target = torch.tensor([4, 4])
                    if (agent.ext_online(test_obs).argmax(-1).flatten() == test_target).all():
                        steps_to_converge = step
                        break
                        
            return steps_to_converge, mu_hist, sigma_hist

        large_results = [train_dqn(1e4, s, check_stats=(s == 0)) for s in range(100)]
        small_results = [train_dqn(1e-4, s, check_stats=False) for s in range(100)]
        
        large_steps = [r[0] for r in large_results]
        small_steps = [r[0] for r in small_results]

        plot_popart_stats("DQN PopArt", "dqn_popart_stats.png", large_results[0][1], large_results[0][2], small_results[0][1], small_results[0][2])

        large_mean, large_std = np.mean(large_steps), np.std(large_steps)
        small_mean, small_std = np.mean(small_steps), np.std(small_steps)

        logging.info(
            f"DQN Convergence steps -> Large: {large_mean:.2f} +- {large_std:.2f}, Small: {small_mean:.2f} +- {small_std:.2f}"
        )
        self.assertTrue(
            abs(large_mean - small_mean) <= 30,
            f"DQN scale invariance failed: {large_mean} vs {small_mean}",
        )
        self.assertLess(large_mean, 395)
        self.assertLess(small_mean, 395)

    def test_sac_popart_integration(self):
        logging.info("Starting SAC PopArt Integration Test (25 Seeds)")

        MAX_STEPS = 700

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
            
            mu_hist = []
            sigma_hist = []

            for step in range(200):
                obs = np.random.uniform(-1, 1, (batch_size, 5)).astype(np.float32)
                next_obs = np.random.uniform(-1, 1, (batch_size, 5)).astype(np.float32)
                
                acts = np.random.uniform(-1, 1, (batch_size, 3)).astype(np.float32)
                n_in_range = np.sum((acts > -0.1) & (acts < 0.1), axis=-1)
                n_out_range = 3 - n_in_range
                r_mean = scale * n_in_range - (1/5) * scale * n_out_range
                r = np.random.normal(r_mean, scale * 0.1).astype(np.float32)
                
                terms = np.zeros(batch_size, dtype=bool)
                truncs = np.zeros(batch_size, dtype=bool)

                agent.observe(obs, acts, r, next_obs, terms, truncs, {})
                agent.update(batch_size=batch_size, global_step=step)
                
                mu_hist.append(agent.qf1.output_layer.mu.item())
                sigma_hist.append(agent.qf1.output_layer.sigma.item())

            agent.qf1_target.load_state_dict(agent.qf1.state_dict())
            agent.qf2_target.load_state_dict(agent.qf2.state_dict())

            sigma_after_burn_in = agent.qf1.output_layer.sigma.item()
            if check_stats:
                logging.info(
                    f"SAC scale={scale:.0e} burn-in sigma={sigma_after_burn_in:.3e}"
                )

            steps_to_converge = MAX_STEPS

            for step in range(200, MAX_STEPS):
                obs = np.random.uniform(-1, 1, (batch_size, 5)).astype(np.float32)
                next_obs = np.random.uniform(-1, 1, (batch_size, 5)).astype(np.float32)
                
                acts = np.random.uniform(-1, 1, (batch_size, 3)).astype(np.float32)
                n_in_range = np.sum((acts > -0.1) & (acts < 0.1), axis=-1)
                n_out_range = 3 - n_in_range
                r_mean = scale * n_in_range - (1/5) * scale * n_out_range
                r = np.random.normal(r_mean, scale * 0.1).astype(np.float32)
                
                terms = np.zeros(batch_size, dtype=bool)
                truncs = np.zeros(batch_size, dtype=bool)

                agent.observe(obs, acts, r, next_obs, terms, truncs, {})
                agent.update(batch_size=batch_size, global_step=step)
                
                mu_hist.append(agent.qf1.output_layer.mu.item())
                sigma_hist.append(agent.qf1.output_layer.sigma.item())

                if check_stats and step == 210:
                    new_sigma = agent.qf1.output_layer.sigma.item()
                    self.assertNotEqual(
                        orig_sigma, new_sigma, "SAC PopArt sigma did not update!"
                    )
                    logging.info(
                        f"SAC scale={scale:.0e} PopArt sigma: {orig_sigma:.3e} -> {new_sigma:.3e}"
                    )

                with torch.no_grad():
                    mean_action, _ = agent.actor(torch.tensor(np.random.uniform(-1, 1, (20, 5)).astype(np.float32)))
                    in_range = (mean_action > -0.1) & (mean_action < 0.1)
                    if in_range.float().mean().item() > 0.9:
                        steps_to_converge = step
                        break

            return steps_to_converge, sigma_after_burn_in, mu_hist, sigma_hist

        large_results = [train_sac(1e4, s, check_stats=(s == 0)) for s in range(100)]
        small_results = [train_sac(1e-4, s, check_stats=(s == 0)) for s in range(100)]

        large_steps = [r[0] for r in large_results]
        small_steps = [r[0] for r in small_results]
        large_sigmas = [r[1] for r in large_results]
        small_sigmas = [r[1] for r in small_results]
        
        plot_popart_stats("SAC PopArt", "sac_popart_stats.png", large_results[0][2], large_results[0][3], small_results[0][2], small_results[0][3])

        sigma_ratio = np.mean(large_sigmas) / np.mean(small_sigmas)
        logging.info(
            f"SAC sigma ratio large/small = {sigma_ratio:.3e} (expected ~1e8)"
        )
        self.assertGreater(
            sigma_ratio, 1e6,
            f"SAC PopArt sigma ratio {sigma_ratio:.3e} too small — "
            "large and small scale agents may be using the same reward scale"
        )

        large_mean, large_std = np.mean(large_steps), np.std(large_steps)
        small_mean, small_std = np.mean(small_steps), np.std(small_steps)

        logging.info(
            f"SAC Convergence steps -> Large: {large_mean:.2f} +- {large_std:.2f}, Small: {small_mean:.2f} +- {small_std:.2f}"
        )
        self.assertTrue(
            abs(large_mean - small_mean) <= 50,
            f"SAC scale invariance failed: {large_mean} vs {small_mean}",
        )
        self.assertLess(large_mean, MAX_STEPS * 0.97)
        self.assertLess(small_mean, MAX_STEPS * 0.97)

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

            steps_to_converge = 200
            
            mu_hist = []
            sigma_hist = []

            for step in range(1, 200):
                obs = np.random.uniform(-1, 1, (batch_size, 5)).astype(np.float32)
                next_obs = np.random.uniform(-1, 1, (batch_size, 5)).astype(np.float32)
                
                actions = np.random.randint(0, 10, batch_size).astype(np.int32)
                r = np.where(
                    actions == 4,
                    np.random.normal(scale, scale*0.1, size=batch_size),
                    np.random.normal(-scale / 9, scale*0.1, size=batch_size)
                ).astype(np.float32)
                
                terms = np.zeros(batch_size, dtype=bool)
                truncs = np.zeros(batch_size, dtype=bool)
                infos = {}

                with torch.no_grad():
                    _, logprobs, _, _, _ = agent.get_action_and_values(
                        torch.tensor(obs), torch.tensor(actions)
                    )

                agent.observe(
                    obs, actions, logprobs.numpy(), r, next_obs, terms, truncs, infos
                )
                agent.update(global_step=step)
                
                mu_hist.append(agent.ext_critic_head.mu.item())
                sigma_hist.append(agent.ext_critic_head.sigma.item())

                if check_stats and step == 5:
                    new_sigma = agent.ext_critic_head.sigma.item()
                    self.assertNotEqual(
                        orig_sigma, new_sigma, "PPO PopArt sigma did not update!"
                    )
                    logging.info(
                        f"PPO PopArt Sigma updated from {orig_sigma} to {new_sigma}"
                    )

                with torch.no_grad():
                    test_obs = torch.tensor(np.random.uniform(-1, 1, (2, 5)).astype(np.float32))
                    logits = agent.actor(test_obs)
                    probs = torch.softmax(logits, dim=-1)
                    if probs[0, 4].item() > 0.8 and probs[1, 4].item() > 0.8:
                        steps_to_converge = step
                        break
                        
            return steps_to_converge, mu_hist, sigma_hist

        large_results = [train_ppo(1e4, s, check_stats=(s == 0)) for s in range(100)]
        small_results = [train_ppo(1e-4, s, check_stats=False) for s in range(100)]

        large_steps = [r[0] for r in large_results]
        small_steps = [r[0] for r in small_results]
        
        plot_popart_stats("PPO PopArt", "ppo_popart_stats.png", large_results[0][1], large_results[0][2], small_results[0][1], small_results[0][2])

        large_mean, large_std = np.mean(large_steps), np.std(large_steps)
        small_mean, small_std = np.mean(small_steps), np.std(small_steps)

        logging.info(
            f"PPO Convergence steps -> Large: {large_mean:.2f} +- {large_std:.2f}, Small: {small_mean:.2f} +- {small_std:.2f}"
        )
        self.assertLess(large_mean, 195)
        self.assertLess(small_mean, 195)


if __name__ == "__main__":
    unittest.main()