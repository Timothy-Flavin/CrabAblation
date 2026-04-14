import torch
import unittest
from learning_algorithms.DQN_Rainbow import EVRainbowDQN
import numpy as np

class DummyBanditEnv:
    def __init__(self, scale=1.0):
        self.scale = scale
        self.state = torch.zeros(1, 1) # simple 1D state
        
    def step(self, action):
        # action 0 is optimal
        reward = self.scale if action == 0 else -self.scale
        return self.state, torch.tensor([reward], dtype=torch.float32), torch.tensor([True]), None

class TestScaleInvariance(unittest.TestCase):
    def test_ev_dqn_scale_invariance(self):
        # We will test two scales: 1e4 and 1e-4
        
        def train_agent(scale, use_popart=True):
            torch.manual_seed(42)
            np.random.seed(42)
            agent = EVRainbowDQN(
                input_dim=1,
                n_action_dims=1,
                n_action_bins=2,
                hidden_layer_sizes=[32],
                lr=0.01,  # Needs higher lr
                popart=use_popart,
                burn_in_updates=0 # turn off burn in
            ).to(torch.device("cpu"))
            
            # Simple training loop
            batch_size = 32
            
            obs = torch.ones(batch_size, 1) # simple static state
            next_obs = torch.ones(batch_size, 1)
            terms = torch.zeros(batch_size)
            
            steps_to_converge = -1
            
            for step in range(1, 500):
                # Generate uniform random actions
                act = torch.randint(0, 2, (batch_size, 1))
                r = torch.where(act[:, 0] == 0, torch.tensor(float(scale)), torch.tensor(-float(scale)))
                
                # update agent
                agent.update(
                    obs, act, r, next_obs, terms, batch_size=None, step=step, extrinsic_only=True
                )
                
                # Check greedy action on state (1)
                with torch.no_grad():
                    q_vals = agent.ext_online(obs)
                    greedy_acts = q_vals.argmax(-1)
                    if (greedy_acts == 0).all():
                        steps_to_converge = step
                        break
            return steps_to_converge

        steps_large_pop = train_agent(1e4, True)
        steps_small_pop = train_agent(1e-4, True)
        
        print(f"PopArt Convergence -> Large scale: {steps_large_pop}, Small scale: {steps_small_pop}")
        self.assertGreater(steps_large_pop, -1, "PopArt Agent failed to converge on large scale.")
        self.assertGreater(steps_small_pop, -1, "PopArt Agent failed to converge on small scale.")
        
        # Convergence should be exactly the same or very close with popart
        self.assertTrue(abs(steps_large_pop - steps_small_pop) <= 5, "Convergence steps diverge significantly with PopArt.")

if __name__ == '__main__':
    unittest.main()