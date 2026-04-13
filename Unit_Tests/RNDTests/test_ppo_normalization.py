import sys
import os
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from learning_algorithms.PG_Rainbow import PPOAgent

class MockSpace:
    def __init__(self, shape):
        self.shape = shape
        self.n = 2

class MockEnv:
    def __init__(self, obs_shape, act_shape):
        self.single_observation_space = MockSpace(obs_shape)
        self.single_action_space = MockSpace(act_shape)
        self.num_envs = 10

def test_ppo_rnd_normalization():
    input_dim = 10
    mock_env = MockEnv((input_dim,), (2,))
    
    agent = PPOAgent(envs=mock_env, Beta=1.0)
    
    huge_state = torch.ones((10, input_dim), dtype=torch.float32) * 1e6
    
    with torch.no_grad():
        raw_out = agent.rnd(huge_state).mean()
        
    agent.update_running_stats(huge_state, r=torch.zeros(10))
    
    with torch.no_grad():
        norm_huge_state = agent.obs_rms.normalize(huge_state.to(torch.float64)).to(torch.float32)
        norm_out = agent.rnd(norm_huge_state).mean()

    print(f"RND raw feed error: {raw_out}")
    print(f"RND fed with internal normalizer rules: {norm_out}")
    
    assert norm_out != raw_out, "The RND outputs are identical despite massive normalization difference! RND isn't consuming normalized arrays!"
    assert norm_out < raw_out / 100, "The normalizer failed to dramatically rescale the massive inputs before feeding them to RND!"
    
    print("PPO Observation Normalization before RND passed mathematically.")

if __name__ == "__main__":
    test_ppo_rnd_normalization()
