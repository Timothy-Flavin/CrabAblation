import sys
import os
import torch
import numpy as np
import gymnasium as gym

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from learning_algorithms.PG_Rainbow import StandardPPOAgent


class MockSpace:
    def __init__(self, shape):
        self.shape = shape
        self.n = 2


class MockEnv:
    def __init__(self, obs_shape, act_shape):
        self.single_observation_space = gym.spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32)
        self.single_action_space = gym.spaces.Box(low=-1, high=1, shape=act_shape, dtype=np.float32)
        self.num_envs = 10


def test_ppo_rnd_normalization():
    input_dim = 10
    mock_env = MockEnv((input_dim,), (2,))
    
    agent = StandardPPOAgent(envs=mock_env, Beta=1.0, num_envs=10)
    
    huge_state = torch.ones((10, input_dim), dtype=torch.float32) * 1e6

    with torch.no_grad():
        raw_out = agent.rnd(huge_state).mean().item()
        
    # Simulate an observation update filling the normalizers
    agent.update_running_stats(huge_state)
    
    with torch.no_grad():
        norm_huge_state = agent.obs_rms.normalize(huge_state.to(torch.float64)).to(torch.float32)
        norm_out = agent.rnd(norm_huge_state).mean().item()

    print(f"RND raw feed error: {raw_out}")
    print(f"RND fed with internal normalizer rules: {norm_out}")

    assert (
        norm_out != raw_out
    ), "The RND outputs are identical despite massive normalization difference! RND isn't consuming normalized arrays!"
    assert (
        norm_out < raw_out / 100
    ), "The normalizer failed to dramatically rescale the massive inputs before feeding them to RND!"

    print("PPO Observation Normalization before RND passed mathematically.")


if __name__ == "__main__":
    test_ppo_rnd_normalization()