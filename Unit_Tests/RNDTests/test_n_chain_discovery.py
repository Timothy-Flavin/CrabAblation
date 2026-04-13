import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from learning_algorithms.DQN_Rainbow import RainbowDQN

def test_n_chain_discovery():
    input_dim = 10
    agent = RainbowDQN(
        input_dim=input_dim,
        n_action_dims=1,
        n_action_bins=2,
        Beta=1.0,
        norm_obs=False 
    )

    # State 0 is repeatedly visited
    state_0 = torch.zeros((10, input_dim), dtype=torch.float32)
    state_0[:, 0] = 1.0

    state_1 = torch.zeros((10, input_dim), dtype=torch.float32)
    state_1[:, 1] = 1.0

    # Burn-in RND on state 0 (supplying batch_norm=False directly to avoid unit test wiping out variance)
    for _ in range(500):
        agent._update_RND(state_0, batch_norm=False)

    # Now verify the raw output of the RND model
    with torch.no_grad():
        r_int_0 = agent.rnd(state_0).mean().item()
        r_int_1 = agent.rnd(state_1).mean().item()

    print(f"RND Error on familiar state 0: {r_int_0}")
    print(f"RND Error on novel state 1: {r_int_1}")
    
    assert r_int_1 > r_int_0 * 5, f"RND reward failed to spike for novel state: familiar {r_int_0}, novel {r_int_1}"
    print("N-Chain Discovery Test Passed!")

    # Check Visitation Decay Test
    for _ in range(500):
        agent._update_RND(state_1, batch_norm=False)
        
    with torch.no_grad():
        r_int_1_decayed = agent.rnd(state_1).mean().item()
        
    print(f"RND Error on state 1 after decay: {r_int_1_decayed}")
    assert r_int_1_decayed < r_int_1 / 5, "Visitation Decay Test failed! Extrinsic decay didn't apply."
    print("Visitation Decay Test Passed!")

if __name__ == "__main__":
    test_n_chain_discovery()
    
def test_full_chain():
    input_dim = 10
    agent = RainbowDQN(
        input_dim=input_dim,
        n_action_dims=1,
        n_action_bins=2,
        Beta=1.0,
        norm_obs=False 
    )
    
    # We step through states 0 to 9 in the chain. Each new state should have huge RND compared to the current known state
    for i in range(input_dim - 1):
        state_current = torch.zeros((10, input_dim), dtype=torch.float32)
        state_current[:, i] = 1.0
        
        state_next = torch.zeros((10, input_dim), dtype=torch.float32)
        state_next[:, i+1] = 1.0
        
        # Train until state_current RND decays
        for _ in range(300):
            agent._update_RND(state_current, batch_norm=False)
            
        with torch.no_grad():
            r_int_known = agent.rnd(state_current).mean().item()
            r_int_novel = agent.rnd(state_next).mean().item()
            
        assert r_int_novel > r_int_known * 5, f"At chain step {i}, novel state {i+1} didn't spike."
    print("Full N-Chain progression passed!")

if __name__ == "__main__":
    test_full_chain()
