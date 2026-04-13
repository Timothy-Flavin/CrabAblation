import sys
import os
import torch
import inspect

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from learning_algorithms.DQN_Rainbow import RainbowDQN

def test_off_policy_dynamic_recalc():
    agent = RainbowDQN(input_dim=10, n_action_dims=1, n_action_bins=2, Beta=1.0)
    
    # 1) Verify update signature does not accept pre-computed intrinsic rewards
    # preventing stale RND rewards leaking from the replay buffer
    sig = inspect.signature(agent.update)
    assert 'r_int' not in sig.parameters, "Agent accepts pre-calculated intrinsic returns! This breaks off-policy RND."
    assert 'rnd_err' not in sig.parameters, "Agent accepts pre-calculated RND errors!"

    print("Signature passes, dynamically calculating intrinsic rewards confirmed.")

    # 2) Verify dynamic update mechanically shifts the calculated r_int
    state = torch.zeros((10, 10), dtype=torch.float32)
    state[:, 0] = 1.0
    
    with torch.no_grad():
        r_int_pre = agent._int_reward(torch.zeros(10), agent.rnd(state)).mean().item()
        
    for _ in range(50):
        agent._update_RND(state, batch_norm=False)
        
    with torch.no_grad():
        r_int_post = agent._int_reward(torch.zeros(10), agent.rnd(state)).mean().item()
        
    assert r_int_pre != r_int_post, "Dynamic recalculation is yielding identical results! Normalization/RND update is broken."
    
    print(f"Post-update r_int is {r_int_post} (was {r_int_pre}). Dynamic Recalculation Confirmed.")

if __name__ == "__main__":
    test_off_policy_dynamic_recalc()
