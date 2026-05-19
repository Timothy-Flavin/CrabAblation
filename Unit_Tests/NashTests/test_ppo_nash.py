import unittest
import numpy as np
import sys
import os

# Add root directory to path to import multiagent_runner
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from multiagent_runner import train_ma, get_ma_args

class TestPPONash(unittest.TestCase):
    def test_tictactoe_ppo_nash(self):
        print("\nStarting PPO Nash Test on Tic-Tac-Toe...")
        args = get_ma_args()
        
        # Override args for the specific test case
        args.algo = "ppo"
        args.ma_env = "tictactoe"
        args.env_name = "tictactoe"
        args.ablation = 6 # No Dist-RL, No RND, Yes GAE, Yes KL Clip
        args.total_episodes = 2000
        args.ent_coef_override = 0.1
        args.device = "cpu"
        
        rewards = train_ma(args)
        r0 = rewards['player_1']
        r1 = rewards['player_2']
        
        # In Tic-Tac-Toe, if agents are learning, they should move towards draws (0 reward)
        # We check the last 500 episodes
        last_500_r0 = np.mean(r0[-500:])
        last_500_r1 = np.mean(r1[-500:])
        
        print(f"Final 500 episodes Avg R0: {last_500_r0:.3f}, Avg R1: {last_500_r1:.3f}")
        
        # They should be relatively close to each other (balanced self-play)
        self.assertLess(abs(last_500_r0 + last_500_r1), 0.1, "Total reward should be zero-sum")
        
    def test_leduc_ppo_nash(self):
        print("\nStarting PPO Nash Test on Leduc Poker...")
        args = get_ma_args()
        
        args.algo = "ppo"
        args.ma_env = "leduc"
        args.env_name = "leduc"
        args.ablation = 6
        args.total_episodes = 1000 # Leduc is harder, but let's see
        args.ent_coef_override = 0.1
        args.device = "cpu"
        
        rewards = train_ma(args)
        # Leduc agents are player_0, player_1
        r0 = rewards['player_0']
        r1 = rewards['player_1']
        
        last_200_r0 = np.mean(r0[-200:])
        last_200_r1 = np.mean(r1[-200:])
        
        print(f"Final 200 episodes Avg R0: {last_200_r0:.3f}, Avg R1: {last_200_r1:.3f}")
        self.assertLess(abs(last_200_r0 + last_200_r1), 0.1)

if __name__ == "__main__":
    unittest.main()
