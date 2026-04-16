import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import unittest
from learning_algorithms.RainbowNetworks import EV_Q_Network, IQN_Network

class TestPreciseOutputPreservation(unittest.TestCase):
    def test_ev_q_network_popart_preservation(self):
        torch.manual_seed(42)
        # Precise Output Preservation for EV_Q_Network
        for dueling in [False, True]:
            net = EV_Q_Network(
                input_dim=10,
                n_action_dims=2,
                n_action_bins=3,
                hidden_layer_sizes=[32],
                dueling=dueling,
                popart=True
            )
            # Dummy batch of states
            dummy_batch = torch.randn(5, 10)
            v_old = net(dummy_batch, normalized=False)
            
            # Manually update PopArt running mean and variance statistics
            # extreme shift and scale
            extreme_targets = torch.randn(10, 6) * 1000 + 5000
            net.output_layer.update_stats(extreme_targets)
            
            v_new = net(dummy_batch, normalized=False)
            
            # Assert V_old == V_new exactly, up to machine precision
            diff = (v_old - v_new).abs().max().item()
            self.assertTrue(diff < 1e-5, f"Outputs not preserved for EV_Q_Network (dueling={dueling}). Max diff: {diff}")

    def test_iqn_network_popart_preservation(self):
        torch.manual_seed(42)
        # Precise Output Preservation for IQN_Network
        for dueling in [False, True]:
            net = IQN_Network(
                input_dim=10,
                n_action_dims=2,
                n_action_bins=3,
                hidden_layer_sizes=[32],
                n_cosines=16,
                dueling=dueling,
                popart=True
            )
            # Dummy batch of states
            dummy_batch = torch.randn(5, 10)
            tau = torch.rand(5, 4)
            v_old = net(dummy_batch, tau, normalized=False)
            
            # Manually update PopArt running mean and variance statistics
            extreme_targets = torch.randn(10, 4, 6) * 1000 + 5000
            net.output_layer.update_stats(extreme_targets)
            
            v_new = net(dummy_batch, tau, normalized=False)
            
            # Assert V_old == V_new exactly, up to machine precision
            diff = (v_old - v_new).abs().max().item()
            self.assertTrue(diff < 1e-5, f"Outputs not preserved for IQN_Network (dueling={dueling}). Max diff: {diff}")

if __name__ == '__main__':
    unittest.main()
