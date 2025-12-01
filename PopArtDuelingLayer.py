import torch
import torch.nn as nn


class PopArtDuelingHead(nn.Module):

    mu: torch.Tensor
    nu: torch.Tensor
    sigma: torch.Tensor

    def __init__(self, input_features, action_dim, beta=0.0005):
        super().__init__()
        self.beta = beta
        self.action_dim = action_dim

        # Two separate heads for Value and Advantage
        # Note: We manually manage weights to apply PopArt updates easily
        self.fc_V = nn.Linear(input_features, 1)
        self.fc_A = nn.Linear(input_features, action_dim)

        # PopArt Statistics (Shared across the whole Q-function)
        self.register_buffer("mu", torch.zeros(1))
        self.register_buffer("nu", torch.ones(1))  # E[x^2]
        self.register_buffer("sigma", torch.ones(1))

    def forward(self, x, normalize=False):
        # 1. Forward pass through heads (Normalized Space)
        v_norm = self.fc_V(x)
        a_norm = self.fc_A(x)

        # 2. Dueling Aggregation (Standard)
        # Q_norm = V_norm + (A_norm - mean(A_norm))
        q_norm = v_norm + (a_norm - a_norm.mean(dim=1, keepdim=True))

        if normalize:
            return q_norm

        # 3. Denormalize on the fly for action selection / targets
        return q_norm * self.sigma + self.mu

    def update_stats_and_weights(self, target_q_unnormalized):
        """
        Updates mu/sigma and adjusts V and A weights to preserve outputs.
        """
        # --- 1. Update Statistics (Same as standard PopArt) ---
        old_mu = self.mu.clone()
        old_sigma = self.sigma.clone()

        batch_mean = torch.mean(target_q_unnormalized)
        batch_sq_mean = torch.mean(target_q_unnormalized**2)

        new_mu = (1 - self.beta) * old_mu + self.beta * batch_mean
        new_nu = (1 - self.beta) * self.nu + self.beta * batch_sq_mean

        # Calculate new sigma = sqrt(E[x^2] - E[x]^2)
        new_sigma = torch.sqrt(torch.clamp(new_nu - new_mu**2, min=1e-4**2))
        new_sigma = torch.clamp(new_sigma, min=1e-4, max=1e6)

        self.mu.copy_(new_mu)
        self.nu.copy_(new_nu)
        self.sigma.copy_(new_sigma)

        # --- 2. Update Weights (The Dueling Logic) ---

        # Calculate ratios
        # We want: new_sigma * new_out + new_mu == old_sigma * old_out + old_mu
        weight_scale = old_sigma / new_sigma
        bias_shift = (old_mu - new_mu) / new_sigma  # Only applies to V

        with torch.no_grad():
            # A. Update ADVANTAGE Stream (Scale only, NO shift)
            # Advantages are relative; they scale with reward magnitude but ignore the mean.
            self.fc_A.weight.mul_(weight_scale)
            self.fc_A.bias.mul_(weight_scale)

            # B. Update VALUE Stream (Scale AND Shift)
            # The Value stream carries the "Base" reward level (Mu).
            self.fc_V.weight.mul_(weight_scale)

            # The math for bias:
            # new_bias = (old_sigma * old_bias + old_mu - new_mu) / new_sigma
            # This simplifies to: old_bias * ratio + shift_term
            self.fc_V.bias.mul_(weight_scale).add_(bias_shift)
