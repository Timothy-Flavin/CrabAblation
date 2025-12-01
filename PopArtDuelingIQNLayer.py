import torch
import torch.nn as nn
import math


class PopArtDuelingIQNLayer(nn.Module):
    """
    PopArt Layer for IQN with Dueling Architecture.
    Combines Implicit Quantile Networks (IQN) with Dueling architecture and PopArt normalization.

    Inputs: [Batch, N_Quantiles, Hidden]
    Outputs: [Batch, N_Quantiles, D, Bins]
    Normalization: Scalar (shared across all outputs) to maintain Dueling constraints easily.
    """

    sigma: torch.Tensor
    mu: torch.Tensor
    nu: torch.Tensor

    def __init__(
        self, in_features, n_action_dims, n_action_bins, beta=0.999, epsilon=1e-5
    ):
        super().__init__()
        self.in_features = in_features
        self.n_action_dims = n_action_dims
        self.n_action_bins = n_action_bins
        self.out_features = n_action_dims * n_action_bins
        self.beta = beta
        self.epsilon = epsilon

        # Dueling Heads
        # Value head outputs scalar value per quantile: [B, N, 1]
        self.value_head = nn.Linear(in_features, 1)
        # Advantage head outputs advantage per action/bin per quantile: [B, N, D*Bins]
        self.adv_head = nn.Linear(in_features, self.out_features)

        # Scalar Statistics (Shared across the whole Q-function)
        # We use scalar stats because Dueling splits Q into V and A.
        # Vector stats would require splitting the shift/scale vector between V (scalar) and A (vector),
        # which is complex. Scalar normalization is standard for Dueling PopArt.
        self.register_buffer("mu", torch.zeros(1))
        self.register_buffer("nu", torch.ones(1))  # E[x^2]
        self.register_buffer("sigma", torch.ones(1))

        self.reset_parameters()

    def reset_parameters(self):
        # Init Value Head
        nn.init.kaiming_uniform_(self.value_head.weight, a=math.sqrt(5))
        if self.value_head.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.value_head.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.value_head.bias, -bound, bound)

        # Init Adv Head
        nn.init.kaiming_uniform_(self.adv_head.weight, a=math.sqrt(5))
        if self.adv_head.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.adv_head.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.adv_head.bias, -bound, bound)

    def forward(self, x, normalized=False):
        """
        Args:
            x: [Batch, N_Quantiles, Hidden]
            normalized: If True, returns normalized Q-values (linear output).
        Returns:
            Q: [Batch, N_Quantiles, D, Bins]
        """
        # Value Stream: [B, N, 1]
        v = self.value_head(x)

        # Advantage Stream: [B, N, D*Bins]
        a = self.adv_head(x)
        # Reshape to [B, N, D, Bins]
        a = a.view(a.shape[0], a.shape[1], self.n_action_dims, self.n_action_bins)

        # Dueling Aggregation: Q = V + (A - mean(A))
        # Mean over action dimensions (D, Bins)
        a_mean = a.mean(dim=(2, 3), keepdim=True)
        q_norm = a - a_mean + v.unsqueeze(-1)  # Broadcast V to [B, N, D, Bins]

        if normalized:
            return q_norm

        # Unnormalize: Q = Q_norm * sigma + mu
        # sigma, mu are scalars, so broadcasting is automatic
        return q_norm * self.sigma + self.mu

    @torch.no_grad()
    def update_stats(self, targets):
        """
        Updates the running statistics based on the targets and adjusts weights/biases
        to preserve the unnormalized output.

        Args:
            targets: Target values (returns). Can be [B, N] or [B, N, D, Bins].
                     We flatten everything to compute scalar stats.
        """
        targets_flat = targets.reshape(-1)

        batch_mean = targets_flat.mean()
        batch_sq_mean = (targets_flat**2).mean()

        # Update running statistics using EMA
        new_mu = self.beta * self.mu + (1 - self.beta) * batch_mean
        new_nu = self.beta * self.nu + (1 - self.beta) * batch_sq_mean

        # Calculate new sigma = sqrt(E[x^2] - E[x]^2)
        new_sigma = torch.sqrt(torch.clamp(new_nu - new_mu**2, min=self.epsilon**2))
        new_sigma = torch.clamp(new_sigma, min=self.epsilon)

        # Update Weights to preserve outputs
        # Q_new = Q_old
        # sigma_new * q_norm_new + mu_new = sigma_old * q_norm_old + mu_old
        # q_norm_new = (sigma_old/sigma_new) * q_norm_old + (mu_old - mu_new)/sigma_new
        # alpha = sigma_old/sigma_new
        # delta = (mu_old - mu_new)/sigma_new

        weight_scale = self.sigma / new_sigma
        bias_shift = (self.mu - new_mu) / new_sigma

        # Update Value Head (Scale + Shift)
        # V' = alpha * V + delta
        self.value_head.weight.mul_(weight_scale)
        self.value_head.bias.mul_(weight_scale).add_(bias_shift)

        # Update Advantage Head (Scale only)
        # A' = alpha * A
        self.adv_head.weight.mul_(weight_scale)
        self.adv_head.bias.mul_(weight_scale)

        # Update Buffers
        self.mu.copy_(new_mu)
        self.nu.copy_(new_nu)
        self.sigma.copy_(new_sigma)

    def normalize(self, x):
        return (x - self.mu) / self.sigma

    def unnormalize(self, x):
        return x * self.sigma + self.mu


if __name__ == "__main__":
    print("Running PopArtDuelingIQNLayer Unit Test...")
    torch.manual_seed(42)

    batch_size = 10
    n_quantiles = 32
    in_dim = 64
    n_action_dims = 2
    n_action_bins = 3

    layer = PopArtDuelingIQNLayer(in_dim, n_action_dims, n_action_bins)

    # Input: [B, N, H]
    x = torch.randn(batch_size, n_quantiles, in_dim)

    # 1. Check Forward Pass
    y_unnorm_before = layer(x, normalized=False)
    assert y_unnorm_before.shape == (
        batch_size,
        n_quantiles,
        n_action_dims,
        n_action_bins,
    )

    # 2. Create targets with shift/scale
    # Targets: [B, N] (e.g. returns)
    targets = torch.randn(batch_size, n_quantiles) * 10.0 + 5.0

    # 3. Update Stats
    layer.update_stats(targets)

    # 4. Check Output Preservation
    y_unnorm_after = layer(x, normalized=False)

    diff = (y_unnorm_before - y_unnorm_after).abs().max().item()
    print(f"Max difference after update: {diff:.6f}")
    assert diff < 1e-5, "Outputs were not preserved!"

    # 5. Check Stats Update
    print(f"Updated Sigma: {layer.sigma.item():.4f}")
    print(f"Updated Mu: {layer.mu.item():.4f}")

    print("PopArtDuelingIQNLayer Test Passed!")
