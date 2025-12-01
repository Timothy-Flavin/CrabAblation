import torch
import torch.nn as nn
import math


class PopArtIQNLayer(nn.Module):
    """
    PopArt Layer for Implicit Quantile Networks (IQN).
    Handles multi-dimensional inputs [Batch, N_Quantiles, Hidden] and outputs [Batch, N_Quantiles, Out_Features].
    Updates statistics based on the mean over quantiles (expected value).
    """

    sigma: torch.Tensor
    mu: torch.Tensor
    nu: torch.Tensor

    def __init__(self, in_features, out_features, beta=0.999, epsilon=1e-5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.beta = beta
        self.epsilon = epsilon

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

        # Running statistics for the targets (mean and mean-squared)
        # Shape: [out_features]
        self.register_buffer("mu", torch.zeros(out_features))
        self.register_buffer("nu", torch.ones(out_features))  # E[x^2]
        self.register_buffer("sigma", torch.ones(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, normalized=False):
        """
        Forward pass.
        Args:
            x: Input tensor of shape [Batch, N_Quantiles, In_Features] or [Batch, In_Features].
            normalized: If True, returns the normalized output (raw linear projection).
                        If False, returns the unnormalized output (scaled and shifted).
        Returns:
            Output tensor of shape [Batch, N_Quantiles, Out_Features] (or [Batch, Out_Features]).
        """
        # x shape: [B, N, H] or [B, H]
        # Linear layer applies to the last dimension
        out_norm = torch.nn.functional.linear(x, self.weight, self.bias)

        if normalized:
            return out_norm

        # Unnormalized output: y = y_norm * sigma + mu
        # Broadcast sigma and mu: [Out] -> [1, 1, Out] or [1, Out]
        if out_norm.ndim == 3:
            # [B, N, Out]
            return out_norm * self.sigma.view(1, 1, -1) + self.mu.view(1, 1, -1)
        else:
            # [B, Out]
            return out_norm * self.sigma + self.mu

    @torch.no_grad()
    def update_stats(self, targets):
        """
        Updates the running statistics based on the targets and adjusts weights/biases
        to preserve the unnormalized output.

        For IQN, targets are typically the expected values (mean over quantiles) or
        the full quantile distribution. PopArt usually tracks the first and second moments
        of the return distribution.

        Args:
            targets: Batch of target values.
                     Can be [Batch, N_Quantiles, Out_Features] or [Batch, Out_Features].
                     If quantiles are provided, we compute stats over the flattened batch/quantile dims
                     or just the mean over quantiles depending on desired behavior.
                     Standard PopArt tracks the return moments.
        """
        # If targets have quantile dimension, we can either:
        # 1. Treat every quantile as a sample (flatten B and N)
        # 2. Compute mean over N first (expected value), then stats over B.
        # Usually, we want to normalize the value scale, so using all quantiles is robust.

        if targets.ndim == 3:
            # [B, N, Out] -> Flatten to [B*N, Out]
            targets_flat = targets.reshape(-1, self.out_features)
        else:
            targets_flat = targets

        # Calculate batch statistics
        batch_mean = targets_flat.mean(dim=0)
        batch_sq_mean = (targets_flat**2).mean(dim=0)

        # Update running statistics using EMA
        new_mu = self.beta * self.mu + (1 - self.beta) * batch_mean
        new_nu = self.beta * self.nu + (1 - self.beta) * batch_sq_mean

        # Calculate new sigma = sqrt(E[x^2] - E[x]^2)
        new_sigma = torch.sqrt(torch.clamp(new_nu - new_mu**2, min=self.epsilon**2))
        new_sigma = torch.clamp(new_sigma, min=self.epsilon)

        # Update weights and biases to preserve unnormalized outputs
        # W_new = W_old * (sigma_old / sigma_new)
        # b_new = (sigma_old * b_old + mu_old - mu_new) / sigma_new
        self.weight.data.mul_(self.sigma.unsqueeze(1) / new_sigma.unsqueeze(1))
        self.bias.data.mul_(self.sigma).add_(self.mu - new_mu).div_(new_sigma)

        # Update buffers
        self.mu.copy_(new_mu)
        self.nu.copy_(new_nu)
        self.sigma.copy_(new_sigma)

    def normalize(self, x):
        """Normalizes values using the current statistics."""
        # Broadcast logic
        if x.ndim == 3:
            return (x - self.mu.view(1, 1, -1)) / self.sigma.view(1, 1, -1)
        return (x - self.mu) / self.sigma

    def unnormalize(self, x_norm):
        """Unnormalizes values using the current statistics."""
        if x_norm.ndim == 3:
            return x_norm * self.sigma.view(1, 1, -1) + self.mu.view(1, 1, -1)
        return x_norm * self.sigma + self.mu


if __name__ == "__main__":
    print("Running PopArtIQNLayer Unit Test...")
    torch.manual_seed(42)

    batch_size = 10
    n_quantiles = 32
    in_dim = 64
    out_dim = 5

    popart_iqn = PopArtIQNLayer(in_dim, out_dim)

    # Input: [B, N, H]
    x = torch.randn(batch_size, n_quantiles, in_dim)

    # 1. Check Forward Pass
    y_unnorm_before = popart_iqn(x, normalized=False)
    assert y_unnorm_before.shape == (batch_size, n_quantiles, out_dim)

    # 2. Create targets with shift/scale
    # Targets: [B, N, Out]
    targets = torch.randn(batch_size, n_quantiles, out_dim) * 10.0 + 5.0

    # 3. Update Stats
    popart_iqn.update_stats(targets)

    # 4. Check Output Preservation
    y_unnorm_after = popart_iqn(x, normalized=False)

    diff = (y_unnorm_before - y_unnorm_after).abs().max().item()
    print(f"Max difference after update: {diff:.6f}")
    assert diff < 1e-5, "Outputs were not preserved!"

    # 5. Check Stats Update
    print(f"Updated Sigma: {popart_iqn.sigma.mean().item():.4f}")
    print(f"Updated Mu: {popart_iqn.mu.mean().item():.4f}")

    print("PopArtIQNLayer Test Passed!")
