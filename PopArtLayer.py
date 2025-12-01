import torch
import torch.nn as nn
import math


class PopArtLayer(nn.Module):
    """
    PopArt (Preserving Outputs Precisely Adaptive Robustness Training) Layer.
    Normalizes the target values and adapts the weights/biases to preserve the unnormalized output.
    """

    mu: torch.Tensor
    sigma: torch.Tensor
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
            x: Input tensor.
            normalized: If True, returns the normalized output (raw linear projection).
                        If False, returns the unnormalized output (scaled and shifted).
        """
        # Normalized output: y_norm = xW^T + b
        out_norm = torch.nn.functional.linear(x, self.weight, self.bias)
        if normalized:
            return out_norm
        # Unnormalized output: y = y_norm * sigma + mu
        return out_norm * self.sigma + self.mu

    @torch.no_grad()
    def update_stats(self, targets):
        """
        Updates the running statistics based on the targets and adjusts weights/biases
        to preserve the unnormalized output.
        Args:
            targets: Batch of target values [batch_size, out_features].
        """
        # Calculate batch statistics
        batch_mean = targets.mean(dim=0)
        batch_sq_mean = (targets**2).mean(dim=0)

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
        return (x - self.mu) / self.sigma

    def unnormalize(self, x_norm):
        """Unnormalizes values using the current statistics."""
        return x_norm * self.sigma + self.mu


if __name__ == "__main__":
    # --- Unit Test ---
    print("Running Unit Test...")
    torch.manual_seed(42)
    in_dim = 5
    out_dim = 3
    popart = PopArtLayer(in_dim, out_dim)

    # Create a random input
    x = torch.randn(10, in_dim)

    # Get unnormalized output before update
    y_before = popart(x, normalized=False)

    # Create some random targets with a different scale/shift to force an update
    # e.g., mean=10, std=5
    targets = torch.randn(10, out_dim) * 5 + 10

    # Update stats
    popart.update_stats(targets)

    # Get unnormalized output after update
    y_after = popart(x, normalized=False)

    # Check if outputs are preserved
    diff = (y_before - y_after).abs().max().item()
    print(f"Max difference in output after update: {diff:.6f}")
    assert diff < 1e-5, "Outputs were not preserved precisely!"
    print("Unit Test Passed!")

    # --- Integration Test ---
    print("\nRunning Integration Test...")
    # Simple regression task: y = 2x + 5
    # We will shift the target distribution over time to see if PopArt adapts.

    model = PopArtLayer(1, 1, beta=0.9)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for step in range(200):
        # Generate data
        x = torch.randn(32, 1)
        # Target function changes scale/shift over time
        scale = 1.0 + step * 0.1
        shift = 5.0 + step * 0.5
        y_true = x * 2 * scale + shift

        # 1. Update stats with targets
        model.update_stats(y_true)

        # 2. Normalize targets
        y_target_norm = model.normalize(y_true)

        # 3. Forward pass (normalized)
        y_pred_norm = model(x, normalized=True)

        # 4. Compute loss
        loss = torch.nn.functional.mse_loss(y_pred_norm, y_target_norm)

        # 5. Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            # Check unnormalized prediction error
            y_pred = model(x, normalized=False)
            real_error = torch.nn.functional.mse_loss(y_pred, y_true).item()
            print(
                f"Step {step}: Loss (norm) = {loss.item():.4f}, Error (real) = {real_error:.4f}, "
                f"Sigma = {model.sigma.item():.2f}, Mu = {model.mu.item():.2f}"
            )

    print("Integration Test Finished!")
