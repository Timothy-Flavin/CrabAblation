import torch
import torch.nn as nn


class RNDModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # 1. Define Target Network (Fixed)
        self.target = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, output_dim)
        )
        # Freeze target weights immediately
        for param in self.target.parameters():
            param.requires_grad = False
        # 2. Define Predictor Network (Trainable)
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),  # Predictor is often slightly deeper/wider
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def _device(self):
        # Helper to get current device of the module's parameters
        try:
            return next(self.parameters()).device
        except StopIteration:
            # Fallback if there are no parameters (shouldn't happen here)
            return torch.device("cpu")

    def forward(self, next_obs):
        # Returns the intrinsic reward for a batch of observations
        # Ensure input is on the same device as the module
        if not isinstance(next_obs, torch.Tensor):
            next_obs = torch.tensor(
                next_obs, dtype=torch.float32, device=self._device()
            )
        else:
            next_obs = next_obs.to(self._device())

        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)

        # Calculate MSE per batch item (no reduction yet)
        error = (predict_feature - target_feature).pow(2).sum(1)
        return error


class RunningMeanStd(nn.Module):
    """
    Calculates the running mean and standard deviation of a data stream
    using Welford's online algorithm. This is essential for normalizing
    inputs or rewards in reinforcement learning.

    Usage:
        - obs_filter = RunningMeanStd(shape=(obs_dim,))
        - obs_filter.update(new_observations_batch)
        - normalized_obs = obs_filter.normalize(obs_to_normalize)
    """

    count: torch.Tensor
    mean: torch.Tensor
    M2: torch.Tensor

    def __init__(self, shape, epsilon=1e-8):
        """
        Initializes the filter.

        Args:
            shape (tuple or int): The shape of the data to be normalized.
                                  e.g., (num_envs, obs_dim) or (obs_dim,)
            epsilon (float): A small value to prevent division by zero.
        """
        super().__init__()
        self.shape = shape
        self.epsilon = epsilon

        # We use register_buffer to make these tensors part of the module's
        # state, but not model parameters (i.e., not trained by optimizer).
        # This ensures they are saved with state_dict and moved to GPU (e.g., .to(device)).

        # 'n' is the count of *individual samples* seen, not batches.
        self.register_buffer("count", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float64))

        # 'M2' stores the sum of squares of differences from the mean
        self.register_buffer("M2", torch.zeros(shape, dtype=torch.float64))

    @property
    def var(self):
        # Variance is M2 / (n - 1) for sample variance, or M2 / n for population.
        # We use n for simplicity; (n-1) is also common.
        # Handle the case where count is 0 or 1 to avoid division by zero.
        return self.M2 / self.count if self.count > 1 else torch.zeros_like(self.mean)

    @property
    def std(self):
        # Standard deviation is the square root of variance
        return torch.sqrt(self.var + self.epsilon)

    def update(self, x):
        """Updates running statistics from data where leading dims are batch and trailing dims match `shape`."""

        # Ensure tensor on correct device/dtype
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float64, device=self.mean.device)
        else:
            x = x.to(dtype=torch.float64, device=self.mean.device)

        feat_shape = tuple(self.mean.shape)
        feat_ndim = self.mean.dim()

        if feat_ndim == 0:
            # Scalar feature per sample: accept x as scalar or any batch of scalars
            if x.ndim == 0:
                x_flat = x.view(1)
            else:
                x_flat = x.reshape(-1)
            batch_size = x_flat.shape[0]
            batch_mean = x_flat.mean(dim=0)  # scalar
            batch_M2 = ((x_flat - batch_mean) ** 2).sum(dim=0)  # scalar
        else:
            # Ensure samples have trailing dims equal to feat_shape
            if x.ndim < feat_ndim:
                raise ValueError(
                    f"Input tensor has ndim={x.ndim} < feature ndim={feat_ndim}"
                )
            if x.shape[-feat_ndim:] != feat_shape:
                # If x is a single sample with exactly feat_shape, add batch dim
                if x.ndim == feat_ndim and tuple(x.shape) == feat_shape:
                    x = x.unsqueeze(0)
                else:
                    raise ValueError(
                        f"Trailing dims {x.shape[-feat_ndim:]} do not match feature shape {feat_shape}"
                    )
            if x.ndim == feat_ndim:
                x = x.unsqueeze(0)
            # Flatten all leading (batch) dims
            x_flat = x.reshape(-1, *feat_shape)
            batch_size = x_flat.shape[0]
            batch_mean = x_flat.mean(dim=0)
            batch_M2 = ((x_flat - batch_mean) ** 2).sum(dim=0)

        # Welford combine
        new_count = self.count + batch_size
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * (batch_size / new_count)
        new_M2 = self.M2 + batch_M2 + (delta**2) * (self.count * batch_size / new_count)

        # Assign new values
        self.count.copy_(new_count)
        self.mean.copy_(new_mean)
        self.M2.copy_(new_M2)

    def normalize(self, x, clip_range=10.0):
        """
        Normalizes an input 'x' using the running statistics.

        Args:
            x (torch.Tensor): The data to be normalized.
            clip_range (float): The range to clip the normalized data to (e.g., [-10, 10]).
                                This prevents extreme values.

        Returns:
            torch.Tensor: The normalized and clipped data.
        """
        if self.count == 0:
            return x  # Can't normalize if we haven't seen any data

        # Ensure x is on the correct device
        x = x.to(self.mean.device)

        # Normalize: (x - mean) / std
        # We use .detach() to ensure no gradients flow back through the filter
        normalized_x = (x - self.mean.detach()) / self.std.detach()

        # Clip the result to be within [-clip_range, clip_range]
        return torch.clamp(normalized_x, -clip_range, clip_range)

    def to(self, *args, **kwargs):
        """Overrides .to() to move all buffers to the correct device."""
        # This is good practice for nn.Modules
        new_module = super().to(*args, **kwargs)
        new_module.count = new_module.count.to(*args, **kwargs)
        new_module.mean = new_module.mean.to(*args, **kwargs)
        new_module.M2 = new_module.M2.to(*args, **kwargs)
        return new_module
