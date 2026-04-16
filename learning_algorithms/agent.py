from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch.utils.tensorboard import SummaryWriter


class Agent(ABC):
    """Common runtime API for training/evaluation agents in this repository."""

    def __init__(self):
        self.tb_writer: SummaryWriter | None = None
        self.tb_prefix: str = "agent"
        self.last_losses: dict[str, Any] = {}
        self.timing: dict[str, float] = {}

    @abstractmethod
    def to(self, device) -> Any:
        """Move all model state and optimizer-linked modules to a target device."""
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> Any:
        """Run a single update/training step and return a scalar loss/metric."""
        pass

    @abstractmethod
    def sample_action(self, *args, **kwargs) -> Any:
        """Return action(s) from observations under the current policy/value rule."""
        pass

    def attach_tensorboard(self, writer: SummaryWriter, prefix: str = "agent") -> Any:
        """Attach TensorBoard writer used by concrete agents for logging."""
        self.tb_writer = writer
        self.tb_prefix = prefix
        pass

    def update_target(self) -> Any:
        """Optional target-network update hook for value-based agents."""
        return None

    def update_running_stats(self, *args, **kwargs) -> Any:
        """Optional running-statistics update hook (obs/reward normalization, etc.)."""
        return None
