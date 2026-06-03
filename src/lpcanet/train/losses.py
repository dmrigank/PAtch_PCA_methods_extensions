"""Loss helpers for PCA-Net training."""

from __future__ import annotations

import torch
from torch import nn


def mse_loss() -> nn.Module:
    """Return mean-squared error loss for latent-space training."""
    return nn.MSELoss()


def relative_l2_error(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute batch-mean relative L2 error."""
    numerator = torch.linalg.vector_norm(pred - true, dim=tuple(range(1, pred.ndim)))
    denominator = torch.linalg.vector_norm(true, dim=tuple(range(1, true.ndim))).clamp_min(eps)
    return torch.mean(numerator / denominator)
