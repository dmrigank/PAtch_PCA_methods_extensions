"""Loss helpers for PCA-Net training."""

from __future__ import annotations

import torch
from torch import nn

from lpcanet.losses.latent import TwoScaleLatentLoss
from lpcanet.losses.terms import (
    DEFAULT_LOSS_WEIGHTS,
    CompositeLoss,
    interface_flux_loss,
    interface_value_loss,
    pde_residual_loss,
    reconstruction_loss,
    spectral_loss,
)


def mse_loss() -> nn.Module:
    """Return mean-squared error loss for latent-space training."""
    return nn.MSELoss()


def relative_l2_error(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute batch-mean relative L2 error."""
    numerator = torch.linalg.vector_norm(pred - true, dim=tuple(range(1, pred.ndim)))
    denominator = torch.linalg.vector_norm(true, dim=tuple(range(1, true.ndim))).clamp_min(eps)
    return torch.mean(numerator / denominator)


__all__ = [
    "CompositeLoss",
    "DEFAULT_LOSS_WEIGHTS",
    "TwoScaleLatentLoss",
    "interface_flux_loss",
    "interface_value_loss",
    "mse_loss",
    "pde_residual_loss",
    "reconstruction_loss",
    "relative_l2_error",
    "spectral_loss",
]
