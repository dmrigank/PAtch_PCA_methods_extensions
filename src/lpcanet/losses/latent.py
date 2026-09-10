"""Latent-space objectives for two-scale PCA outputs."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class TwoScaleLatentLoss(nn.Module):
    """MSE variants for concatenated coarse-global and residual-local codes."""

    MODES = {"mse", "block_balanced", "score_normalized"}

    def __init__(
        self,
        *,
        mode: str = "mse",
        coarse_dim: int | None = None,
        score_variances: Sequence[float] | torch.Tensor | None = None,
        coarse_weight: float = 0.5,
        residual_weight: float = 0.5,
        variance_floor: float = 1.0e-8,
    ) -> None:
        super().__init__()
        normalized_mode = str(mode).lower()
        if normalized_mode not in self.MODES:
            raise ValueError(
                f"Unsupported latent loss mode {mode!r}; expected one of {sorted(self.MODES)}."
            )
        if coarse_weight < 0.0 or residual_weight < 0.0:
            raise ValueError("Latent block weights must be non-negative.")
        if coarse_weight + residual_weight <= 0.0:
            raise ValueError("At least one latent block weight must be positive.")
        if variance_floor <= 0.0:
            raise ValueError("variance_floor must be positive.")
        if normalized_mode != "mse" and (coarse_dim is None or coarse_dim <= 0):
            raise ValueError(f"{normalized_mode} requires a positive coarse_dim.")

        self.mode = normalized_mode
        self.coarse_dim = None if coarse_dim is None else int(coarse_dim)
        weight_sum = float(coarse_weight + residual_weight)
        self.coarse_weight = float(coarse_weight) / weight_sum
        self.residual_weight = float(residual_weight) / weight_sum
        self.variance_floor = float(variance_floor)

        variances = (
            torch.empty(0, dtype=torch.float32)
            if score_variances is None
            else torch.as_tensor(score_variances, dtype=torch.float32).reshape(-1)
        )
        if normalized_mode == "score_normalized":
            if variances.numel() == 0:
                raise ValueError("score_normalized requires PCA score_variances.")
            if not bool(torch.all(torch.isfinite(variances))):
                raise ValueError("score_variances must be finite.")
            if not bool(torch.all(variances > 0.0)):
                raise ValueError("score_variances must be positive.")
        self.register_buffer("score_variances", variances)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Evaluate the selected objective on matching latent vectors."""
        if pred.shape != target.shape:
            raise ValueError(
                f"Latent prediction and target shapes differ: {tuple(pred.shape)} vs "
                f"{tuple(target.shape)}."
            )
        if pred.ndim != 2:
            raise ValueError(f"Expected latent tensors with shape (B, Z), got {tuple(pred.shape)}.")
        error_squared = (pred - target).square()
        if self.mode == "mse":
            return torch.mean(error_squared)

        assert self.coarse_dim is not None
        if not 0 < self.coarse_dim < pred.shape[1]:
            raise ValueError(
                f"coarse_dim={self.coarse_dim} must split latent dimension {pred.shape[1]} "
                "into non-empty coarse and residual blocks."
            )
        if self.mode == "score_normalized":
            if self.score_variances.numel() != pred.shape[1]:
                raise ValueError(
                    f"Expected {pred.shape[1]} score variances, got "
                    f"{self.score_variances.numel()}."
                )
            denominator = self.score_variances.to(
                device=pred.device,
                dtype=pred.dtype,
            ).clamp_min(self.variance_floor)
            error_squared = error_squared / denominator

        coarse_loss = torch.mean(error_squared[:, : self.coarse_dim])
        residual_loss = torch.mean(error_squared[:, self.coarse_dim :])
        return self.coarse_weight * coarse_loss + self.residual_weight * residual_loss


__all__ = ["TwoScaleLatentLoss"]
