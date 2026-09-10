"""Differentiable physical-field loss terms.

All functions operate on assembled scalar fields, not latent codes or decoded
patch tensors. The stencil/interface work is delegated to
``lpcanet.metrics.torch_ops`` so metric and loss definitions stay aligned.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from lpcanet.metrics.residuals import POISSON_RESIDUAL_CONVENTION
from lpcanet.metrics.torch_ops import (
    darcy_residual_torch,
    interface_flux_jump_torch,
    interface_flux_traces_torch,
    interface_value_jump_torch,
    interface_value_traces_torch,
    mse_torch,
    poisson_residual_torch,
    relative_l2_torch,
    relative_spectrum_error_torch,
    residual_norm_torch,
)

DEFAULT_LOSS_WEIGHTS: dict[str, float] = {
    "recon": 1.0,
    "interface_value": 0.0,
    "interface_flux": 0.0,
    "pde_residual": 0.0,
    "spectral": 0.0,
}


def reconstruction_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    *,
    mode: str = "relative_l2",
    eps: float = 1e-12,
) -> torch.Tensor:
    """Reconstruction loss as global relative L2 or MSE."""
    if mode == "relative_l2":
        return relative_l2_torch(pred, true, eps=eps)
    if mode == "mse":
        return mse_torch(pred, true)
    raise ValueError(f"Unsupported reconstruction loss mode {mode!r}.")


def interface_value_loss(
    pred: torch.Tensor,
    true: torch.Tensor | None = None,
    *,
    patch_size: int,
    stride: int,
    include_edges: bool = False,
    target: str = "zero",
) -> torch.Tensor:
    """Value-interface loss against zero jump or the ground-truth seam trace."""
    if target == "truth":
        if true is None:
            raise ValueError("Truth-referenced interface_value loss requires true.")
        pred_trace = interface_value_traces_torch(
            pred,
            patch_size,
            stride,
            include_edges=include_edges,
        )
        true_trace = interface_value_traces_torch(
            true,
            patch_size,
            stride,
            include_edges=include_edges,
        )
        if pred_trace.numel() == 0:
            return pred.sum() * 0.0
        return torch.mean(torch.abs(pred_trace - true_trace))
    if target != "zero":
        raise ValueError(f"Unsupported interface target {target!r}; expected 'zero' or 'truth'.")
    return interface_value_jump_torch(
        pred,
        patch_size,
        stride,
        include_edges=include_edges,
    )


def interface_flux_loss(
    pred: torch.Tensor,
    true: torch.Tensor | None = None,
    *,
    patch_size: int,
    stride: int,
    dx: float,
    include_edges: bool = False,
    target: str = "zero",
) -> torch.Tensor:
    """Flux-interface loss against zero jump or the ground-truth seam trace."""
    if target == "truth":
        if true is None:
            raise ValueError("Truth-referenced interface_flux loss requires true.")
        pred_trace = interface_flux_traces_torch(
            pred,
            patch_size,
            stride,
            dx=dx,
            include_edges=include_edges,
        )
        true_trace = interface_flux_traces_torch(
            true,
            patch_size,
            stride,
            dx=dx,
            include_edges=include_edges,
        )
        if pred_trace.numel() == 0:
            return pred.sum() * 0.0
        return torch.mean(torch.abs(pred_trace - true_trace))
    if target != "zero":
        raise ValueError(f"Unsupported interface target {target!r}; expected 'zero' or 'truth'.")
    return interface_flux_jump_torch(
        pred,
        patch_size,
        stride,
        dx=dx,
        include_edges=include_edges,
    )


def pde_residual_loss(
    pred: torch.Tensor,
    *,
    dx: float,
    forcing: torch.Tensor | float,
    equation: str = "poisson",
    coefficient: torch.Tensor | float | None = None,
    reduction: str = "rms",
    poisson_convention: str = POISSON_RESIDUAL_CONVENTION,
) -> torch.Tensor:
    """Finite-difference PDE residual loss for Poisson or Darcy equations."""
    if equation == "poisson":
        residual = poisson_residual_torch(
            pred,
            forcing,
            dx,
            convention=poisson_convention,
        )
    elif equation == "darcy":
        if coefficient is None:
            raise ValueError("Darcy residual loss requires coefficient.")
        residual = darcy_residual_torch(pred, coefficient, forcing, dx)
    else:
        raise ValueError(f"Unsupported PDE equation {equation!r}.")
    return residual_norm_torch(residual, reduction=reduction)


def spectral_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    *,
    eps: float = 1e-12,
    high_k_weight_power: float = 1.0,
) -> torch.Tensor:
    """Relative radial energy-spectrum error, optionally weighted toward high k."""
    return relative_spectrum_error_torch(
        pred,
        true,
        eps=eps,
        high_k_weight_power=high_k_weight_power,
    )


class CompositeLoss(nn.Module):
    """Weighted sum of differentiable physical-field loss terms."""

    def __init__(
        self,
        *,
        weights: Mapping[str, float] | None = None,
        reconstruction: str = "relative_l2",
        patch_size: int | None = None,
        stride: int | None = None,
        dx: float | None = None,
        include_edges: bool = False,
        pde: str = "poisson",
        pde_reduction: str = "rms",
        poisson_convention: str = POISSON_RESIDUAL_CONVENTION,
        spectral_high_k_weight_power: float = 1.0,
        interface_target: str = "zero",
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        merged = dict(DEFAULT_LOSS_WEIGHTS)
        if weights is not None:
            merged.update({name: float(value) for name, value in weights.items()})
        unknown = set(merged) - set(DEFAULT_LOSS_WEIGHTS)
        if unknown:
            raise ValueError(f"Unsupported loss weights: {sorted(unknown)}.")
        self.weights = merged
        self.reconstruction = reconstruction
        self.patch_size = patch_size
        self.stride = stride
        self.dx = dx
        self.include_edges = include_edges
        self.pde = pde
        self.pde_reduction = pde_reduction
        self.poisson_convention = poisson_convention
        self.spectral_high_k_weight_power = spectral_high_k_weight_power
        if interface_target not in {"zero", "truth"}:
            raise ValueError("interface_target must be 'zero' or 'truth'.")
        self.interface_target = interface_target
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        true: torch.Tensor,
        *,
        forcing: torch.Tensor | float | None = None,
        coefficient: torch.Tensor | float | None = None,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return weighted total loss, optionally with unweighted components."""
        components: dict[str, torch.Tensor] = {}
        total = pred.sum() * 0.0

        if self.weights["recon"] != 0.0:
            components["recon"] = reconstruction_loss(
                pred,
                true,
                mode=self.reconstruction,
                eps=self.eps,
            )
            total = total + self.weights["recon"] * components["recon"]

        if self.weights["interface_value"] != 0.0:
            patch_size, stride = self._require_patch_config("interface_value")
            components["interface_value"] = interface_value_loss(
                pred,
                true,
                patch_size=patch_size,
                stride=stride,
                include_edges=self.include_edges,
                target=self.interface_target,
            )
            total = total + self.weights["interface_value"] * components["interface_value"]

        if self.weights["interface_flux"] != 0.0:
            patch_size, stride = self._require_patch_config("interface_flux")
            dx = self._require_dx("interface_flux")
            components["interface_flux"] = interface_flux_loss(
                pred,
                true,
                patch_size=patch_size,
                stride=stride,
                dx=dx,
                include_edges=self.include_edges,
                target=self.interface_target,
            )
            total = total + self.weights["interface_flux"] * components["interface_flux"]

        if self.weights["pde_residual"] != 0.0:
            dx = self._require_dx("pde_residual")
            if forcing is None:
                raise ValueError("pde_residual loss requires forcing.")
            components["pde_residual"] = pde_residual_loss(
                pred,
                dx=dx,
                forcing=forcing,
                equation=self.pde,
                coefficient=coefficient,
                reduction=self.pde_reduction,
                poisson_convention=self.poisson_convention,
            )
            total = total + self.weights["pde_residual"] * components["pde_residual"]

        if self.weights["spectral"] != 0.0:
            components["spectral"] = spectral_loss(
                pred,
                true,
                eps=self.eps,
                high_k_weight_power=self.spectral_high_k_weight_power,
            )
            total = total + self.weights["spectral"] * components["spectral"]

        if return_components:
            return total, components
        return total

    def _require_patch_config(self, term: str) -> tuple[int, int]:
        if self.patch_size is None or self.stride is None:
            raise ValueError(f"{term} loss requires patch_size and stride.")
        return int(self.patch_size), int(self.stride)

    def _require_dx(self, term: str) -> float:
        if self.dx is None:
            raise ValueError(f"{term} loss requires dx.")
        return float(self.dx)
