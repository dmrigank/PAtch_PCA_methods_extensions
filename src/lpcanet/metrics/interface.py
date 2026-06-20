"""Patch-interface diagnostics."""

from __future__ import annotations

import numpy as np
import torch

from lpcanet.metrics.torch_ops import interface_flux_jump_torch, interface_value_jump_torch


def interface_jump(
    pred: np.ndarray,
    patch_size: int,
    stride: int,
    *,
    include_edges: bool = False,
) -> float:
    """Return mean absolute value jump across patch-grid interfaces.

    Boundaries are inferred from all patch slice starts/ends. For overlapping
    patches this measures jumps on the denser overlap grid induced by ``stride``.
    """
    return interface_value_jump(pred, patch_size, stride, include_edges=include_edges)


def interface_value_jump(
    pred: np.ndarray,
    patch_size: int,
    stride: int,
    *,
    include_edges: bool = False,
) -> float:
    """Return mean absolute value jump across stride-induced seams."""
    pred = np.asarray(pred, dtype=np.float64)
    tensor = torch.as_tensor(pred, dtype=torch.float64)
    return float(
        interface_value_jump_torch(
            tensor,
            patch_size,
            stride,
            include_edges=include_edges,
        ).detach()
    )


def interface_flux_jump(
    pred: np.ndarray,
    patch_size: int,
    stride: int,
    *,
    dx: float,
    include_edges: bool = False,
) -> float:
    """Return mean absolute normal derivative jump across stride-induced seams."""
    pred = np.asarray(pred, dtype=np.float64)
    tensor = torch.as_tensor(pred, dtype=torch.float64)
    return float(
        interface_flux_jump_torch(
            tensor,
            patch_size,
            stride,
            dx=dx,
            include_edges=include_edges,
        ).detach()
    )
