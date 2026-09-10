"""Patch-interface diagnostics."""

from __future__ import annotations

import numpy as np
import torch

from lpcanet.metrics.torch_ops import (
    interface_flux_jump_torch,
    interface_flux_traces_torch,
    interface_value_jump_torch,
    interface_value_traces_torch,
)


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


def interface_value_trace_error(
    pred: np.ndarray,
    true: np.ndarray,
    patch_size: int,
    stride: int,
    *,
    include_edges: bool = False,
) -> float:
    """Return the MAE between predicted and true signed seam traces."""
    pred_tensor = torch.as_tensor(np.asarray(pred, dtype=np.float64), dtype=torch.float64)
    true_tensor = torch.as_tensor(np.asarray(true, dtype=np.float64), dtype=torch.float64)
    pred_trace = interface_value_traces_torch(
        pred_tensor,
        patch_size,
        stride,
        include_edges=include_edges,
    )
    true_trace = interface_value_traces_torch(
        true_tensor,
        patch_size,
        stride,
        include_edges=include_edges,
    )
    if pred_trace.numel() == 0:
        return 0.0
    return float(torch.mean(torch.abs(pred_trace - true_trace)).detach())


def interface_flux_trace_error(
    pred: np.ndarray,
    true: np.ndarray,
    patch_size: int,
    stride: int,
    *,
    dx: float,
    include_edges: bool = False,
) -> float:
    """Return the MAE between predicted and true signed flux seam traces."""
    pred_tensor = torch.as_tensor(np.asarray(pred, dtype=np.float64), dtype=torch.float64)
    true_tensor = torch.as_tensor(np.asarray(true, dtype=np.float64), dtype=torch.float64)
    pred_trace = interface_flux_traces_torch(
        pred_tensor,
        patch_size,
        stride,
        dx=dx,
        include_edges=include_edges,
    )
    true_trace = interface_flux_traces_torch(
        true_tensor,
        patch_size,
        stride,
        dx=dx,
        include_edges=include_edges,
    )
    if pred_trace.numel() == 0:
        return 0.0
    return float(torch.mean(torch.abs(pred_trace - true_trace)).detach())
