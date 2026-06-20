"""Coarse-grid restriction and prolongation operators."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _as_nchw(fields: torch.Tensor) -> tuple[torch.Tensor, int]:
    if fields.ndim == 2:
        return fields.unsqueeze(0).unsqueeze(0), 2
    if fields.ndim == 3:
        return fields.unsqueeze(1), 3
    if fields.ndim == 4:
        return fields, 4
    raise ValueError(f"fields must have shape (H, W), (B, H, W), or (B, C, H, W); got {tuple(fields.shape)}.")


def _restore_shape(fields: torch.Tensor, original_ndim: int) -> torch.Tensor:
    if original_ndim == 2:
        return fields[0, 0]
    if original_ndim == 3:
        return fields[:, 0]
    return fields


def restrict_average(fields: torch.Tensor, factor: int) -> torch.Tensor:
    """Average-restrict fields by an integer factor.

    Args:
        fields: Tensor with shape ``(H, W)``, ``(B, H, W)``, or ``(B, C, H, W)``.
        factor: Integer downsampling factor.

    Returns:
        Restricted tensor with spatial shape ``(H/factor, W/factor)``.
    """
    if factor <= 0:
        raise ValueError("factor must be positive.")
    nchw, original_ndim = _as_nchw(fields)
    height, width = int(nchw.shape[-2]), int(nchw.shape[-1])
    if height % factor != 0 or width % factor != 0:
        raise ValueError(f"Spatial shape {(height, width)} must be divisible by factor={factor}.")
    restricted = F.avg_pool2d(nchw, kernel_size=factor, stride=factor)
    return _restore_shape(restricted, original_ndim)


def prolongate_bilinear(fields: torch.Tensor, factor: int, *, align_corners: bool = False) -> torch.Tensor:
    """Bilinearly prolongate fields by an integer factor."""
    if factor <= 0:
        raise ValueError("factor must be positive.")
    nchw, original_ndim = _as_nchw(fields)
    prolongated = F.interpolate(
        nchw,
        scale_factor=factor,
        mode="bilinear",
        align_corners=align_corners,
    )
    return _restore_shape(prolongated, original_ndim)
