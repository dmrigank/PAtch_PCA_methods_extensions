"""Differentiable patch mosaic assembly."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
import torch

from lpcanet.assembly.patching import _as_unflattened_patches, _validate_grid_shape
from lpcanet.assembly.windows import safe_hann2d


@dataclass(frozen=True)
class PatchIndexMap:
    """Precomputed patch-to-global pixel mapping.

    Attributes:
        grid_shape: Output field shape ``(H, W)``.
        patch_size: Patch side length ``p``.
        stride: Patch stride in pixels.
        include_edges: Whether edge-shifted patches are included.
        flat_indices: Flattened global pixel indices with shape ``(P * p * p,)``.
        n_patches: Number of patches ``P``.
    """

    grid_shape: tuple[int, int]
    patch_size: int
    stride: int
    include_edges: bool
    flat_indices: torch.Tensor
    n_patches: int


@lru_cache(maxsize=128)
def _cached_flat_indices(
    grid_shape: tuple[int, int],
    patch_size: int,
    stride: int,
    include_edges: bool,
) -> tuple[tuple[int, ...], int]:
    from lpcanet.assembly.patching import get_patch_slices

    height, width = _validate_grid_shape(grid_shape)
    slices = get_patch_slices((height, width), patch_size, stride, include_edges=include_edges)
    indices: list[int] = []
    for row_slice, col_slice in slices:
        for row in range(int(row_slice.start), int(row_slice.stop)):
            row_offset = row * width
            for col in range(int(col_slice.start), int(col_slice.stop)):
                indices.append(row_offset + col)
    return tuple(indices), len(slices)


def make_patch_index_map(
    grid_shape: tuple[int, int],
    patch_size: int,
    stride: int,
    *,
    include_edges: bool = False,
    device: torch.device | str | None = None,
) -> PatchIndexMap:
    """Return precomputed global pixel indices for patch assembly."""
    height, width = _validate_grid_shape(grid_shape)
    flat_indices, n_patches = _cached_flat_indices(
        (height, width), patch_size, stride, include_edges
    )
    return PatchIndexMap(
        grid_shape=(height, width),
        patch_size=int(patch_size),
        stride=int(stride),
        include_edges=bool(include_edges),
        flat_indices=torch.as_tensor(flat_indices, dtype=torch.long, device=device),
        n_patches=n_patches,
    )


def hann_weights(
    patch_size: int,
    *,
    safe: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return prior-paper Hann reconstruction weights as a Torch tensor."""
    weights_np = safe_hann2d(patch_size) if safe else np.hanning(patch_size)[:, None] * np.hanning(patch_size)[None, :]
    return torch.as_tensor(weights_np, dtype=dtype, device=device)


def assemble_mosaic(
    patches: torch.Tensor,
    grid_shape: tuple[int, int],
    patch_size: int,
    stride: int,
    weights: torch.Tensor | None = None,
    *,
    include_edges: bool = False,
    index_map: PatchIndexMap | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Assemble decoded patches into full fields with differentiable scatter-add.

    Args:
        patches: Decoded patches with shape ``(B, P, p, p)`` or ``(B, P, p*p)``.
        grid_shape: Output field shape ``(H, W)``.
        patch_size: Patch side length ``p``.
        stride: Patch stride in pixels.
        weights: Optional per-patch weights with shape ``(p, p)``. If omitted,
            uniform averaging is used.
        include_edges: Whether edge-shifted patches are included.
        index_map: Optional precomputed map from ``make_patch_index_map``.
        eps: Positive denominator stabilizer.

    Returns:
        Assembled fields with shape ``(B, H, W)``.
    """
    if patches.ndim == 3:
        if patches.shape[-1] != patch_size * patch_size:
            raise ValueError(
                f"Flattened patches must have trailing size {patch_size * patch_size}, "
                f"got {patches.shape[-1]}."
            )
        patches_4d = patches.reshape(patches.shape[0], patches.shape[1], patch_size, patch_size)
    elif patches.ndim == 4:
        if tuple(patches.shape[-2:]) != (patch_size, patch_size):
            raise ValueError(
                f"Patch tensor trailing shape must be {(patch_size, patch_size)}, "
                f"got {tuple(patches.shape[-2:])}."
            )
        patches_4d = patches
    else:
        raise ValueError(
            "patches must have shape (B, P, p*p) or (B, P, p, p), "
            f"got {tuple(patches.shape)}."
        )

    height, width = _validate_grid_shape(grid_shape)
    if index_map is None:
        index_map = make_patch_index_map(
            (height, width),
            patch_size,
            stride,
            include_edges=include_edges,
            device=patches.device,
        )
    elif (
        index_map.grid_shape != (height, width)
        or index_map.patch_size != patch_size
        or index_map.stride != stride
        or index_map.include_edges != include_edges
    ):
        raise ValueError("index_map configuration does not match assembly arguments.")

    if patches_4d.shape[1] != index_map.n_patches:
        raise ValueError(
            f"Expected {index_map.n_patches} patches for grid_shape={(height, width)}, "
            f"patch_size={patch_size}, stride={stride}; got {patches_4d.shape[1]}."
        )

    flat_indices = index_map.flat_indices.to(device=patches.device)
    if weights is None:
        weights_flat = torch.ones(
            patch_size * patch_size,
            dtype=patches.dtype,
            device=patches.device,
        )
    else:
        if tuple(weights.shape) != (patch_size, patch_size):
            raise ValueError(f"weights must have shape {(patch_size, patch_size)}, got {tuple(weights.shape)}.")
        if torch.any(weights < 0):
            raise ValueError("weights must be nonnegative.")
        weights_flat = weights.to(dtype=patches.dtype, device=patches.device).reshape(-1)

    repeated_weights = weights_flat.repeat(index_map.n_patches)
    weighted_patches = patches_4d.reshape(patches_4d.shape[0], -1) * repeated_weights
    output_flat = patches.new_zeros((patches_4d.shape[0], height * width))
    output_flat.scatter_add_(1, flat_indices.expand(patches_4d.shape[0], -1), weighted_patches)

    weight_sums = patches.new_zeros((height * width,))
    weight_sums.scatter_add_(0, flat_indices, repeated_weights)
    if torch.any(weight_sums <= 0):
        zero_count = int(torch.count_nonzero(weight_sums <= 0).item())
        raise ValueError(
            "Patch weights leave uncovered or zero-weight pixels "
            f"({zero_count} pixels). Use nonzero boundary weights, uniform averaging, "
            "or an explicit boundary policy."
        )

    assembled = output_flat / weight_sums.clamp_min(eps).unsqueeze(0)
    return assembled.reshape(patches_4d.shape[0], height, width)


def assemble_mosaic_numpy(
    patches: np.ndarray,
    grid_shape: tuple[int, int],
    patch_size: int,
    stride: int,
    weights: np.ndarray | None = None,
    *,
    include_edges: bool = False,
    dtype_policy: Literal["preserve", "float64"] = "preserve",
) -> np.ndarray:
    """NumPy-facing wrapper around the differentiable assembly path."""
    patches_4d = _as_unflattened_patches(patches, patch_size)
    torch_dtype = torch.float64 if patches_4d.dtype == np.float64 else torch.float32
    patch_tensor = torch.as_tensor(patches_4d, dtype=torch_dtype)
    weight_tensor = None if weights is None else torch.as_tensor(weights, dtype=torch_dtype)
    assembled = assemble_mosaic(
        patch_tensor,
        grid_shape,
        patch_size,
        stride,
        weight_tensor,
        include_edges=include_edges,
    ).detach().cpu().numpy()
    if dtype_policy == "float64":
        return assembled.astype(np.float64, copy=False)
    return assembled.astype(patches_4d.dtype, copy=False)
