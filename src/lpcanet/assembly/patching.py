"""Robust 2D patch extraction and reconstruction utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

PatchSlice = tuple[slice, slice]


def get_patch_slices(
    grid_shape: tuple[int, int] | Sequence[int],
    patch_size: int,
    stride: int,
    *,
    include_edges: bool = False,
) -> list[PatchSlice]:
    """Return row-major patch slices that exactly cover a 2D grid.

    The current experiments require exact tiling/coverage by the configured
    ``patch_size`` and ``stride``. If the final row/column would be uncovered,
    this function raises a clear ``ValueError`` unless ``include_edges`` is set.
    With ``include_edges=True``, the final patch on each axis is shifted to end
    exactly at the domain boundary.
    """
    height, width = _validate_grid_shape(grid_shape)
    _validate_patch_params(patch_size, stride)
    if patch_size > height or patch_size > width:
        raise ValueError(
            f"patch_size={patch_size} cannot exceed grid shape {(height, width)}."
        )

    if not include_edges:
        _validate_exact_coverage(height, patch_size, stride, axis="height")
        _validate_exact_coverage(width, patch_size, stride, axis="width")

    slices: list[PatchSlice] = []
    for row in _axis_starts(height, patch_size, stride, include_edges=include_edges):
        for col in _axis_starts(width, patch_size, stride, include_edges=include_edges):
            slices.append((slice(row, row + patch_size), slice(col, col + patch_size)))
    return slices


def extract_patches_2d(
    fields: np.ndarray,
    patch_size: int,
    stride: int,
    flatten: bool = True,
    *,
    include_edges: bool = False,
) -> np.ndarray:
    """Extract 2D patches from fields with shape ``(N, H, W)``."""
    fields = np.asarray(fields)
    if fields.ndim != 3:
        raise ValueError(f"fields must have shape (N, H, W), got {fields.shape}.")

    slices = get_patch_slices(fields.shape[1:], patch_size, stride, include_edges=include_edges)
    patches = np.empty(
        (fields.shape[0], len(slices), patch_size, patch_size),
        dtype=fields.dtype,
    )
    for patch_index, (row_slice, col_slice) in enumerate(slices):
        patches[:, patch_index] = fields[:, row_slice, col_slice]

    if flatten:
        return patches.reshape(fields.shape[0], len(slices), patch_size * patch_size)
    return patches


def assemble_patches_2d(
    patches: np.ndarray,
    grid_shape: tuple[int, int] | Sequence[int],
    patch_size: int,
    stride: int,
    mode: str = "average",
    *,
    include_edges: bool = False,
) -> np.ndarray:
    """Assemble patches into fields using average overlap handling."""
    if mode != "average":
        raise ValueError(f"Unsupported assembly mode {mode!r}; expected 'average'.")
    return assemble_weighted_patches_2d(
        patches,
        grid_shape,
        patch_size,
        stride,
        np.ones((patch_size, patch_size), dtype=np.float64),
        include_edges=include_edges,
    )


def assemble_weighted_patches_2d(
    patches: np.ndarray,
    grid_shape: tuple[int, int] | Sequence[int],
    patch_size: int,
    stride: int,
    weights: np.ndarray,
    *,
    include_edges: bool = False,
) -> np.ndarray:
    """Assemble patches into fields using a per-patch 2D weight matrix."""
    from lpcanet.assembly.mosaic import assemble_mosaic_numpy

    height, width = _validate_grid_shape(grid_shape)
    slices = get_patch_slices((height, width), patch_size, stride, include_edges=include_edges)
    patches_4d = _as_unflattened_patches(patches, patch_size)
    if patches_4d.shape[1] != len(slices):
        raise ValueError(
            f"Expected {len(slices)} patches for grid_shape={(height, width)}, "
            f"patch_size={patch_size}, stride={stride}; got {patches_4d.shape[1]}."
        )

    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (patch_size, patch_size):
        raise ValueError(
            f"weights must have shape {(patch_size, patch_size)}, got {weights.shape}."
        )
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights must contain only finite values.")
    if np.any(weights < 0):
        raise ValueError("weights must be nonnegative.")

    return assemble_mosaic_numpy(
        patches_4d,
        (height, width),
        patch_size,
        stride,
        weights,
        include_edges=include_edges,
    )


def _as_unflattened_patches(patches: np.ndarray, patch_size: int) -> np.ndarray:
    patches = np.asarray(patches)
    if patches.ndim == 4:
        if patches.shape[2:] != (patch_size, patch_size):
            raise ValueError(
                f"Unflattened patches must have trailing shape "
                f"{(patch_size, patch_size)}, got {patches.shape[2:]}."
            )
        return patches
    if patches.ndim == 3:
        expected = patch_size * patch_size
        if patches.shape[2] != expected:
            raise ValueError(
                f"Flattened patches must have trailing size {expected}, "
                f"got {patches.shape[2]}."
            )
        return patches.reshape(patches.shape[0], patches.shape[1], patch_size, patch_size)
    raise ValueError(
        "patches must have shape (N, P, patch_size * patch_size) or "
        f"(N, P, patch_size, patch_size), got {patches.shape}."
    )


def _validate_grid_shape(grid_shape: tuple[int, int] | Sequence[int]) -> tuple[int, int]:
    if len(grid_shape) != 2:
        raise ValueError(f"grid_shape must contain two dimensions, got {grid_shape}.")
    height, width = int(grid_shape[0]), int(grid_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"grid_shape dimensions must be positive, got {grid_shape}.")
    return height, width


def _validate_patch_params(patch_size: int, stride: int) -> None:
    if not isinstance(patch_size, int) or not isinstance(stride, int):
        raise TypeError("patch_size and stride must be integers.")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")


def _axis_starts(size: int, patch_size: int, stride: int, *, include_edges: bool) -> list[int]:
    starts = list(range(0, size - patch_size + 1, stride))
    if include_edges:
        edge_start = size - patch_size
        if not starts or starts[-1] != edge_start:
            starts.append(edge_start)
    return sorted(set(starts))


def _validate_exact_coverage(size: int, patch_size: int, stride: int, axis: str) -> None:
    remainder = (size - patch_size) % stride
    if remainder != 0:
        raise ValueError(
            f"Patch configuration does not exactly cover {axis}={size}: "
            f"patch_size={patch_size}, stride={stride}. "
            "Choose values where (size - patch_size) is divisible by stride."
        )
