"""Patch-interface diagnostics."""

from __future__ import annotations

import numpy as np

from lpcanet.assembly.patching import get_patch_slices


def interface_jump(
    pred: np.ndarray,
    patch_size: int,
    stride: int,
    *,
    include_edges: bool = False,
) -> float:
    """Return mean absolute jump across patch-grid interfaces.

    Boundaries are inferred from all patch slice starts/ends. For overlapping
    patches this measures jumps on the denser overlap grid induced by ``stride``.
    """
    pred = np.asarray(pred, dtype=np.float64)
    if pred.ndim != 3:
        raise ValueError(f"pred must have shape (N, H, W), got {pred.shape}.")
    _, height, width = pred.shape
    slices = get_patch_slices((height, width), patch_size, stride, include_edges=include_edges)
    row_boundaries: set[int] = set()
    col_boundaries: set[int] = set()
    for row_slice, col_slice in slices:
        for boundary in (row_slice.start, row_slice.stop):
            if 0 < boundary < height:
                row_boundaries.add(int(boundary))
        for boundary in (col_slice.start, col_slice.stop):
            if 0 < boundary < width:
                col_boundaries.add(int(boundary))

    jumps: list[np.ndarray] = []
    for row in sorted(row_boundaries):
        jumps.append(np.abs(pred[:, row, :] - pred[:, row - 1, :]))
    for col in sorted(col_boundaries):
        jumps.append(np.abs(pred[:, :, col] - pred[:, :, col - 1]))
    if not jumps:
        return 0.0
    return float(np.mean(np.concatenate([jump.ravel() for jump in jumps])))
