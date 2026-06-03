"""Window functions used for patch reconstruction."""

from __future__ import annotations

import numpy as np


def hann1d(patch_size: int) -> np.ndarray:
    """Return a 1D Hann window of length ``patch_size``."""
    _validate_patch_size(patch_size)
    return np.hanning(patch_size)


def hann2d(patch_size: int) -> np.ndarray:
    """Return a separable 2D Hann window."""
    window = hann1d(patch_size)
    return np.outer(window, window)


def safe_hann2d(patch_size: int, eps: float = 1e-6) -> np.ndarray:
    """Return a Hann-like 2D window with nonzero endpoint weights."""
    if eps <= 0:
        raise ValueError("eps must be positive.")
    return np.maximum(hann2d(patch_size), eps)


def _validate_patch_size(patch_size: int) -> None:
    if not isinstance(patch_size, int):
        raise TypeError("patch_size must be an integer.")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")
