"""Helpers for creating latent PCA-Net training arrays."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def create_latent_arrays(
    encoder: Any,
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, np.ndarray]:
    """Transform input/output fields into latent arrays for model training."""
    return {
        "x_latent": encoder.transform_inputs(x).astype(np.float32, copy=False),
        "y_latent": encoder.transform_outputs(y).astype(np.float32, copy=False),
    }


def select_latent_split(
    latent_arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Select a split from latent arrays using integer sample indices."""
    indices = np.asarray(indices, dtype=np.int64)
    return {name: array[indices] for name, array in latent_arrays.items()}
