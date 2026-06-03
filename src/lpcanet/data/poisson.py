"""Poisson dataset loading helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from lpcanet.data.io import load_mat, summarize_mat

DEFAULT_INPUT_KEY = "f_data"
DEFAULT_OUTPUT_KEY = "phi_data"


def _expected_grid_shape(config: Mapping[str, Any]) -> tuple[int, int] | None:
    grid_size = config.get("grid_size")
    if grid_size is not None:
        return (int(grid_size), int(grid_size))
    expected_shape = config.get("expected_shape")
    if expected_shape is None:
        return (128, 128)
    if len(expected_shape) != 2:
        raise ValueError("expected_shape must contain exactly two entries.")
    return (int(expected_shape[0]), int(expected_shape[1]))


def load_poisson_raw(
    config: Mapping[str, Any],
    *,
    limit_samples: int | None = None,
) -> dict[str, np.ndarray]:
    """Load raw Poisson MAT arrays into internal ``f`` and ``u`` arrays."""
    raw_path = config.get("raw_path")
    if raw_path is None:
        raise KeyError("Poisson dataset config must define raw_path.")

    input_key = str(config.get("input_key", DEFAULT_INPUT_KEY))
    output_key = str(config.get("output_key", DEFAULT_OUTPUT_KEY))
    summary = summarize_mat(Path(str(raw_path)))
    missing = [key for key in (input_key, output_key) if key not in summary]
    if missing:
        raise KeyError(
            f"Poisson MAT file is missing keys {missing}. "
            f"Available keys: {sorted(summary)}"
        )

    arrays = load_mat(raw_path, [input_key, output_key], limit_samples=limit_samples)
    f = np.asarray(arrays[input_key], dtype=np.float32)
    u = np.asarray(arrays[output_key], dtype=np.float32)

    _validate_field("f", f, _expected_grid_shape(config))
    _validate_field("u", u, _expected_grid_shape(config))
    if f.shape[0] != u.shape[0]:
        raise ValueError(f"Input/output sample counts differ: {f.shape[0]} != {u.shape[0]}")

    return {"f": f, "u": u}


def _validate_field(name: str, array: np.ndarray, grid_shape: tuple[int, int] | None) -> None:
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape (N, H, W), got {array.shape}.")
    if grid_shape is not None and tuple(array.shape[1:]) != grid_shape:
        raise ValueError(
            f"{name} must have grid shape {grid_shape}, got {tuple(array.shape[1:])}."
        )
