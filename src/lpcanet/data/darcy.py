"""Darcy dataset loading helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from lpcanet.data.io import load_mat, summarize_mat

FIELD_ALIASES = {
    "a": ("a", "coeff", "coefficient", "permeability"),
    "f": ("f", "forcing", "source"),
    "u": ("u", "solution", "phi", "pressure"),
}


def load_darcy_raw(
    config: Mapping[str, Any],
    *,
    limit_samples: int | None = None,
) -> dict[str, np.ndarray]:
    """Load raw Darcy arrays using explicit config key mappings."""
    raw_path = config.get("raw_path")
    if raw_path is None:
        raise KeyError("Darcy dataset config must define raw_path.")

    summary = summarize_mat(Path(str(raw_path)))
    key_mapping = _key_mapping(config)
    if not key_mapping:
        raise _missing_mapping_error(summary)

    solution_key = _resolve_field_key(key_mapping, "u")
    if solution_key is None:
        raise KeyError(
            "Darcy key_mapping must define a solution field using one of "
            f"{FIELD_ALIASES['u']}. Available MAT keys: {sorted(summary)}"
        )

    input_key = _resolve_field_key(key_mapping, "a")
    forcing_key = _resolve_field_key(key_mapping, "f")
    if input_key is None and forcing_key is None:
        raise KeyError(
            "Darcy key_mapping must define at least one input field: coefficient/a "
            f"or forcing/f. Available MAT keys: {sorted(summary)}"
        )

    requested = [key for key in (input_key, forcing_key, solution_key) if key is not None]
    missing = [key for key in requested if key not in summary]
    if missing:
        raise KeyError(
            f"Darcy MAT file is missing configured keys {missing}. "
            f"Available keys: {sorted(summary)}"
        )

    arrays = load_mat(raw_path, requested, limit_samples=limit_samples)
    result: dict[str, np.ndarray] = {"u": np.asarray(arrays[solution_key], dtype=np.float32)}
    if input_key is not None:
        result["a"] = np.asarray(arrays[input_key], dtype=np.float32)
    if forcing_key is not None:
        result["f"] = np.asarray(arrays[forcing_key], dtype=np.float32)
    elif config.get("constant_forcing") is not None:
        result["f"] = np.full_like(result["u"], float(config["constant_forcing"]))

    expected = _expected_grid_shape(config)
    for name, array in result.items():
        _validate_field(name, array, expected)
    _validate_sample_counts(result)
    return result


def _key_mapping(config: Mapping[str, Any]) -> Mapping[str, Any]:
    mapping = config.get("key_mapping", {})
    if not isinstance(mapping, Mapping):
        raise TypeError("Darcy key_mapping must be a mapping.")
    return mapping


def _resolve_field_key(mapping: Mapping[str, Any], canonical: str) -> str | None:
    for alias in FIELD_ALIASES[canonical]:
        value = mapping.get(alias)
        if value:
            return str(value)
    return None


def _expected_grid_shape(config: Mapping[str, Any]) -> tuple[int, int]:
    grid_size = config.get("grid_size", 128)
    return (int(grid_size), int(grid_size))


def _validate_field(name: str, array: np.ndarray, grid_shape: tuple[int, int]) -> None:
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape (N, H, W), got {array.shape}.")
    if tuple(array.shape[1:]) != grid_shape:
        raise ValueError(
            f"{name} must have grid shape {grid_shape}, got {tuple(array.shape[1:])}."
        )


def _validate_sample_counts(arrays: Mapping[str, np.ndarray]) -> None:
    counts = {name: array.shape[0] for name, array in arrays.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"Darcy sample counts differ: {counts}")


def _missing_mapping_error(summary: Mapping[str, Any]) -> KeyError:
    return KeyError(
        "Darcy config must include key_mapping for coefficient/input and solution. "
        f"Available MAT keys: {sorted(summary)}"
    )
