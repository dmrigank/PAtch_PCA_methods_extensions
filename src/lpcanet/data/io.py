"""Input/output helpers for MATLAB and NumPy dataset files."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.io

from lpcanet.utils.paths import normalize_path

MatSummary = dict[str, dict[str, Any]]


def _normalize_variable_names(variable_names: Iterable[str] | None) -> list[str] | None:
    if variable_names is None:
        return None
    return list(variable_names)


def _slice_limit(array: np.ndarray, limit_samples: int | None) -> np.ndarray:
    if limit_samples is None:
        return array
    if limit_samples < 0:
        raise ValueError("limit_samples must be nonnegative.")
    if array.ndim == 0:
        return array
    return array[:limit_samples]


def _is_hdf5_error(error: Exception) -> bool:
    message = str(error).lower()
    return (
        isinstance(error, NotImplementedError)
        or "please use hdf reader" in message
        or "unknown mat file type" in message
        or "hdf5" in message
    )


def load_mat(
    path: str | Path,
    variable_names: Iterable[str] | None = None,
    *,
    limit_samples: int | None = None,
) -> dict[str, np.ndarray]:
    """Load variables from a MATLAB MAT file with SciPy and h5py fallback."""
    mat_path = normalize_path(path)
    names = _normalize_variable_names(variable_names)
    try:
        loaded = scipy.io.loadmat(mat_path, variable_names=names)
        return {
            key: _slice_limit(np.asarray(value), limit_samples)
            for key, value in loaded.items()
            if not key.startswith("__")
        }
    except Exception as exc:
        if not _is_hdf5_error(exc):
            raise
        return _load_hdf5_mat(mat_path, names, limit_samples=limit_samples)


def _load_hdf5_mat(
    path: Path,
    variable_names: list[str] | None,
    *,
    limit_samples: int | None,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    wanted = set(variable_names) if variable_names is not None else None
    with h5py.File(path, "r") as handle:
        def visit(name: str, obj: Any) -> None:
            key = name.split("/")[-1]
            if key.startswith("__") or not hasattr(obj, "shape"):
                return
            if wanted is not None and key not in wanted and name not in wanted:
                return
            if limit_samples is not None and obj.shape:
                result[key] = np.asarray(obj[:limit_samples])
            else:
                result[key] = np.asarray(obj[()])

        handle.visititems(visit)
    return result


def summarize_mat(path: str | Path) -> MatSummary:
    """Return variable names, shapes, dtypes, and loader type for a MAT file."""
    mat_path = normalize_path(path)
    try:
        summary: MatSummary = {}
        for name, shape, dtype_name in scipy.io.whosmat(mat_path):
            if name.startswith("__"):
                continue
            summary[name] = {
                "shape": tuple(shape),
                "dtype": dtype_name,
                "loader": "scipy",
            }
        return summary
    except Exception as exc:
        if not _is_hdf5_error(exc):
            raise

    summary = {}
    with h5py.File(mat_path, "r") as handle:
        def visit(name: str, obj: Any) -> None:
            key = name.split("/")[-1]
            if key.startswith("__") or not hasattr(obj, "shape"):
                return
            summary[key] = {
                "shape": tuple(obj.shape),
                "dtype": str(obj.dtype),
                "loader": "h5py",
            }

        handle.visititems(visit)
    return summary


def save_npz(path: str | Path, **arrays: np.ndarray) -> Path:
    """Save arrays to a compressed NPZ file, creating parent directories."""
    out_path = normalize_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    return out_path


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load an NPZ file into an eager dictionary of arrays."""
    npz_path = normalize_path(path)
    with np.load(npz_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def available_keys(summary: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Return sorted MAT variable names from a summary mapping."""
    return sorted(summary)
