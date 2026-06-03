"""Deterministic train/validation/test split helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lpcanet.data.io import save_npz


def make_splits(
    n: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
    shuffle: bool = True,
) -> dict[str, np.ndarray]:
    """Create deterministic train/validation/test index splits."""
    if n < 0:
        raise ValueError("n must be nonnegative.")
    total = train_frac + val_frac + test_frac
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split fractions must sum to 1.0, got {total}.")
    for name, frac in {
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": test_frac,
    }.items():
        if frac < 0:
            raise ValueError(f"{name} must be nonnegative.")

    indices = np.arange(n, dtype=np.int64)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    return {
        "train_idx": train_idx.astype(np.int64, copy=False),
        "val_idx": val_idx.astype(np.int64, copy=False),
        "test_idx": test_idx.astype(np.int64, copy=False),
    }


def save_splits(path: str | Path, splits: dict[str, np.ndarray]) -> Path:
    """Save split indices to an NPZ file."""
    required = {"train_idx", "val_idx", "test_idx"}
    missing = required.difference(splits)
    if missing:
        raise KeyError(f"Missing split arrays: {sorted(missing)}")
    return save_npz(path, **{key: splits[key].astype(np.int64) for key in required})
