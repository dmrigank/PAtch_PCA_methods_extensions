"""Evaluation metrics for field predictions."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from skimage.metrics import structural_similarity


def mse(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean squared error over all samples and pixels."""
    pred, true = _as_matching_arrays(pred, true)
    return float(np.mean((pred - true) ** 2))


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean absolute error over all samples and pixels."""
    pred, true = _as_matching_arrays(pred, true)
    return float(np.mean(np.abs(pred - true)))


def mre(pred: np.ndarray, true: np.ndarray, eps: float = 1e-12) -> float:
    """Relative L2 error over the whole batch."""
    pred, true = _as_matching_arrays(pred, true)
    numerator = np.linalg.norm((pred - true).reshape(pred.shape[0], -1))
    denominator = np.linalg.norm(true.reshape(true.shape[0], -1))
    return float(numerator / max(float(denominator), eps))


def ssim_batch(
    pred: np.ndarray,
    true: np.ndarray,
    *,
    data_range: float | None = None,
) -> tuple[float, float]:
    """Compute sample-wise SSIM and return ``(mean, std)``."""
    pred, true = _as_matching_arrays(pred, true)
    if pred.ndim != 3:
        raise ValueError(f"SSIM expects scalar fields with shape (N, H, W), got {pred.shape}.")

    scores: list[float] = []
    for pred_i, true_i in zip(pred, true):
        sample_range = data_range
        if sample_range is None:
            sample_range = float(np.max(true_i) - np.min(true_i))
            if sample_range == 0.0:
                sample_range = 1.0
        scores.append(
            float(
                structural_similarity(
                    true_i,
                    pred_i,
                    data_range=sample_range,
                )
            )
        )
    return float(np.mean(scores)), float(np.std(scores))


def aggregate_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    metrics: Iterable[str] | None = None,
    *,
    data_range: float | None = None,
) -> dict[str, float]:
    """Compute a configured set of aggregate metrics."""
    requested = list(metrics) if metrics is not None else ["mse", "mae", "mre", "ssim"]
    results: dict[str, float] = {}
    for name in requested:
        if name == "mse":
            results["mse"] = mse(pred, true)
        elif name == "mae":
            results["mae"] = mae(pred, true)
        elif name == "mre":
            results["mre"] = mre(pred, true)
        elif name == "ssim":
            mean, std = ssim_batch(pred, true, data_range=data_range)
            results["ssim_mean"] = mean
            results["ssim_std"] = std
        else:
            raise ValueError(f"Unsupported metric {name!r}.")
    return results


def _as_matching_arrays(pred: np.ndarray, true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError(f"Prediction and target shapes differ: {pred.shape} != {true.shape}.")
    if pred.ndim == 0:
        raise ValueError("Metrics expect at least one sample dimension.")
    return pred, true
