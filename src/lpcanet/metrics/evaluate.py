#!/usr/bin/env python
"""Evaluate a saved PCA-Net run directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from lpcanet.data.io import load_npz
from lpcanet.metrics.interface import interface_flux_jump, interface_jump
from lpcanet.metrics.metrics import aggregate_metrics
from lpcanet.metrics.residuals import darcy_residual, poisson_residual
from lpcanet.metrics.spectral import relative_spectrum_error
from lpcanet.train.checkpointing import save_metrics
from lpcanet.utils.config import load_config
from lpcanet.utils.paths import normalize_path


def evaluate_run(run_dir: str | Path) -> dict[str, float]:
    """Compute evaluation metrics for a run directory and update metrics files."""
    run_path = normalize_path(run_dir)
    predictions = load_npz(run_path / "predictions_test.npz")
    pred = predictions["y_pred"]
    true = predictions["y_true"]

    config = _load_run_config(run_path)
    requested = config.get("evaluation", {}).get(
        "metrics",
        ["mse", "mae", "mre", "ssim", "relative_spectrum_error", "interface_jump", "poisson_residual"],
    )
    core_requested = [metric for metric in requested if metric in {"mse", "mae", "mre", "ssim"}]
    metrics = aggregate_metrics(pred, true, core_requested)

    if "spectrum_error" in requested or "relative_spectrum_error" in requested:
        metrics["relative_spectrum_error"] = relative_spectrum_error(pred, true)

    patches = config.get("patches", {})
    dx = _resolve_dx(config, pred.shape[-1])
    if patches and ("interface_jump" in requested or "interface_jump_mean" in requested):
        metrics["interface_jump"] = interface_jump(
            pred,
            patch_size=int(patches["patch_size"]),
            stride=int(patches["stride"]),
            include_edges=bool(patches.get("include_edges", False)),
        )
    if patches and "interface_flux_jump" in requested:
        metrics["interface_flux_jump"] = interface_flux_jump(
            pred,
            patch_size=int(patches["patch_size"]),
            stride=int(patches["stride"]),
            dx=dx,
            include_edges=bool(patches.get("include_edges", False)),
        )

    dataset_name = str(config.get("dataset", {}).get("name", "")).lower()
    if "poisson_residual" in requested and dataset_name == "poisson" and "x_input" in predictions:
        residual = poisson_residual(pred, predictions["x_input"], dx)
        metrics["poisson_residual_mean_abs"] = float(np.mean(np.abs(residual)))
        metrics["poisson_residual_rms"] = float(np.sqrt(np.mean(residual**2)))

    if "darcy_residual" in requested:
        coeff = _first_existing(predictions, ["a", "coeff", "coefficient", "permeability"])
        forcing = _first_existing(predictions, ["f", "source", "forcing", "x_input"])
        if coeff is not None and forcing is not None:
            residual = darcy_residual(pred, coeff, forcing, dx)
            metrics["darcy_residual_mean_abs"] = float(np.mean(np.abs(residual)))
            metrics["darcy_residual_rms"] = float(np.sqrt(np.mean(residual**2)))

    previous = _load_existing_metrics(run_path / "metrics.json")
    previous.update(metrics)
    save_metrics(previous, run_path)
    return previous


def _load_run_config(run_path: Path) -> dict[str, Any]:
    config_path = run_path / "config.yaml"
    if not config_path.exists():
        return {}
    return load_config(config_path)


def _load_existing_metrics(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return dict(json.load(handle))


def _resolve_dx(config: dict[str, Any], grid_size: int) -> float:
    evaluation = config.get("evaluation", {})
    if "dx" in evaluation:
        return float(evaluation["dx"])
    dataset = config.get("dataset", {})
    configured_grid = int(dataset.get("grid_size", grid_size))
    return 1.0 / float(configured_grid - 1)


def _first_existing(predictions: dict[str, np.ndarray], keys: list[str]) -> np.ndarray | None:
    for key in keys:
        if key in predictions:
            return predictions[key]
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Training run directory to evaluate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_run(args.run_dir)
    print(f"Updated metrics for {args.run_dir}")
    for key in sorted(metrics):
        print(f"{key}: {metrics[key]}")


if __name__ == "__main__":
    main()
