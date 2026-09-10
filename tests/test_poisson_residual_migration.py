"""Tests for targeted Poisson residual archive migration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lpcanet.metrics.migration import (
    migrate_poisson_residuals,
    plan_poisson_residual_migration,
)
from lpcanet.train.checkpointing import save_metrics
from lpcanet.utils.config import save_config


def test_migration_updates_only_residual_fields_and_rollup(tmp_path: Path) -> None:
    study = tmp_path / "study"
    run = study / "method" / "seed_0"
    affected = study / "a5" / "seed_0"
    run.mkdir(parents=True)
    affected.mkdir(parents=True)
    config = _config(run, in_loop=False, pde_weight=0.0)
    save_config(config, run)
    save_config(_config(affected, in_loop=True, pde_weight=0.01), affected)

    resolution = 10
    dx = 1.0 / float(resolution - 1)
    prediction = np.zeros((2, resolution, resolution), dtype=np.float64)
    prediction[:, 1:-1, 1:-1] = 0.25
    forcing = np.zeros_like(prediction)
    forcing[:, 1:-1, 1:-1] = _laplacian(prediction, dx)
    np.savez(
        run / "predictions_test.npz",
        x_input=forcing,
        y_true=prediction,
        y_pred=prediction,
    )
    save_metrics(
        {
            "mre": 0.125,
            "poisson_residual_mean_abs": 8.0,
            "poisson_residual_rms": 9.0,
        },
        run,
    )
    records = [
        _record(run, "method"),
        _record(affected, "a5_full_pde"),
    ]
    with (study / "results.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    plan = plan_poisson_residual_migration(study)
    assert [action.recovery for action in plan] == ["retrain", "saved_predictions"] or [
        action.recovery for action in plan
    ] == ["saved_predictions", "retrain"]

    audit = migrate_poisson_residuals(
        study,
        apply=True,
        include_validation=False,
        device="cpu",
    )
    assert {entry["recovery"] for entry in audit} == {"saved_predictions", "retrain"}

    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["mre"] == 0.125
    assert metrics["poisson_residual_mean_abs"] < 1.0e-12
    assert metrics["poisson_residual_rms"] < 1.0e-12
    assert metrics["pde_metrics_version"] == 2
    assert metrics["poisson_residual_convention"] == "delta_u_equals_f"

    rollup = pd.read_csv(study / "results.csv")
    migrated = rollup[rollup["baseline_model"] == "method"].iloc[0]
    assert migrated["metric_mre"] == 0.125
    assert migrated["metric_poisson_residual_rms"] < 1.0e-12
    assert migrated["metric_pde_metrics_version"] == 2


def _config(run_dir: Path, *, in_loop: bool, pde_weight: float) -> dict[str, object]:
    return {
        "experiment": {"seed": 0, "output_dir": str(run_dir)},
        "dataset": {
            "name": "poisson",
            "grid_size": 10,
            "poisson_convention": "delta_u_equals_f",
        },
        "mechanisms": {"in_loop_loss": in_loop},
        "loss": {"weights": {"pde_residual": pde_weight}},
        "evaluation": {"metrics": ["poisson_residual"]},
    }


def _record(run_dir: Path, method: str) -> dict[str, object]:
    return {
        "run_id": f"{method}_seed0",
        "config_hash": "fixture",
        "run_dir": str(run_dir),
        "seed": 0,
        "resolution": 10,
        "dataset": "poisson",
        "baseline_model": method,
        "mechanisms": {"two_scale": False, "coupling": False, "in_loop_loss": False},
        "metrics": {
            "mre": 0.125,
            "poisson_residual_mean_abs": 8.0,
            "poisson_residual_rms": 9.0,
        },
        "validation_metrics": {},
        "pca": {},
        "timings": {},
        "invocation_timings": {},
        "timing_accounting": {},
        "times": {},
    }


def _laplacian(field: np.ndarray, dx: float) -> np.ndarray:
    return (
        field[:, 2:, 1:-1]
        + field[:, :-2, 1:-1]
        + field[:, 1:-1, 2:]
        + field[:, 1:-1, :-2]
        - 4.0 * field[:, 1:-1, 1:-1]
    ) / (dx**2)
