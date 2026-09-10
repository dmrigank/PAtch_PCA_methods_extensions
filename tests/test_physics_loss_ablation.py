from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from torch import nn

from lpcanet.train.factorial import (
    FactorialRun,
    _select_physics_candidate,
    plan_physics_loss_ablation_runs,
)
from lpcanet.train.loop import calibrate_physical_loss_weights


def test_physics_loss_ablation_plan_has_41_paired_cells() -> None:
    root = Path(__file__).resolve().parents[1]
    plan = plan_physics_loss_ablation_runs(
        root=root,
        experiment="physics_loss_ablation",
    )

    assert len(plan) == 41
    assert len([item for item in plan if item["phase"] == "reference"]) == 6
    assert len([item for item in plan if item["phase"] == "screen"]) == 10
    assert len([item for item in plan if item["phase"] == "confirm"]) == 25
    assert {(item["resolution"], item["seed"]) for item in plan if item["phase"] == "reference"} == {
        (128, 0),
        (128, 1),
        (128, 2),
        (256, 0),
        (256, 1),
        (256, 2),
    }
    screen_cells = {
        item["cell"] for item in plan if item["phase"] == "screen"
    }
    assert screen_cells == {
        "a1_reconstruction",
        "a2_value_low",
        "a2_value_default",
        "a2_value_high",
        "a3_value_flux_low",
        "a3_value_flux_default",
        "a3_value_flux_high",
        "a4_value_flux_spectral_default",
        "a4_value_flux_spectral_high",
        "a5_full_pde_low",
    }


def test_loss_calibration_matches_requested_initial_contribution_ratios() -> None:
    rng = np.random.default_rng(9)
    prediction = rng.normal(size=(12, 8, 8)).astype(np.float32)
    truth = (0.8 * prediction + 0.2 * rng.normal(size=prediction.shape)).astype(
        np.float32
    )
    forcing = rng.normal(size=prediction.shape).astype(np.float32)
    ratios = {
        "interface_value": 0.05,
        "interface_flux": 0.025,
        "spectral": 0.1,
        "pde_residual": 0.01,
    }
    weights, components = calibrate_physical_loss_weights(
        nn.Identity(),
        nn.Identity(),
        prediction,
        truth,
        forcing_val=forcing,
        loss_config={
            "reconstruction": "relative_l2",
            "interface_target": "truth",
            "active_terms": [
                "recon",
                "interface_value",
                "interface_flux",
                "spectral",
                "pde_residual",
            ],
            "weights": {"recon": 1.0},
            "patch_size": 4,
            "stride": 4,
            "dx": 1.0 / 7.0,
            "pde": "poisson",
            "pde_reduction": "rms",
            "spectral_high_k_weight_power": 1.0,
        },
        target_ratios=ratios,
        device="cpu",
        batch_size=4,
        max_samples=12,
    )

    reconstruction = components["recon"]
    for term, ratio in ratios.items():
        contribution = weights[term] * components[term] / reconstruction
        assert contribution == pytest.approx(ratio, rel=1.0e-6)


def test_weight_selection_uses_validation_metrics_and_mre_guardrail() -> None:
    candidates = [
        (
            _candidate("value_low", 0.0125),
            _record(
                "value_low",
                validation_mre=0.101,
                validation_target=0.03,
                test_target=0.001,
            ),
        ),
        (
            _candidate("value_default", 0.05),
            _record(
                "value_default",
                validation_mre=0.103,
                validation_target=0.02,
                test_target=0.5,
            ),
        ),
        (
            _candidate("value_high", 0.2),
            _record(
                "value_high",
                validation_mre=0.12,
                validation_target=0.001,
                test_target=0.0001,
            ),
        ),
    ]

    selected, _record_value, report = _select_physics_candidate(
        candidates,
        target_metric="interface_value_trace_error",
        reference_mre=0.1,
        guardrail_fraction=0.05,
    )

    assert selected.baseline_model == "value_default"
    assert report["guardrail_had_eligible_candidate"] is True
    assert report["selected_target_ratio"] == pytest.approx(0.05)


def _candidate(name: str, ratio: float) -> FactorialRun:
    config = {
        "loss": {
            "weight_calibration": {
                "target_ratios": {"interface_value": ratio}
            }
        }
    }
    return FactorialRun(
        run_id=name,
        config=config,
        config_hash=name,
        seed=0,
        mechanisms={
            "two_scale": True,
            "coupling": False,
            "in_loop_loss": True,
        },
        resolution=128,
        run_dir=Path(name),
        baseline_model=name,
    )


def _record(
    name: str,
    *,
    validation_mre: float,
    validation_target: float,
    test_target: float,
) -> dict[str, object]:
    return {
        "run_id": name,
        "validation_metrics": {
            "mre": validation_mre,
            "interface_value_trace_error": validation_target,
        },
        "metrics": {
            "interface_value_trace_error": test_target,
        },
    }
