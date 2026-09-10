from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_physics_ablation_manuscript_outputs_run_on_fixture(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    results_dir = tmp_path / "physics_loss_ablation"
    _write_fixture(results_dir)

    command = [
        sys.executable,
        str(root / "scripts" / "make_physics_ablation_outputs.py"),
        "--results-dir",
        str(results_dir),
        "--formats",
        "png",
        "--expected-seeds",
        "2",
        "--skip-qualitative",
    ]
    completed = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout

    expected = (
        "figures/main/physics_loss_ladder.png",
        "figures/main/physics_accuracy_continuity_pareto.png",
        "figures/appendix/physics_cost_breakdown.png",
        "figures/appendix/physics_weight_screen.png",
        "tables/main/physics_loss_ablation_main.csv",
        "tables/main/physics_loss_ablation_main.tex",
        "tables/appendix/physics_loss_ablation_full.csv",
        "tables/appendix/physics_weight_screen.tex",
        "tables/appendix/physics_loss_ablation_per_seed.csv",
        "physics_ablation_manifest.json",
    )
    for relative_path in expected:
        assert (results_dir / relative_path).is_file()
    latex = (results_dir / "tables/main/physics_loss_ablation_main.tex").read_text(
        encoding="utf-8"
    )
    assert r"\ensuremath{\pm}" in latex


def _write_fixture(results_dir: Path) -> None:
    stages = (
        "a0_latent_baseline",
        "a1_reconstruction",
        "a2_value",
        "a3_value_flux",
        "a4_value_flux_spectral",
        "a5_full_pde",
    )
    rows: list[dict[str, object]] = []
    for resolution in (128, 256):
        for stage_index, stage in enumerate(stages):
            for seed in (0, 1):
                scale = 1.0 + 0.02 * seed + 0.01 * stage_index
                rows.append(
                    {
                        "run_id": f"{resolution}_{stage}_{seed}",
                        "run_dir": str(results_dir / stage / f"seed_{seed}"),
                        "config_hash": f"hash-{resolution}-{stage}-{seed}",
                        "resolution": resolution,
                        "seed": seed,
                        "baseline_model": stage,
                        "ablation_active_terms": "recon",
                        "metric_mre": 0.011 * scale,
                        "metric_ssim_mean": 0.997 / scale,
                        "metric_mse": 2.0e-9 * scale,
                        "metric_mae": 3.0e-5 * scale,
                        "metric_interface_jump": 1.0e-4 / scale,
                        "metric_interface_flux_jump": 4.0e-3 / scale,
                        "metric_interface_value_trace_error": 2.0e-5 / scale,
                        "metric_interface_flux_trace_error": 3.5e-3 / scale,
                        "metric_relative_spectrum_error": 5.0e-3 / scale,
                        "metric_poisson_residual_rms": 0.62 / scale,
                        "timing_pca_fit": 5.0,
                        "timing_latent_transform": 2.0,
                        "timing_nn_train": 170.0 + 20.0 * stage_index,
                        "timing_inference": 0.3,
                        "timing_end_to_end": 200.0 + 25.0 * stage_index,
                        "invocation_timing_nn_train": 80.0 + 20.0 * stage_index,
                        "invocation_timing_end_to_end": 100.0 + 25.0 * stage_index,
                        "metric_calibrated_weight_interface_value": 1.0,
                        "metric_calibrated_weight_interface_flux": 0.1,
                        "metric_calibrated_weight_spectral": 0.2,
                        "metric_calibrated_weight_pde_residual": 1.0e-4,
                    }
                )
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(results_dir / "results.csv", index=False)

    screen_rows: list[dict[str, object]] = []
    specs = (
        (
            "interface_value",
            "a2_value",
            "validation_metric_interface_value_trace_error",
        ),
        (
            "interface_flux",
            "a3_value_flux",
            "validation_metric_interface_flux_trace_error",
        ),
        (
            "spectral",
            "a4_value_flux_spectral",
            "validation_metric_relative_spectrum_error",
        ),
        (
            "pde_residual",
            "a5_full_pde",
            "validation_metric_poisson_residual_rms",
        ),
    )
    for term, prefix, target_column in specs:
        ratios = (0.01,) if term == "pde_residual" else (0.0125, 0.05, 0.2)
        for index, ratio in enumerate(ratios):
            row: dict[str, object] = {
                "baseline_model": f"{prefix}_{index}",
                "validation_metric_mre": 0.0105 + 0.0002 * index,
                f"ablation_target_ratio_{term}": ratio,
                f"metric_calibrated_weight_{term}": ratio * 10.0,
                target_column: 0.01 / (index + 1),
            }
            screen_rows.append(row)
    screen_dir = results_dir / "screen"
    screen_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(screen_rows).to_csv(screen_dir / "results.csv", index=False)

    selection = {
        "reconstruction_validation_mre": 0.0105,
        "mre_guardrail_fraction": 0.05,
        "terms": {
            "interface_value": {
                "selected_target_ratio": 0.2,
                "guardrail_had_eligible_candidate": True,
            },
            "interface_flux": {
                "selected_target_ratio": 0.2,
                "guardrail_had_eligible_candidate": True,
            },
            "spectral": {
                "selected_target_ratio": 0.2,
                "guardrail_had_eligible_candidate": False,
            },
            "pde_residual": {
                "selected_target_ratio": 0.01,
                "guardrail_passed": False,
            },
        },
    }
    (results_dir / "selection.json").write_text(
        json.dumps(selection),
        encoding="utf-8",
    )
