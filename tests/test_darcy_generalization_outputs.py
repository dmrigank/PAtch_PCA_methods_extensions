from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def test_darcy_generalization_outputs_run_on_fixture(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    results_dir = tmp_path / "darcy"
    _write_fixture(results_dir)

    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "make_darcy_generalization_outputs.py"),
            "--results-dir",
            str(results_dir),
            "--formats",
            "png",
            "--expected-seeds",
            "2",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout

    expected = (
        "figures/main/darcy_accuracy_physics.png",
        "figures/main/darcy_interfaces.png",
        "figures/main/darcy_representation_gap.png",
        "figures/main/darcy_qualitative.png",
        "figures/appendix/darcy_quality_cost.png",
        "figures/appendix/darcy_common_seams_seed0.png",
        "figures/appendix/darcy_pca_breakdown.png",
        "tables/main/darcy_generalization_main.csv",
        "tables/main/darcy_generalization_main.tex",
        "tables/appendix/darcy_generalization_full.csv",
        "tables/appendix/darcy_paired_effects.csv",
        "tables/appendix/darcy_per_seed.csv",
        "tables/appendix/darcy_common_seams_seed0.csv",
        "tables/appendix/darcy_pca_breakdown.csv",
        "summary.md",
        "darcy_generalization_manifest.json",
    )
    for relative_path in expected:
        assert (results_dir / relative_path).is_file()

    summary = (results_dir / "summary.md").read_text(encoding="utf-8")
    assert "latent prediction, not output compression" in summary
    assert "2 paired seeds" in summary
    latex = (results_dir / "tables/main/darcy_generalization_main.tex").read_text(
        encoding="utf-8"
    )
    assert r"\ensuremath{\pm}" in latex


def _write_fixture(results_dir: Path) -> None:
    methods = (
        "global_pca",
        "plain_l2l",
        "overlap_l2l",
        "fno",
        "two_scale",
        "two_scale_interface",
    )
    scales = {
        "global_pca": 0.8,
        "plain_l2l": 1.0,
        "overlap_l2l": 0.7,
        "fno": 0.1,
        "two_scale": 0.75,
        "two_scale_interface": 0.7,
    }
    rows: list[dict[str, object]] = []
    coordinate = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    xx, yy = np.meshgrid(coordinate, coordinate, indexing="xy")
    truth = np.stack(
        [
            np.sin(np.pi * xx) * np.sin(np.pi * yy),
            np.sin(2.0 * np.pi * xx) * np.sin(np.pi * yy),
        ]
    ).astype(np.float32)
    coefficient = np.stack(
        [(xx > 0.5).astype(np.float32), (yy > 0.5).astype(np.float32)]
    )
    for method_index, method in enumerate(methods):
        for seed in (0, 1):
            scale = scales[method] * (1.0 + 0.01 * seed)
            mre = 0.05 * scale
            oracle = None if method == "fno" else 0.4 * mre
            row: dict[str, object] = {
                "run_id": f"{method}_{seed}",
                "run_dir": str(results_dir / method / f"seed_{seed}"),
                "config_hash": f"hash-{method}-{seed}",
                "seed": seed,
                "resolution": 256,
                "dataset": "darcy",
                "baseline_model": method,
                "metric_mre": mre,
                "metric_ssim_mean": 1.0 - mre,
                "metric_mse": mre**2,
                "metric_mae": mre * 0.1,
                "metric_pca_oracle_mre": oracle,
                "metric_relative_spectrum_error": mre * 0.2,
                "metric_interface_jump": mre * 0.01,
                "metric_interface_flux_jump": mre * 0.1,
                "metric_interface_value_trace_error": mre * 0.008,
                "metric_interface_flux_trace_error": mre * 0.08,
                "metric_darcy_residual_mean_abs": mre * 5.0,
                "metric_darcy_residual_rms": mre * 6.0,
                "timing_pca_fit": 0.0 if method == "fno" else 2.0 + method_index,
                "timing_latent_transform": 0.0 if method == "fno" else 1.0,
                "timing_nn_train": 20.0 + method_index,
                "timing_inference": 0.2,
                "timing_end_to_end": 35.0 + 5.0 * method_index,
                "invocation_timing_nn_train": 15.0 + method_index,
                "invocation_timing_end_to_end": 25.0 + method_index,
                "pca_timings_fit_input_pca": 0.5,
                "pca_timings_fit_output_pca": 0.7,
                "pca_timings_fit_coarse_svd": (
                    0.2 if "two_scale" in method else None
                ),
                "pca_timings_fit_output_residual_pca": (
                    1.0 if "two_scale" in method else None
                ),
            }
            rows.append(row)
            if seed == 0:
                run_dir = Path(str(row["run_dir"]))
                run_dir.mkdir(parents=True, exist_ok=True)
                prediction = truth * np.float32(1.0 - mre)
                np.savez_compressed(
                    run_dir / "predictions_test.npz",
                    a=coefficient,
                    x_input=coefficient,
                    y_true=truth,
                    y_pred=prediction,
                    sample_indices=np.array([10, 11], dtype=np.int64),
                )
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(results_dir / "results.csv", index=False)
