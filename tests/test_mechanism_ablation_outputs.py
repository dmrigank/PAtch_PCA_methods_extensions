from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def test_mechanism_ablation_outputs_run_on_fixture(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    results_dir = tmp_path / "mechanism"
    _write_fixture(results_dir)

    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "make_mechanism_ablation_outputs.py"),
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
        "figures/main/mechanism_outcomes.png",
        "figures/main/mechanism_interaction.png",
        "figures/main/mechanism_interfaces.png",
        "figures/main/mechanism_qualitative.png",
        "figures/appendix/mechanism_quality_cost.png",
        "figures/appendix/mechanism_common_seams_seed0.png",
        "figures/appendix/mechanism_pca_breakdown.png",
        "tables/main/mechanism_ablation_main.csv",
        "tables/main/mechanism_ablation_main.tex",
        "tables/appendix/mechanism_ablation_full.csv",
        "tables/appendix/mechanism_paired_effects.csv",
        "tables/appendix/mechanism_per_seed.csv",
        "tables/appendix/mechanism_common_seams_seed0.csv",
        "tables/appendix/mechanism_pca_breakdown.csv",
        "summary.md",
        "mechanism_ablation_manifest.json",
    )
    for relative_path in expected:
        assert (results_dir / relative_path).is_file()

    summary = (results_dir / "summary.md").read_text(encoding="utf-8")
    assert "Two-scale is the dominant accuracy mechanism" in summary
    assert "2 paired seeds" in summary
    latex = (results_dir / "tables/main/mechanism_ablation_main.tex").read_text(
        encoding="utf-8"
    )
    assert r"\ensuremath{\pm}" in latex


def _write_fixture(results_dir: Path) -> None:
    methods = (
        "plain_l2l",
        "plain_l2l_interface",
        "two_scale",
        "two_scale_interface",
        "overlap_l2l",
    )
    scales = {
        "plain_l2l": 1.0,
        "plain_l2l_interface": 0.9,
        "two_scale": 0.3,
        "two_scale_interface": 0.2,
        "overlap_l2l": 0.5,
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
    for method_index, method in enumerate(methods):
        for seed in (0, 1):
            scale = scales[method] * (1.0 + 0.01 * seed)
            mre = 0.05 * scale
            row: dict[str, object] = {
                "run_id": f"{method}_{seed}",
                "run_dir": str(results_dir / method / f"seed_{seed}"),
                "config_hash": f"hash-{method}-{seed}",
                "seed": seed,
                "resolution": 128,
                "baseline_model": method,
                "metric_mre": mre,
                "metric_ssim_mean": 1.0 - mre,
                "metric_mse": mre**2,
                "metric_mae": mre * 0.1,
                "metric_pca_oracle_mre": 0.8 * mre,
                "metric_relative_spectrum_error": mre * 0.2,
                "metric_interface_jump": mre * 0.01,
                "metric_interface_flux_jump": mre * 0.1,
                "metric_interface_value_trace_error": mre * 0.008,
                "metric_interface_flux_trace_error": mre * 0.08,
                "metric_poisson_residual_mean_abs": mre * 5.0,
                "metric_poisson_residual_rms": mre * 6.0,
                "timing_pca_fit": 2.0 + method_index,
                "timing_latent_transform": 1.0,
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
                    y_true=truth,
                    y_pred=prediction,
                    sample_indices=np.array([10, 11], dtype=np.int64),
                )
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(results_dir / "results.csv", index=False)
