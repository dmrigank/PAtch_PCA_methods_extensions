from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def test_headline_poisson_outputs_run_on_fixture(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    results_dir = tmp_path / "headline"
    _write_fixture(results_dir)

    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "make_headline_poisson_outputs.py"),
            "--results-dir",
            str(results_dir),
            "--formats",
            "png",
            "--expected-seeds",
            "2",
            "--skip-qualitative",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout

    expected = (
        "figures/main/headline_accuracy.png",
        "figures/main/headline_interfaces.png",
        "figures/main/headline_quality_cost.png",
        "figures/appendix/headline_representation_gap.png",
        "figures/appendix/headline_pca_breakdown.png",
        "tables/main/headline_poisson_128.csv",
        "tables/main/headline_poisson_128.tex",
        "tables/appendix/headline_poisson_128_full.csv",
        "tables/appendix/headline_paired_comparisons.csv",
        "tables/appendix/headline_per_seed.csv",
        "tables/appendix/headline_common_seam_seed0.csv",
        "tables/appendix/headline_pca_breakdown.csv",
        "headline_poisson_128_manifest.json",
    )
    for relative_path in expected:
        assert (results_dir / relative_path).is_file()
    latex = (results_dir / "tables/main/headline_poisson_128.tex").read_text(
        encoding="utf-8"
    )
    assert r"\ensuremath{\pm}" in latex


def _write_fixture(results_dir: Path) -> None:
    methods = (
        "global_pca",
        "plain_l2l",
        "overlap_l2l",
        "l2l_refinement",
        "fno",
        "two_scale",
        "two_scale_interface",
    )
    rows: list[dict[str, object]] = []
    for method_index, method in enumerate(methods):
        for seed in (0, 1):
            mre = (0.08 / (method_index + 1)) * (1.0 + 0.01 * seed)
            oracle = None if method in {"fno", "l2l_refinement"} else 0.8 * mre
            row: dict[str, object] = {
                "run_id": f"{method}_{seed}",
                "run_dir": str(results_dir / method / f"seed_{seed}"),
                "config_hash": f"hash-{method}-{seed}",
                "seed": seed,
                "resolution": 128,
                "baseline_model": method,
                "metric_mre": mre,
                "metric_pca_oracle_mre": oracle,
                "metric_ssim_mean": 1.0 - mre,
                "metric_pca_oracle_ssim_mean": None if oracle is None else 1.0 - oracle,
                "metric_mse": mre**2,
                "metric_mae": mre * 0.1,
                "metric_relative_spectrum_error": mre * 0.2,
                "metric_interface_jump": mre * 0.01,
                "metric_interface_flux_jump": mre * 0.1,
                "metric_interface_value_trace_error": mre * 0.008,
                "metric_interface_flux_trace_error": mre * 0.08,
                "metric_poisson_residual_mean_abs": mre * 5.0,
                "metric_poisson_residual_rms": mre * 6.0,
                "timing_pca_fit": 0.0 if method == "fno" else 2.0 + method_index,
                "timing_latent_transform": 1.0,
                "timing_nn_train": 20.0 + method_index,
                "timing_inference": 0.2,
                "timing_end_to_end": 35.0 + 5.0 * method_index,
                "invocation_timing_nn_train": 15.0 + method_index,
                "invocation_timing_end_to_end": 25.0 + method_index,
                "pca_timings_fit_input_pca": 0.5,
                "pca_timings_fit_output_pca": 0.7,
                "pca_timings_fit_coarse_svd": 0.2 if "two_scale" in method else None,
                "pca_timings_fit_output_residual_pca": 1.0 if "two_scale" in method else None,
            }
            rows.append(row)
            if seed == 0:
                run_dir = Path(str(row["run_dir"]))
                run_dir.mkdir(parents=True, exist_ok=True)
                coordinate = np.linspace(0.0, 1.0, 64, dtype=np.float32)
                xx, yy = np.meshgrid(coordinate, coordinate, indexing="xy")
                truth = np.stack(
                    [
                        np.sin(np.pi * xx) * np.sin(np.pi * yy),
                        np.sin(2.0 * np.pi * xx) * np.sin(np.pi * yy),
                    ]
                ).astype(np.float32)
                prediction = truth * np.float32(1.0 - mre)
                np.savez_compressed(
                    run_dir / "predictions_test.npz",
                    y_true=truth,
                    y_pred=prediction,
                    sample_indices=np.array([10, 11], dtype=np.int64),
                )
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(results_dir / "results.csv", index=False)
