from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def test_sample_efficiency_manuscript_outputs_run_on_fixture(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    results_dir = tmp_path / "sample_efficiency"
    _write_fixture(results_dir)

    command = [
        sys.executable,
        str(root / "scripts" / "make_sample_efficiency_outputs.py"),
        "--results-dir",
        str(results_dir),
        "--formats",
        "png",
        "--expected-seeds",
        "2",
        "--common-seam-stride",
        "4",
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
        "figures/main/sample_efficiency_accuracy.png",
        "figures/main/sample_efficiency_representation_gap.png",
        "figures/main/sample_efficiency_artifacts.png",
        "figures/appendix/sample_efficiency_cost.png",
        "figures/appendix/sample_efficiency_pca_breakdown.png",
        "figures/appendix/sample_efficiency_qualitative.png",
        "tables/main/sample_efficiency_main.csv",
        "tables/main/sample_efficiency_main.tex",
        "tables/appendix/sample_efficiency_full.csv",
        "tables/appendix/sample_efficiency_cost.tex",
        "tables/appendix/sample_efficiency_relative_changes.csv",
        "tables/appendix/sample_efficiency_per_seed.csv",
        "tables/appendix/sample_efficiency_common_seam_seed0.csv",
        "tables/appendix/sample_efficiency_pca_breakdown.csv",
        "tables/appendix/sample_efficiency_pca_breakdown.tex",
        "sample_efficiency_manifest.json",
    )
    for relative_path in expected:
        assert (results_dir / relative_path).is_file()
    latex = (results_dir / "tables/main/sample_efficiency_main.tex").read_text(
        encoding="utf-8"
    )
    assert r"\ensuremath{\pm}" in latex

    common = pd.read_csv(
        results_dir / "tables/appendix/sample_efficiency_common_seam_seed0.csv"
    )
    assert len(common) == 8
    assert set(common["Diagnostic stride"]) == {4}


def _write_fixture(results_dir: Path) -> None:
    methods = (
        "plain_l2l",
        "overlap_l2l",
        "two_scale",
        "two_scale_interface",
    )
    rows: list[dict[str, object]] = []
    grid = np.linspace(0.0, 1.0, 16, dtype=np.float32)
    xx, yy = np.meshgrid(grid, grid, indexing="ij")
    truth_base = np.sin(np.pi * xx) * np.sin(np.pi * yy)
    for size_index, sample_size in enumerate((1250, 8000)):
        for method_index, method in enumerate(methods):
            for seed in (0, 1):
                scale = 1.0 + 0.02 * seed
                error = (0.05 - 0.006 * method_index) / (1.0 + 0.4 * size_index)
                run_dir = (
                    results_dir
                    / "poisson"
                    / "resolution_16"
                    / f"train_{sample_size}"
                    / method
                    / f"seed_{seed}"
                )
                run_dir.mkdir(parents=True, exist_ok=True)
                if seed == 0:
                    truth = np.stack([truth_base, 0.9 * truth_base]).astype(np.float32)
                    prediction = truth + error * np.stack(
                        [
                            np.cos(2.0 * np.pi * xx),
                            np.cos(2.0 * np.pi * yy),
                        ]
                    ).astype(np.float32)
                    np.savez(
                        run_dir / "predictions_test.npz",
                        y_true=truth,
                        y_pred=prediction,
                        sample_indices=np.array([101, 102]),
                    )
                rows.append(
                    {
                        "run_id": f"{sample_size}_{method}_{seed}",
                        "run_dir": str(run_dir),
                        "config_hash": f"hash-{sample_size}-{method}-{seed}",
                        "seed": seed,
                        "resolution": 16,
                        "baseline_model": method,
                        "ablation_sample_size": sample_size,
                        "metric_mre": error * scale,
                        "metric_pca_oracle_mre": 0.01 * scale,
                        "metric_ssim_mean": 0.98 / scale,
                        "metric_pca_oracle_ssim_mean": 0.995 / scale,
                        "metric_mse": error**2 * scale,
                        "metric_mae": error * 0.1 * scale,
                        "metric_relative_spectrum_error": error * 0.2 * scale,
                        "metric_interface_jump": error * 0.01 * scale,
                        "metric_interface_flux_jump": error * scale,
                        "metric_interface_value_trace_error": error * 0.008 * scale,
                        "metric_interface_flux_trace_error": error * 0.8 * scale,
                        "metric_poisson_residual_mean_abs": error * 5.0 * scale,
                        "metric_poisson_residual_rms": error * 6.0 * scale,
                        "timing_pca_fit": 1.0 + method_index,
                        "timing_latent_transform": 2.0,
                        "timing_nn_train": 20.0 + 2.0 * method_index,
                        "timing_inference": 0.2,
                        "timing_end_to_end": 30.0 + 4.0 * method_index,
                        "invocation_timing_nn_train": 15.0 + method_index,
                        "invocation_timing_end_to_end": 22.0 + method_index,
                        "pca_timings_fit_input_pca": (
                            0.8 if method == "overlap_l2l" else 0.4
                        ),
                        "pca_timings_fit_output_pca": (
                            0.9
                            if method == "overlap_l2l"
                            else (2.2 if "two_scale" in method else 0.5)
                        ),
                        "pca_timings_fit_coarse_svd": (
                            0.2 if "two_scale" in method else None
                        ),
                        "pca_timings_fit_output_residual_pca": (
                            2.0 if "two_scale" in method else None
                        ),
                    }
                )
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(results_dir / "results.csv", index=False)
