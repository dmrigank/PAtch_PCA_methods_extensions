from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_svd_ablation_manuscript_outputs_run_on_fixture(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    results_dir = tmp_path / "svd_solver_ablation"
    _write_fixture(results_dir)

    command = [
        sys.executable,
        str(root / "scripts" / "make_svd_ablation_outputs.py"),
        "--results-dir",
        str(results_dir),
        "--formats",
        "png",
        "--expected-seeds",
        "2",
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
        "figures/main/svd_quality_equivalence.png",
        "figures/main/svd_runtime_speedup.png",
        "figures/appendix/svd_paired_mre.png",
        "figures/appendix/svd_stage_costs.png",
        "figures/appendix/svd_pca_breakdown.png",
        "figures/appendix/svd_trainability_gap.png",
        "tables/main/svd_solver_quality.csv",
        "tables/main/svd_solver_quality.tex",
        "tables/main/svd_solver_timing.csv",
        "tables/main/svd_solver_timing.tex",
        "tables/appendix/svd_solver_full_metrics.csv",
        "tables/appendix/svd_solver_equivalence.csv",
        "tables/appendix/svd_solver_pca_diagnostics.csv",
        "tables/appendix/svd_solver_per_seed.csv",
        "tables/appendix/svd_pca_breakdown.csv",
        "tables/appendix/svd_pca_breakdown.tex",
        "svd_ablation_manifest.json",
    )
    for relative_path in expected:
        assert (results_dir / relative_path).is_file()
    latex = (results_dir / "tables/main/svd_solver_quality.tex").read_text(
        encoding="utf-8"
    )
    assert r"\ensuremath{\pm}" in latex

    equivalence = pd.read_csv(
        results_dir / "tables/appendix/svd_solver_equivalence.csv"
    )
    assert len(equivalence) == 45
    assert set(equivalence["Comparison"]).issuperset(
        {
            "Plain: randomized vs full",
            "Overlap: randomized vs full",
            "Two-scale: randomized vs full",
        }
    )


def _write_fixture(results_dir: Path) -> None:
    cells = (
        ("plain_full", "plain_l2l", "full", 9.0),
        ("plain_randomized", "plain_l2l", "randomized", 3.2),
        ("overlap_full", "overlap_l2l", "full", 28.0),
        ("overlap_randomized", "overlap_l2l", "randomized", 10.0),
        ("two_scale_full", "two_scale", "full", 13.0),
        ("two_scale_hybrid", "two_scale", "hybrid", 12.5),
        ("two_scale_randomized", "two_scale", "randomized", 4.7),
    )
    base_mre = {
        "plain_l2l": 0.048,
        "overlap_l2l": 0.023,
        "two_scale": 0.0115,
    }
    rows: list[dict[str, object]] = []
    for cell, method, solver, pca_time in cells:
        for seed in (0, 1):
            scale = 1.0 + 0.01 * seed
            mre = base_mre[method] * scale
            if solver == "randomized":
                mre *= 1.002
            elif solver == "hybrid":
                mre *= 1.01
            rows.append(
                {
                    "seed": seed,
                    "resolution": 128,
                    "ablation_cell": cell,
                    "ablation_method": method,
                    "ablation_solver_class": solver,
                    "config_hash": f"{cell}-{seed}",
                    "metric_mre": mre,
                    "metric_pca_oracle_mre": base_mre[method] * 0.7 * scale,
                    "metric_ssim_mean": 0.99 / scale,
                    "metric_mse": mre**2,
                    "metric_mae": mre * 0.1,
                    "metric_relative_spectrum_error": mre * 0.2,
                    "metric_interface_value_trace_error": mre * 0.01,
                    "metric_interface_flux_trace_error": mre * 0.8,
                    "metric_poisson_residual_rms": mre * 6.0,
                    "metric_pca_fit_seconds": pca_time * 0.9,
                    "timing_pca_fit": pca_time,
                    "timing_latent_transform": 2.0,
                    "timing_nn_train": 150.0,
                    "timing_inference": 0.2,
                    "timing_end_to_end": 175.0 + pca_time,
                    "timing_coarse_svd": 0.5 if method == "two_scale" else 0.0,
                    "equivalence_gate_minimum_pca_speedup": 1.2,
                    "pca_component_counts_input_total": 144,
                    "pca_component_counts_output_total": (
                        48 if method == "plain_l2l" else 147
                    ),
                    "pca_component_counts_coarse_output": (
                        10 if method == "two_scale" else None
                    ),
                    "pca_component_counts_output_residual_total": (
                        137 if method == "two_scale" else None
                    ),
                    "pca_groups_input_explained_variance_ratio_mean": 0.995,
                    "pca_groups_output_explained_variance_ratio_mean": 0.997,
                    "pca_groups_coarse_output_explained_variance_ratio_mean": (
                        0.999 if method == "two_scale" else None
                    ),
                    "pca_groups_output_residual_explained_variance_ratio_mean": (
                        0.996 if method == "two_scale" else None
                    ),
                    "pca_timings_fit_input_pca": 0.48 * pca_time,
                    "pca_timings_fit_output_pca": 0.52 * pca_time,
                    "pca_timings_fit_coarse_svd": (
                        0.05 * pca_time if method == "two_scale" else None
                    ),
                    "pca_timings_fit_output_residual_pca": (
                        0.47 * pca_time if method == "two_scale" else None
                    ),
                }
            )
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(results_dir / "results.csv", index=False)
