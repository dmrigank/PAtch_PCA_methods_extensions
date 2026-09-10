from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_resolution_sweep_manuscript_outputs_run_on_fixture(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    results_dir = tmp_path / "resolution_sweep"
    _write_fixture(results_dir)

    command = [
        sys.executable,
        str(root / "scripts" / "make_resolution_sweep_outputs.py"),
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
        "figures/main/resolution_accuracy.png",
        "figures/main/resolution_artifacts.png",
        "figures/main/resolution_quality_cost.png",
        "figures/main/resolution_time_projection.png",
        "figures/appendix/resolution_representation_gap.png",
        "figures/appendix/resolution_stage_costs.png",
        "figures/appendix/resolution_pca_breakdown.png",
        "tables/main/resolution_sweep_main.csv",
        "tables/main/resolution_sweep_main.tex",
        "tables/appendix/resolution_sweep_full.csv",
        "tables/appendix/resolution_stage_costs.tex",
        "tables/appendix/resolution_relative_changes.csv",
        "tables/appendix/resolution_pca_capacity.csv",
        "tables/appendix/resolution_pca_breakdown.csv",
        "tables/appendix/resolution_pca_breakdown.tex",
        "tables/appendix/resolution_time_projection.csv",
        "tables/appendix/resolution_time_projection.tex",
        "tables/appendix/resolution_per_seed.csv",
        "resolution_sweep_manifest.json",
    )
    for relative_path in expected:
        assert (results_dir / relative_path).is_file()
    latex = (results_dir / "tables/main/resolution_sweep_main.tex").read_text(
        encoding="utf-8"
    )
    assert r"\ensuremath{\pm}" in latex


def _write_fixture(results_dir: Path) -> None:
    methods = (
        "global_pca",
        "plain_l2l",
        "overlap_l2l",
        "two_scale",
        "two_scale_interface",
        "fno",
    )
    base_mre = {
        "global_pca": 0.08,
        "plain_l2l": 0.048,
        "overlap_l2l": 0.023,
        "two_scale": 0.012,
        "two_scale_interface": 0.011,
        "fno": 0.0025,
    }
    rows: list[dict[str, object]] = []
    for resolution in (16, 32, 64):
        for method_index, method in enumerate(methods):
            for seed in (0, 1):
                scale = 1.0 + 0.01 * seed + 0.001 * resolution
                mre = base_mre[method] * scale
                oracle = (
                    None
                    if method == "fno"
                    else base_mre[method] * (0.45 if "two_scale" in method else 0.98)
                )
                rows.append(
                    {
                        "run_id": f"{resolution}_{method}_{seed}",
                        "run_dir": str(
                            results_dir
                            / f"resolution_{resolution}"
                            / method
                            / f"seed_{seed}"
                        ),
                        "config_hash": f"hash-{resolution}-{method}-{seed}",
                        "seed": seed,
                        "resolution": resolution,
                        "ablation_method": method,
                        "metric_mre": mre,
                        "metric_pca_oracle_mre": oracle,
                        "metric_ssim_mean": 1.0 - mre,
                        "metric_pca_oracle_ssim_mean": (
                            None if oracle is None else 1.0 - oracle
                        ),
                        "metric_mse": mre**2,
                        "metric_mae": mre * 0.1,
                        "metric_relative_spectrum_error": mre * 0.2,
                        "metric_interface_jump": mre * 0.01,
                        "metric_interface_flux_jump": mre * resolution,
                        "metric_interface_value_trace_error": mre * 0.008,
                        "metric_interface_flux_trace_error": mre * 0.8,
                        "metric_poisson_residual_mean_abs": mre * 5.0,
                        "metric_poisson_residual_rms": mre * 6.0,
                        "timing_pca_fit": 1.0 + method_index + resolution / 64,
                        "timing_latent_transform": 2.0,
                        "timing_nn_train": 20.0 + 2.0 * method_index,
                        "timing_inference": 0.2 + resolution / 1000,
                        "timing_end_to_end": 35.0 + 5.0 * method_index + resolution,
                        "invocation_timing_nn_train": 15.0 + method_index,
                        "invocation_timing_end_to_end": 25.0 + method_index,
                        "pca_component_counts_input_total": (
                            None if method in {"global_pca", "fno"} else 144
                        ),
                        "pca_component_counts_output_total": (
                            None
                            if method == "fno"
                            else (160 if "two_scale" in method else 48)
                        ),
                        "pca_component_counts_coarse_output": (
                            10 if "two_scale" in method else None
                        ),
                        "pca_component_counts_output_residual_total": (
                            150 if "two_scale" in method else None
                        ),
                        "pca_timings_fit_input_pca": (
                            None if method == "fno" else 0.5 + resolution / 128
                        ),
                        "pca_timings_fit_output_pca": (
                            None
                            if method == "fno"
                            else (1.8 if "two_scale" in method else 0.7)
                        ),
                        "pca_timings_fit_coarse_svd": (
                            0.2 if "two_scale" in method else None
                        ),
                        "pca_timings_fit_output_residual_pca": (
                            1.6 if "two_scale" in method else None
                        ),
                    }
                )
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(results_dir / "results.csv", index=False)
