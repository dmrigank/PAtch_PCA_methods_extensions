from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_coarse_ablation_manuscript_outputs_run_on_fixture(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    results_dir = tmp_path / "coarse_ablation"
    _write_fixture(results_dir)

    command = [
        sys.executable,
        str(root / "scripts" / "make_coarse_ablation_outputs.py"),
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
        "figures/coarse_ablation_reconstruction.png",
        "figures/coarse_ablation_artifacts.png",
        "figures/coarse_ablation_capacity_cost.png",
        "figures/coarse_rank_trainability_gap.png",
        "tables/coarse_ablation_summary_numeric.csv",
        "tables/coarse_ablation_quality_table.csv",
        "tables/coarse_ablation_quality_table.tex",
        "tables/coarse_ablation_cost_table.csv",
        "tables/coarse_ablation_cost_table.tex",
        "coarse_ablation_manifest.json",
    )
    for relative_path in expected:
        assert (results_dir / relative_path).is_file()
    latex = (results_dir / "tables/coarse_ablation_quality_table.tex").read_text(
        encoding="utf-8"
    )
    assert r"\ensuremath{\pm}" in latex


def _write_fixture(results_dir: Path) -> None:
    records: list[dict[str, object]] = []
    cells = (
        ("shared_baseline", "baseline_exact_rank10", 4, 10, 1.0, 0.99),
        ("coarse_rank", "coarse_rank_5", 4, 5, 1.0, 0.99),
        ("coarse_factor", "coarse_factor_8", 8, 10, 1.0, 0.99),
        ("residual_variance", "residual_variance_0p95", 4, 10, 1.0, 0.95),
        ("adaptive_reference", "adaptive_99pct_cap20", 4, 20, 0.99, 0.99),
    )
    for resolution in (128, 256):
        for family, cell, factor, components, coarse_variance, residual_variance in cells:
            for seed in (0, 1):
                run_dir = (
                    results_dir
                    / "poisson"
                    / f"resolution_{resolution}"
                    / cell
                    / f"seed_{seed}"
                )
                run_dir.mkdir(parents=True, exist_ok=True)
                actual_coarse = 10 if family == "adaptive_reference" else components
                residual_dim = 120 + int(100 * (residual_variance - 0.95))
                (run_dir / "pca_summary.json").write_text(
                    json.dumps(
                        {
                            "component_counts": {
                                "input_total": 140,
                                "coarse_output": actual_coarse,
                                "output_residual_total": residual_dim,
                                "output_total": actual_coarse + residual_dim,
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                scale = (
                    1.0
                    + 0.02 * seed
                    + 0.001 * resolution
                    + 0.01 * components
                )
                records.append(
                    {
                        "run_id": f"{resolution}_{cell}_{seed}",
                        "run_dir": str(run_dir),
                        "seed": seed,
                        "resolution": resolution,
                        "ablation": {
                            "family": family,
                            "cell": cell,
                            "coarse_factor": factor,
                            "coarse_components": components,
                            "coarse_variance": coarse_variance,
                            "residual_variance": residual_variance,
                        },
                        "metrics": {
                            "mre": 0.01 * scale,
                            "pca_oracle_mre": 0.006 / scale,
                            "ssim_mean": 0.99 / scale,
                            "interface_jump": 1.0e-4 * scale,
                            "interface_flux_jump": 7.0e-3 * scale,
                            "relative_spectrum_error": 4.0e-3 * scale,
                        },
                        "timings": {
                            "coarse_svd": 0.3 * scale,
                            "pca_fit": 5.0 * scale,
                            "latent_transform": 2.0 * scale,
                            "nn_train": 20.0 * scale,
                            "inference": 0.2 * scale,
                            "end_to_end": 35.0 * scale,
                        },
                    }
                )

    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
