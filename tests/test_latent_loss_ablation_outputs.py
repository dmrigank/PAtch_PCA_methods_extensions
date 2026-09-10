from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def test_latent_loss_ablation_outputs_run_on_fixture(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    results_dir = tmp_path / "latent_loss"
    _write_fixture(results_dir)

    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "make_latent_loss_ablation_outputs.py"),
            "--results-dir",
            str(results_dir),
            "--formats",
            "png",
            "--expected-seeds",
            "1",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout

    expected = (
        "figures/main/latent_loss_accuracy.png",
        "figures/main/latent_loss_tradeoffs.png",
        "figures/appendix/latent_loss_stage_costs.png",
        "figures/appendix/latent_loss_qualitative_poisson_256.png",
        "figures/appendix/latent_loss_qualitative_darcy_256.png",
        "tables/main/latent_loss_quality.csv",
        "tables/main/latent_loss_quality.tex",
        "tables/main/latent_loss_relative_effects.csv",
        "tables/main/latent_loss_relative_effects.tex",
        "tables/appendix/latent_loss_full_metrics.csv",
        "tables/appendix/latent_loss_relative_effects_numeric.csv",
        "tables/appendix/latent_loss_capacity.csv",
        "tables/appendix/latent_loss_capacity.tex",
        "summary.md",
        "latent_loss_ablation_manifest.json",
    )
    for relative_path in expected:
        assert (results_dir / relative_path).is_file()

    summary = (results_dir / "summary.md").read_text(encoding="utf-8")
    assert "one run (seed 0)" in summary
    assert "equal block balancing" in summary
    effects = pd.read_csv(results_dir / "tables/appendix/latent_loss_relative_effects_numeric.csv")
    assert len(effects) == 8
    assert (effects["metric_mre_change_percent"] < 0.0).all()


def _write_fixture(results_dir: Path) -> None:
    modes = ("mse", "block_balanced", "score_normalized")
    mode_scale = {"mse": 1.0, "block_balanced": 0.6, "score_normalized": 0.7}
    coordinate = np.linspace(0.0, 1.0, 32, dtype=np.float32)
    xx, yy = np.meshgrid(coordinate, coordinate, indexing="xy")
    truth = np.stack(
        [
            np.sin(np.pi * xx) * np.sin(np.pi * yy),
            np.sin(2.0 * np.pi * xx) * np.sin(np.pi * yy),
        ]
    ).astype(np.float32)
    rows: list[dict[str, object]] = []
    for dataset in ("poisson", "darcy"):
        for resolution in (128, 256):
            for mode_index, mode in enumerate(modes):
                scale = mode_scale[mode]
                run_dir = results_dir / dataset / f"resolution_{resolution}" / mode / "seed_0"
                run_dir.mkdir(parents=True, exist_ok=True)
                prediction = truth * np.float32(1.0 - 0.05 * scale)
                input_field = np.stack([xx + yy, xx - yy]).astype(np.float32)
                archive: dict[str, np.ndarray] = {
                    "x_input": input_field,
                    "y_true": truth,
                    "y_pred": prediction,
                    "sample_indices": np.array([10, 11], dtype=np.int64),
                }
                if dataset == "poisson":
                    archive["f"] = input_field
                else:
                    archive["a"] = (input_field > 0.5).astype(np.float32)
                np.savez_compressed(run_dir / "predictions_test.npz", **archive)
                (run_dir / "pca_summary.json").write_text(
                    json.dumps(
                        {
                            "component_counts": {
                                "input_total": 100,
                                "coarse_output": 10,
                                "output_residual_total": 120,
                                "output_total": 130,
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                mre = 0.05 * scale
                rows.append(
                    {
                        "run_dir": str(run_dir),
                        "seed": 0,
                        "dataset": dataset,
                        "resolution": resolution,
                        "latent_loss_mode": mode,
                        "metric_mre": mre,
                        "metric_ssim_mean": 1.0 - 0.1 * scale,
                        "metric_mse": mre**2,
                        "metric_mae": 0.1 * mre,
                        "metric_relative_spectrum_error": 0.2 * scale,
                        "metric_interface_jump": 0.01 * scale,
                        "metric_interface_flux_jump": 0.1 * scale,
                        "metric_poisson_residual_rms": (
                            0.5 * scale if dataset == "poisson" else np.nan
                        ),
                        "metric_darcy_residual_rms": (
                            3.0 * scale if dataset == "darcy" else np.nan
                        ),
                        "metric_best_val_loss": 0.2 * scale,
                        "metric_num_parameters": 1000,
                        "timing_pca_fit": 5.0,
                        "timing_latent_transform": 2.0,
                        "timing_nn_train": 20.0 + mode_index,
                        "timing_inference": 0.2,
                        "timing_end_to_end": 30.0 + mode_index,
                    }
                )
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(results_dir / "results.csv", index=False)
