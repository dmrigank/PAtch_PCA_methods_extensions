from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def test_256_study_plot_script_writes_all_comparisons(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    results_dir = tmp_path / "study"
    methods = (
        "global_pca",
        "plain_l2l",
        "overlap_l2l",
        "fno",
        "two_scale",
        "two_scale_in_loop",
    )
    grid = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    x, y = np.meshgrid(grid, grid, indexing="ij")
    truth = np.stack(
        [np.sin(np.pi * x) * np.sin(np.pi * y) + offset for offset in (0.0, 0.01)]
    )
    inputs = np.stack([x + y, x - y])
    for index, method in enumerate(methods):
        run_dir = results_dir / "resolution_256" / method / "seed_0"
        run_dir.mkdir(parents=True)
        np.savez_compressed(
            run_dir / "predictions_test.npz",
            x_input=inputs,
            y_true=truth,
            y_pred=truth + np.float32(0.001 * (index + 1)),
            sample_indices=np.asarray([10, 11], dtype=np.int64),
        )
    gnn_dir = tmp_path / "gnn_interface"
    gnn_dir.mkdir()
    np.savez_compressed(
        gnn_dir / "predictions_test.npz",
        x_input=inputs,
        y_true=truth,
        y_pred=truth + np.float32(0.0005),
        sample_indices=np.asarray([10, 11], dtype=np.int64),
    )
    (gnn_dir / "runtime.json").write_text(
        '{"stages": {"pca_fit": 0.0, "latent_transform": 0.1, '
        '"nn_train": 0.2, "inference": 0.01, "end_to_end": 0.4}}',
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {
                "method": method,
                "resolution": 256,
                "seed": 0,
                "config_hash": method,
                "includes_warm_start": method == "two_scale_in_loop",
                "pca_fit": 1.0,
                "latent_transform": 0.5,
                "nn_train": 2.0,
                "inference": 0.1,
                "end_to_end": 4.0,
            }
            for method in methods
        ]
    ).to_csv(results_dir / "stage_costs.csv", index=False)

    output = tmp_path / "figures" / "darcy_256_reconstruction_comparison"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "plot_256_reconstructions.py"),
            "--dataset",
            "darcy",
            "--results-dir",
            str(results_dir),
            "--output",
            str(output),
            "--gnn-interface-dir",
            str(gnn_dir),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    expected = (
        "darcy_256_reconstruction_comparison",
        "darcy_256_stage_costs",
        "darcy_256_qualitative_diagnostics",
    )
    for stem in expected:
        assert (output.parent / f"{stem}.png").is_file()
        assert (output.parent / f"{stem}.pdf").is_file()
