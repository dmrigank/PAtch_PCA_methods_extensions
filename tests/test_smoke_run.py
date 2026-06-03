from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np


def test_run_experiment_tiny_end_to_end(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n_samples = 24
    grid_size = 8
    forcing = rng.normal(size=(n_samples, grid_size, grid_size)).astype(np.float32)
    solution = (0.1 * forcing + 0.01 * rng.normal(size=forcing.shape)).astype(np.float32)
    data_path = tmp_path / "tiny_poisson.npz"
    np.savez_compressed(data_path, f=forcing, u=solution)

    run_dir = tmp_path / "run"
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(root / "scripts" / "run_experiment.py"),
        "dataset=poisson",
        "model=l2l",
        f"dataset.processed_path={data_path}",
        "dataset.grid_size=8",
        "patches.patch_size=4",
        "patches.stride=4",
        "pca.input_variance=2",
        "pca.output_variance=2",
        "model.hidden_size=8",
        "model.num_layers=2",
        "training.batch_size=4",
        "training.eval_batch_size=8",
        "evaluation.metrics=[mse,mae,mre,ssim]",
        "--epochs",
        "1",
        "--limit-samples",
        str(n_samples),
        "--output-dir",
        str(run_dir),
        "--device",
        "cpu",
        "--overwrite",
    ]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr or result.stdout
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "runtime.json").exists()
    assert (run_dir / "predictions_test.npz").exists()
