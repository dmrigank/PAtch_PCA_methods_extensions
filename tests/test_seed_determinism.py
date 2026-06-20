from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lpcanet.metrics.evaluate import evaluate_run
from lpcanet.train.experiment import train_from_config
from lpcanet.train.factorial import FactorialRun, build_result_record, config_hash_for
from lpcanet.utils.config import compose_config, save_config


def test_fixed_seed_reproduces_metrics_and_result_schema(tmp_path: Path) -> None:
    data_path = tmp_path / "determinism_poisson.npz"
    forcing, solution = _make_tiny_poisson(36)
    np.savez_compressed(data_path, f=forcing, u=solution)

    run_dirs = []
    metrics = []
    records = []
    for index in range(2):
        config = _make_config(data_path, tmp_path / f"run_{index}")
        config_path = save_config(config, tmp_path, filename=f"config_{index}.yaml")
        run_dir = train_from_config(config_path, output_dir=config["experiment"]["output_dir"], device="cpu", overwrite=True)
        run_metrics = evaluate_run(run_dir)
        run = FactorialRun(
            run_id=f"determinism_{index}",
            config=config,
            config_hash=config_hash_for(config),
            seed=11,
            mechanisms={key: bool(config["mechanisms"][key]) for key in ("two_scale", "coupling", "in_loop_loss")},
            resolution=8,
            run_dir=Path(run_dir),
        )
        records.append(build_result_record(run, metrics=run_metrics, run_dir=run_dir))
        run_dirs.append(run_dir)
        metrics.append(run_metrics)

    del run_dirs
    for key in ("mse", "mae", "mre", "relative_spectrum_error", "interface_jump"):
        assert metrics[0][key] == pytest.approx(metrics[1][key], abs=1e-8, rel=1e-8)
    assert records[0]["metrics"]["mre"] == pytest.approx(records[1]["metrics"]["mre"], abs=1e-8)
    assert set(records[0]["timings"]) >= {"pca_fit", "latent_transform", "nn_train", "inference", "end_to_end"}
    assert records[0]["mechanisms"] == {
        "two_scale": False,
        "coupling": False,
        "in_loop_loss": False,
    }


def _make_config(data_path: Path, output_dir: Path) -> dict:
    root = Path(__file__).resolve().parents[1]
    config = compose_config(root=root, dataset="poisson", model="l2l", experiment="single")
    config["dataset"]["processed_path"] = str(data_path)
    config["dataset"]["grid_size"] = 8
    config["experiment"]["seed"] = 11
    config["experiment"]["output_dir"] = str(output_dir)
    config["mechanisms"]["two_scale"] = False
    config["mechanisms"]["coupling"] = False
    config["mechanisms"]["in_loop_loss"] = False
    config["patches"]["patch_size"] = 4
    config["patches"]["stride"] = 4
    config["patches"]["assembly"] = "average"
    config["pca"]["input_variance"] = 2
    config["pca"]["output_variance"] = 2
    config["model"]["hidden_size"] = 8
    config["model"]["num_layers"] = 2
    config["training"]["batch_size"] = 4
    config["training"]["eval_batch_size"] = 8
    config["training"]["epochs"] = 3
    config["training"]["lr"] = 1.0e-3
    config["training"]["weight_decay"] = 0.0
    config["training"]["scheduler"] = "none"
    config["training"]["device"] = "cpu"
    config["evaluation"]["metrics"] = [
        "mse",
        "mae",
        "mre",
        "relative_spectrum_error",
        "interface_jump",
    ]
    return config


def _make_tiny_poisson(n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2)
    forcing = rng.normal(size=(n_samples, 8, 8)).astype(np.float32)
    solution = (0.2 * forcing + 0.01 * rng.normal(size=forcing.shape)).astype(np.float32)
    return forcing, solution
