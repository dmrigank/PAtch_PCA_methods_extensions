from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lpcanet.metrics.evaluate import evaluate_run
from lpcanet.train.experiment import train_from_config
from lpcanet.utils.config import compose_config, save_config


def test_in_loop_training_reduces_spectral_and_interface_metrics(tmp_path: Path) -> None:
    data_path = tmp_path / "poisson128_in_loop.npz"
    forcing, solution = _make_poisson128_fields(48)
    np.savez_compressed(data_path, f=forcing, u=solution)

    baseline_dir = _run_case(tmp_path, data_path, in_loop_loss=False)
    in_loop_dir = _run_case(
        tmp_path,
        data_path,
        in_loop_loss=True,
        warm_start_dir=baseline_dir,
    )
    baseline_metrics = evaluate_run(baseline_dir)
    in_loop_metrics = evaluate_run(in_loop_dir)

    assert in_loop_metrics["relative_spectrum_error"] < baseline_metrics["relative_spectrum_error"]
    assert in_loop_metrics["interface_jump"] < baseline_metrics["interface_jump"]

    with (in_loop_dir / "config.yaml").open("r", encoding="utf-8") as handle:
        saved_config = handle.read()
    assert "in_loop_loss: true" in saved_config
    assert "postprocessing_stage: none" in saved_config
    assert "refinementnet_used: false" in saved_config
    with (in_loop_dir / "runtime.json").open("r", encoding="utf-8") as handle:
        runtime = json.load(handle)
    assert "refinement" not in runtime["times"]
    assert "postprocess" not in runtime["times"]


def _run_case(
    tmp_path: Path,
    data_path: Path,
    *,
    in_loop_loss: bool,
    warm_start_dir: Path | None = None,
) -> Path:
    root = Path(__file__).resolve().parents[1]
    config = compose_config(root=root, dataset="poisson", model="l2l", experiment="single")
    config["dataset"]["processed_path"] = str(data_path)
    config["dataset"]["grid_size"] = 128
    config["experiment"]["seed"] = 3
    config["experiment"]["output_dir"] = str(tmp_path / ("in_loop" if in_loop_loss else "baseline"))
    config["mechanisms"]["two_scale"] = False
    config["mechanisms"]["coupling"] = False
    config["mechanisms"]["in_loop_loss"] = in_loop_loss
    if warm_start_dir is not None:
        config["warm_start"] = {
            "enabled": True,
            "run_dir": str(warm_start_dir),
        }
    config["patches"]["patch_size"] = 32
    config["patches"]["stride"] = 32
    config["patches"]["assembly"] = "average"
    config["pca"]["input_variance"] = 2
    config["pca"]["output_variance"] = 2
    config["pca"]["solver"] = "full"
    config["model"]["hidden_size"] = 16
    config["model"]["num_layers"] = 2
    config["training"]["batch_size"] = 8
    config["training"]["eval_batch_size"] = 16
    config["training"]["epochs"] = 40
    config["training"]["lr"] = 3.0e-3
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
    config["loss"]["reconstruction"] = "mse"
    config["loss"]["active_terms"] = ["recon", "interface_value", "interface_flux", "spectral"]
    config["loss"]["weights"] = {
        "recon": 1.0,
        "interface_value": 2.0e-1,
        "interface_flux": 5.0e-4,
        "pde_residual": 0.0,
        "spectral": 1.0e-1,
    }
    config["loss"]["warmup"] = {
        "reconstruction_epochs": 1,
        "ramp_epochs": 2,
    }
    config_path = save_config(config, tmp_path, filename=f"{'in_loop' if in_loop_loss else 'baseline'}.yaml")
    return train_from_config(
        config_path,
        output_dir=config["experiment"]["output_dir"],
        device="cpu",
        overwrite=True,
    )


def _make_poisson128_fields(n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(13)
    axis = np.linspace(0.0, 1.0, 128, dtype=np.float64)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    modes = [
        np.sin(np.pi * xx) * np.sin(np.pi * yy),
        np.sin(2.0 * np.pi * xx) * np.sin(np.pi * yy),
        np.sin(np.pi * xx) * np.sin(2.0 * np.pi * yy),
        np.sin(3.0 * np.pi * xx) * np.sin(np.pi * yy),
    ]
    coeffs = rng.normal(size=(n_samples, len(modes)))
    solution = np.zeros((n_samples, 128, 128), dtype=np.float64)
    forcing = np.zeros_like(solution)
    for index, mode in enumerate(modes):
        amplitude = coeffs[:, index, None, None]
        k_scale = float(index + 1)
        solution += amplitude * mode
        forcing += (np.pi * k_scale) ** 2 * amplitude * mode
    solution += 0.01 * rng.normal(size=solution.shape)
    forcing += 0.01 * rng.normal(size=forcing.shape)
    return forcing.astype(np.float32), solution.astype(np.float32)
