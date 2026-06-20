"""RefinementNet post-processing baseline routed through the new harness."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import torch

from lpcanet.data.io import load_npz, save_npz
from lpcanet.data.splits import make_splits
from lpcanet.metrics.metrics import aggregate_metrics
from lpcanet.models.mlp import MLP, count_parameters
from lpcanet.models.refinement import RefinementNet
from lpcanet.train.checkpointing import save_environment, save_json, save_metrics, save_model
from lpcanet.train.loop import predict_images, predict_latent, train_image_model
from lpcanet.train.seeding import set_seed
from lpcanet.utils.config import load_config, save_config
from lpcanet.utils.paths import ensure_dir, normalize_path, run_dir_has_artifacts
from lpcanet.utils.timing import END_TO_END, INFERENCE, NN_TRAIN, TimingCollector


def train_refinement_from_config(
    config_path: str | Path,
    *,
    limit_samples: int | None = None,
    epochs: int | None = None,
    output_dir: str | Path | None = None,
    device: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Train RefinementNet on top of a plain L2L PCA-Net base run."""
    start_all = perf_counter()
    config = load_config(config_path)
    experiment = config.setdefault("experiment", {})
    training_cfg = config.setdefault("training", {})
    refinement_cfg = config.setdefault("refinement", {})
    if epochs is not None:
        training_cfg["epochs"] = int(epochs)
    if output_dir is not None:
        experiment["output_dir"] = str(output_dir)
    if device is not None:
        training_cfg["device"] = device

    run_dir = normalize_path(experiment.get("output_dir", "results/l2l_refinement"))
    if run_dir_has_artifacts(run_dir) and not overwrite:
        raise FileExistsError(f"Output directory exists and is not empty: {run_dir}")
    ensure_dir(run_dir)

    seed = int(experiment.get("seed", 0))
    set_seed(seed)
    device_name = _resolve_device(str(training_cfg.get("device", "auto")))
    timer = TimingCollector()

    base_run_dir = _ensure_base_run(
        config,
        limit_samples=limit_samples,
        device=device_name,
        overwrite=overwrite,
    )
    base = _load_base_reconstructions(base_run_dir, limit_samples=limit_samples, device=device_name)
    model = RefinementNet(
        in_channels=1,
        hidden_channels=int(refinement_cfg.get("hidden_channels", 32)),
        num_layers=int(refinement_cfg.get("num_layers", 4)),
        kernel_size=int(refinement_cfg.get("kernel_size", 5)),
        activation=str(refinement_cfg.get("activation", "relu")),
        residual=bool(refinement_cfg.get("residual", False)),
    )
    normalizer = _RefinementNormalizer.fit(
        base["blocky_train"],
        base["y_train"],
        enabled=bool(refinement_cfg.get("normalize_fields", True)),
        eps=float(refinement_cfg.get("normalization_eps", 1e-8)),
    )

    with timer.time(NN_TRAIN):
        result = train_image_model(
            model,
            normalizer.transform_input(base["blocky_train"]),
            normalizer.transform_target(base["y_train"]),
            normalizer.transform_input(base["blocky_val"]),
            normalizer.transform_target(base["y_val"]),
            device=device_name,
            batch_size=int(training_cfg.get("batch_size", 32)),
            epochs=int(training_cfg.get("epochs", 25)),
            lr=float(training_cfg.get("lr", 1e-3)),
            weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
            scheduler=training_cfg.get("scheduler"),
            patience=int(training_cfg.get("patience", 10)),
        )

    with timer.time(INFERENCE):
        refined_model_output = predict_images(
            model,
            normalizer.transform_input(base["blocky_test"]),
            device=device_name,
            batch_size=int(training_cfg.get("eval_batch_size", 256)),
        )
        y_refined = normalizer.inverse_transform_target(refined_model_output)

    metrics = aggregate_metrics(y_refined, base["y_test"], ["mse", "mae", "mre", "ssim"])
    blocky_metrics = aggregate_metrics(base["blocky_test"], base["y_test"], ["mse", "mae", "mre", "ssim"])
    metrics.update({f"blocky_{key}": value for key, value in blocky_metrics.items()})
    metrics.update(
        {
            "best_val_loss": result.best_val_loss,
            "num_parameters": count_parameters(model),
            "n_train": int(len(base["y_train"])),
            "n_val": int(len(base["y_val"])),
            "n_test": int(len(base["y_test"])),
            "refinementnet_used": 1.0,
            "external_postprocessing": 1.0,
        }
    )
    config.setdefault("resolved", {})
    config["resolved"].update(
        {
            "device": device_name,
            "base_run_dir": str(base_run_dir),
            "postprocessing_stage": "refinementnet",
            "refinementnet_used": True,
            "refinement_normalization": normalizer.to_dict(),
        }
    )
    save_model(model, run_dir / "refinement_model.pt")
    save_metrics(metrics, run_dir)
    save_npz(
        run_dir / "predictions_test.npz",
        x_input=base["x_test"].astype(np.float32, copy=False),
        y_true=base["y_test"].astype(np.float32, copy=False),
        y_pred=y_refined.astype(np.float32, copy=False),
        y_blocky=base["blocky_test"].astype(np.float32, copy=False),
        sample_indices=base["test_idx"].astype(np.int64, copy=False),
        **{key: value.astype(np.float32, copy=False) for key, value in base["optional_test_arrays"].items()},
    )
    save_npz(run_dir / "loss_curves.npz", train_loss=result.train_loss, val_loss=result.val_loss)
    save_config(config, run_dir)
    save_environment(run_dir / "environment.txt")
    base_runtime = _load_runtime(base_run_dir)
    for key, value in base_runtime.get("stages", {}).items():
        timer.add(f"base_{key}", float(value))
    timer.add(END_TO_END, perf_counter() - start_all)
    timer.save(run_dir / "runtime.json")
    save_json({"base_run_dir": str(base_run_dir)}, run_dir / "refinement_summary.json")
    return run_dir


def _ensure_base_run(
    config: dict[str, Any],
    *,
    limit_samples: int | None,
    device: str,
    overwrite: bool,
) -> Path:
    base_run_dir = normalize_path(config.get("base_run_dir", "results/baselines/l2l_base"))
    required = [base_run_dir / "config.yaml", base_run_dir / "pca_encoder.joblib", base_run_dir / "model.pt"]
    if all(path.exists() for path in required):
        return base_run_dir
    if not bool(config.get("train_base_if_missing", True)):
        missing = [str(path) for path in required if not path.exists()]
        raise FileNotFoundError(f"Base L2L run artifacts are missing: {missing}")
    from lpcanet.train.experiment import train_from_config

    base_config = dict(config.get("base_config_resolved", {}))
    if not base_config:
        raise ValueError("Refinement baseline requires base_config_resolved when training base.")
    base_config.setdefault("experiment", {})["output_dir"] = str(base_run_dir)
    base_config_path = save_config(base_config, base_run_dir / "resolved_configs", filename="config.yaml")
    train_from_config(
        base_config_path,
        limit_samples=limit_samples,
        output_dir=base_run_dir,
        device=device,
        overwrite=overwrite,
    )
    return base_run_dir


def _load_base_reconstructions(
    base_run_dir: Path,
    *,
    limit_samples: int | None,
    device: str,
) -> dict[str, Any]:
    base_config = load_config(base_run_dir / "config.yaml")
    dataset_cfg = base_config["dataset"]
    data = load_npz(dataset_cfg["processed_path"])
    if limit_samples is not None:
        data = {
            key: value[:limit_samples] if value.ndim > 0 and key not in _SPLIT_KEYS else value
            for key, value in data.items()
        }
    x_all = _select_input_array(data, dataset_cfg.get("input_keys", ["f"]))
    y_all = np.asarray(data[dataset_cfg.get("output_key", "u")], dtype=np.float32)
    splits = _select_splits(
        data,
        len(y_all),
        seed=int(base_config.get("experiment", {}).get("seed", 0)),
        force_regenerate=limit_samples is not None,
    )
    x_train, y_train = x_all[splits["train_idx"]], y_all[splits["train_idx"]]
    x_val, y_val = x_all[splits["val_idx"]], y_all[splits["val_idx"]]
    x_test, y_test = x_all[splits["test_idx"]], y_all[splits["test_idx"]]
    encoder = joblib.load(base_run_dir / "pca_encoder.joblib")
    model = _load_base_model(base_config, encoder, base_run_dir / "model.pt", x_train, y_train)
    blocky_train = _predict_blocky(model, encoder, x_train, base_config, device)
    blocky_val = _predict_blocky(model, encoder, x_val, base_config, device)
    blocky_test = _predict_blocky(model, encoder, x_test, base_config, device)
    optional_test_arrays = {}
    for optional_key in ("a", "f", "coeff", "coefficient", "permeability", "source", "forcing"):
        if optional_key in data:
            optional_test_arrays[optional_key] = np.asarray(data[optional_key][splits["test_idx"]], dtype=np.float32)
    return {
        "x_test": x_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "blocky_train": blocky_train,
        "blocky_val": blocky_val,
        "blocky_test": blocky_test,
        "test_idx": splits["test_idx"],
        "optional_test_arrays": optional_test_arrays,
    }


def _load_base_model(
    config: dict[str, Any],
    encoder: Any,
    model_path: Path,
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> MLP:
    resolved = config.get("resolved", {})
    input_dim = int(resolved.get("input_latent_dim", encoder.transform_inputs(x_train[:1]).shape[1]))
    output_dim = int(resolved.get("output_latent_dim", encoder.transform_outputs(y_train[:1]).shape[1]))
    model_cfg = config.get("model", {})
    model = MLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_size=int(model_cfg.get("hidden_size", 128)),
        num_layers=int(model_cfg.get("num_layers", 4)),
        activation=str(model_cfg.get("activation", "relu")),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )
    try:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model


def _predict_blocky(model: MLP, encoder: Any, x: np.ndarray, config: dict[str, Any], device: str) -> np.ndarray:
    x_latent = encoder.transform_inputs(x)
    y_latent = predict_latent(
        model,
        x_latent,
        device=device,
        batch_size=int(config.get("training", {}).get("eval_batch_size", 256)),
    )
    return encoder.inverse_transform_outputs(y_latent).astype(np.float32, copy=False)


class _RefinementNormalizer:
    def __init__(
        self,
        *,
        enabled: bool,
        input_mean: float,
        input_std: float,
        target_mean: float,
        target_std: float,
        eps: float,
    ) -> None:
        self.enabled = enabled
        self.input_mean = input_mean
        self.input_std = input_std
        self.target_mean = target_mean
        self.target_std = target_std
        self.eps = eps

    @classmethod
    def fit(cls, x_train: np.ndarray, y_train: np.ndarray, *, enabled: bool, eps: float) -> _RefinementNormalizer:
        if not enabled:
            return cls(enabled=False, input_mean=0.0, input_std=1.0, target_mean=0.0, target_std=1.0, eps=eps)
        return cls(
            enabled=True,
            input_mean=float(np.mean(x_train)),
            input_std=max(float(np.std(x_train)), eps),
            target_mean=float(np.mean(y_train)),
            target_std=max(float(np.std(y_train)), eps),
            eps=eps,
        )

    def transform_input(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if not self.enabled:
            return values
        return ((values - self.input_mean) / self.input_std).astype(np.float32, copy=False)

    def transform_target(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if not self.enabled:
            return values
        return ((values - self.target_mean) / self.target_std).astype(np.float32, copy=False)

    def inverse_transform_target(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if not self.enabled:
            return values
        return (values * self.target_std + self.target_mean).astype(np.float32, copy=False)

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "enabled": self.enabled,
            "input_mean": self.input_mean,
            "input_std": self.input_std,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
            "eps": self.eps,
        }


_SPLIT_KEYS = {"train_idx", "val_idx", "test_idx"}


def _select_input_array(data: dict[str, np.ndarray], input_keys: list[str] | tuple[str, ...]) -> np.ndarray:
    if isinstance(input_keys, str):
        input_keys = [input_keys]
    if len(input_keys) != 1:
        raise ValueError("Refinement training currently supports exactly one scalar input key.")
    key = input_keys[0]
    if key not in data:
        raise KeyError(f"Input key {key!r} not found. Available keys: {sorted(data)}")
    return np.asarray(data[key], dtype=np.float32)


def _select_splits(data: dict[str, np.ndarray], n_samples: int, *, seed: int, force_regenerate: bool) -> dict[str, np.ndarray]:
    if not force_regenerate and _SPLIT_KEYS.issubset(data):
        splits = {key: np.asarray(data[key], dtype=np.int64) for key in _SPLIT_KEYS}
        if all(np.max(indices, initial=-1) < n_samples for indices in splits.values()):
            return splits
    return make_splits(n_samples, seed=seed)


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false.")
    return device


def _load_runtime(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "runtime.json"
    if not path.exists():
        return {}
    import json

    with path.open("r", encoding="utf-8") as handle:
        return dict(json.load(handle))
