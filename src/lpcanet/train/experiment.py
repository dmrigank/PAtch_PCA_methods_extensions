#!/usr/bin/env python
"""Train PCA-Net variants from YAML configs."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from lpcanet.assembly.windows import hann2d, safe_hann2d
from lpcanet.data.io import load_npz, save_npz
from lpcanet.data.splits import make_splits
from lpcanet.models.mlp import MLP, count_parameters
from lpcanet.models.patchwise_heads import (
    FlatPatchwiseHeadAdapter,
    IndependentPatchMLP,
    NeighborAwarePatchMLP,
    SharedPatchMLP,
    SingleConcatMLP,
)
from lpcanet.pca.pca import (
    GlobalPCAEncoder,
    LocalToGlobalPCAEncoder,
    LocalToLocalPCAEncoder,
)
from lpcanet.train.checkpointing import (
    save_environment,
    save_json,
    save_metrics,
    save_model,
)
from lpcanet.train.loop import predict_latent, train_latent_model
from lpcanet.train.seeding import set_seed
from lpcanet.utils.config import load_config, save_config
from lpcanet.utils.paths import ensure_dir, normalize_path
from lpcanet.utils.timing import (
    END_TO_END,
    INFERENCE,
    LATENT_TRANSFORM,
    NN_TRAIN,
    PCA_FIT,
    TimingCollector,
)


def train_from_config(
    config_path: str | Path,
    *,
    limit_samples: int | None = None,
    epochs: int | None = None,
    output_dir: str | Path | None = None,
    device: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Train a PCA-Net run and return the output directory."""
    end_to_end_start = perf_counter()
    config = load_config(config_path)
    experiment = config.setdefault("experiment", {})
    dataset_cfg = config["dataset"]
    training_cfg = config.setdefault("training", {})

    if epochs is not None:
        training_cfg["epochs"] = int(epochs)
    if output_dir is not None:
        experiment["output_dir"] = str(output_dir)
    if device is not None:
        training_cfg["device"] = device

    run_dir = normalize_path(experiment.get("output_dir", f"outputs/{experiment.get('name', 'run')}"))
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory exists and is not empty: {run_dir}")
    ensure_dir(run_dir)

    seed = int(experiment.get("seed", 0))
    set_seed(seed)
    device_name = _resolve_device(str(training_cfg.get("device", "auto")))
    timer = TimingCollector()

    with timer.time("load_dataset"):
        data = load_npz(dataset_cfg["processed_path"])
        if limit_samples is not None:
            data = {
                key: value[:limit_samples] if value.ndim > 0 and key not in _SPLIT_KEYS else value
                for key, value in data.items()
            }

    x_all = _select_input_array(data, dataset_cfg.get("input_keys", ["f"]))
    y_all = np.asarray(data[dataset_cfg.get("output_key", "u")], dtype=np.float32)
    splits = _select_splits(data, len(y_all), seed=seed, force_regenerate=limit_samples is not None)

    x_train, y_train = x_all[splits["train_idx"]], y_all[splits["train_idx"]]
    x_val, y_val = x_all[splits["val_idx"]], y_all[splits["val_idx"]]
    x_test, y_test = x_all[splits["test_idx"]], y_all[splits["test_idx"]]

    with timer.time(PCA_FIT):
        encoder = _make_encoder(config)
        encoder.fit(x_train, y_train)

    with timer.time(LATENT_TRANSFORM):
        x_train_latent = encoder.transform_inputs(x_train)
        y_train_latent = encoder.transform_outputs(y_train)
        x_val_latent = encoder.transform_inputs(x_val)
        y_val_latent = encoder.transform_outputs(y_val)
        x_test_latent = encoder.transform_inputs(x_test)

    model_cfg = config.setdefault("model", {})
    model = _make_model(
        model_cfg,
        encoder,
        input_dim=int(x_train_latent.shape[1]),
        output_dim=int(y_train_latent.shape[1]),
    )

    with timer.time(NN_TRAIN):
        result = train_latent_model(
            model,
            x_train_latent,
            y_train_latent,
            x_val_latent,
            y_val_latent,
            device=device_name,
            batch_size=int(training_cfg.get("batch_size", 32)),
            epochs=int(training_cfg.get("epochs", 100)),
            lr=float(training_cfg.get("lr", 1e-3)),
            weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
            scheduler=training_cfg.get("scheduler"),
            patience=int(training_cfg.get("patience", 30)),
        )

    with timer.time(INFERENCE):
        y_pred_latent = predict_latent(
            model,
            x_test_latent,
            device=device_name,
            batch_size=int(training_cfg.get("eval_batch_size", 256)),
        )
        y_pred = encoder.inverse_transform_outputs(y_pred_latent)

    metrics = _compute_metrics(y_pred, y_test)
    metrics.update(
        {
            "best_val_loss": result.best_val_loss,
            "num_parameters": count_parameters(model),
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "n_test": int(len(y_test)),
        }
    )
    metrics["pca_fit_seconds"] = float(
        encoder.timings.get("fit_input_pca", 0.0) + encoder.timings.get("fit_output_pca", 0.0)
    )

    config.setdefault("resolved", {})
    config["resolved"].update(
        {
            "device": device_name,
            "limit_samples": limit_samples,
            "input_latent_dim": int(x_train_latent.shape[1]),
            "output_latent_dim": int(y_train_latent.shape[1]),
            "pca_component_counts": encoder.component_counts,
        }
    )

    encoder.save(run_dir / "pca_encoder.joblib")
    save_model(model, run_dir / "model.pt")
    save_metrics(metrics, run_dir)
    prediction_arrays = {
        "x_input": x_test.astype(np.float32, copy=False),
        "y_true": y_test.astype(np.float32, copy=False),
        "y_pred": y_pred.astype(np.float32, copy=False),
        "sample_indices": splits["test_idx"].astype(np.int64),
    }
    for optional_key in ("a", "f", "coeff", "coefficient", "permeability", "source", "forcing"):
        if optional_key in data:
            prediction_arrays[optional_key] = np.asarray(data[optional_key][splits["test_idx"]], dtype=np.float32)
    save_npz(run_dir / "predictions_test.npz", **prediction_arrays)
    save_npz(run_dir / "loss_curves.npz", train_loss=result.train_loss, val_loss=result.val_loss)
    save_config(config, run_dir)
    save_environment(run_dir / "environment.txt")
    timer.times.update({f"pca_{key}": value for key, value in encoder.timings.items()})
    timer.add(END_TO_END, perf_counter() - end_to_end_start)
    timer.save(run_dir / "runtime.json")
    save_json({"component_counts": encoder.component_counts}, run_dir / "pca_summary.json")
    return run_dir


_SPLIT_KEYS = {"train_idx", "val_idx", "test_idx"}


def _select_input_array(data: dict[str, np.ndarray], input_keys: list[str] | tuple[str, ...]) -> np.ndarray:
    if isinstance(input_keys, str):
        input_keys = [input_keys]
    if len(input_keys) != 1:
        raise ValueError("Current training path supports exactly one scalar input key.")
    key = input_keys[0]
    if key not in data:
        raise KeyError(f"Input key {key!r} not found. Available keys: {sorted(data)}")
    return np.asarray(data[key], dtype=np.float32)


def _select_splits(
    data: dict[str, np.ndarray],
    n_samples: int,
    *,
    seed: int,
    force_regenerate: bool,
) -> dict[str, np.ndarray]:
    if not force_regenerate and _SPLIT_KEYS.issubset(data):
        splits = {key: np.asarray(data[key], dtype=np.int64) for key in _SPLIT_KEYS}
        if all(np.max(indices, initial=-1) < n_samples for indices in splits.values()):
            return splits
    return make_splits(n_samples, seed=seed)


def _make_encoder(config: dict[str, Any]):
    pca_cfg = config.setdefault("pca", {})
    patches_cfg = config.setdefault("patches", {})
    common = {
        "input_variance": pca_cfg.get("input_variance", 0.99),
        "output_variance": pca_cfg.get("output_variance", 0.99),
        "solver": pca_cfg.get("solver", "full"),
        "standardize": bool(pca_cfg.get("standardize", True)),
        "oversampling": int(pca_cfg.get("randomized", {}).get("oversampling", 20)),
        "n_iter": int(pca_cfg.get("randomized", {}).get("n_iter", 4)),
        "random_state": config.get("experiment", {}).get("seed", 0),
    }
    pca_type = str(pca_cfg.get("type", "global"))
    if pca_type == "global":
        return GlobalPCAEncoder(**common)
    if pca_type == "local_to_global":
        return LocalToGlobalPCAEncoder(
            patch_size=int(patches_cfg["patch_size"]),
            stride=int(patches_cfg["stride"]),
            include_edges=bool(patches_cfg.get("include_edges", False)),
            **common,
        )
    if pca_type == "local_to_local":
        weights = _assembly_weights(patches_cfg)
        return LocalToLocalPCAEncoder(
            patch_size=int(patches_cfg["patch_size"]),
            stride=int(patches_cfg["stride"]),
            assembly_weights=weights,
            include_edges=bool(patches_cfg.get("include_edges", False)),
            **common,
        )
    raise ValueError(f"Unsupported pca.type {pca_type!r}.")


def _make_model(model_cfg: dict[str, Any], encoder: Any, *, input_dim: int, output_dim: int):
    model_type = str(model_cfg.get("type", "mlp"))
    hidden_size = int(model_cfg.get("hidden_size", 128))
    num_layers = int(model_cfg.get("num_layers", 4))
    activation = str(model_cfg.get("activation", "relu"))
    dropout = float(model_cfg.get("dropout", 0.0))
    if model_type in {"mlp", "pca_net"}:
        return MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
        )
    patchwise_types = {
        "single_concat",
        "independent_patch",
        "shared_patch",
        "neighbor_aware_4",
        "neighbor_aware_8",
    }
    if model_type not in patchwise_types:
        raise ValueError(f"Unsupported model.type {model_type!r}.")
    counts = getattr(encoder, "component_counts", {})
    input_counts = [int(value) for value in counts.get("input_patches", [])]
    output_counts = [int(value) for value in counts.get("output_patches", [])]
    if not input_counts or not output_counts:
        raise ValueError("Patchwise heads require a fitted local-to-local PCA encoder.")
    num_patches = len(input_counts)
    input_dim_per_patch = max(input_counts)
    output_dim_per_patch = max(output_counts)
    head: torch.nn.Module
    if model_type == "single_concat":
        head = SingleConcatMLP(
            num_patches=num_patches,
            input_dim_per_patch=input_dim_per_patch,
            output_dim_per_patch=output_dim_per_patch,
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
        )
    elif model_type == "independent_patch":
        head = IndependentPatchMLP(
            num_patches=num_patches,
            input_dim_per_patch=input_dim_per_patch,
            output_dim_per_patch=output_dim_per_patch,
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
        )
    elif model_type == "shared_patch":
        head = SharedPatchMLP(
            num_patches=num_patches,
            input_dim_per_patch=input_dim_per_patch,
            output_dim_per_patch=output_dim_per_patch,
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
        )
    else:
        patch_slices = getattr(encoder, "patch_slices", [])
        grid_shape = _patch_grid_shape(patch_slices)
        head = NeighborAwarePatchMLP(
            grid_shape=grid_shape,
            input_dim_per_patch=input_dim_per_patch,
            output_dim_per_patch=output_dim_per_patch,
            neighborhood=4 if model_type == "neighbor_aware_4" else 8,
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
        )
    return FlatPatchwiseHeadAdapter(head, input_counts, output_counts)


def _patch_grid_shape(patch_slices: list[tuple[slice, slice]]) -> tuple[int, int]:
    if not patch_slices:
        raise ValueError("Cannot infer patch grid shape from empty patch_slices.")
    rows = sorted({patch[0].start for patch in patch_slices})
    cols = sorted({patch[1].start for patch in patch_slices})
    return (len(rows), len(cols))


def _assembly_weights(patches_cfg: dict[str, Any]) -> np.ndarray | None:
    assembly = str(patches_cfg.get("assembly", "average"))
    patch_size = int(patches_cfg.get("patch_size", 0))
    if assembly in ("average", "none"):
        return None
    if assembly in ("hann_safe", "safe_hann", "hann_weighted_safe"):
        return safe_hann2d(patch_size)
    if assembly in ("hann", "hann_weighted"):
        return hann2d(patch_size)
    raise ValueError(f"Unsupported patch assembly {assembly!r}.")


def _compute_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    error = pred - true
    mse = float(np.mean(error**2))
    mae = float(np.mean(np.abs(error)))
    denom = float(np.linalg.norm(true.reshape(true.shape[0], -1)))
    mre = float(np.linalg.norm(error.reshape(error.shape[0], -1)) / max(denom, 1e-12))
    return {"mse": mse, "mae": mae, "mre": mre}


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false.")
    return device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to model YAML config.")
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    start = perf_counter()
    args = parse_args()
    run_dir = train_from_config(
        args.config,
        limit_samples=args.limit_samples,
        epochs=args.epochs,
        output_dir=args.output_dir,
        device=args.device,
        overwrite=args.overwrite,
    )
    print(f"Training complete: {run_dir} ({perf_counter() - start:.2f}s)")


if __name__ == "__main__":
    main()
