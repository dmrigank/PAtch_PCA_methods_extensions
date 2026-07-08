#!/usr/bin/env python
"""Train PCA-Net variants from YAML configs."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import torch

from lpcanet.assembly.windows import hann2d, safe_hann2d
from lpcanet.data.io import load_npz, save_npz
from lpcanet.data.splits import make_splits
from lpcanet.models.boundary_correction import BoundaryCorrectionGNN, PassthroughDecoder
from lpcanet.models.coupling import CouplingOperator
from lpcanet.models.fno import FNO2d
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
    TwoScaleLocalToLocalPCAEncoder,
)
from lpcanet.pca.torch_decoder import TorchPCADecoder
from lpcanet.train.checkpointing import (
    save_environment,
    save_json,
    save_metrics,
    save_model,
)
from lpcanet.train.loop import (
    predict_images,
    predict_latent,
    train_image_model,
    train_in_loop_physical_model,
    train_latent_model,
)
from lpcanet.train.normalization import FieldNormalizer
from lpcanet.train.seeding import set_seed
from lpcanet.utils.config import load_config, save_config
from lpcanet.utils.paths import ensure_dir, normalize_path, run_dir_has_artifacts
from lpcanet.utils.timing import (
    COARSE_SVD,
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
    if run_dir_has_artifacts(run_dir) and not overwrite:
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

    if _is_image_operator_config(config):
        return _train_image_operator_from_arrays(
            config,
            run_dir,
            timer,
            end_to_end_start,
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
            splits,
            data,
            device_name,
        )

    warm_start_dir = _warm_start_run_dir(config)
    if warm_start_dir is None:
        with timer.time(PCA_FIT):
            encoder = _make_encoder(
                config,
                n_train=int(len(y_train)),
                grid_shape=tuple(int(dim) for dim in y_train.shape[1:]),
            )
            encoder.fit(x_train, y_train)
    else:
        encoder = joblib.load(warm_start_dir / "pca_encoder.joblib")
        timer.add(PCA_FIT, 0.0)

    with timer.time(LATENT_TRANSFORM):
        x_train_latent = encoder.transform_inputs(x_train)
        y_train_latent = encoder.transform_outputs(y_train)
        x_val_latent = encoder.transform_inputs(x_val)
        y_val_latent = encoder.transform_outputs(y_val)
        x_test_latent = encoder.transform_inputs(x_test)

    model_cfg = config.setdefault("model", {})
    boundary_correction_enabled = bool(config.get("mechanisms", {}).get("boundary_correction", False))
    model = _make_model(
        model_cfg,
        encoder,
        input_dim=int(x_train_latent.shape[1]),
        output_dim=int(y_train_latent.shape[1]),
        coupling_enabled=bool(config.get("mechanisms", {}).get("coupling", False)),
        boundary_correction_enabled=boundary_correction_enabled,
    )
    if warm_start_dir is not None:
        state_dict = torch.load(
            warm_start_dir / "model.pt",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict)

    with timer.time(NN_TRAIN):
        in_loop_loss = bool(config.get("mechanisms", {}).get("in_loop_loss", False))
        if in_loop_loss:
            loss_config = _make_in_loop_loss_config(config, y_train.shape[-1])
            forcing_all, coefficient_all = _select_pde_arrays(data, x_all, loss_config["pde"])
            # BoundaryCorrectionGNN returns field directly; use PassthroughDecoder
            field_decoder = PassthroughDecoder() if boundary_correction_enabled else TorchPCADecoder(encoder)
            # Per-group LR: correction_heads optionally get a separate (lower) lr
            _lr = float(training_cfg.get("lr", 1e-3))
            _wd = float(training_cfg.get("weight_decay", 1e-4))
            _corr_lr_raw = training_cfg.get("correction_heads_lr")
            _loop_optimizer: torch.optim.Optimizer | None = None
            if boundary_correction_enabled and _corr_lr_raw is not None:
                _corr_lr = float(_corr_lr_raw)
                _corr_ids = {id(p) for p in model.correction_heads.parameters()}
                _loop_optimizer = torch.optim.Adam(
                    [
                        {"params": [p for p in model.parameters() if id(p) not in _corr_ids],
                         "lr": _lr, "weight_decay": _wd},
                        {"params": [p for p in model.parameters() if id(p) in _corr_ids],
                         "lr": _corr_lr, "weight_decay": _wd},
                    ]
                )
                print(f"[train] Per-group optimizer: main lr={_lr}, correction_heads lr={_corr_lr}")
            result = train_in_loop_physical_model(
                model,
                field_decoder,
                x_train_latent,
                y_train,
                x_val_latent,
                y_val,
                forcing_train=None if forcing_all is None else forcing_all[splits["train_idx"]],
                forcing_val=None if forcing_all is None else forcing_all[splits["val_idx"]],
                coefficient_train=None if coefficient_all is None else coefficient_all[splits["train_idx"]],
                coefficient_val=None if coefficient_all is None else coefficient_all[splits["val_idx"]],
                loss_config=loss_config,
                device=device_name,
                batch_size=int(training_cfg.get("batch_size", 32)),
                epochs=int(training_cfg.get("epochs", 100)),
                lr=_lr,
                weight_decay=_wd,
                scheduler=training_cfg.get("scheduler"),
                patience=int(training_cfg.get("patience", 30)),
                grad_clip_norm=(
                    float(training_cfg["grad_clip_norm"])
                    if training_cfg.get("grad_clip_norm") is not None
                    else None
                ),
                optimizer=_loop_optimizer,
            )
        else:
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
        if boundary_correction_enabled:
            # model(x) -> (B, H, W) field directly; no PCA inverse transform needed
            y_pred = predict_latent(
                model,
                x_test_latent,
                device=device_name,
                batch_size=int(training_cfg.get("eval_batch_size", 256)),
            )
        else:
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
            "in_loop_loss": float(bool(config.get("mechanisms", {}).get("in_loop_loss", False))),
            "external_postprocessing": 0.0,
            "refinementnet_used": 0.0,
            "num_parameters": count_parameters(model),
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "n_test": int(len(y_test)),
        }
    )
    metrics["pca_fit_seconds"] = 0.0 if warm_start_dir is not None else float(
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
            "in_loop_loss": bool(config.get("mechanisms", {}).get("in_loop_loss", False)),
            "boundary_correction": boundary_correction_enabled,
            "postprocessing_stage": "none",
            "refinementnet_used": False,
            "warm_start_run_dir": None if warm_start_dir is None else str(warm_start_dir),
        }
    )

    encoder.save(run_dir / "pca_encoder.joblib")
    save_model(model, run_dir / "model.pt")
    save_metrics(metrics, run_dir)
    prediction_arrays: dict[str, np.ndarray] = {
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
    if result.term_losses:
        save_npz(run_dir / "term_losses.npz", **result.term_losses)
    save_config(config, run_dir)
    save_environment(run_dir / "environment.txt")
    timer.times.update({f"pca_{key}": value for key, value in encoder.timings.items()})
    if "fit_coarse_svd" in encoder.timings:
        timer.add(COARSE_SVD, float(encoder.timings["fit_coarse_svd"]))
    timer.add(END_TO_END, perf_counter() - end_to_end_start)
    timer.save(run_dir / "runtime.json")
    save_json({"component_counts": encoder.component_counts}, run_dir / "pca_summary.json")
    return run_dir


_SPLIT_KEYS = {"train_idx", "val_idx", "test_idx"}


def _is_image_operator_config(config: dict[str, Any]) -> bool:
    model_type = str(config.get("model", {}).get("type", "")).lower()
    pca_type = str(config.get("pca", {}).get("type", "")).lower()
    return model_type == "fno" or pca_type in {"none", "identity", "image"}


def _train_image_operator_from_arrays(
    config: dict[str, Any],
    run_dir: Path,
    timer: TimingCollector,
    end_to_end_start: float,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    splits: dict[str, np.ndarray],
    data: dict[str, np.ndarray],
    device_name: str,
) -> Path:
    training_cfg = config.setdefault("training", {})
    model_cfg = config.setdefault("model", {})
    model = _make_image_model(model_cfg)
    normalization_cfg = model_cfg.get("normalization", {})
    if not isinstance(normalization_cfg, dict):
        raise TypeError("model.normalization must be a mapping.")
    normalizer = FieldNormalizer.fit(
        x_train,
        y_train,
        enabled=bool(normalization_cfg.get("enabled", True)),
        eps=float(normalization_cfg.get("eps", 1.0e-8)),
    )
    with timer.time(NN_TRAIN):
        result = train_image_model(
            model,
            normalizer.transform_input(x_train),
            normalizer.transform_target(y_train),
            normalizer.transform_input(x_val),
            normalizer.transform_target(y_val),
            device=device_name,
            batch_size=int(training_cfg.get("batch_size", 32)),
            epochs=int(training_cfg.get("epochs", 100)),
            lr=float(training_cfg.get("lr", 1e-3)),
            weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
            scheduler=training_cfg.get("scheduler"),
            patience=int(training_cfg.get("patience", 30)),
            **_early_stopping_kwargs(training_cfg),
        )
    with timer.time(INFERENCE):
        y_pred = normalizer.inverse_transform_target(
            predict_images(
                model,
                normalizer.transform_input(x_test),
                device=device_name,
                batch_size=int(training_cfg.get("eval_batch_size", 256)),
            )
        )

    metrics = _compute_metrics(y_pred, y_test)
    metrics.update(
        {
            "best_val_loss": result.best_val_loss,
            "best_val_loss_normalized": result.best_val_loss,
            "epochs_ran": result.epochs_ran,
            "stopped_early": float(result.stopped_early),
            "num_parameters": count_parameters(model),
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "n_test": int(len(y_test)),
            "pca_fit_seconds": 0.0,
            "external_postprocessing": 0.0,
            "refinementnet_used": 0.0,
        }
    )
    config.setdefault("resolved", {})
    config["resolved"].update(
        {
            "device": device_name,
            "input_latent_dim": 0,
            "output_latent_dim": 0,
            "pca_component_counts": {},
            "image_operator": True,
            "normalization": normalizer.to_dict(),
            "epochs_ran": result.epochs_ran,
            "stopped_early": result.stopped_early,
            "postprocessing_stage": "none",
            "refinementnet_used": False,
        }
    )
    save_model(model, run_dir / "model.pt")
    save_json(normalizer.to_dict(), run_dir / "normalization.json")
    save_metrics(metrics, run_dir)
    prediction_arrays: dict[str, np.ndarray] = {
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
    timer.add(END_TO_END, perf_counter() - end_to_end_start)
    timer.save(run_dir / "runtime.json")
    save_json({"component_counts": {}}, run_dir / "pca_summary.json")
    return run_dir


def _make_image_model(model_cfg: dict[str, Any]) -> torch.nn.Module:
    model_type = str(model_cfg.get("type", "fno")).lower()
    if model_type != "fno":
        raise ValueError(f"Unsupported image operator model.type {model_type!r}.")
    fno_cfg = model_cfg.get("fno", {})
    return FNO2d(
        in_channels=int(fno_cfg.get("in_channels", 1)),
        out_channels=int(fno_cfg.get("out_channels", 1)),
        width=int(fno_cfg.get("width", 32)),
        modes=int(fno_cfg.get("modes", 12)),
        num_layers=int(fno_cfg.get("num_layers", 4)),
        hidden_size=int(fno_cfg.get("hidden_size", 128)),
        padding=int(fno_cfg.get("padding", 8)),
    )


def _early_stopping_kwargs(training_cfg: dict[str, Any]) -> dict[str, Any]:
    early_stopping = training_cfg.get("early_stopping", {})
    if not isinstance(early_stopping, dict) or not bool(early_stopping.get("enabled", False)):
        return {
            "early_stopping_patience": None,
            "early_stopping_min_delta": 0.0,
        }
    return {
        "early_stopping_patience": int(early_stopping.get("patience", 15)),
        "early_stopping_min_delta": float(early_stopping.get("min_delta", 0.0)),
    }


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


def _make_encoder(
    config: dict[str, Any],
    *,
    n_train: int | None = None,
    grid_shape: tuple[int, ...] | None = None,
):
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
        if bool(config.get("mechanisms", {}).get("two_scale", False)):
            two_scale_cfg = pca_cfg.get("two_scale", {})
            grid_size = int(grid_shape[0]) if grid_shape is not None and len(grid_shape) >= 1 else None
            return TwoScaleLocalToLocalPCAEncoder(
                patch_size=int(patches_cfg["patch_size"]),
                stride=int(patches_cfg["stride"]),
                assembly_weights=weights,
                include_edges=bool(patches_cfg.get("include_edges", False)),
                coarse_factor=int(two_scale_cfg.get("coarse_factor", 4)),
                coarse_components=int(two_scale_cfg.get("coarse_components", 20)),
                coarse_variance=float(two_scale_cfg.get("coarse_variance", 0.99)),
                grid_size=grid_size,
                n_samples=n_train,
                **common,
            )
        return LocalToLocalPCAEncoder(
            patch_size=int(patches_cfg["patch_size"]),
            stride=int(patches_cfg["stride"]),
            assembly_weights=weights,
            include_edges=bool(patches_cfg.get("include_edges", False)),
            guard_band=int(pca_cfg.get("guard_band", 0)),
            **common,
        )
    raise ValueError(f"Unsupported pca.type {pca_type!r}.")


def _make_model(
    model_cfg: dict[str, Any],
    encoder: Any,
    *,
    input_dim: int,
    output_dim: int,
    coupling_enabled: bool = False,
    boundary_correction_enabled: bool = False,
):
    model_type = str(model_cfg.get("type", "mlp"))
    hidden_size = int(model_cfg.get("hidden_size", 128))
    num_layers = int(model_cfg.get("num_layers", 4))
    activation = str(model_cfg.get("activation", "relu"))
    dropout = float(model_cfg.get("dropout", 0.0))
    counts = getattr(encoder, "component_counts", {})
    input_counts = [int(value) for value in counts.get("input_patches", [])]
    output_counts = [int(value) for value in counts.get("output_patches", [])]
    if boundary_correction_enabled:
        if not input_counts or not output_counts:
            raise ValueError("BoundaryCorrectionGNN requires a fitted local-to-local PCA encoder.")
        patch_slices = getattr(encoder, "patch_slices", [])
        coupling_cfg = model_cfg.get("coupling", {})
        bc_cfg = model_cfg.get("boundary_correction", {})
        patch_size = int(getattr(encoder, "patch_size", 32))
        pca_decoder = TorchPCADecoder(encoder)
        model = BoundaryCorrectionGNN(
            input_component_counts=input_counts,
            output_component_counts=output_counts,
            grid_shape=_patch_grid_shape(patch_slices),
            embed_dim=int(coupling_cfg.get("embed_dim", hidden_size)),
            num_layers=int(coupling_cfg.get("num_layers", 3)),
            neighborhood=int(coupling_cfg.get("neighborhood", 8)),
            dropout=dropout,
            activation=activation,
            patch_size=patch_size,
            delta=int(bc_cfg.get("delta", 4)),
            pca_decoder=pca_decoder,
        )
        print(
            f"[_make_model] BoundaryCorrectionGNN "
            f"patches={model.num_patches} patch_size={patch_size} "
            f"delta={model.delta} embed_dim={model.embed_dim}"
        )
        return model
    if coupling_enabled:
        if not input_counts or not output_counts:
            raise ValueError("Coupling requires a fitted local-to-local PCA encoder.")
        patch_slices = getattr(encoder, "patch_slices", [])
        coupling_cfg = model_cfg.get("coupling", {})
        global_output_dim = int(counts.get("coarse_output", 0))
        model = CouplingOperator(
            input_component_counts=input_counts,
            output_component_counts=output_counts,
            grid_shape=_patch_grid_shape(patch_slices),
            embed_dim=int(coupling_cfg.get("embed_dim", hidden_size)),
            backend=str(coupling_cfg.get("backend", "gnn")),
            num_layers=int(coupling_cfg.get("num_layers", 3)),
            neighborhood=int(coupling_cfg.get("neighborhood", 8)),
            attention_heads=int(coupling_cfg.get("attention_heads", 4)),
            attention_n_max=int(coupling_cfg.get("attention_n_max", 256)),
            dropout=dropout,
            activation=activation,
            global_output_dim=global_output_dim,
            local_residual=bool(coupling_cfg.get("local_residual", False)),
            local_hidden_size=int(coupling_cfg.get("local_hidden_size", 64)),
            local_num_layers=int(coupling_cfg.get("local_num_layers", 2)),
            zero_init_correction=bool(coupling_cfg.get("zero_init_correction", False)),
            gat_negative_slope=float(coupling_cfg.get("gat_negative_slope", 0.2)),
        )
        if model.input_dim != input_dim or model.output_dim != output_dim:
            raise ValueError(
                f"Coupling model dimensions {model.input_dim}->{model.output_dim} "
                f"do not match training data {input_dim}->{output_dim}."
            )
        print(
            f"[_make_model] CouplingOperator backend={model.backend!r} "
            f"({type(model.backend_module).__name__}) "
            f"patches={model.num_patches} embed_dim={model.embed_dim}"
        )
        return model

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


def _make_in_loop_loss_config(config: dict[str, Any], grid_size: int) -> dict[str, Any]:
    loss_cfg = dict(config.get("loss", {}))
    patches_cfg = config.get("patches", {})
    loss_cfg["patch_size"] = int(patches_cfg["patch_size"])
    loss_cfg["stride"] = int(patches_cfg["stride"])
    loss_cfg["include_edges"] = bool(patches_cfg.get("include_edges", False))
    loss_cfg["dx"] = float(loss_cfg.get("dx", 1.0 / float(grid_size - 1)))
    return loss_cfg


def _warm_start_run_dir(config: dict[str, Any]) -> Path | None:
    warm_start = config.get("warm_start", {})
    if not isinstance(warm_start, dict) or not bool(warm_start.get("enabled", False)):
        return None
    raw_run_dir = warm_start.get("run_dir")
    if not raw_run_dir:
        raise ValueError("warm_start.enabled=true requires warm_start.run_dir.")
    run_dir = normalize_path(raw_run_dir)
    required = [run_dir / "pca_encoder.joblib", run_dir / "model.pt"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Warm-start run is missing required artifacts: {missing}")
    return run_dir


def _select_pde_arrays(
    data: dict[str, np.ndarray],
    x_all: np.ndarray,
    equation: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    equation = equation.lower()
    if equation == "poisson":
        forcing = _first_existing(data, ["f", "source", "forcing"])
        return (np.asarray(forcing, dtype=np.float32) if forcing is not None else x_all, None)
    if equation == "darcy":
        forcing = _first_existing(data, ["f", "source", "forcing", "x_input"])
        coefficient = _first_existing(data, ["a", "coeff", "coefficient", "permeability"])
        if forcing is None or coefficient is None:
            return None, None
        return np.asarray(forcing, dtype=np.float32), np.asarray(coefficient, dtype=np.float32)
    return None, None


def _first_existing(data: dict[str, np.ndarray], keys: list[str]) -> np.ndarray | None:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _resolve_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
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
