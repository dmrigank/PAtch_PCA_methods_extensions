"""Targeted migration of legacy Poisson residual metrics to convention v2."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from lpcanet.data.io import load_npz
from lpcanet.data.splits import make_splits
from lpcanet.metrics.residuals import (
    PDE_METRICS_VERSION,
    POISSON_RESIDUAL_CONVENTION,
    poisson_residual,
)
from lpcanet.models.refinement import RefinementNet
from lpcanet.pca.torch_decoder import TorchPCADecoder
from lpcanet.train.checkpointing import save_json, save_metrics
from lpcanet.train.experiment import _make_image_model, _make_model
from lpcanet.train.loop import (
    predict_images,
    predict_latent,
    predict_physical_fields,
)
from lpcanet.train.normalization import FieldNormalizer
from lpcanet.train.refinement import _RefinementNormalizer
from lpcanet.utils.config import load_config
from lpcanet.utils.paths import normalize_path

RESIDUAL_KEYS = (
    "poisson_residual_mean_abs",
    "poisson_residual_rms",
    "pca_oracle_poisson_residual_mean_abs",
    "pca_oracle_poisson_residual_rms",
)
PROVENANCE_KEYS = (
    "pde_metrics_version",
    "poisson_residual_convention",
)


@dataclass(frozen=True)
class MigrationAction:
    """One archived run and the recovery path needed to repair it."""

    study_dir: Path
    run_dir: Path
    method: str
    seed: int
    resolution: int
    recovery: str
    reason: str


class DatasetCache:
    """Keep at most one processed dataset resident during checkpoint inference."""

    def __init__(self) -> None:
        self._path: Path | None = None
        self._data: dict[str, np.ndarray] | None = None

    def load(self, path: str | Path) -> dict[str, np.ndarray]:
        normalized = normalize_path(path)
        if normalized != self._path:
            self._data = load_npz(normalized)
            self._path = normalized
        assert self._data is not None
        return self._data


def plan_poisson_residual_migration(results_root: str | Path) -> list[MigrationAction]:
    """Inventory Poisson records and classify their migration path."""
    actions: dict[Path, MigrationAction] = {}
    for study_dir, record in _iter_records(results_root):
        if str(record.get("dataset", "")).lower() != "poisson":
            continue
        run_dir = normalize_path(str(record["run_dir"]))
        config = load_config(run_dir / "config.yaml")
        metrics = _load_json(run_dir / "metrics.json")
        if _has_current_provenance(metrics):
            recovery = "current"
            reason = "PDE metrics version 2 already recorded"
        elif _pde_training_affected(config):
            recovery = "retrain"
            reason = "nonzero in-loop Poisson PDE loss"
        elif (run_dir / "predictions_test.npz").is_file():
            recovery = "saved_predictions"
            reason = "test predictions retained"
        elif _has_checkpoint_path(run_dir, config):
            recovery = "checkpoint_inference"
            reason = "test predictions pruned"
        else:
            recovery = "blocked"
            reason = "missing predictions or compatible checkpoint"
        actions.setdefault(
            run_dir,
            MigrationAction(
                study_dir=study_dir,
                run_dir=run_dir,
                method=str(record.get("baseline_model", "")),
                seed=int(record.get("seed", 0)),
                resolution=int(record.get("resolution", 0)),
                recovery=recovery,
                reason=reason,
            )
        )
    return list(actions.values())


def migrate_poisson_residuals(
    results_root: str | Path,
    *,
    apply: bool = False,
    include_validation: bool = True,
    device: str = "auto",
) -> list[dict[str, Any]]:
    """Repair eligible runs and rebuild study rollups when ``apply`` is true."""
    actions = plan_poisson_residual_migration(results_root)
    if not apply:
        return [_action_record(action) for action in actions]

    resolved_device = _resolve_device(device)
    cache = DatasetCache()
    audit: list[dict[str, Any]] = []
    for action in actions:
        if action.recovery in {"current", "retrain", "blocked"}:
            audit.append(_action_record(action))
            continue
        audit.append(
            _migrate_run(
                action,
                cache=cache,
                include_validation=include_validation,
                device=resolved_device,
            )
        )
    for study_dir in _poisson_study_dirs(results_root):
        _refresh_study_rollups(study_dir)
    audit_path = normalize_path(results_root) / "poisson_residual_migration_v2.jsonl"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w", encoding="utf-8") as handle:
        for record in audit:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return audit


def summarize_actions(actions: list[dict[str, Any]]) -> dict[str, int]:
    """Count migration records by recovery path/status."""
    counts: dict[str, int] = {}
    for action in actions:
        recovery = str(action["recovery"])
        counts[recovery] = counts.get(recovery, 0) + 1
    return counts


def _migrate_run(
    action: MigrationAction,
    *,
    cache: DatasetCache,
    include_validation: bool,
    device: str,
) -> dict[str, Any]:
    config = load_config(action.run_dir / "config.yaml")
    old_metrics = _load_json(action.run_dir / "metrics.json")
    predictions_path = action.run_dir / "predictions_test.npz"
    inferred: dict[str, np.ndarray] | None = None
    if predictions_path.is_file():
        test_arrays = load_npz(predictions_path)
    else:
        inferred = _infer_split(action.run_dir, config, "test", cache, device)
        test_arrays = inferred

    updates: dict[str, Any] = _residual_metrics(
        test_arrays["y_pred"],
        test_arrays["x_input"],
        config,
    )
    oracle = test_arrays.get("y_pca_oracle")
    if oracle is None and bool(config.get("evaluation", {}).get("pca_oracle", False)):
        oracle = _pca_oracle(action.run_dir, test_arrays["y_true"])
    if oracle is not None:
        updates.update(
            {
                f"pca_oracle_{key}": value
                for key, value in _residual_metrics(
                    oracle,
                    test_arrays["x_input"],
                    config,
                ).items()
            }
        )
    updates.update(_provenance())
    new_metrics = dict(old_metrics)
    new_metrics.update(updates)
    _assert_only_expected_changes(old_metrics, new_metrics)
    save_metrics(new_metrics, action.run_dir)

    validation_updates: dict[str, float] = {}
    validation_path = action.run_dir / "validation_metrics.json"
    if include_validation and validation_path.is_file():
        validation_arrays = _infer_split(
            action.run_dir,
            config,
            "validation",
            cache,
            device,
        )
        old_validation = _load_json(validation_path)
        validation_updates = _residual_metrics(
            validation_arrays["y_pred"],
            validation_arrays["x_input"],
            config,
        )
        new_validation = dict(old_validation)
        new_validation.update(validation_updates)
        _assert_only_expected_changes(old_validation, new_validation)
        save_json(new_validation, validation_path)

    return {
        **_action_record(action),
        "status": "migrated",
        "old": {key: old_metrics.get(key) for key in RESIDUAL_KEYS if key in old_metrics},
        "new": {key: new_metrics.get(key) for key in RESIDUAL_KEYS if key in new_metrics},
        "validation_updated": bool(validation_updates),
        "inference_used": inferred is not None or bool(validation_updates),
    }


def _infer_split(
    run_dir: Path,
    config: dict[str, Any],
    split: str,
    cache: DatasetCache,
    device: str,
) -> dict[str, np.ndarray]:
    dataset_cfg = config["dataset"]
    data = cache.load(dataset_cfg["processed_path"])
    indices = _split_indices(run_dir, config, data, split)
    input_key = _single_input_key(dataset_cfg)
    output_key = str(dataset_cfg.get("output_key", "u"))
    x = np.asarray(data[input_key][indices], dtype=np.float32)
    y = np.asarray(data[output_key][indices], dtype=np.float32)

    if _is_refinement_run(run_dir, config):
        pred = _predict_refinement(run_dir, config, x, device)
    elif _is_image_run(config):
        pred = _predict_fno(run_dir, config, x, device)
    else:
        pred = _predict_pca_net(run_dir, config, x, device)
    return {"x_input": x, "y_true": y, "y_pred": pred}


def _predict_pca_net(
    run_dir: Path,
    config: dict[str, Any],
    x: np.ndarray,
    device: str,
) -> np.ndarray:
    encoder = joblib.load(run_dir / "pca_encoder.joblib")
    x_latent = encoder.transform_inputs(x)
    resolved = config.get("resolved", {})
    input_dim = int(resolved.get("input_latent_dim", x_latent.shape[1]))
    output_dim = int(
        resolved.get(
            "output_latent_dim",
            encoder.component_counts.get("output_total", encoder.component_counts.get("output", 0)),
        )
    )
    model = _make_model(
        config.get("model", {}),
        encoder,
        input_dim=input_dim,
        output_dim=output_dim,
        coupling_enabled=bool(config.get("mechanisms", {}).get("coupling", False)),
    )
    model.load_state_dict(_load_state_dict(run_dir / "model.pt"))
    batch_size = int(config.get("training", {}).get("eval_batch_size", 256))
    if hasattr(model, "predict_physical"):
        return predict_physical_fields(
            model,
            TorchPCADecoder(encoder),
            x_latent,
            device=device,
            batch_size=batch_size,
        )
    latent = predict_latent(
        model,
        x_latent,
        device=device,
        batch_size=batch_size,
    )
    return encoder.inverse_transform_outputs(latent).astype(np.float32, copy=False)


def _predict_fno(
    run_dir: Path,
    config: dict[str, Any],
    x: np.ndarray,
    device: str,
) -> np.ndarray:
    model = _make_image_model(config.get("model", {}))
    model.load_state_dict(_load_state_dict(run_dir / "model.pt"))
    normalizer_payload = _load_json(run_dir / "normalization.json")
    if not normalizer_payload:
        normalizer_payload = dict(config.get("resolved", {}).get("normalization", {}))
    normalizer = FieldNormalizer(**normalizer_payload)
    normalized = predict_images(
        model,
        normalizer.transform_input(x),
        device=device,
        batch_size=int(config.get("training", {}).get("eval_batch_size", 256)),
    )
    return normalizer.inverse_transform_target(normalized)


def _predict_refinement(
    run_dir: Path,
    config: dict[str, Any],
    x: np.ndarray,
    device: str,
) -> np.ndarray:
    raw_base = config.get("base_run_dir") or config.get("resolved", {}).get("base_run_dir")
    if not raw_base:
        raise ValueError(f"Refinement run is missing base_run_dir: {run_dir}")
    base_run = normalize_path(str(raw_base))
    base_config = load_config(base_run / "config.yaml")
    blocky = _predict_pca_net(base_run, base_config, x, device)
    refinement_cfg = config.get("refinement", {})
    model = RefinementNet(
        in_channels=1,
        hidden_channels=int(refinement_cfg.get("hidden_channels", 32)),
        num_layers=int(refinement_cfg.get("num_layers", 4)),
        kernel_size=int(refinement_cfg.get("kernel_size", 5)),
        activation=str(refinement_cfg.get("activation", "relu")),
        residual=bool(refinement_cfg.get("residual", False)),
    )
    model.load_state_dict(_load_state_dict(run_dir / "refinement_model.pt"))
    payload = dict(config.get("resolved", {}).get("refinement_normalization", {}))
    normalizer = _RefinementNormalizer(**payload)
    normalized = predict_images(
        model,
        normalizer.transform_input(blocky),
        device=device,
        batch_size=int(config.get("training", {}).get("eval_batch_size", 256)),
    )
    return normalizer.inverse_transform_target(normalized)


def _pca_oracle(run_dir: Path, truth: np.ndarray) -> np.ndarray | None:
    path = run_dir / "pca_encoder.joblib"
    if not path.is_file():
        return None
    encoder = joblib.load(path)
    latent = encoder.transform_outputs(truth)
    return encoder.inverse_transform_outputs(latent).astype(np.float32, copy=False)


def _residual_metrics(
    prediction: np.ndarray,
    forcing: np.ndarray,
    config: dict[str, Any],
) -> dict[str, float]:
    resolution = int(config.get("dataset", {}).get("grid_size", prediction.shape[-1]))
    dx = float(config.get("evaluation", {}).get("dx", 1.0 / float(resolution - 1)))
    convention = str(
        config.get("dataset", {}).get(
            "poisson_convention",
            POISSON_RESIDUAL_CONVENTION,
        )
    )
    residual = poisson_residual(
        prediction,
        forcing,
        dx,
        convention=convention,
    )
    return {
        "poisson_residual_mean_abs": float(np.mean(np.abs(residual))),
        "poisson_residual_rms": float(np.sqrt(np.mean(residual**2))),
    }


def _split_indices(
    run_dir: Path,
    config: dict[str, Any],
    data: dict[str, np.ndarray],
    split: str,
) -> np.ndarray:
    key = "val_idx" if split == "validation" else "test_idx"
    candidates = [run_dir / "split_indices.npz"]
    raw_base = config.get("base_run_dir") or config.get("resolved", {}).get("base_run_dir")
    if raw_base:
        candidates.append(normalize_path(str(raw_base)) / "split_indices.npz")
    for path in candidates:
        if path.is_file():
            return np.asarray(load_npz(path)[key], dtype=np.int64)

    output_key = str(config.get("dataset", {}).get("output_key", "u"))
    dataset_cfg = config.get("dataset", {})
    embedded = {name: np.asarray(data[name], dtype=np.int64) for name in ("train_idx", "val_idx", "test_idx") if name in data}
    if bool(dataset_cfg.get("use_embedded_splits", True)) and key in embedded:
        return embedded[key]
    splits = make_splits(
        len(data[output_key]),
        seed=int(config.get("experiment", {}).get("seed", 0)),
    )
    return np.asarray(splits[key], dtype=np.int64)


def _refresh_study_rollups(study_dir: Path) -> None:
    from lpcanet.train.factorial import write_rollup

    jsonl_path = study_dir / "results.jsonl"
    records = _read_jsonl(jsonl_path)
    refreshed: list[dict[str, Any]] = []
    for original in records:
        record = copy.deepcopy(original)
        run_dir = normalize_path(str(record["run_dir"]))
        metrics = _load_json(run_dir / "metrics.json")
        validation = _load_json(run_dir / "validation_metrics.json")
        record.setdefault("metrics", {}).update(
            {
                key: value
                for key, value in metrics.items()
                if key in RESIDUAL_KEYS or key in PROVENANCE_KEYS
            }
        )
        record.setdefault("validation_metrics", {}).update(
            {key: value for key, value in validation.items() if key in RESIDUAL_KEYS}
        )
        refreshed.append(record)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in refreshed:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    write_rollup(refreshed, study_dir)


def _iter_records(results_root: str | Path):
    root = normalize_path(results_root)
    paths = sorted(root.rglob("results.jsonl"))
    for path in paths:
        for record in _read_jsonl(path):
            yield path.parent, record


def _poisson_study_dirs(results_root: str | Path) -> list[Path]:
    directories: set[Path] = set()
    for study_dir, record in _iter_records(results_root):
        if str(record.get("dataset", "")).lower() == "poisson":
            directories.add(study_dir)
    return sorted(directories)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return dict(json.load(handle))


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _pde_training_affected(config: dict[str, Any]) -> bool:
    in_loop = bool(config.get("mechanisms", {}).get("in_loop_loss", False))
    weight = float(config.get("loss", {}).get("weights", {}).get("pde_residual", 0.0) or 0.0)
    return in_loop and weight != 0.0


def _has_current_provenance(metrics: dict[str, Any]) -> bool:
    return (
        int(metrics.get("pde_metrics_version", 0)) == PDE_METRICS_VERSION
        and str(metrics.get("poisson_residual_convention", ""))
        == POISSON_RESIDUAL_CONVENTION
    )


def _has_checkpoint_path(run_dir: Path, config: dict[str, Any]) -> bool:
    if _is_refinement_run(run_dir, config):
        return (run_dir / "refinement_model.pt").is_file()
    return (run_dir / "model.pt").is_file()


def _is_refinement_run(run_dir: Path, config: dict[str, Any]) -> bool:
    return (run_dir / "refinement_model.pt").is_file() or bool(
        config.get("resolved", {}).get("refinementnet_used", False)
    )


def _is_image_run(config: dict[str, Any]) -> bool:
    return str(config.get("model", {}).get("type", "")).lower() == "fno"


def _single_input_key(dataset_cfg: dict[str, Any]) -> str:
    keys = dataset_cfg.get("input_keys", ["f"])
    if isinstance(keys, str):
        return keys
    if len(keys) != 1:
        raise ValueError(f"Expected one Poisson input key, got {keys}.")
    return str(keys[0])


def _provenance() -> dict[str, int | str]:
    return {
        "pde_metrics_version": PDE_METRICS_VERSION,
        "poisson_residual_convention": POISSON_RESIDUAL_CONVENTION,
    }


def _assert_only_expected_changes(
    old: dict[str, Any],
    new: dict[str, Any],
) -> None:
    allowed = set(RESIDUAL_KEYS) | set(PROVENANCE_KEYS)
    changed = {key for key in set(old) | set(new) if old.get(key) != new.get(key)}
    unexpected = changed - allowed
    if unexpected:
        raise AssertionError(f"Migration changed unrelated metrics: {sorted(unexpected)}")


def _action_record(action: MigrationAction) -> dict[str, Any]:
    return {
        "study_dir": str(action.study_dir),
        "run_dir": str(action.run_dir),
        "method": action.method,
        "seed": action.seed,
        "resolution": action.resolution,
        "recovery": action.recovery,
        "reason": action.reason,
        "status": (
            "planned"
            if action.recovery not in {"current", "retrain", "blocked"}
            else action.recovery
        ),
    }


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false.")
    return device
