"""Multirun expansion and result recording for mechanism factorials."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from lpcanet.metrics.evaluate import evaluate_run
from lpcanet.train.experiment import train_from_config
from lpcanet.train.refinement import train_refinement_from_config
from lpcanet.utils.config import Config, compose_config, deep_merge, load_config, save_config
from lpcanet.utils.paths import ensure_dir

MECHANISM_KEYS = ("two_scale", "coupling", "in_loop_loss")


@dataclass(frozen=True)
class FactorialRun:
    """One expanded multirun cell."""

    run_id: str
    config: Config
    config_hash: str
    seed: int
    mechanisms: dict[str, bool]
    resolution: int
    run_dir: Path
    baseline_model: str = ""


def expand_factorial_runs(
    *,
    root: str | Path,
    dataset: str = "poisson",
    model: str = "l2l",
    experiment: str = "factorial",
    overrides: list[str] | None = None,
    output_dir: str | Path | None = None,
) -> list[FactorialRun]:
    """Expand a factorial experiment config into concrete single-run configs."""
    base = compose_config(
        root=root,
        dataset=dataset,
        model=model,
        experiment=experiment,
        overrides=overrides,
    )
    matrix = base.get("matrix", {})
    mechanisms_matrix = matrix.get("mechanisms", {})
    seeds = [int(seed) for seed in matrix.get("seeds", [base.get("experiment", {}).get("seed", 0)])]
    resolutions = [int(value) for value in matrix.get("resolutions", [base.get("dataset", {}).get("grid_size", 128)])]
    output_root = Path(
        output_dir
        or matrix.get("output_root")
        or base.get("experiment", {}).get("output_dir")
        or "results/factorial"
    )

    mechanism_values = [
        [bool(value) for value in mechanisms_matrix.get(key, [base.get("mechanisms", {}).get(key, False)])]
        for key in MECHANISM_KEYS
    ]
    runs: list[FactorialRun] = []
    for resolution in resolutions:
        for flags_tuple in itertools.product(*mechanism_values):
            mechanisms = dict(zip(MECHANISM_KEYS, flags_tuple))
            cell_name = _cell_name(mechanisms)
            for seed in seeds:
                config = copy.deepcopy(base)
                config.pop("matrix", None)
                config.setdefault("mechanisms", {}).update(mechanisms)
                config.setdefault("experiment", {})["seed"] = seed
                config["experiment"]["name"] = f"factorial_{cell_name}_seed{seed}_r{resolution}"
                config.setdefault("dataset", {})["grid_size"] = resolution
                run_dir = output_root / f"resolution_{resolution}" / cell_name / f"seed_{seed}"
                config["experiment"]["output_dir"] = str(run_dir)
                config.setdefault("resolved", {}).update(
                    {
                        "multirun": True,
                        "run_id": config["experiment"]["name"],
                        "mechanism_cell": cell_name,
                    }
                )
                config_hash = config_hash_for(config)
                runs.append(
                    FactorialRun(
                        run_id=config["experiment"]["name"],
                        config=config,
                        config_hash=config_hash,
                        seed=seed,
                        mechanisms=mechanisms,
                        resolution=resolution,
                        run_dir=run_dir,
                    )
                )
    return runs


def run_factorial(
    *,
    root: str | Path,
    dataset: str = "poisson",
    model: str = "l2l",
    experiment: str = "factorial",
    overrides: list[str] | None = None,
    limit_samples: int | None = None,
    epochs: int | None = None,
    output_dir: str | Path | None = None,
    device: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> list[FactorialRun]:
    """Expand and optionally execute a mechanism factorial."""
    runs = expand_factorial_runs(
        root=root,
        dataset=dataset,
        model=model,
        experiment=experiment,
        overrides=overrides,
        output_dir=output_dir,
    )
    if dry_run:
        for run in runs:
            print(_dry_run_line(run))
        return runs

    result_root = Path(output_dir or runs[0].run_dir.parents[2] if runs else "results/factorial")
    result_root = ensure_dir(result_root)
    records: list[dict[str, Any]] = []
    jsonl_path = result_root / "results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for run in runs:
            run_config = copy.deepcopy(run.config)
            if epochs is not None:
                run_config.setdefault("training", {})["epochs"] = int(epochs)
            if device is not None:
                run_config.setdefault("training", {})["device"] = device
            resolved_dir = ensure_dir(run.run_dir / "resolved_configs")
            config_path = save_config(run_config, resolved_dir, filename="config.yaml")
            trained_dir = train_from_config(
                config_path,
                limit_samples=limit_samples,
                epochs=epochs,
                output_dir=run.run_dir,
                device=device,
                overwrite=overwrite,
            )
            metrics = evaluate_run(trained_dir)
            record = build_result_record(run, metrics=metrics, run_dir=trained_dir)
            records.append(record)
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
    write_rollup(records, result_root)
    return runs


def expand_baseline_runs(
    *,
    root: str | Path,
    dataset: str = "poisson",
    experiment: str = "baselines",
    overrides: list[str] | None = None,
    output_dir: str | Path | None = None,
) -> list[FactorialRun]:
    """Expand configured prior-paper/FNO baseline runs."""
    root_path = Path(root)
    experiment_cfg = compose_config(
        root=root,
        dataset=dataset,
        model="l2l",
        experiment=experiment,
        overrides=overrides,
    )
    matrix = experiment_cfg.get("matrix", {})
    baseline_models = [str(value) for value in matrix.get("models", [])]
    seeds = [int(seed) for seed in matrix.get("seeds", [0])]
    resolutions = [int(value) for value in matrix.get("resolutions", [128])]
    output_root = Path(output_dir or matrix.get("output_root") or "results/baselines")
    runs: list[FactorialRun] = []
    for resolution in resolutions:
        for baseline_model in baseline_models:
            for seed in seeds:
                config = compose_config(
                    root=root_path,
                    dataset=dataset,
                    model=baseline_model,
                    experiment=experiment,
                    overrides=overrides,
                )
                config.pop("matrix", None)
                config.setdefault("mechanisms", {}).update(
                    {"two_scale": False, "coupling": False, "in_loop_loss": False}
                )
                config.setdefault("dataset", {})["grid_size"] = resolution
                config.setdefault("experiment", {})["seed"] = seed
                config["experiment"]["name"] = f"baseline_{baseline_model}_seed{seed}_r{resolution}"
                run_dir = output_root / f"resolution_{resolution}" / baseline_model / f"seed_{seed}"
                config["experiment"]["output_dir"] = str(run_dir)
                if baseline_model == "l2l_refinement":
                    base_config = compose_config(
                        root=root_path,
                        dataset=dataset,
                        model="l2l",
                        experiment="single",
                    )
                    base_config.setdefault("experiment", {})["seed"] = seed
                    base_config.setdefault("dataset", {})["grid_size"] = resolution
                    base_run_dir = output_root / f"resolution_{resolution}" / "l2l" / f"seed_{seed}"
                    base_config["experiment"]["output_dir"] = str(base_run_dir)
                    config["base_run_dir"] = str(base_run_dir)
                    config["base_config_resolved"] = base_config
                config.setdefault("resolved", {}).update(
                    {
                        "multirun": True,
                        "baseline_model": baseline_model,
                        "run_id": config["experiment"]["name"],
                    }
                )
                config_hash = config_hash_for(config)
                runs.append(
                    FactorialRun(
                        run_id=config["experiment"]["name"],
                        config=config,
                        config_hash=config_hash,
                        seed=seed,
                        mechanisms={key: bool(config["mechanisms"][key]) for key in MECHANISM_KEYS},
                        resolution=resolution,
                        run_dir=run_dir,
                        baseline_model=baseline_model,
                    )
                )
    return runs


def expand_resolution_sweep_runs(
    *,
    root: str | Path,
    dataset: str = "poisson",
    experiment: str = "resolution_sweep",
    overrides: list[str] | None = None,
    output_dir: str | Path | None = None,
    include_optional: bool = False,
) -> list[FactorialRun]:
    """Expand the paper-grade Poisson resolution and cost study."""
    root_path = Path(root)
    if dataset != "poisson":
        raise ValueError("The paper resolution sweep is defined for the Poisson dataset.")
    study = compose_config(
        root=root_path,
        dataset=dataset,
        model="l2l",
        experiment=experiment,
        overrides=overrides,
    )
    matrix = study.get("matrix", {})
    expected_methods = [
        "global_pca",
        "plain_l2l",
        "overlap_l2l",
        "two_scale",
        "two_scale_interface",
        "fno",
    ]
    methods = [str(value) for value in matrix.get("methods", [])]
    if methods != expected_methods:
        raise ValueError(
            "matrix.methods must preserve the dependency and manuscript order "
            f"{expected_methods}, got {methods}."
        )
    seeds = [int(seed) for seed in matrix.get("seeds", [0, 1, 2])]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("matrix.seeds must contain unique values.")
    n_train = int(matrix.get("n_train", 8000))
    output_root = Path(
        output_dir
        or matrix.get("output_root")
        or study.get("experiment", {}).get("output_dir")
        or "paper_results/resolution_sweep"
    )
    processed_paths = matrix.get("processed_paths", {})
    resolution_settings = matrix.get("resolutions", {})
    if not isinstance(processed_paths, Mapping) or not processed_paths:
        raise TypeError("matrix.processed_paths must be a non-empty mapping.")
    if not isinstance(resolution_settings, Mapping) or not resolution_settings:
        raise TypeError("matrix.resolutions must be a non-empty mapping.")

    representation = matrix.get("representation", {})
    latent_training = matrix.get("latent_training", {})
    interface = matrix.get("interface_fine_tune", {})
    fno_training = matrix.get("fno_training", {})
    retention = matrix.get("prediction_retention", {})
    for name, value in (
        ("representation", representation),
        ("latent_training", latent_training),
        ("interface_fine_tune", interface),
        ("fno_training", fno_training),
        ("prediction_retention", retention),
    ):
        if not isinstance(value, Mapping):
            raise TypeError(f"matrix.{name} must be a mapping.")
    retained_seeds = {int(value) for value in retention.get("seeds", [0])}
    retained_resolutions = {
        int(value) for value in retention.get("resolutions", [64, 128, 256])
    }
    if not retained_seeds.issubset(set(seeds)):
        raise ValueError("Prediction-retention seeds must be part of matrix.seeds.")

    blocks: list[tuple[str, int, Mapping[str, Any], list[str], str]] = []
    for raw_resolution, settings in resolution_settings.items():
        resolution = int(raw_resolution)
        if not isinstance(settings, Mapping):
            raise TypeError(f"Resolution settings for {resolution} must be a mapping.")
        processed_path = _mapping_value(processed_paths, resolution)
        if processed_path is None:
            raise ValueError(f"No processed path configured for resolution {resolution}.")
        blocks.append(("main", resolution, settings, methods, str(processed_path)))

    optional = matrix.get("optional", {})
    if not isinstance(optional, Mapping):
        raise TypeError("matrix.optional must be a mapping.")
    if include_optional or bool(optional.get("enabled", False)):
        optional_resolution = int(optional.get("resolution", 512))
        optional_methods = [str(value) for value in optional.get("methods", [])]
        if not optional_methods or not set(optional_methods).issubset(
            {"plain_l2l", "two_scale"}
        ):
            raise ValueError(
                "matrix.optional.methods must be a non-empty subset of "
                "['plain_l2l', 'two_scale']."
            )
        blocks.append(
            (
                "optional",
                optional_resolution,
                optional,
                optional_methods,
                str(
                    optional.get(
                        "processed_path",
                        f"data/processed/poisson_{optional_resolution}.npz",
                    )
                ),
            )
        )

    runs: list[FactorialRun] = []
    for block_name, resolution, settings, block_methods, processed_path in blocks:
        patch_size = int(settings.get("patch_size", resolution // 4))
        plain_stride = int(settings.get("plain_stride", patch_size))
        overlap_stride = int(settings.get("overlap_stride", patch_size // 2))
        coarse_factor = int(settings.get("coarse_factor", 4))
        _assert_coarse_factor_legal(resolution, coarse_factor, n_train)
        if resolution % patch_size != 0:
            raise ValueError(
                f"resolution={resolution} must be divisible by patch_size={patch_size}."
            )
        for seed in seeds:
            two_scale_dir = (
                output_root
                / "poisson"
                / f"resolution_{resolution}"
                / "two_scale"
                / f"seed_{seed}"
            )
            for method in block_methods:
                model_name = {
                    "global_pca": "global_pca",
                    "plain_l2l": "l2l",
                    "overlap_l2l": "l2l_overlap",
                    "two_scale": "l2l",
                    "two_scale_interface": "l2l",
                    "fno": "fno",
                }[method]
                config = compose_config(
                    root=root_path,
                    dataset=dataset,
                    model=model_name,
                    experiment=experiment,
                    overrides=overrides,
                )
                config.pop("matrix", None)
                is_two_scale = method in {"two_scale", "two_scale_interface"}
                is_interface = method == "two_scale_interface"
                is_fno = method == "fno"
                mechanisms = {
                    "two_scale": is_two_scale,
                    "coupling": False,
                    "in_loop_loss": is_interface,
                }
                config.setdefault("mechanisms", {}).update(mechanisms)
                config.setdefault("dataset", {}).update(
                    {
                        "name": "poisson",
                        "grid_size": resolution,
                        "processed_path": processed_path,
                        "train_samples": n_train,
                        "use_embedded_splits": False,
                        "save_split_indices": True,
                    }
                )
                stride = overlap_stride if method == "overlap_l2l" else plain_stride
                config.setdefault("patches", {}).update(
                    {
                        "patch_size": patch_size,
                        "stride": stride,
                        "assembly": (
                            "hann_safe" if method == "overlap_l2l" else "average"
                        ),
                    }
                )

                if not is_fno:
                    output_variance = float(
                        representation.get(
                            "residual_variance"
                            if is_two_scale
                            else "local_output_variance",
                            0.995 if is_two_scale else 0.99,
                        )
                    )
                    config.setdefault("pca", {}).update(
                        {
                            "solver": str(representation.get("solver", "randomized")),
                            "input_variance": float(
                                representation.get("input_variance", 0.99)
                            ),
                            "output_variance": output_variance,
                            "randomized": {
                                "oversampling": int(
                                    representation.get("randomized_oversampling", 20)
                                ),
                                "n_iter": int(
                                    representation.get("randomized_n_iter", 4)
                                ),
                            },
                        }
                    )
                    config["pca"].setdefault("two_scale", {}).update(
                        {
                            "coarse_factor": coarse_factor,
                            "coarse_components": int(
                                representation.get("coarse_components", 10)
                            ),
                            "coarse_variance": float(
                                representation.get("coarse_variance", 1.0)
                            ),
                            "coarse_solver": str(
                                representation.get("solver", "randomized")
                            ),
                        }
                    )
                    config.setdefault("training", {}).update(
                        {
                            key: copy.deepcopy(value)
                            for key, value in latent_training.items()
                        }
                    )
                    config["training"].setdefault("latent_loss", {}).update(
                        {
                            "mode": "block_balanced" if is_two_scale else "mse",
                            "coarse_weight": 0.5,
                            "residual_weight": 0.5,
                        }
                    )
                else:
                    config.setdefault("training", {}).update(
                        {
                            key: copy.deepcopy(value)
                            for key, value in fno_training.items()
                            if key != "early_stopping"
                        }
                    )
                    if isinstance(fno_training.get("early_stopping"), Mapping):
                        config["training"].setdefault("early_stopping", {}).update(
                            copy.deepcopy(fno_training["early_stopping"])
                        )

                if is_interface:
                    config.setdefault("training", {}).update(
                        {
                            "epochs": int(interface.get("epochs", 50)),
                            "lr": float(interface.get("lr", 1.0e-4)),
                            "weight_decay": float(interface.get("weight_decay", 0.0)),
                            "scheduler": str(interface.get("scheduler", "none")),
                        }
                    )
                    config["warm_start"] = {
                        "enabled": True,
                        "run_dir": str(two_scale_dir),
                    }
                    config.setdefault("loss", {}).update(
                        {
                            "reconstruction": "relative_l2",
                            "interface_target": "truth",
                            "active_terms": [
                                "recon",
                                "interface_value",
                                "interface_flux",
                            ],
                            "pde": "poisson",
                            "weights": {
                                "recon": 1.0,
                                "interface_value": 0.0,
                                "interface_flux": 0.0,
                                "spectral": 0.0,
                                "pde_residual": 0.0,
                            },
                            "warmup": {
                                "reconstruction_epochs": int(
                                    interface.get("reconstruction_epochs", 5)
                                ),
                                "ramp_epochs": int(interface.get("ramp_epochs", 10)),
                            },
                            "weight_calibration": {
                                "enabled": True,
                                "target_ratios": {
                                    "interface_value": float(
                                        interface.get("value_target_ratio", 0.2)
                                    ),
                                    "interface_flux": float(
                                        interface.get("flux_target_ratio", 0.2)
                                    ),
                                },
                                "batch_size": int(
                                    interface.get("calibration_batch_size", 16)
                                ),
                                "max_samples": int(
                                    interface.get("calibration_samples", 128)
                                ),
                                "minimum_weight": 1.0e-12,
                                "maximum_weight": 1.0e6,
                            },
                        }
                    )

                retain_predictions = (
                    seed in retained_seeds and resolution in retained_resolutions
                )
                config.setdefault("evaluation", {}).update(
                    {
                        "pca_oracle": not is_fno,
                        "validation_metrics": True,
                        "retain_predictions": retain_predictions,
                        "metrics": [
                            "mse",
                            "mae",
                            "mre",
                            "ssim",
                            "relative_spectrum_error",
                            "interface_jump",
                            "interface_flux_jump",
                            "interface_value_trace_error",
                            "interface_flux_trace_error",
                            "poisson_residual",
                        ],
                    }
                )

                run_id = f"{experiment}_poisson_{method}_seed{seed}_r{resolution}"
                run_dir = (
                    output_root
                    / "poisson"
                    / f"resolution_{resolution}"
                    / method
                    / f"seed_{seed}"
                )
                ablation = {
                    "family": "resolution_sweep",
                    "cell": method,
                    "method": method,
                    "block": block_name,
                    "n_train": n_train,
                    "patch_size": patch_size,
                    "stride": stride,
                    "coarse_factor": coarse_factor,
                    "pca_solver": (
                        "none"
                        if is_fno
                        else str(representation.get("solver", "randomized"))
                    ),
                    "common_seam_stride": plain_stride,
                    "split_policy": "regenerate_from_run_seed",
                }
                config.setdefault("experiment", {}).update(
                    {
                        "name": run_id,
                        "seed": seed,
                        "output_dir": str(run_dir),
                    }
                )
                config.setdefault("resolved", {}).update(
                    {
                        "multirun": True,
                        "study": experiment,
                        "study_type": "resolution_sweep",
                        "study_method": method,
                        "dataset": "poisson",
                        "n_train": n_train,
                        "coarse_factor": coarse_factor,
                        "ablation": ablation,
                        "run_id": run_id,
                    }
                )
                runs.append(
                    FactorialRun(
                        run_id=run_id,
                        config=config,
                        config_hash=config_hash_for(config),
                        seed=seed,
                        mechanisms=mechanisms,
                        resolution=resolution,
                        run_dir=run_dir,
                        baseline_model=method,
                    )
                )
    return runs


def run_resolution_sweep(
    *,
    root: str | Path,
    dataset: str = "poisson",
    experiment: str = "resolution_sweep",
    overrides: list[str] | None = None,
    limit_samples: int | None = None,
    epochs: int | None = None,
    output_dir: str | Path | None = None,
    device: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    include_optional: bool = False,
) -> list[FactorialRun]:
    """Run the controlled three-seed Poisson resolution study."""
    if limit_samples is not None:
        raise ValueError(
            "resolution_sweep fixes m through matrix.n_train; "
            "do not use --limit-samples."
        )
    if epochs is not None:
        raise ValueError(
            "resolution_sweep uses method-specific fixed training schedules; "
            "do not use --epochs."
        )
    study = compose_config(
        root=Path(root),
        dataset=dataset,
        model="l2l",
        experiment=experiment,
        overrides=overrides,
    )
    matrix = study.get("matrix", {})
    result_root = Path(
        output_dir
        or matrix.get("output_root")
        or study.get("experiment", {}).get("output_dir")
        or "paper_results/resolution_sweep"
    )
    runs = expand_resolution_sweep_runs(
        root=root,
        dataset=dataset,
        experiment=experiment,
        overrides=overrides,
        output_dir=output_dir,
        include_optional=include_optional,
    )
    if dry_run:
        for run in runs:
            print(_dry_run_line(run))
        return runs
    _validate_study_dataset_paths(runs)
    result_root = ensure_dir(result_root)
    write_pca_projection_table(
        resolutions=sorted({run.resolution for run in runs}),
        output_dir=result_root,
        n_samples=int(matrix.get("n_train", 8000)),
    )
    records = _execute_runs(
        runs,
        result_root=result_root,
        limit_samples=None,
        epochs=None,
        device=device,
        overwrite=overwrite,
    )
    write_stage_cost_table(records, result_root)
    return runs


def expand_headline_poisson_runs(
    *,
    root: str | Path,
    dataset: str = "poisson",
    experiment: str = "headline_poisson_128",
    overrides: list[str] | None = None,
    output_dir: str | Path | None = None,
) -> list[FactorialRun]:
    """Expand the controlled five-seed Poisson 128 headline comparison."""
    root_path = Path(root)
    if dataset != "poisson":
        raise ValueError("The headline study is defined for the Poisson dataset.")
    study = compose_config(
        root=root_path,
        dataset=dataset,
        model="l2l",
        experiment=experiment,
        overrides=overrides,
    )
    matrix = study.get("matrix", {})
    expected_order = [
        "global_pca",
        "plain_l2l",
        "overlap_l2l",
        "l2l_refinement",
        "fno",
        "two_scale",
        "two_scale_interface",
    ]
    method_order = [str(value) for value in matrix.get("headline_methods", [])]
    if method_order != expected_order:
        raise ValueError(
            "matrix.headline_methods must preserve the dependency and manuscript "
            f"order {expected_order}, got {method_order}."
        )
    seeds = [int(seed) for seed in matrix.get("seeds", [])]
    if seeds != [0, 1, 2, 3, 4]:
        raise ValueError(
            "The headline study requires the paired seeds [0, 1, 2, 3, 4]."
        )
    resolutions = {int(value) for value in matrix.get("resolutions", {})}
    if resolutions != {128}:
        raise ValueError("The headline study must contain only resolution 128.")

    core_runs = expand_resolution_sweep_runs(
        root=root_path,
        dataset=dataset,
        experiment=experiment,
        overrides=overrides,
        output_dir=output_dir,
    )
    output_root = Path(
        output_dir
        or matrix.get("output_root")
        or study.get("experiment", {}).get("output_dir")
        or "paper_results/headline_poisson_128"
    )
    core_by_seed: dict[int, dict[str, FactorialRun]] = {seed: {} for seed in seeds}
    for run in core_runs:
        config = copy.deepcopy(run.config)
        config.setdefault("resolved", {}).update(
            {
                "study": experiment,
                "study_type": "headline_poisson_128",
                "study_method": run.baseline_model,
            }
        )
        config["resolved"].setdefault("ablation", {}).update(
            {
                "family": "headline_poisson_128",
                "cell": run.baseline_model,
                "method": run.baseline_model,
            }
        )
        normalized = FactorialRun(
            run_id=run.run_id,
            config=config,
            config_hash=config_hash_for(config),
            seed=run.seed,
            mechanisms=dict(run.mechanisms),
            resolution=run.resolution,
            run_dir=run.run_dir,
            baseline_model=run.baseline_model,
        )
        core_by_seed[run.seed][run.baseline_model] = normalized

    refinement_training = matrix.get("refinement_training", {})
    if not isinstance(refinement_training, Mapping):
        raise TypeError("matrix.refinement_training must be a mapping.")
    runs: list[FactorialRun] = []
    for seed in seeds:
        seed_runs = core_by_seed[seed]
        plain = seed_runs["plain_l2l"]
        refinement = compose_config(
            root=root_path,
            dataset=dataset,
            model="l2l_refinement",
            experiment=experiment,
            overrides=overrides,
        )
        refinement.pop("matrix", None)
        refinement["mechanisms"] = {
            "two_scale": False,
            "coupling": False,
            "in_loop_loss": False,
        }
        refinement["dataset"] = copy.deepcopy(plain.config["dataset"])
        refinement["patches"] = copy.deepcopy(plain.config["patches"])
        refinement["pca"] = copy.deepcopy(plain.config["pca"])
        refinement.setdefault("training", {}).update(
            {key: copy.deepcopy(value) for key, value in refinement_training.items()}
        )
        refinement["base_run_dir"] = str(plain.run_dir)
        refinement["base_config_resolved"] = copy.deepcopy(plain.config)
        refinement["train_base_if_missing"] = True
        refinement["evaluation"] = copy.deepcopy(plain.config["evaluation"])
        run_id = f"{experiment}_poisson_l2l_refinement_seed{seed}_r128"
        run_dir = (
            output_root
            / "poisson"
            / "resolution_128"
            / "l2l_refinement"
            / f"seed_{seed}"
        )
        refinement.setdefault("experiment", {}).update(
            {"name": run_id, "seed": seed, "output_dir": str(run_dir)}
        )
        refinement.setdefault("resolved", {}).update(
            {
                "multirun": True,
                "study": experiment,
                "study_type": "headline_poisson_128",
                "study_method": "l2l_refinement",
                "dataset": "poisson",
                "n_train": int(matrix.get("n_train", 8000)),
                "postprocessing_stage": "refinementnet",
                "refinementnet_used": True,
                "run_id": run_id,
                "ablation": {
                    "family": "headline_poisson_128",
                    "cell": "l2l_refinement",
                    "method": "l2l_refinement",
                    "n_train": int(matrix.get("n_train", 8000)),
                    "patch_size": 32,
                    "stride": 32,
                    "pca_solver": str(refinement["pca"].get("solver", "randomized")),
                    "common_seam_stride": 32,
                    "split_policy": "regenerate_from_run_seed",
                },
            }
        )
        refinement_run = FactorialRun(
            run_id=run_id,
            config=refinement,
            config_hash=config_hash_for(refinement),
            seed=seed,
            mechanisms={key: False for key in MECHANISM_KEYS},
            resolution=128,
            run_dir=run_dir,
            baseline_model="l2l_refinement",
        )
        seed_runs["l2l_refinement"] = refinement_run
        runs.extend(seed_runs[method] for method in method_order)
    return runs


def run_headline_poisson_study(
    *,
    root: str | Path,
    dataset: str = "poisson",
    experiment: str = "headline_poisson_128",
    overrides: list[str] | None = None,
    limit_samples: int | None = None,
    epochs: int | None = None,
    output_dir: str | Path | None = None,
    device: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> list[FactorialRun]:
    """Run the frozen five-seed Poisson 128 headline study."""
    if limit_samples is not None:
        raise ValueError(
            "headline_poisson_128 fixes m through matrix.n_train; "
            "do not use --limit-samples."
        )
    if epochs is not None:
        raise ValueError(
            "headline_poisson_128 uses method-specific fixed schedules; "
            "do not use --epochs."
        )
    study = compose_config(
        root=Path(root),
        dataset=dataset,
        model="l2l",
        experiment=experiment,
        overrides=overrides,
    )
    matrix = study.get("matrix", {})
    result_root = Path(
        output_dir
        or matrix.get("output_root")
        or study.get("experiment", {}).get("output_dir")
        or "paper_results/headline_poisson_128"
    )
    runs = expand_headline_poisson_runs(
        root=root,
        dataset=dataset,
        experiment=experiment,
        overrides=overrides,
        output_dir=output_dir,
    )
    if dry_run:
        for run in runs:
            print(_dry_run_line(run))
        return runs
    _validate_study_dataset_paths(runs)
    result_root = ensure_dir(result_root)
    records = _execute_runs(
        runs,
        result_root=result_root,
        limit_samples=None,
        epochs=None,
        device=device,
        overwrite=overwrite,
    )
    write_stage_cost_table(records, result_root)
    return runs


def expand_darcy_generalization_runs(
    *,
    root: str | Path,
    dataset: str = "darcy",
    experiment: str = "darcy_generalization",
    overrides: list[str] | None = None,
    output_dir: str | Path | None = None,
) -> list[FactorialRun]:
    """Expand the frozen five-seed Darcy 256 generalization study."""
    root_path = Path(root)
    if dataset != "darcy":
        raise ValueError("The Darcy generalization study requires dataset=darcy.")
    study = compose_config(
        root=root_path,
        dataset=dataset,
        model="l2l",
        experiment=experiment,
        overrides=overrides,
    )
    matrix = study.get("matrix", {})
    expected_core = [
        "global_pca",
        "plain_l2l",
        "overlap_l2l",
        "two_scale",
        "two_scale_interface",
        "fno",
    ]
    core_methods = [str(value) for value in matrix.get("methods", [])]
    if core_methods != expected_core:
        raise ValueError(
            "matrix.methods must preserve the shared expansion order "
            f"{expected_core}, got {core_methods}."
        )
    expected_order = [
        "global_pca",
        "plain_l2l",
        "overlap_l2l",
        "fno",
        "two_scale",
        "two_scale_interface",
    ]
    method_order = [str(value) for value in matrix.get("darcy_methods", [])]
    if method_order != expected_order:
        raise ValueError(
            "matrix.darcy_methods must preserve the manuscript and dependency "
            f"order {expected_order}, got {method_order}."
        )
    seeds = [int(seed) for seed in matrix.get("seeds", [])]
    if seeds != [0, 1, 2, 3, 4]:
        raise ValueError(
            "The Darcy generalization study requires paired seeds [0, 1, 2, 3, 4]."
        )
    resolutions = {int(value) for value in matrix.get("resolutions", {})}
    if resolutions != {256}:
        raise ValueError("The primary Darcy study must contain only resolution 256.")

    # Reuse the frozen method construction and replace only dataset/PDE semantics.
    core_runs = expand_resolution_sweep_runs(
        root=root_path,
        dataset="poisson",
        experiment=experiment,
        overrides=overrides,
        output_dir=output_dir,
    )
    output_root = Path(
        output_dir
        or matrix.get("output_root")
        or study.get("experiment", {}).get("output_dir")
        or "paper_results/darcy_generalization"
    )
    dataset_template = copy.deepcopy(study.get("dataset", {}))
    by_seed: dict[int, dict[str, FactorialRun]] = {seed: {} for seed in seeds}
    for run in core_runs:
        config = copy.deepcopy(run.config)
        source_dataset = config.get("dataset", {})
        darcy_dataset = copy.deepcopy(dataset_template)
        for key in (
            "grid_size",
            "processed_path",
            "train_samples",
            "use_embedded_splits",
            "save_split_indices",
        ):
            if key in source_dataset:
                darcy_dataset[key] = copy.deepcopy(source_dataset[key])
        darcy_dataset["name"] = "darcy"
        darcy_dataset["input_keys"] = ["a"]
        darcy_dataset["output_key"] = "u"
        darcy_dataset["constant_forcing"] = 1.0
        config["dataset"] = darcy_dataset
        config.setdefault("loss", {})["pde"] = "darcy"
        config.setdefault("evaluation", {})["metrics"] = [
            "mse",
            "mae",
            "mre",
            "ssim",
            "relative_spectrum_error",
            "interface_jump",
            "interface_flux_jump",
            "interface_value_trace_error",
            "interface_flux_trace_error",
            "darcy_residual",
        ]

        method = run.baseline_model
        run_id = f"{experiment}_darcy_{method}_seed{run.seed}_r{run.resolution}"
        run_dir = (
            output_root
            / "darcy"
            / f"resolution_{run.resolution}"
            / method
            / f"seed_{run.seed}"
        )
        config.setdefault("experiment", {}).update(
            {"name": run_id, "seed": run.seed, "output_dir": str(run_dir)}
        )
        if method == "two_scale_interface":
            config["warm_start"] = {
                "enabled": True,
                "run_dir": str(
                    output_root
                    / "darcy"
                    / f"resolution_{run.resolution}"
                    / "two_scale"
                    / f"seed_{run.seed}"
                ),
            }
        ablation = copy.deepcopy(config.get("resolved", {}).get("ablation", {}))
        ablation.update(
            {
                "family": "darcy_generalization",
                "cell": method,
                "method": method,
                "dataset": "darcy",
            }
        )
        config.setdefault("resolved", {}).update(
            {
                "dataset_config": "darcy",
                "dataset": "darcy",
                "study": experiment,
                "study_type": "darcy_generalization",
                "study_method": method,
                "ablation": ablation,
                "run_id": run_id,
            }
        )
        normalized = FactorialRun(
            run_id=run_id,
            config=config,
            config_hash=config_hash_for(config),
            seed=run.seed,
            mechanisms=dict(run.mechanisms),
            resolution=run.resolution,
            run_dir=run_dir,
            baseline_model=method,
        )
        by_seed[run.seed][method] = normalized

    runs: list[FactorialRun] = []
    for seed in seeds:
        runs.extend(by_seed[seed][method] for method in method_order)
    return runs


def run_darcy_generalization_study(
    *,
    root: str | Path,
    dataset: str = "darcy",
    experiment: str = "darcy_generalization",
    overrides: list[str] | None = None,
    limit_samples: int | None = None,
    epochs: int | None = None,
    output_dir: str | Path | None = None,
    device: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> list[FactorialRun]:
    """Run the frozen five-seed Darcy 256 generalization study."""
    if limit_samples is not None:
        raise ValueError(
            "darcy_generalization fixes m through matrix.n_train; "
            "do not use --limit-samples."
        )
    if epochs is not None:
        raise ValueError(
            "darcy_generalization uses method-specific fixed schedules; "
            "do not use --epochs."
        )
    study = compose_config(
        root=Path(root),
        dataset=dataset,
        model="l2l",
        experiment=experiment,
        overrides=overrides,
    )
    matrix = study.get("matrix", {})
    result_root = Path(
        output_dir
        or matrix.get("output_root")
        or study.get("experiment", {}).get("output_dir")
        or "paper_results/darcy_generalization"
    )
    runs = expand_darcy_generalization_runs(
        root=root,
        dataset=dataset,
        experiment=experiment,
        overrides=overrides,
        output_dir=output_dir,
    )
    if dry_run:
        for run in runs:
            print(_dry_run_line(run))
        return runs
    _validate_study_dataset_paths(runs)
    result_root = ensure_dir(result_root)
    records = _execute_runs(
        runs,
        result_root=result_root,
        limit_samples=None,
        epochs=None,
        device=device,
        overwrite=overwrite,
    )
    write_stage_cost_table(records, result_root)
    return runs


def expand_mechanism_ablation_runs(
    *,
    root: str | Path,
    dataset: str = "poisson",
    experiment: str = "mechanism_ablation",
    overrides: list[str] | None = None,
    output_dir: str | Path | None = None,
) -> list[FactorialRun]:
    """Expand the paired representation-versus-in-loop mechanism study."""
    root_path = Path(root)
    if dataset != "poisson":
        raise ValueError("The mechanism ablation is defined for Poisson 128.")
    study = compose_config(
        root=root_path,
        dataset=dataset,
        model="l2l",
        experiment=experiment,
        overrides=overrides,
    )
    matrix = study.get("matrix", {})
    expected_order = [
        "plain_l2l",
        "plain_l2l_interface",
        "two_scale",
        "two_scale_interface",
        "overlap_l2l",
    ]
    method_order = [str(value) for value in matrix.get("mechanism_methods", [])]
    if method_order != expected_order:
        raise ValueError(
            "matrix.mechanism_methods must preserve the dependency and manuscript "
            f"order {expected_order}, got {method_order}."
        )
    seeds = [int(seed) for seed in matrix.get("seeds", [])]
    if seeds != [0, 1, 2, 3, 4]:
        raise ValueError(
            "The mechanism ablation requires the paired seeds [0, 1, 2, 3, 4]."
        )
    resolutions = {int(value) for value in matrix.get("resolutions", {})}
    if resolutions != {128}:
        raise ValueError("The mechanism ablation must contain only resolution 128.")

    core_runs = expand_resolution_sweep_runs(
        root=root_path,
        dataset=dataset,
        experiment=experiment,
        overrides=overrides,
        output_dir=output_dir,
    )
    output_root = Path(
        output_dir
        or matrix.get("output_root")
        or study.get("experiment", {}).get("output_dir")
        or "paper_results/mechanism_ablation"
    )
    retained_core = {
        "plain_l2l",
        "overlap_l2l",
        "two_scale",
        "two_scale_interface",
    }
    core_by_seed: dict[int, dict[str, FactorialRun]] = {seed: {} for seed in seeds}
    for run in core_runs:
        if run.baseline_model not in retained_core:
            continue
        config = copy.deepcopy(run.config)
        config.setdefault("resolved", {}).update(
            {
                "study": experiment,
                "study_type": "mechanism_ablation",
                "study_method": run.baseline_model,
            }
        )
        config["resolved"].setdefault("ablation", {}).update(
            {
                "family": "mechanism_ablation",
                "cell": run.baseline_model,
                "method": run.baseline_model,
            }
        )
        normalized = FactorialRun(
            run_id=run.run_id,
            config=config,
            config_hash=config_hash_for(config),
            seed=run.seed,
            mechanisms=dict(run.mechanisms),
            resolution=run.resolution,
            run_dir=run.run_dir,
            baseline_model=run.baseline_model,
        )
        core_by_seed[run.seed][run.baseline_model] = normalized

    interface = matrix.get("interface_fine_tune", {})
    if not isinstance(interface, Mapping):
        raise TypeError("matrix.interface_fine_tune must be a mapping.")
    runs: list[FactorialRun] = []
    for seed in seeds:
        seed_runs = core_by_seed[seed]
        plain = seed_runs["plain_l2l"]
        config = copy.deepcopy(plain.config)
        mechanisms = {
            "two_scale": False,
            "coupling": False,
            "in_loop_loss": True,
        }
        config["mechanisms"] = mechanisms
        config.setdefault("training", {}).update(
            {
                "epochs": int(interface.get("epochs", 50)),
                "lr": float(interface.get("lr", 1.0e-4)),
                "weight_decay": float(interface.get("weight_decay", 0.0)),
                "scheduler": str(interface.get("scheduler", "none")),
            }
        )
        config["training"]["latent_loss"] = copy.deepcopy(
            plain.config["training"]["latent_loss"]
        )
        config["warm_start"] = {
            "enabled": True,
            "run_dir": str(plain.run_dir),
        }
        config.setdefault("loss", {}).update(
            {
                "reconstruction": "relative_l2",
                "interface_target": "truth",
                "active_terms": [
                    "recon",
                    "interface_value",
                    "interface_flux",
                ],
                "pde": "poisson",
                "weights": {
                    "recon": 1.0,
                    "interface_value": 0.0,
                    "interface_flux": 0.0,
                    "spectral": 0.0,
                    "pde_residual": 0.0,
                },
                "warmup": {
                    "reconstruction_epochs": int(
                        interface.get("reconstruction_epochs", 5)
                    ),
                    "ramp_epochs": int(interface.get("ramp_epochs", 10)),
                },
                "weight_calibration": {
                    "enabled": True,
                    "target_ratios": {
                        "interface_value": float(
                            interface.get("value_target_ratio", 0.2)
                        ),
                        "interface_flux": float(
                            interface.get("flux_target_ratio", 0.2)
                        ),
                    },
                    "batch_size": int(interface.get("calibration_batch_size", 16)),
                    "max_samples": int(interface.get("calibration_samples", 128)),
                    "minimum_weight": 1.0e-12,
                    "maximum_weight": 1.0e6,
                },
            }
        )
        run_id = f"{experiment}_poisson_plain_l2l_interface_seed{seed}_r128"
        run_dir = (
            output_root
            / "poisson"
            / "resolution_128"
            / "plain_l2l_interface"
            / f"seed_{seed}"
        )
        config.setdefault("experiment", {}).update(
            {"name": run_id, "seed": seed, "output_dir": str(run_dir)}
        )
        config.setdefault("resolved", {}).update(
            {
                "multirun": True,
                "study": experiment,
                "study_type": "mechanism_ablation",
                "study_method": "plain_l2l_interface",
                "dataset": "poisson",
                "n_train": int(matrix.get("n_train", 8000)),
                "run_id": run_id,
                "ablation": {
                    "family": "mechanism_ablation",
                    "cell": "plain_l2l_interface",
                    "method": "plain_l2l_interface",
                    "n_train": int(matrix.get("n_train", 8000)),
                    "patch_size": 32,
                    "stride": 32,
                    "coarse_factor": 4,
                    "pca_solver": str(config["pca"].get("solver", "randomized")),
                    "common_seam_stride": 32,
                    "split_policy": "regenerate_from_run_seed",
                },
            }
        )
        interface_run = FactorialRun(
            run_id=run_id,
            config=config,
            config_hash=config_hash_for(config),
            seed=seed,
            mechanisms=mechanisms,
            resolution=128,
            run_dir=run_dir,
            baseline_model="plain_l2l_interface",
        )
        seed_runs["plain_l2l_interface"] = interface_run
        runs.extend(seed_runs[method] for method in method_order)
    return runs


def run_mechanism_ablation(
    *,
    root: str | Path,
    dataset: str = "poisson",
    experiment: str = "mechanism_ablation",
    overrides: list[str] | None = None,
    limit_samples: int | None = None,
    epochs: int | None = None,
    output_dir: str | Path | None = None,
    device: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> list[FactorialRun]:
    """Run the frozen five-seed mechanism ablation."""
    if limit_samples is not None:
        raise ValueError(
            "mechanism_ablation fixes m through matrix.n_train; "
            "do not use --limit-samples."
        )
    if epochs is not None:
        raise ValueError(
            "mechanism_ablation uses method-specific fixed schedules; "
            "do not use --epochs."
        )
    study = compose_config(
        root=Path(root),
        dataset=dataset,
        model="l2l",
        experiment=experiment,
        overrides=overrides,
    )
    matrix = study.get("matrix", {})
    result_root = Path(
        output_dir
        or matrix.get("output_root")
        or study.get("experiment", {}).get("output_dir")
        or "paper_results/mechanism_ablation"
    )
    runs = expand_mechanism_ablation_runs(
        root=root,
        dataset=dataset,
        experiment=experiment,
        overrides=overrides,
        output_dir=output_dir,
    )
    if dry_run:
        for run in runs:
            print(_dry_run_line(run))
        return runs
    _validate_study_dataset_paths(runs)
    result_root = ensure_dir(result_root)
    records = _execute_runs(
        runs,
        result_root=result_root,
        limit_samples=None,
        epochs=None,
        device=device,
        overwrite=overwrite,
    )
    write_stage_cost_table(records, result_root)
    return runs


def run_baselines(
    *,
    root: str | Path,
    dataset: str = "poisson",
    experiment: str = "baselines",
    overrides: list[str] | None = None,
    limit_samples: int | None = None,
    epochs: int | None = None,
    output_dir: str | Path | None = None,
    device: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> list[FactorialRun]:
    """Expand and optionally execute the baseline matrix."""
    runs = expand_baseline_runs(
        root=root,
        dataset=dataset,
        experiment=experiment,
        overrides=overrides,
        output_dir=output_dir,
    )
    if dry_run:
        for run in runs:
            print(_dry_run_line(run))
        return runs
    result_root = Path(output_dir or "results/baselines")
    result_root = ensure_dir(result_root)
    _execute_runs(
        runs,
        result_root=result_root,
        limit_samples=limit_samples,
        epochs=epochs,
        device=device,
        overwrite=overwrite,
    )
    return runs


def expand_latent_loss_ablation_runs(
    *,
    root: str | Path,
    experiment: str = "two_scale_latent_loss_ablation",
    overrides: list[str] | None = None,
    output_dir: str | Path | None = None,
) -> list[FactorialRun]:
    """Expand the cross-dataset two-scale latent-objective ablation."""
    root_path = Path(root)
    study = compose_config(
        root=root_path,
        dataset="poisson",
        model="l2l",
        experiment=experiment,
        overrides=overrides,
    )
    matrix = study.get("matrix", {})
    datasets = matrix.get("datasets", {})
    resolutions = matrix.get("resolutions", {})
    seeds = [int(seed) for seed in matrix.get("seeds", [0])]
    modes = [str(mode) for mode in matrix.get("latent_loss_modes", [])]
    n_train = int(matrix.get("n_train", 8000))
    pca_solver = str(matrix.get("pca_solver", "randomized"))
    output_root = Path(
        output_dir
        or matrix.get("output_root")
        or study.get("experiment", {}).get("output_dir")
        or "paper_results/two_scale_latent_loss_ablation"
    )
    if not isinstance(datasets, Mapping) or not datasets:
        raise ValueError(f"Experiment {experiment!r} must define matrix.datasets.")
    if not isinstance(resolutions, Mapping) or not resolutions:
        raise ValueError(f"Experiment {experiment!r} must define matrix.resolutions.")
    if not modes:
        raise ValueError(f"Experiment {experiment!r} must define matrix.latent_loss_modes.")

    allowed_modes = {"mse", "block_balanced", "score_normalized"}
    unknown_modes = sorted(set(modes) - allowed_modes)
    if unknown_modes:
        raise ValueError(f"Unsupported latent loss modes in matrix: {unknown_modes}.")

    runs: list[FactorialRun] = []
    for dataset_name, dataset_settings in datasets.items():
        if not isinstance(dataset_settings, Mapping):
            raise TypeError(f"Dataset matrix entry {dataset_name!r} must be a mapping.")
        processed_paths = dataset_settings.get("processed_paths", {})
        if not isinstance(processed_paths, Mapping):
            raise TypeError(f"{dataset_name}.processed_paths must be a mapping.")
        for raw_resolution, resolution_settings in resolutions.items():
            resolution = int(raw_resolution)
            if not isinstance(resolution_settings, Mapping):
                raise TypeError(f"Resolution matrix entry {raw_resolution!r} must be a mapping.")
            coarse_factor = int(resolution_settings.get("coarse_factor", 4))
            _assert_coarse_factor_legal(resolution, coarse_factor, n_train)
            processed_path = _mapping_value(processed_paths, resolution)
            if processed_path is None:
                raise ValueError(
                    f"Dataset {dataset_name!r} is missing a processed path for resolution "
                    f"{resolution}."
                )
            patch_size = int(resolution_settings["patch_size"])
            stride = int(resolution_settings.get("stride", patch_size))
            for seed in seeds:
                for mode in modes:
                    config = compose_config(
                        root=root_path,
                        dataset=str(dataset_name),
                        model="l2l",
                        experiment=experiment,
                        overrides=overrides,
                    )
                    config.pop("matrix", None)
                    config.setdefault("mechanisms", {}).update(
                        {"two_scale": True, "coupling": False, "in_loop_loss": False}
                    )
                    config.setdefault("dataset", {}).update(
                        {
                            "name": str(dataset_name),
                            "grid_size": resolution,
                            "processed_path": str(processed_path),
                        }
                    )
                    config.setdefault("patches", {}).update(
                        {
                            "patch_size": patch_size,
                            "stride": stride,
                            "assembly": "average",
                        }
                    )
                    config.setdefault("pca", {})["solver"] = pca_solver
                    config["pca"].setdefault("two_scale", {})[
                        "coarse_factor"
                    ] = coarse_factor
                    config.setdefault("training", {}).setdefault("latent_loss", {})[
                        "mode"
                    ] = mode
                    run_id = (
                        f"{experiment}_{dataset_name}_{mode}_seed{seed}_r{resolution}"
                    )
                    run_dir = (
                        output_root
                        / str(dataset_name)
                        / f"resolution_{resolution}"
                        / mode
                        / f"seed_{seed}"
                    )
                    config.setdefault("experiment", {}).update(
                        {
                            "name": run_id,
                            "seed": seed,
                            "output_dir": str(run_dir),
                        }
                    )
                    config.setdefault("resolved", {}).update(
                        {
                            "multirun": True,
                            "study": experiment,
                            "study_type": "latent_loss_ablation",
                            "study_method": mode,
                            "dataset": str(dataset_name),
                            "n_train": n_train,
                            "coarse_factor": coarse_factor,
                            "run_id": run_id,
                        }
                    )
                    mechanisms = {
                        key: bool(config["mechanisms"][key])
                        for key in MECHANISM_KEYS
                    }
                    runs.append(
                        FactorialRun(
                            run_id=run_id,
                            config=config,
                            config_hash=config_hash_for(config),
                            seed=seed,
                            mechanisms=mechanisms,
                            resolution=resolution,
                            run_dir=run_dir,
                            baseline_model=mode,
                        )
                    )
    return runs


def run_latent_loss_ablation(
    *,
    root: str | Path,
    experiment: str = "two_scale_latent_loss_ablation",
    overrides: list[str] | None = None,
    limit_samples: int | None = None,
    epochs: int | None = None,
    output_dir: str | Path | None = None,
    device: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> list[FactorialRun]:
    """Run the cross-dataset two-scale latent-objective ablation."""
    runs = expand_latent_loss_ablation_runs(
        root=root,
        experiment=experiment,
        overrides=overrides,
        output_dir=output_dir,
    )
    if dry_run:
        for run in runs:
            print(_dry_run_line(run))
        return runs
    _validate_study_dataset_paths(runs)
    result_root = ensure_dir(
        output_dir
        or runs[0].run_dir.parents[3]
        if runs
        else "paper_results/two_scale_latent_loss_ablation"
    )
    records = _execute_runs(
        runs,
        result_root=result_root,
        limit_samples=limit_samples,
        epochs=epochs,
        device=device,
        overwrite=overwrite,
    )
    write_stage_cost_table(records, result_root)
    return runs


def expand_coarse_representation_ablation_runs(
    *,
    root: str | Path,
    experiment: str = "coarse_representation_ablation",
    overrides: list[str] | None = None,
    output_dir: str | Path | None = None,
) -> list[FactorialRun]:
    """Expand the deduplicated Poisson coarse-representation ablation."""
    root_path = Path(root)
    study = compose_config(
        root=root_path,
        dataset="poisson",
        model="l2l",
        experiment=experiment,
        overrides=overrides,
    )
    matrix = study.get("matrix", {})
    resolutions = matrix.get("resolutions", {})
    processed_paths = matrix.get("processed_paths", {})
    seeds = [int(seed) for seed in matrix.get("seeds", [0])]
    exact_ranks = [int(rank) for rank in matrix.get("exact_coarse_ranks", [])]
    residual_variances = [
        float(value) for value in matrix.get("residual_variances", [])
    ]
    factors = matrix.get("coarse_factors", {})
    n_train = int(matrix.get("n_train", 8000))
    pca_solver = str(matrix.get("pca_solver", "randomized"))
    default_rank = int(matrix.get("default_exact_rank", 10))
    default_factor = int(matrix.get("default_coarse_factor", 4))
    default_residual_variance = float(
        matrix.get("default_residual_variance", 0.99)
    )
    adaptive = matrix.get("adaptive_reference", {})
    output_root = Path(
        output_dir
        or matrix.get("output_root")
        or study.get("experiment", {}).get("output_dir")
        or "paper_results/coarse_representation_ablation"
    )
    if not isinstance(resolutions, Mapping) or not resolutions:
        raise ValueError(f"Experiment {experiment!r} must define matrix.resolutions.")
    if not isinstance(processed_paths, Mapping):
        raise TypeError("matrix.processed_paths must be a mapping.")
    if not isinstance(factors, Mapping):
        raise TypeError("matrix.coarse_factors must be a mapping.")
    if default_rank not in exact_ranks:
        raise ValueError("default_exact_rank must be included in exact_coarse_ranks.")
    if default_residual_variance not in residual_variances:
        raise ValueError(
            "default_residual_variance must be included in residual_variances."
        )

    runs: list[FactorialRun] = []
    for raw_resolution, resolution_settings in resolutions.items():
        resolution = int(raw_resolution)
        if not isinstance(resolution_settings, Mapping):
            raise TypeError(f"Resolution matrix entry {raw_resolution!r} must be a mapping.")
        processed_path = _mapping_value(processed_paths, resolution)
        if processed_path is None:
            raise ValueError(f"Missing processed path for Poisson resolution {resolution}.")
        resolution_factors = _mapping_value(factors, resolution)
        if not isinstance(resolution_factors, list):
            raise TypeError(f"coarse_factors[{resolution}] must be a list.")
        resolution_factors = [int(value) for value in resolution_factors]
        if default_factor not in resolution_factors:
            raise ValueError(
                f"default_coarse_factor={default_factor} is missing at resolution "
                f"{resolution}."
            )
        for factor in resolution_factors:
            _assert_coarse_factor_legal(resolution, factor, n_train)

        cells = _coarse_ablation_cells(
            exact_ranks=exact_ranks,
            coarse_factors=resolution_factors,
            residual_variances=residual_variances,
            default_rank=default_rank,
            default_factor=default_factor,
            default_residual_variance=default_residual_variance,
            adaptive=adaptive,
        )
        for cell in cells:
            for seed in seeds:
                config = compose_config(
                    root=root_path,
                    dataset="poisson",
                    model="l2l",
                    experiment=experiment,
                    overrides=overrides,
                )
                config.pop("matrix", None)
                config.setdefault("mechanisms", {}).update(
                    {"two_scale": True, "coupling": False, "in_loop_loss": False}
                )
                config.setdefault("dataset", {}).update(
                    {
                        "name": "poisson",
                        "grid_size": resolution,
                        "processed_path": str(processed_path),
                    }
                )
                patch_size = int(resolution_settings["patch_size"])
                config.setdefault("patches", {}).update(
                    {
                        "patch_size": patch_size,
                        "stride": int(resolution_settings.get("stride", patch_size)),
                        "assembly": "average",
                    }
                )
                config.setdefault("pca", {}).update(
                    {
                        "solver": pca_solver,
                        "output_variance": float(cell["residual_variance"]),
                    }
                )
                config["pca"].setdefault("two_scale", {}).update(
                    {
                        "coarse_factor": int(cell["coarse_factor"]),
                        "coarse_components": int(cell["coarse_components"]),
                        "coarse_variance": float(cell["coarse_variance"]),
                    }
                )
                config.setdefault("training", {}).setdefault("latent_loss", {}).update(
                    {
                        "mode": "block_balanced",
                        "coarse_weight": 0.5,
                        "residual_weight": 0.5,
                    }
                )
                config.setdefault("evaluation", {})["pca_oracle"] = True
                cell_name = str(cell["name"])
                run_id = (
                    f"{experiment}_poisson_{cell_name}_seed{seed}_r{resolution}"
                )
                run_dir = (
                    output_root
                    / "poisson"
                    / f"resolution_{resolution}"
                    / cell_name
                    / f"seed_{seed}"
                )
                ablation = {
                    "family": str(cell["family"]),
                    "cell": cell_name,
                    "coarse_factor": int(cell["coarse_factor"]),
                    "coarse_components": int(cell["coarse_components"]),
                    "coarse_variance": float(cell["coarse_variance"]),
                    "residual_variance": float(cell["residual_variance"]),
                }
                config.setdefault("experiment", {}).update(
                    {"name": run_id, "seed": seed, "output_dir": str(run_dir)}
                )
                config.setdefault("resolved", {}).update(
                    {
                        "multirun": True,
                        "study": experiment,
                        "study_type": "coarse_representation_ablation",
                        "study_method": cell_name,
                        "dataset": "poisson",
                        "n_train": n_train,
                        "ablation": ablation,
                        "run_id": run_id,
                    }
                )
                mechanisms = {
                    key: bool(config["mechanisms"][key]) for key in MECHANISM_KEYS
                }
                runs.append(
                    FactorialRun(
                        run_id=run_id,
                        config=config,
                        config_hash=config_hash_for(config),
                        seed=seed,
                        mechanisms=mechanisms,
                        resolution=resolution,
                        run_dir=run_dir,
                        baseline_model=cell_name,
                    )
                )
    return runs


def run_coarse_representation_ablation(
    *,
    root: str | Path,
    experiment: str = "coarse_representation_ablation",
    overrides: list[str] | None = None,
    limit_samples: int | None = None,
    epochs: int | None = None,
    output_dir: str | Path | None = None,
    device: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> list[FactorialRun]:
    """Run the three-seed Poisson coarse-representation ablation."""
    runs = expand_coarse_representation_ablation_runs(
        root=root,
        experiment=experiment,
        overrides=overrides,
        output_dir=output_dir,
    )
    if dry_run:
        for run in runs:
            print(_dry_run_line(run))
        return runs
    _validate_study_dataset_paths(runs)
    result_root = ensure_dir(
        output_dir
        or runs[0].run_dir.parents[3]
        if runs
        else "paper_results/coarse_representation_ablation"
    )
    records = _execute_runs(
        runs,
        result_root=result_root,
        limit_samples=limit_samples,
        epochs=epochs,
        device=device,
        overwrite=overwrite,
    )
    write_stage_cost_table(records, result_root)
    return runs


def expand_sample_efficiency_runs(
    *,
    root: str | Path,
    experiment: str = "sample_efficiency",
    overrides: list[str] | None = None,
    output_dir: str | Path | None = None,
) -> list[FactorialRun]:
    """Expand the paired, nested Poisson-128 sample-efficiency matrix."""
    root_path = Path(root)
    study = compose_config(
        root=root_path,
        dataset="poisson",
        model="l2l",
        experiment=experiment,
        overrides=overrides,
    )
    matrix = study.get("matrix", {})
    resolution = int(matrix.get("resolution", 128))
    sample_sizes = [int(value) for value in matrix.get("sample_sizes", [])]
    seeds = [int(value) for value in matrix.get("seeds", [0])]
    methods = [str(value) for value in matrix.get("methods", [])]
    expected_methods = [
        "plain_l2l",
        "overlap_l2l",
        "two_scale",
        "two_scale_interface",
    ]
    if methods != expected_methods:
        raise ValueError(
            "matrix.methods must preserve the dependency order "
            f"{expected_methods}, got {methods}."
        )
    if not sample_sizes or sample_sizes != sorted(set(sample_sizes)):
        raise ValueError("matrix.sample_sizes must be a non-empty increasing list.")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("matrix.seeds must contain unique values.")

    processed_path = str(
        matrix.get("processed_path", f"data/processed/poisson_{resolution}.npz")
    )
    output_root = Path(
        output_dir
        or matrix.get("output_root")
        or study.get("experiment", {}).get("output_dir")
        or "paper_results/sample_efficiency"
    )
    pca_solver = str(matrix.get("pca_solver", "randomized"))
    patch_size = int(matrix.get("patch_size", 32))
    plain_stride = int(matrix.get("plain_stride", patch_size))
    overlap_stride = int(matrix.get("overlap_stride", patch_size // 2))
    coarse_factor = int(matrix.get("coarse_factor", 4))
    coarse_components = int(matrix.get("coarse_components", 10))
    residual_variance = float(matrix.get("residual_variance", 0.995))
    optimization = matrix.get("optimization", {})
    interface = matrix.get("interface_fine_tune", {})
    retention = matrix.get("prediction_retention", {})
    if not isinstance(optimization, Mapping):
        raise TypeError("matrix.optimization must be a mapping.")
    if not isinstance(interface, Mapping):
        raise TypeError("matrix.interface_fine_tune must be a mapping.")
    if not isinstance(retention, Mapping):
        raise TypeError("matrix.prediction_retention must be a mapping.")
    retained_seeds = {int(value) for value in retention.get("seeds", [0])}
    retained_sample_sizes = {
        int(value)
        for value in retention.get(
            "sample_sizes",
            [sample_sizes[0], sample_sizes[-1]],
        )
    }
    if not retained_seeds.issubset(set(seeds)):
        raise ValueError("Prediction-retention seeds must be part of matrix.seeds.")
    if not retained_sample_sizes.issubset(set(sample_sizes)):
        raise ValueError(
            "Prediction-retention sample sizes must be part of matrix.sample_sizes."
        )
    for sample_size in sample_sizes:
        _assert_coarse_factor_legal(resolution, coarse_factor, sample_size)

    runs: list[FactorialRun] = []
    for sample_size in sample_sizes:
        budget = _sample_efficiency_budget(sample_size, optimization)
        for seed in seeds:
            warm_start_dir = (
                output_root
                / "poisson"
                / f"resolution_{resolution}"
                / f"train_{sample_size}"
                / "two_scale"
                / f"seed_{seed}"
            )
            for method in methods:
                model_name = "l2l_overlap" if method == "overlap_l2l" else "l2l"
                config = compose_config(
                    root=root_path,
                    dataset="poisson",
                    model=model_name,
                    experiment=experiment,
                    overrides=overrides,
                )
                config.pop("matrix", None)
                is_two_scale = method in {"two_scale", "two_scale_interface"}
                is_interface = method == "two_scale_interface"
                mechanisms = {
                    "two_scale": is_two_scale,
                    "coupling": False,
                    "in_loop_loss": is_interface,
                }
                config.setdefault("mechanisms", {}).update(mechanisms)
                config.setdefault("dataset", {}).update(
                    {
                        "name": "poisson",
                        "grid_size": resolution,
                        "processed_path": processed_path,
                        "train_samples": sample_size,
                        "save_split_indices": True,
                    }
                )
                config.setdefault("patches", {}).update(
                    {
                        "patch_size": patch_size,
                        "stride": overlap_stride if method == "overlap_l2l" else plain_stride,
                        "assembly": "hann_safe" if method == "overlap_l2l" else "average",
                    }
                )
                config.setdefault("pca", {}).update(
                    {
                        "solver": pca_solver,
                        "output_variance": residual_variance if is_two_scale else 0.99,
                    }
                )
                config["pca"].setdefault("two_scale", {}).update(
                    {
                        "coarse_factor": coarse_factor,
                        "coarse_components": coarse_components,
                        "coarse_variance": 1.0,
                    }
                )
                config.setdefault("training", {}).update(
                    {
                        "batch_size": int(budget["batch_size"]),
                        "epochs": int(
                            budget["in_loop_epochs"]
                            if is_interface
                            else budget["latent_epochs"]
                        ),
                        "patience": int(budget["scheduler_patience_epochs"]),
                    }
                )
                config["training"].setdefault("latent_loss", {}).update(
                    {
                        "mode": "block_balanced" if is_two_scale else "mse",
                        "coarse_weight": 0.5,
                        "residual_weight": 0.5,
                    }
                )
                if is_interface:
                    config["training"].update(
                        {
                            "lr": float(interface.get("lr", 1.0e-4)),
                            "weight_decay": float(interface.get("weight_decay", 0.0)),
                            "scheduler": "none",
                        }
                    )
                    config["warm_start"] = {
                        "enabled": True,
                        "run_dir": str(warm_start_dir),
                    }
                    config.setdefault("loss", {}).update(
                        {
                            "reconstruction": "relative_l2",
                            "interface_target": "truth",
                            "active_terms": [
                                "recon",
                                "interface_value",
                                "interface_flux",
                            ],
                            "pde": "poisson",
                            "weights": {
                                "recon": 1.0,
                                "interface_value": 0.0,
                                "interface_flux": 0.0,
                                "spectral": 0.0,
                                "pde_residual": 0.0,
                            },
                            "warmup": {
                                "reconstruction_epochs": int(
                                    budget["reconstruction_warmup_epochs"]
                                ),
                                "ramp_epochs": int(budget["auxiliary_ramp_epochs"]),
                            },
                            "weight_calibration": {
                                "enabled": True,
                                "target_ratios": {
                                    "interface_value": float(
                                        interface.get("value_target_ratio", 0.2)
                                    ),
                                    "interface_flux": float(
                                        interface.get("flux_target_ratio", 0.2)
                                    ),
                                },
                                "batch_size": int(
                                    interface.get("calibration_batch_size", 16)
                                ),
                                "max_samples": int(
                                    interface.get("calibration_samples", 128)
                                ),
                                "minimum_weight": 1.0e-12,
                                "maximum_weight": 1.0e6,
                            },
                        }
                    )
                config.setdefault("evaluation", {}).update(
                    {
                        "pca_oracle": True,
                        "validation_metrics": True,
                        "retain_predictions": (
                            seed in retained_seeds
                            and sample_size in retained_sample_sizes
                        ),
                        "metrics": [
                            "mse",
                            "mae",
                            "mre",
                            "ssim",
                            "relative_spectrum_error",
                            "interface_jump",
                            "interface_flux_jump",
                            "interface_value_trace_error",
                            "interface_flux_trace_error",
                            "poisson_residual",
                        ],
                    }
                )

                run_id = (
                    f"{experiment}_poisson_m{sample_size}_{method}_"
                    f"seed{seed}_r{resolution}"
                )
                run_dir = (
                    output_root
                    / "poisson"
                    / f"resolution_{resolution}"
                    / f"train_{sample_size}"
                    / method
                    / f"seed_{seed}"
                )
                ablation = {
                    "family": "sample_efficiency",
                    "cell": method,
                    "sample_size": sample_size,
                    "nested_subset": True,
                    "target_optimizer_steps": int(
                        budget[
                            "in_loop_target_steps"
                            if is_interface
                            else "latent_target_steps"
                        ]
                    ),
                    "planned_optimizer_steps": int(
                        budget[
                            "in_loop_planned_steps"
                            if is_interface
                            else "latent_planned_steps"
                        ]
                    ),
                }
                config.setdefault("experiment", {}).update(
                    {"name": run_id, "seed": seed, "output_dir": str(run_dir)}
                )
                config.setdefault("resolved", {}).update(
                    {
                        "multirun": True,
                        "study": experiment,
                        "study_type": "sample_efficiency",
                        "study_method": method,
                        "dataset": "poisson",
                        "n_train": sample_size,
                        "coarse_factor": coarse_factor,
                        "optimization_budget": budget,
                        "ablation": ablation,
                        "run_id": run_id,
                    }
                )
                runs.append(
                    FactorialRun(
                        run_id=run_id,
                        config=config,
                        config_hash=config_hash_for(config),
                        seed=seed,
                        mechanisms=mechanisms,
                        resolution=resolution,
                        run_dir=run_dir,
                        baseline_model=method,
                    )
                )
    return runs


def run_sample_efficiency(
    *,
    root: str | Path,
    experiment: str = "sample_efficiency",
    overrides: list[str] | None = None,
    limit_samples: int | None = None,
    epochs: int | None = None,
    output_dir: str | Path | None = None,
    device: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> list[FactorialRun]:
    """Run the nested three-seed Poisson sample-efficiency study."""
    if limit_samples is not None:
        raise ValueError(
            "sample_efficiency controls training size through "
            "matrix.sample_sizes; do not use --limit-samples."
        )
    if epochs is not None:
        raise ValueError(
            "sample_efficiency uses a fixed optimizer-step budget derived from "
            "matrix.optimization; do not use --epochs."
        )
    runs = expand_sample_efficiency_runs(
        root=root,
        experiment=experiment,
        overrides=overrides,
        output_dir=output_dir,
    )
    if dry_run:
        for run in runs:
            print(_dry_run_line(run))
        return runs
    _validate_study_dataset_paths(runs)
    result_root = ensure_dir(
        output_dir
        or (
            runs[0].run_dir.parents[4]
            if runs
            else "paper_results/sample_efficiency"
        )
    )
    records = _execute_runs(
        runs,
        result_root=result_root,
        limit_samples=None,
        epochs=epochs,
        device=device,
        overwrite=overwrite,
    )
    write_stage_cost_table(records, result_root)
    return runs


def _sample_efficiency_budget(
    sample_size: int,
    optimization: Mapping[str, Any],
) -> dict[str, int]:
    """Translate the reference epochs into an approximately fixed step budget."""
    batch_size = int(optimization.get("batch_size", 32))
    reference_samples = int(optimization.get("reference_samples", 8000))
    if batch_size <= 0 or reference_samples <= 0:
        raise ValueError("Sample-efficiency batch/reference sizes must be positive.")
    steps_per_epoch = (sample_size + batch_size - 1) // batch_size
    reference_steps_per_epoch = (
        reference_samples + batch_size - 1
    ) // batch_size

    def scaled_epochs(reference_epochs: int) -> int:
        target_steps = reference_epochs * reference_steps_per_epoch
        return max(1, (target_steps + steps_per_epoch - 1) // steps_per_epoch)

    latent_reference_epochs = int(
        optimization.get("latent_reference_epochs", 500)
    )
    in_loop_reference_epochs = int(
        optimization.get("in_loop_reference_epochs", 50)
    )
    latent_epochs = scaled_epochs(latent_reference_epochs)
    in_loop_epochs = scaled_epochs(in_loop_reference_epochs)
    scheduler_patience = scaled_epochs(
        int(optimization.get("scheduler_patience_reference_epochs", 30))
    )
    reconstruction_warmup = scaled_epochs(
        int(optimization.get("reconstruction_warmup_reference_epochs", 5))
    )
    auxiliary_ramp = scaled_epochs(
        int(optimization.get("auxiliary_ramp_reference_epochs", 10))
    )
    return {
        "batch_size": batch_size,
        "reference_samples": reference_samples,
        "steps_per_epoch": steps_per_epoch,
        "reference_steps_per_epoch": reference_steps_per_epoch,
        "latent_target_steps": latent_reference_epochs * reference_steps_per_epoch,
        "latent_epochs": latent_epochs,
        "latent_planned_steps": latent_epochs * steps_per_epoch,
        "in_loop_target_steps": in_loop_reference_epochs * reference_steps_per_epoch,
        "in_loop_epochs": in_loop_epochs,
        "in_loop_planned_steps": in_loop_epochs * steps_per_epoch,
        "scheduler_patience_epochs": scheduler_patience,
        "reconstruction_warmup_epochs": reconstruction_warmup,
        "auxiliary_ramp_epochs": auxiliary_ramp,
    }


def expand_svd_solver_ablation_runs(
    *,
    root: str | Path,
    experiment: str = "svd_solver_ablation",
    overrides: list[str] | None = None,
    output_dir: str | Path | None = None,
) -> list[FactorialRun]:
    """Expand the paired Poisson-128 full/randomized SVD comparison."""
    root_path = Path(root)
    study = compose_config(
        root=root_path,
        dataset="poisson",
        model="l2l",
        experiment=experiment,
        overrides=overrides,
    )
    matrix = study.get("matrix", {})
    resolution = int(matrix.get("resolution", 128))
    processed_path = str(
        matrix.get("processed_path", f"data/processed/poisson_{resolution}.npz")
    )
    train_samples = int(matrix.get("train_samples", 8000))
    seeds = [int(value) for value in matrix.get("seeds", [0, 1, 2, 3, 4])]
    raw_cells = matrix.get("cells", [])
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("matrix.seeds must contain unique values.")
    if not isinstance(raw_cells, list) or not all(
        isinstance(cell, Mapping) for cell in raw_cells
    ):
        raise TypeError("matrix.cells must be a list of solver-cell mappings.")
    cells = [dict(cell) for cell in raw_cells]
    expected_names = [
        "plain_full",
        "plain_randomized",
        "overlap_full",
        "overlap_randomized",
        "two_scale_full",
        "two_scale_hybrid",
        "two_scale_randomized",
    ]
    names = [str(cell.get("name", "")) for cell in cells]
    if names != expected_names:
        raise ValueError(
            "matrix.cells must preserve the paired solver order "
            f"{expected_names}, got {names}."
        )

    output_root = Path(
        output_dir
        or matrix.get("output_root")
        or study.get("experiment", {}).get("output_dir")
        or "paper_results/svd_solver_ablation"
    )
    patch_size = int(matrix.get("patch_size", 32))
    plain_stride = int(matrix.get("plain_stride", patch_size))
    overlap_stride = int(matrix.get("overlap_stride", patch_size // 2))
    coarse_factor = int(matrix.get("coarse_factor", 4))
    coarse_components = int(matrix.get("coarse_components", 10))
    input_variance = float(matrix.get("input_variance", 0.99))
    local_output_variance = float(matrix.get("local_output_variance", 0.99))
    residual_variance = float(matrix.get("residual_variance", 0.995))
    training = matrix.get("training", {})
    if not isinstance(training, Mapping):
        raise TypeError("matrix.training must be a mapping.")
    randomized = matrix.get("randomized", {})
    if not isinstance(randomized, Mapping):
        raise TypeError("matrix.randomized must be a mapping.")
    equivalence_gates = matrix.get("equivalence_gates", {})
    if not isinstance(equivalence_gates, Mapping):
        raise TypeError("matrix.equivalence_gates must be a mapping.")
    oversampling = int(randomized.get("oversampling", 20))
    n_iter = int(randomized.get("n_iter", 4))
    if oversampling <= 0 or n_iter < 0:
        raise ValueError("Randomized oversampling must be positive and n_iter nonnegative.")
    _assert_coarse_factor_legal(resolution, coarse_factor, train_samples)

    retention = matrix.get("prediction_retention", {})
    if not isinstance(retention, Mapping):
        raise TypeError("matrix.prediction_retention must be a mapping.")
    retain_predictions = bool(retention.get("enabled", False))
    runs: list[FactorialRun] = []
    for seed in seeds:
        for cell in cells:
            cell_name = str(cell["name"])
            method = str(cell.get("method", ""))
            solver_class = str(cell.get("solver_class", ""))
            local_solver = str(cell.get("local_solver", ""))
            coarse_solver = str(cell.get("coarse_solver", "none"))
            if method not in {"plain_l2l", "overlap_l2l", "two_scale"}:
                raise ValueError(f"Unsupported SVD-ablation method {method!r}.")
            if local_solver not in {"full", "randomized"}:
                raise ValueError(f"Unsupported local solver {local_solver!r}.")
            is_two_scale = method == "two_scale"
            if is_two_scale and coarse_solver not in {"full", "randomized"}:
                raise ValueError(f"Unsupported coarse solver {coarse_solver!r}.")
            if not is_two_scale and coarse_solver != "none":
                raise ValueError(
                    f"Non-two-scale cell {cell_name!r} must use coarse_solver=none."
                )

            model_name = "l2l_overlap" if method == "overlap_l2l" else "l2l"
            config = compose_config(
                root=root_path,
                dataset="poisson",
                model=model_name,
                experiment=experiment,
                overrides=overrides,
            )
            config.pop("matrix", None)
            mechanisms = {
                "two_scale": is_two_scale,
                "coupling": False,
                "in_loop_loss": False,
            }
            config.setdefault("mechanisms", {}).update(mechanisms)
            config.setdefault("dataset", {}).update(
                {
                    "name": "poisson",
                    "grid_size": resolution,
                    "processed_path": processed_path,
                    "train_samples": train_samples,
                    "save_split_indices": True,
                }
            )
            config.setdefault("patches", {}).update(
                {
                    "patch_size": patch_size,
                    "stride": overlap_stride if method == "overlap_l2l" else plain_stride,
                    "assembly": "hann_safe" if method == "overlap_l2l" else "average",
                }
            )
            config.setdefault("pca", {}).update(
                {
                    "solver": local_solver,
                    "input_variance": input_variance,
                    "output_variance": (
                        residual_variance if is_two_scale else local_output_variance
                    ),
                    "randomized": {
                        "oversampling": oversampling,
                        "n_iter": n_iter,
                    },
                }
            )
            config["pca"].setdefault("two_scale", {}).update(
                {
                    "coarse_factor": coarse_factor,
                    "coarse_components": coarse_components,
                    "coarse_variance": 1.0,
                    "coarse_solver": (
                        coarse_solver if is_two_scale else "randomized"
                    ),
                }
            )
            config["training"].setdefault("latent_loss", {}).update(
                {
                    "mode": "block_balanced" if is_two_scale else "mse",
                    "coarse_weight": 0.5,
                    "residual_weight": 0.5,
                }
            )
            config["training"].update(
                {
                    "batch_size": int(training.get("batch_size", 32)),
                    "epochs": int(training.get("epochs", 500)),
                    "lr": float(training.get("lr", 1.0e-3)),
                    "weight_decay": float(training.get("weight_decay", 1.0e-4)),
                    "scheduler": str(training.get("scheduler", "reduce_on_plateau")),
                    "patience": int(training.get("patience", 30)),
                }
            )
            config.setdefault("evaluation", {}).update(
                {
                    "pca_oracle": True,
                    "validation_metrics": True,
                    "retain_predictions": retain_predictions,
                    "metrics": [
                        "mse",
                        "mae",
                        "mre",
                        "ssim",
                        "relative_spectrum_error",
                        "interface_jump",
                        "interface_flux_jump",
                        "interface_value_trace_error",
                        "interface_flux_trace_error",
                        "poisson_residual",
                    ],
                }
            )

            run_id = f"{experiment}_{cell_name}_seed{seed}_r{resolution}"
            run_dir = (
                output_root
                / "poisson"
                / f"resolution_{resolution}"
                / cell_name
                / f"seed_{seed}"
            )
            ablation = {
                "family": "svd_solver",
                "cell": cell_name,
                "method": method,
                "solver_class": solver_class,
                "local_solver": local_solver,
                "coarse_solver": coarse_solver,
                "randomized_oversampling": oversampling,
                "randomized_n_iter": n_iter,
            }
            config.setdefault("experiment", {}).update(
                {"name": run_id, "seed": seed, "output_dir": str(run_dir)}
            )
            config.setdefault("resolved", {}).update(
                {
                    "multirun": True,
                    "study": experiment,
                    "study_type": "svd_solver_ablation",
                    "study_method": method,
                    "ablation": ablation,
                    "equivalence_gates": dict(equivalence_gates),
                    "run_id": run_id,
                }
            )
            runs.append(
                FactorialRun(
                    run_id=run_id,
                    config=config,
                    config_hash=config_hash_for(config),
                    seed=seed,
                    mechanisms=mechanisms,
                    resolution=resolution,
                    run_dir=run_dir,
                    baseline_model=cell_name,
                )
            )
    return runs


def run_svd_solver_ablation(
    *,
    root: str | Path,
    experiment: str = "svd_solver_ablation",
    overrides: list[str] | None = None,
    limit_samples: int | None = None,
    epochs: int | None = None,
    output_dir: str | Path | None = None,
    device: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> list[FactorialRun]:
    """Run the five-seed, seven-cell SVD solver equivalence study."""
    if limit_samples is not None:
        raise ValueError(
            "svd_solver_ablation fixes m through matrix.train_samples; "
            "do not use --limit-samples."
        )
    if epochs is not None:
        raise ValueError(
            "svd_solver_ablation fixes the training schedule; do not use --epochs."
        )
    runs = expand_svd_solver_ablation_runs(
        root=root,
        experiment=experiment,
        overrides=overrides,
        output_dir=output_dir,
    )
    if dry_run:
        for run in runs:
            print(_dry_run_line(run))
        return runs
    _validate_study_dataset_paths(runs)
    result_root = ensure_dir(
        output_dir
        or (
            runs[0].run_dir.parents[3]
            if runs
            else "paper_results/svd_solver_ablation"
        )
    )
    records = _execute_runs(
        runs,
        result_root=result_root,
        limit_samples=None,
        epochs=None,
        device=device,
        overwrite=overwrite,
    )
    write_stage_cost_table(records, result_root)
    return runs


def plan_physics_loss_ablation_runs(
    *,
    root: str | Path,
    experiment: str = "physics_loss_ablation",
    overrides: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return the planned 41 reference/screen/confirmation cells."""
    study = compose_config(
        root=Path(root),
        dataset="poisson",
        model="l2l",
        experiment=experiment,
        overrides=overrides,
    )
    matrix = study.get("matrix", {})
    resolutions = [int(value) for value in matrix.get("resolutions", [128, 256])]
    seeds = [int(value) for value in matrix.get("seeds", [0, 1, 2])]
    screen_resolution = int(matrix.get("screen_resolution", 128))
    screen_seed = int(matrix.get("screen_seed", 0))
    levels = matrix.get("weight_levels", {})
    value_levels = list(levels.get("interface_value", {}).keys())
    flux_levels = list(levels.get("interface_flux", {}).keys())
    spectral_levels = [
        name
        for name, value in levels.get("spectral", {}).items()
        if str(name).lower() != "off" and float(value) != 0.0
    ]
    pde_levels = [
        name
        for name, value in levels.get("pde_residual", {}).items()
        if str(name).lower() != "off" and float(value) != 0.0
    ]

    plan: list[dict[str, Any]] = []
    for resolution in resolutions:
        for seed in seeds:
            plan.append(
                {
                    "phase": "reference",
                    "resolution": resolution,
                    "seed": seed,
                    "cell": "a0_latent_baseline",
                }
            )
    plan.append(
        {
            "phase": "screen",
            "resolution": screen_resolution,
            "seed": screen_seed,
            "cell": "a1_reconstruction",
        }
    )
    for stage, names in (
        ("a2_value", value_levels),
        ("a3_value_flux", flux_levels),
        ("a4_value_flux_spectral", spectral_levels),
        ("a5_full_pde", pde_levels),
    ):
        for name in names:
            plan.append(
                {
                    "phase": "screen",
                    "resolution": screen_resolution,
                    "seed": screen_seed,
                    "cell": f"{stage}_{name}",
                }
            )
    for resolution in resolutions:
        for seed in seeds:
            if resolution == screen_resolution and seed == screen_seed:
                continue
            for stage in (
                "a1_reconstruction",
                "a2_value",
                "a3_value_flux",
                "a4_value_flux_spectral",
                "a5_full_pde",
            ):
                plan.append(
                    {
                        "phase": "confirm",
                        "resolution": resolution,
                        "seed": seed,
                        "cell": stage,
                    }
                )
    return plan


def run_physics_loss_ablation(
    *,
    root: str | Path,
    experiment: str = "physics_loss_ablation",
    overrides: list[str] | None = None,
    limit_samples: int | None = None,
    epochs: int | None = None,
    output_dir: str | Path | None = None,
    device: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> list[FactorialRun]:
    """Run validation-only weight screening followed by the locked loss ladder."""
    root_path = Path(root)
    study = compose_config(
        root=root_path,
        dataset="poisson",
        model="l2l",
        experiment=experiment,
        overrides=overrides,
    )
    matrix = study.get("matrix", {})
    result_root = Path(
        output_dir
        or matrix.get("output_root")
        or study.get("experiment", {}).get("output_dir")
        or "paper_results/physics_loss_ablation"
    )
    if dry_run:
        for item in plan_physics_loss_ablation_runs(
            root=root_path,
            experiment=experiment,
            overrides=overrides,
        ):
            print(
                f"physics_loss_ablation phase={item['phase']} "
                f"resolution={item['resolution']} seed={item['seed']} "
                f"cell={item['cell']}"
            )
        return []
    result_root = ensure_dir(result_root)

    resolutions = [int(value) for value in matrix.get("resolutions", [128, 256])]
    seeds = [int(value) for value in matrix.get("seeds", [0, 1, 2])]
    screen_resolution = int(matrix.get("screen_resolution", 128))
    screen_seed = int(matrix.get("screen_seed", 0))
    guardrail = float(matrix.get("mre_guardrail_fraction", 0.05))
    if screen_resolution not in resolutions or screen_seed not in seeds:
        raise ValueError("The physics screen resolution/seed must be in the confirmation matrix.")
    _validate_physics_ablation_inputs(root_path, matrix, resolutions, seeds)

    all_runs: list[FactorialRun] = []
    screen_records: list[dict[str, Any]] = []
    screen_root = ensure_dir(result_root / "screen")
    baseline_run = _physics_reference_run(
        root=root_path,
        experiment=experiment,
        matrix=matrix,
        resolution=screen_resolution,
        seed=screen_seed,
    )
    all_runs.append(baseline_run)
    screen_records.append(_physics_reference_record(baseline_run))

    recon_run = _physics_finetune_run(
        root=root_path,
        experiment=experiment,
        matrix=matrix,
        output_root=result_root,
        phase="screen",
        resolution=screen_resolution,
        seed=screen_seed,
        cell="a1_reconstruction",
        active_terms=["recon"],
        target_ratios={},
    )
    all_runs.append(recon_run)
    recon_record = _execute_run(
        recon_run,
        limit_samples=limit_samples,
        epochs=epochs,
        device=device,
        overwrite=overwrite,
    )
    screen_records.append(recon_record)
    recon_validation_mre = _validation_metric(recon_record, "mre")

    selection: dict[str, Any] = {
        "screen_resolution": screen_resolution,
        "screen_seed": screen_seed,
        "mre_guardrail_fraction": guardrail,
        "reconstruction_validation_mre": recon_validation_mre,
        "terms": {},
    }
    selected_ratios: dict[str, float] = {}
    selected_records: dict[str, dict[str, Any]] = {
        "a1_reconstruction": recon_record
    }

    stage_specs = (
        (
            "a2_value",
            "interface_value",
            ["recon", "interface_value"],
            "interface_value_trace_error",
        ),
        (
            "a3_value_flux",
            "interface_flux",
            ["recon", "interface_value", "interface_flux"],
            "interface_flux_trace_error",
        ),
        (
            "a4_value_flux_spectral",
            "spectral",
            ["recon", "interface_value", "interface_flux", "spectral"],
            "relative_spectrum_error",
        ),
    )
    for stage, term, active_terms, target_metric in stage_specs:
        candidates: list[tuple[FactorialRun, dict[str, Any]]] = []
        for level, multiplier in _physics_weight_levels(matrix, term):
            ratios = dict(selected_ratios)
            ratios[term] = _physics_default_ratio(matrix, term) * multiplier
            candidate = _physics_finetune_run(
                root=root_path,
                experiment=experiment,
                matrix=matrix,
                output_root=result_root,
                phase="screen",
                resolution=screen_resolution,
                seed=screen_seed,
                cell=f"{stage}_{level}",
                active_terms=active_terms,
                target_ratios=ratios,
            )
            all_runs.append(candidate)
            record = _execute_run(
                candidate,
                limit_samples=limit_samples,
                epochs=epochs,
                device=device,
                overwrite=overwrite,
            )
            screen_records.append(record)
            candidates.append((candidate, record))
        selected_run, selected_record, report = _select_physics_candidate(
            candidates,
            target_metric=target_metric,
            reference_mre=recon_validation_mre,
            guardrail_fraction=guardrail,
        )
        selected_ratios[term] = float(
            selected_run.config["loss"]["weight_calibration"]["target_ratios"][term]
        )
        selected_records[stage] = selected_record
        selection["terms"][term] = report
        _write_physics_progress(screen_records, screen_root, selection, result_root)

    pde_level, pde_multiplier = _single_physics_weight_level(
        matrix,
        "pde_residual",
    )
    selected_ratios["pde_residual"] = (
        _physics_default_ratio(matrix, "pde_residual") * pde_multiplier
    )
    pde_run = _physics_finetune_run(
        root=root_path,
        experiment=experiment,
        matrix=matrix,
        output_root=result_root,
        phase="screen",
        resolution=screen_resolution,
        seed=screen_seed,
        cell=f"a5_full_pde_{pde_level}",
        active_terms=[
            "recon",
            "interface_value",
            "interface_flux",
            "spectral",
            "pde_residual",
        ],
        target_ratios=selected_ratios,
    )
    all_runs.append(pde_run)
    pde_record = _execute_run(
        pde_run,
        limit_samples=limit_samples,
        epochs=epochs,
        device=device,
        overwrite=overwrite,
    )
    screen_records.append(pde_record)
    selected_records["a5_full_pde"] = pde_record
    selection["terms"]["pde_residual"] = {
        "selected_level": pde_level,
        "selected_target_ratio": selected_ratios["pde_residual"],
        "validation_mre": _validation_metric(pde_record, "mre"),
        "validation_target_metric": _validation_metric(
            pde_record,
            "poisson_residual_rms",
        ),
        "guardrail_passed": (
            _validation_metric(pde_record, "mre")
            <= recon_validation_mre * (1.0 + guardrail)
        ),
    }
    _write_physics_progress(screen_records, screen_root, selection, result_root)

    confirmation_records: list[dict[str, Any]] = []
    stage_terms = {
        "a1_reconstruction": ["recon"],
        "a2_value": ["recon", "interface_value"],
        "a3_value_flux": ["recon", "interface_value", "interface_flux"],
        "a4_value_flux_spectral": [
            "recon",
            "interface_value",
            "interface_flux",
            "spectral",
        ],
        "a5_full_pde": [
            "recon",
            "interface_value",
            "interface_flux",
            "spectral",
            "pde_residual",
        ],
    }
    stage_ratios = {
        stage: {
            term: selected_ratios[term]
            for term in terms
            if term != "recon"
        }
        for stage, terms in stage_terms.items()
    }
    for resolution in resolutions:
        for seed in seeds:
            reference = _physics_reference_run(
                root=root_path,
                experiment=experiment,
                matrix=matrix,
                resolution=resolution,
                seed=seed,
            )
            if resolution != screen_resolution or seed != screen_seed:
                all_runs.append(reference)
            confirmation_records.append(_physics_reference_record(reference))
            for stage, active_terms in stage_terms.items():
                if resolution == screen_resolution and seed == screen_seed:
                    confirmation_records.append(
                        _physics_confirmation_alias(selected_records[stage], stage)
                    )
                    continue
                run = _physics_finetune_run(
                    root=root_path,
                    experiment=experiment,
                    matrix=matrix,
                    output_root=result_root,
                    phase="confirm",
                    resolution=resolution,
                    seed=seed,
                    cell=stage,
                    active_terms=active_terms,
                    target_ratios=stage_ratios[stage],
                )
                all_runs.append(run)
                confirmation_records.append(
                    _execute_run(
                        run,
                        limit_samples=limit_samples,
                        epochs=epochs,
                        device=device,
                        overwrite=overwrite,
                    )
                )

    _write_record_set(confirmation_records, result_root)
    write_stage_cost_table(confirmation_records, result_root)
    save_path = result_root / "selection.json"
    save_path.write_text(json.dumps(selection, indent=2, sort_keys=True), encoding="utf-8")
    return all_runs


def _physics_finetune_run(
    *,
    root: Path,
    experiment: str,
    matrix: Mapping[str, Any],
    output_root: Path,
    phase: str,
    resolution: int,
    seed: int,
    cell: str,
    active_terms: list[str],
    target_ratios: Mapping[str, float],
) -> FactorialRun:
    config = compose_config(
        root=root,
        dataset="poisson",
        model="l2l",
        experiment=experiment,
    )
    config.pop("matrix", None)
    settings = _physics_resolution_settings(matrix, resolution)
    representation = matrix.get("representation", {})
    fine_tune = matrix.get("fine_tune", {})
    calibration = matrix.get("calibration", {})
    processed_path = _mapping_value(matrix.get("processed_paths", {}), resolution)
    warm_start_dir = _physics_baseline_dir(matrix, resolution, seed)
    config.setdefault("dataset", {}).update(
        {
            "name": "poisson",
            "grid_size": resolution,
            "processed_path": str(processed_path),
        }
    )
    config.setdefault("patches", {}).update(
        {
            "patch_size": int(settings["patch_size"]),
            "stride": int(settings.get("stride", settings["patch_size"])),
            "assembly": "average",
        }
    )
    config.setdefault("pca", {}).update(
        {
            "solver": str(representation.get("solver", "randomized")),
            "output_variance": float(
                representation.get("residual_variance", 0.995)
            ),
        }
    )
    config["pca"].setdefault("two_scale", {}).update(
        {
            "coarse_factor": int(representation.get("coarse_factor", 4)),
            "coarse_components": int(representation.get("coarse_components", 10)),
            "coarse_variance": float(representation.get("coarse_variance", 1.0)),
        }
    )
    config.setdefault("mechanisms", {}).update(
        {"two_scale": True, "coupling": False, "in_loop_loss": True}
    )
    config.setdefault("training", {}).update(
        {
            "epochs": int(fine_tune.get("epochs", 50)),
            "lr": float(fine_tune.get("lr", 1.0e-4)),
            "weight_decay": float(fine_tune.get("weight_decay", 0.0)),
            "scheduler": str(fine_tune.get("scheduler", "none")),
        }
    )
    config["training"].setdefault("latent_loss", {}).update(
        {"mode": "block_balanced", "coarse_weight": 0.5, "residual_weight": 0.5}
    )
    config["warm_start"] = {"enabled": True, "run_dir": str(warm_start_dir)}
    config.setdefault("loss", {}).update(
        {
            "reconstruction": "relative_l2",
            "interface_target": "truth",
            "active_terms": active_terms,
            "pde": "poisson",
            "weights": {
                "recon": 1.0,
                "interface_value": 0.0,
                "interface_flux": 0.0,
                "spectral": 0.0,
                "pde_residual": 0.0,
            },
            "warmup": {
                "reconstruction_epochs": int(
                    fine_tune.get("reconstruction_epochs", 5)
                ),
                "ramp_epochs": int(fine_tune.get("ramp_epochs", 10)),
            },
            "weight_calibration": {
                "enabled": len(active_terms) > 1,
                "target_ratios": {
                    str(name): float(value) for name, value in target_ratios.items()
                },
                "batch_size": int(calibration.get("batch_size", 16)),
                "max_samples": int(calibration.get("max_samples", 128)),
                "minimum_weight": float(
                    calibration.get("minimum_weight", 1.0e-12)
                ),
                "maximum_weight": float(calibration.get("maximum_weight", 1.0e6)),
            },
        }
    )
    config.setdefault("evaluation", {}).update(
        {
            "validation_metrics": True,
            "metrics": [
                "mse",
                "mae",
                "mre",
                "ssim",
                "relative_spectrum_error",
                "interface_jump",
                "interface_flux_jump",
                "interface_value_trace_error",
                "interface_flux_trace_error",
                "poisson_residual",
            ],
        }
    )
    run_id = f"{experiment}_{phase}_{cell}_seed{seed}_r{resolution}"
    run_dir = output_root / phase / f"resolution_{resolution}" / cell / f"seed_{seed}"
    ablation = {
        "family": "physics_loss",
        "phase": phase,
        "cell": cell,
        "active_terms": "+".join(active_terms),
        **{
            f"target_ratio_{name}": float(value)
            for name, value in target_ratios.items()
        },
    }
    config.setdefault("experiment", {}).update(
        {"name": run_id, "seed": seed, "output_dir": str(run_dir)}
    )
    config.setdefault("resolved", {}).update(
        {
            "multirun": True,
            "study": experiment,
            "study_type": "physics_loss_ablation",
            "study_method": cell,
            "ablation": ablation,
            "run_id": run_id,
        }
    )
    mechanisms = {"two_scale": True, "coupling": False, "in_loop_loss": True}
    return FactorialRun(
        run_id=run_id,
        config=config,
        config_hash=config_hash_for(config),
        seed=seed,
        mechanisms=mechanisms,
        resolution=resolution,
        run_dir=run_dir,
        baseline_model=cell,
    )


def _physics_reference_run(
    *,
    root: Path,
    experiment: str,
    matrix: Mapping[str, Any],
    resolution: int,
    seed: int,
) -> FactorialRun:
    config = compose_config(
        root=root,
        dataset="poisson",
        model="l2l",
        experiment=experiment,
    )
    config.pop("matrix", None)
    config.setdefault("mechanisms", {}).update(
        {"two_scale": True, "coupling": False, "in_loop_loss": False}
    )
    config.setdefault("resolved", {})["ablation"] = {
        "family": "physics_loss",
        "phase": "confirm",
        "cell": "a0_latent_baseline",
        "active_terms": "block_balanced_latent",
    }
    run_id = f"{experiment}_confirm_a0_latent_baseline_seed{seed}_r{resolution}"
    config.setdefault("experiment", {}).update({"name": run_id, "seed": seed})
    run_dir = _physics_baseline_dir(matrix, resolution, seed)
    return FactorialRun(
        run_id=run_id,
        config=config,
        config_hash=config_hash_for(config),
        seed=seed,
        mechanisms={"two_scale": True, "coupling": False, "in_loop_loss": False},
        resolution=resolution,
        run_dir=run_dir,
        baseline_model="a0_latent_baseline",
    )


def _physics_reference_record(run: FactorialRun) -> dict[str, Any]:
    metrics = evaluate_run(run.run_dir)
    return build_result_record(run, metrics=metrics, run_dir=run.run_dir)


def _select_physics_candidate(
    candidates: list[tuple[FactorialRun, dict[str, Any]]],
    *,
    target_metric: str,
    reference_mre: float,
    guardrail_fraction: float,
) -> tuple[FactorialRun, dict[str, Any], dict[str, Any]]:
    threshold = reference_mre * (1.0 + guardrail_fraction)
    eligible = [
        pair
        for pair in candidates
        if _validation_metric(pair[1], "mre") <= threshold
    ]
    pool = eligible if eligible else candidates
    selected_run, selected_record = min(
        pool,
        key=lambda pair: _validation_metric(pair[1], target_metric),
    )
    target_ratios = selected_run.config["loss"]["weight_calibration"][
        "target_ratios"
    ]
    term = next(reversed(target_ratios))
    report = {
        "target_metric": target_metric,
        "mre_threshold": threshold,
        "guardrail_had_eligible_candidate": bool(eligible),
        "selected_cell": selected_run.baseline_model,
        "selected_level": selected_run.baseline_model.rsplit("_", 1)[-1],
        "selected_target_ratio": float(target_ratios[term]),
        "validation_mre": _validation_metric(selected_record, "mre"),
        "validation_target_metric": _validation_metric(
            selected_record,
            target_metric,
        ),
        "candidates": [
            {
                "cell": run.baseline_model,
                "validation_mre": _validation_metric(record, "mre"),
                "validation_target_metric": _validation_metric(
                    record,
                    target_metric,
                ),
                "guardrail_passed": (
                    _validation_metric(record, "mre") <= threshold
                ),
            }
            for run, record in candidates
        ],
    }
    return selected_run, selected_record, report


def _validation_metric(record: Mapping[str, Any], name: str) -> float:
    metrics = record.get("validation_metrics", {})
    if not isinstance(metrics, Mapping) or name not in metrics:
        raise ValueError(
            f"Run {record.get('run_id', '')!r} is missing validation metric {name!r}."
        )
    return float(metrics[name])


def _physics_confirmation_alias(
    record: Mapping[str, Any],
    stage: str,
) -> dict[str, Any]:
    alias = copy.deepcopy(dict(record))
    alias["run_id"] = str(alias["run_id"]).replace("_screen_", "_confirm_selected_")
    alias["baseline_model"] = stage
    ablation = dict(alias.get("ablation", {}))
    ablation.update({"phase": "confirm", "cell": stage, "selected_from_screen": True})
    alias["ablation"] = ablation
    return alias


def _write_physics_progress(
    records: list[dict[str, Any]],
    screen_root: Path,
    selection: Mapping[str, Any],
    result_root: Path,
) -> None:
    _write_record_set(records, screen_root)
    write_stage_cost_table(records, screen_root)
    (result_root / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_record_set(records: list[dict[str, Any]], output_dir: Path) -> None:
    ensure_dir(output_dir)
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    write_rollup(records, output_dir)


def _physics_weight_levels(
    matrix: Mapping[str, Any],
    term: str,
) -> list[tuple[str, float]]:
    levels = matrix.get("weight_levels", {})
    term_levels = levels.get(term, {}) if isinstance(levels, Mapping) else {}
    if not isinstance(term_levels, Mapping) or not term_levels:
        raise ValueError(f"Missing weight levels for physics term {term!r}.")
    return [
        (str(name), float(value))
        for name, value in term_levels.items()
        if str(name).lower() != "off" and float(value) != 0.0
    ]


def _single_physics_weight_level(
    matrix: Mapping[str, Any],
    term: str,
) -> tuple[str, float]:
    levels = _physics_weight_levels(matrix, term)
    if len(levels) != 1:
        raise ValueError(f"Expected exactly one nonzero level for {term}, got {levels}.")
    return levels[0]


def _physics_default_ratio(matrix: Mapping[str, Any], term: str) -> float:
    ratios = matrix.get("default_target_ratios", {})
    if not isinstance(ratios, Mapping) or term not in ratios:
        raise ValueError(f"Missing default target ratio for physics term {term!r}.")
    return float(ratios[term])


def _physics_resolution_settings(
    matrix: Mapping[str, Any],
    resolution: int,
) -> Mapping[str, Any]:
    settings = _mapping_value(matrix.get("resolution_settings", {}), resolution)
    if not isinstance(settings, Mapping):
        raise ValueError(f"Missing physics-loss settings for resolution {resolution}.")
    return settings


def _physics_baseline_dir(
    matrix: Mapping[str, Any],
    resolution: int,
    seed: int,
) -> Path:
    baseline_root = Path(
        str(
            matrix.get(
                "baseline_root",
                "paper_results/coarse_representation_ablation/poisson",
            )
        )
    )
    baseline_cell = str(matrix.get("baseline_cell", "residual_variance_0p995"))
    return (
        baseline_root
        / f"resolution_{resolution}"
        / baseline_cell
        / f"seed_{seed}"
    )


def _validate_physics_ablation_inputs(
    root: Path,
    matrix: Mapping[str, Any],
    resolutions: list[int],
    seeds: list[int],
) -> None:
    processed_paths = matrix.get("processed_paths", {})
    missing_data = [
        str(root / str(_mapping_value(processed_paths, resolution)))
        for resolution in resolutions
        if not (root / str(_mapping_value(processed_paths, resolution))).is_file()
    ]
    if missing_data:
        raise FileNotFoundError(f"Missing physics-loss datasets: {missing_data}")
    missing_baselines: list[str] = []
    for resolution in resolutions:
        for seed in seeds:
            run_dir = root / _physics_baseline_dir(matrix, resolution, seed)
            for filename in ("model.pt", "pca_encoder.joblib", "predictions_test.npz"):
                if not (run_dir / filename).is_file():
                    missing_baselines.append(str(run_dir / filename))
    if missing_baselines:
        raise FileNotFoundError(
            "Missing selected ablation-5 warm-start artifacts: "
            f"{missing_baselines}"
        )


def expand_method_study_runs(
    *,
    root: str | Path,
    dataset: str = "poisson",
    experiment: str = "poisson_256_seed0",
    overrides: list[str] | None = None,
    output_dir: str | Path | None = None,
) -> list[FactorialRun]:
    """Expand an ordered, single-seed method comparison study."""
    root_path = Path(root)
    study = compose_config(
        root=root_path,
        dataset=dataset,
        model="l2l",
        experiment=experiment,
        overrides=overrides,
    )
    matrix = study.get("matrix", {})
    resolution = int(matrix.get("resolution", 256))
    seed = int(matrix.get("seed", study.get("experiment", {}).get("seed", 0)))
    n_train = int(matrix.get("n_train", 8000))
    coarse_factor = int(matrix.get("coarse_factor", 4))
    pca_solver = str(matrix.get("pca_solver", "randomized"))
    processed_path = str(matrix.get("processed_path", f"data/processed/{dataset}_{resolution}.npz"))
    output_root = Path(
        output_dir
        or matrix.get("output_root")
        or study.get("experiment", {}).get("output_dir")
        or f"results/{experiment}"
    )
    _assert_coarse_factor_legal(resolution, coarse_factor, n_train)

    entries = matrix.get("runs", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Experiment {experiment!r} must define a non-empty matrix.runs list.")

    runs: list[FactorialRun] = []
    known_methods: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise TypeError("Each matrix.runs entry must be a mapping.")
        method_name = str(entry.get("name", "")).strip()
        model_name = str(entry.get("model", "")).strip()
        if not method_name or not model_name:
            raise ValueError("Each matrix.runs entry requires non-empty name and model fields.")
        if method_name in known_methods:
            raise ValueError(f"Duplicate method name in study: {method_name!r}.")

        config = compose_config(
            root=root_path,
            dataset=dataset,
            model=model_name,
            experiment=experiment,
            overrides=overrides,
        )
        config.pop("matrix", None)
        config = deep_merge(config, entry.get("config", {}))
        mechanisms = {"two_scale": False, "coupling": False, "in_loop_loss": False}
        mechanisms.update(
            {
                key: bool(value)
                for key, value in entry.get("mechanisms", {}).items()
                if key in MECHANISM_KEYS
            }
        )
        config.setdefault("mechanisms", {}).update(mechanisms)
        config.setdefault("dataset", {}).update(
            {"grid_size": resolution, "processed_path": processed_path}
        )
        config.setdefault("experiment", {}).update(
            {
                "name": f"{experiment}_{method_name}_seed{seed}_r{resolution}",
                "seed": seed,
            }
        )
        run_dir = output_root / f"resolution_{resolution}" / method_name / f"seed_{seed}"
        config["experiment"]["output_dir"] = str(run_dir)
        if str(config.get("pca", {}).get("type", "")).lower() not in {
            "none",
            "identity",
            "image",
        }:
            config.setdefault("pca", {})["solver"] = pca_solver
        config.setdefault("pca", {}).setdefault("two_scale", {})[
            "coarse_factor"
        ] = coarse_factor

        warm_start_from = entry.get("warm_start_from")
        if warm_start_from is not None:
            source_method = str(warm_start_from)
            if source_method not in known_methods:
                raise ValueError(
                    f"Method {method_name!r} warm-starts from {source_method!r}, "
                    "which must appear earlier in matrix.runs."
                )
            config["warm_start"] = {
                "enabled": True,
                "run_dir": str(
                    output_root
                    / f"resolution_{resolution}"
                    / source_method
                    / f"seed_{seed}"
                ),
            }

        config.setdefault("resolved", {}).update(
            {
                "multirun": True,
                "study": experiment,
                "study_method": method_name,
                "n_train": n_train,
                "coarse_factor": coarse_factor,
                "run_id": config["experiment"]["name"],
            }
        )
        runs.append(
            FactorialRun(
                run_id=config["experiment"]["name"],
                config=config,
                config_hash=config_hash_for(config),
                seed=seed,
                mechanisms=mechanisms,
                resolution=resolution,
                run_dir=run_dir,
                baseline_model=method_name,
            )
        )
        known_methods.add(method_name)
    return runs


def run_method_study(
    *,
    root: str | Path,
    dataset: str = "poisson",
    experiment: str = "poisson_256_seed0",
    overrides: list[str] | None = None,
    limit_samples: int | None = None,
    epochs: int | None = None,
    output_dir: str | Path | None = None,
    device: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> list[FactorialRun]:
    """Run an ordered method study, including warm-start dependencies."""
    runs = expand_method_study_runs(
        root=root,
        dataset=dataset,
        experiment=experiment,
        overrides=overrides,
        output_dir=output_dir,
    )
    if dry_run:
        for run in runs:
            print(_dry_run_line(run))
        return runs
    result_root = ensure_dir(
        output_dir
        or runs[0].run_dir.parents[2]
        if runs
        else f"results/{experiment}"
    )
    records = _execute_runs(
        runs,
        result_root=result_root,
        limit_samples=limit_samples,
        epochs=epochs,
        device=device,
        overwrite=overwrite,
    )
    write_stage_cost_table(records, result_root)
    return runs


def _execute_runs(
    runs: list[FactorialRun],
    *,
    result_root: Path,
    limit_samples: int | None,
    epochs: int | None,
    device: str | None,
    overwrite: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    jsonl_path = result_root / "results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for run in runs:
            record = _execute_run(
                run,
                limit_samples=limit_samples,
                epochs=epochs,
                device=device,
                overwrite=overwrite,
            )
            records.append(record)
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
    write_rollup(records, result_root)
    return records


def _execute_run(
    run: FactorialRun,
    *,
    limit_samples: int | None,
    epochs: int | None,
    device: str | None,
    overwrite: bool,
) -> dict[str, Any]:
    run_config = copy.deepcopy(run.config)
    if epochs is not None:
        run_config.setdefault("training", {})["epochs"] = int(epochs)
    if device is not None:
        run_config.setdefault("training", {})["device"] = device
    resolved_dir = ensure_dir(run.run_dir / "resolved_configs")
    config_path = save_config(run_config, resolved_dir, filename="config.yaml")
    if not overwrite and _run_is_complete(run.run_dir):
        prediction_path = run.run_dir / "predictions_test.npz"
        metrics = (
            evaluate_run(run.run_dir)
            if prediction_path.is_file()
            else {
                key: _jsonable(value)
                for key, value in _load_json_mapping(
                    run.run_dir / "metrics.json"
                ).items()
            }
        )
        return build_result_record(run, metrics=metrics, run_dir=run.run_dir)
    if run.baseline_model == "l2l_refinement":
        trained_dir = train_refinement_from_config(
            config_path,
            limit_samples=limit_samples,
            epochs=epochs,
            output_dir=run.run_dir,
            device=device,
            overwrite=overwrite,
        )
    else:
        trained_dir = train_from_config(
            config_path,
            limit_samples=limit_samples,
            epochs=epochs,
            output_dir=run.run_dir,
            device=device,
            overwrite=overwrite,
        )
    metrics = evaluate_run(trained_dir)
    if not bool(run_config.get("evaluation", {}).get("retain_predictions", True)):
        (trained_dir / "predictions_test.npz").unlink(missing_ok=True)
    return build_result_record(run, metrics=metrics, run_dir=trained_dir)


def build_result_record(
    run: FactorialRun,
    *,
    metrics: Mapping[str, Any],
    run_dir: str | Path,
) -> dict[str, Any]:
    """Build one stable result record with metrics and timings."""
    run_path = Path(run_dir)
    runtime = _load_runtime(run_path / "runtime.json")
    validation_metrics = _load_json_mapping(run_path / "validation_metrics.json")
    pca_summary = _load_json_mapping(run_path / "pca_summary.json")
    invocation_timings = dict(runtime.get("stages", {}))
    cumulative_timings = dict(invocation_timings)
    warm_start = run.config.get("warm_start", {})
    warm_start_dir: Path | None = None
    base_run_dir: Path | None = None
    if isinstance(warm_start, Mapping) and bool(warm_start.get("enabled", False)):
        raw_source = warm_start.get("run_dir")
        if raw_source:
            warm_start_dir = Path(str(raw_source))
            source_runtime = _load_runtime(warm_start_dir / "runtime.json")
            cumulative_timings = _sum_pipeline_timings(
                source_runtime.get("stages", {}),
                invocation_timings,
            )
    elif run.baseline_model == "l2l_refinement":
        raw_source = run.config.get("base_run_dir")
        if raw_source:
            base_run_dir = Path(str(raw_source))
            source_runtime = _load_runtime(base_run_dir / "runtime.json")
            cumulative_timings = _sum_pipeline_timings(
                source_runtime.get("stages", {}),
                invocation_timings,
            )
    return {
        "run_id": run.run_id,
        "config_hash": run.config_hash,
        "run_dir": str(run_path),
        "seed": run.seed,
        "resolution": run.resolution,
        "dataset": str(run.config.get("dataset", {}).get("name", "")),
        "baseline_model": run.baseline_model,
        "latent_loss_mode": str(
            run.config.get("training", {}).get("latent_loss", {}).get("mode", "mse")
        ),
        "ablation": dict(run.config.get("resolved", {}).get("ablation", {})),
        "equivalence_gates": dict(
            run.config.get("resolved", {}).get("equivalence_gates", {})
        ),
        "mechanisms": dict(run.mechanisms),
        "metrics": {key: _jsonable(value) for key, value in metrics.items()},
        "validation_metrics": {
            key: _jsonable(value) for key, value in validation_metrics.items()
        },
        "pca": pca_summary,
        "timings": cumulative_timings,
        "invocation_timings": invocation_timings,
        "timing_accounting": {
            "includes_warm_start": warm_start_dir is not None,
            "warm_start_run_dir": None if warm_start_dir is None else str(warm_start_dir),
            "includes_base_run": base_run_dir is not None,
            "base_run_dir": None if base_run_dir is None else str(base_run_dir),
        },
        "times": runtime.get("times", {}),
    }


def write_rollup(records: Iterable[Mapping[str, Any]], output_dir: str | Path) -> Path:
    """Write a flat CSV roll-up for quick tabulation."""
    rows = [_flatten_record(record) for record in records]
    output_path = Path(output_dir) / "results.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def write_stage_cost_table(records: Iterable[Mapping[str, Any]], output_dir: str | Path) -> Path:
    """Write cumulative and invocation-only stage costs by method/resolution/seed."""
    rows: list[dict[str, Any]] = []
    stages = (
        "coarse_svd",
        "pca_fit",
        "latent_transform",
        "nn_train",
        "inference",
        "end_to_end",
    )
    for record in records:
        timing = record.get("timings", {})
        invocation_timing = record.get("invocation_timings", timing)
        accounting = record.get("timing_accounting", {})
        row = {
            "method": record.get("baseline_model", ""),
            "dataset": record.get("dataset", ""),
            "latent_loss_mode": record.get("latent_loss_mode", "mse"),
            "resolution": record.get("resolution", ""),
            "seed": record.get("seed", ""),
            "config_hash": record.get("config_hash", ""),
            "includes_warm_start": (
                bool(accounting.get("includes_warm_start", False))
                if isinstance(accounting, Mapping)
                else False
            ),
        }
        ablation = record.get("ablation", {})
        if isinstance(ablation, Mapping):
            for key in ("method", "solver_class", "local_solver", "coarse_solver"):
                if key in ablation:
                    row[key] = ablation[key]
        for stage in stages:
            row[stage] = timing.get(stage, 0.0)
            row[f"invocation_{stage}"] = invocation_timing.get(stage, 0.0)
        rows.append(row)
    output_path = Path(output_dir) / "stage_costs.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def write_pca_projection_table(
    *,
    resolutions: Iterable[int],
    output_dir: str | Path,
    n_samples: int,
) -> Path:
    """Write projected global-vs-two-scale PCA feature scaling table."""
    rows = []
    for resolution in sorted({int(value) for value in resolutions}):
        coarse_factor = 8 if resolution >= 512 else 4
        _assert_coarse_factor_legal(resolution, coarse_factor, n_samples)
        global_features = resolution * resolution
        coarse_features = (resolution // coarse_factor) ** 2
        rows.append(
            {
                "resolution": resolution,
                "n_samples": int(n_samples),
                "global_features": global_features,
                "two_scale_coarse_factor": coarse_factor,
                "two_scale_coarse_features": coarse_features,
                "global_feature_ratio_vs_128": global_features / float(128 * 128),
                "coarse_feature_ratio_vs_128": coarse_features / float((128 // 4) ** 2),
                "coarse_invariant_legal": True,
            }
        )
    output_path = Path(output_dir) / "pca_projection.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def config_hash_for(config: Mapping[str, Any]) -> str:
    """Return a stable hash for the run-defining config content."""
    canonical = copy.deepcopy(dict(config))
    canonical.get("experiment", {}).pop("output_dir", None)
    canonical.get("resolved", {}).pop("run_id", None)
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _assert_coarse_factor_legal(resolution: int, coarse_factor: int, n_samples: int) -> None:
    if resolution % coarse_factor != 0:
        raise ValueError(
            f"resolution={resolution} must be divisible by coarse_factor={coarse_factor}."
        )
    coarse_features = (resolution // coarse_factor) ** 2
    if coarse_features >= n_samples:
        raise ValueError(
            "Coarse factor invariant violated: "
            f"(D/c)^2={coarse_features} must be < m={n_samples}; "
            f"got D={resolution}, c={coarse_factor}."
        )


def _dry_run_line(run: FactorialRun) -> str:
    flags = " ".join(f"{key}={str(value).lower()}" for key, value in run.mechanisms.items())
    baseline = f" baseline_model={run.baseline_model}" if run.baseline_model else ""
    dataset = str(run.config.get("dataset", {}).get("name", ""))
    train_samples = run.config.get("dataset", {}).get("train_samples")
    sample_label = f" train_samples={train_samples}" if train_samples is not None else ""
    latent_loss = str(
        run.config.get("training", {}).get("latent_loss", {}).get("mode", "mse")
    )
    pca_type = str(run.config.get("pca", {}).get("type", "")).lower()
    local_solver = (
        "none"
        if pca_type in {"none", "identity", "image"}
        else str(run.config.get("pca", {}).get("solver", ""))
    )
    coarse_solver = str(
        run.config.get("pca", {}).get("two_scale", {}).get("coarse_solver", "")
    )
    solver_label = f" local_solver={local_solver}"
    if bool(run.mechanisms.get("two_scale", False)):
        solver_label += f" coarse_solver={coarse_solver}"
    return (
        f"{run.run_id} dataset={dataset} seed={run.seed} resolution={run.resolution}"
        f"{sample_label}{baseline} latent_loss={latent_loss}{solver_label} "
        f"{flags} config_hash={run.config_hash} output_dir={run.run_dir}"
    )


def _cell_name(mechanisms: Mapping[str, bool]) -> str:
    pieces = []
    for key in MECHANISM_KEYS:
        pieces.append(f"{key}-{'on' if mechanisms[key] else 'off'}")
    return "__".join(pieces)


def _load_runtime(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"stages": {}, "times": {}}
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        return {"stages": {}, "times": {}}
    return loaded


def _load_json_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    return loaded if isinstance(loaded, dict) else {}


def _mapping_value(mapping: Mapping[Any, Any], integer_key: int) -> Any:
    if integer_key in mapping:
        return mapping[integer_key]
    return mapping.get(str(integer_key))


def _validate_study_dataset_paths(runs: Iterable[FactorialRun]) -> None:
    missing = sorted(
        {
            str(run.config.get("dataset", {}).get("processed_path", ""))
            for run in runs
            if not Path(
                str(run.config.get("dataset", {}).get("processed_path", ""))
            ).is_file()
        }
    )
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "The study requires processed datasets that do not exist:\n"
            f"{formatted}\n"
            "Generate the missing datasets before starting the multirun."
        )


def _sum_pipeline_timings(
    prerequisite: Mapping[str, Any],
    invocation: Mapping[str, Any],
) -> dict[str, float]:
    """Combine prerequisite and invocation costs for a from-scratch comparison."""
    stages = ("pca_fit", "latent_transform", "nn_train", "inference", "end_to_end")
    combined = {
        stage: float(prerequisite.get(stage, 0.0)) + float(invocation.get(stage, 0.0))
        for stage in stages
    }
    combined["coarse_svd"] = float(prerequisite.get("coarse_svd", 0.0))
    if float(invocation.get("pca_fit", 0.0)) > 0.0:
        combined["coarse_svd"] += float(invocation.get("coarse_svd", 0.0))
    return combined


def _run_is_complete(run_dir: str | Path) -> bool:
    run_path = Path(run_dir)
    required = [
        "config.yaml",
        "metrics.json",
        "runtime.json",
    ]
    config_path = run_path / "config.yaml"
    if config_path.is_file():
        config = load_config(config_path)
        if bool(config.get("evaluation", {}).get("retain_predictions", True)):
            required.append("predictions_test.npz")
    else:
        required.append("predictions_test.npz")
    return all((run_path / filename).is_file() for filename in required)


def _flatten_record(record: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_id": record["run_id"],
        "config_hash": record["config_hash"],
        "run_dir": record["run_dir"],
        "seed": record["seed"],
        "resolution": record["resolution"],
        "dataset": record.get("dataset", ""),
        "baseline_model": record.get("baseline_model", ""),
        "latent_loss_mode": record.get("latent_loss_mode", "mse"),
    }
    for key, value in record["mechanisms"].items():
        row[f"mechanism_{key}"] = value
    for key, value in record.get("ablation", {}).items():
        row[f"ablation_{key}"] = value
    for key, value in record.get("equivalence_gates", {}).items():
        row[f"equivalence_gate_{key}"] = value
    for key, value in record["metrics"].items():
        row[f"metric_{key}"] = value
    for key, value in record.get("validation_metrics", {}).items():
        row[f"validation_metric_{key}"] = value
    for key, value in _flatten_diagnostic_mapping(record.get("pca", {})).items():
        row[f"pca_{key}"] = value
    for key, value in record["timings"].items():
        row[f"timing_{key}"] = value
    for key, value in record.get("invocation_timings", {}).items():
        row[f"invocation_timing_{key}"] = value
    accounting = record.get("timing_accounting", {})
    if isinstance(accounting, Mapping):
        row["timing_includes_warm_start"] = bool(
            accounting.get("includes_warm_start", False)
        )
        row["timing_warm_start_run_dir"] = accounting.get("warm_start_run_dir")
    return row


def _flatten_diagnostic_mapping(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested PCA diagnostics while keeping vector values stable."""
    if not isinstance(value, Mapping):
        return {}
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            flattened.update(_flatten_diagnostic_mapping(item, name))
        elif isinstance(item, (list, tuple)):
            flattened[name] = json.dumps(list(item), sort_keys=True)
        else:
            flattened[name] = item
    return flattened


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _coarse_ablation_cells(
    *,
    exact_ranks: list[int],
    coarse_factors: list[int],
    residual_variances: list[float],
    default_rank: int,
    default_factor: int,
    default_residual_variance: float,
    adaptive: Any,
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = [
        {
            "name": "baseline_exact_rank10",
            "family": "shared_baseline",
            "coarse_factor": default_factor,
            "coarse_components": default_rank,
            "coarse_variance": 1.0,
            "residual_variance": default_residual_variance,
        }
    ]
    for rank in exact_ranks:
        if rank == default_rank:
            continue
        cells.append(
            {
                "name": f"coarse_rank_{rank}",
                "family": "coarse_rank",
                "coarse_factor": default_factor,
                "coarse_components": rank,
                "coarse_variance": 1.0,
                "residual_variance": default_residual_variance,
            }
        )
    for factor in coarse_factors:
        if factor == default_factor:
            continue
        cells.append(
            {
                "name": f"coarse_factor_{factor}",
                "family": "coarse_factor",
                "coarse_factor": factor,
                "coarse_components": default_rank,
                "coarse_variance": 1.0,
                "residual_variance": default_residual_variance,
            }
        )
    for variance in residual_variances:
        if variance == default_residual_variance:
            continue
        variance_label = str(variance).replace(".", "p")
        cells.append(
            {
                "name": f"residual_variance_{variance_label}",
                "family": "residual_variance",
                "coarse_factor": default_factor,
                "coarse_components": default_rank,
                "coarse_variance": 1.0,
                "residual_variance": variance,
            }
        )
    if not isinstance(adaptive, Mapping):
        raise TypeError("matrix.adaptive_reference must be a mapping.")
    cells.append(
        {
            "name": "adaptive_99pct_cap20",
            "family": "adaptive_reference",
            "coarse_factor": default_factor,
            "coarse_components": int(adaptive.get("coarse_components", 20)),
            "coarse_variance": float(adaptive.get("coarse_variance", 0.99)),
            "residual_variance": default_residual_variance,
        }
    )
    return cells
