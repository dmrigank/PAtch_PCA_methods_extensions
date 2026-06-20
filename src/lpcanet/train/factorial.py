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

from lpcanet.data.cache import dataset_cache_path
from lpcanet.metrics.evaluate import evaluate_run
from lpcanet.train.experiment import train_from_config
from lpcanet.train.refinement import train_refinement_from_config
from lpcanet.utils.config import Config, compose_config, deep_merge, save_config
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
    """Expand the 64/128/256 resolution sweep plus optional 512 configs."""
    root_path = Path(root)
    sweep_config = compose_config(
        root=root_path,
        dataset=dataset,
        model="l2l",
        experiment=experiment,
        overrides=overrides,
    )
    matrix = sweep_config.get("matrix", {})
    output_root = Path(output_dir or matrix.get("output_root") or "results/resolution_sweep")
    seeds = [int(seed) for seed in matrix.get("seeds", [0])]
    n_samples = int(matrix.get("n_samples", 8000))
    data_cache_cfg = sweep_config.get("data_cache", {})
    cache_root = Path(data_cache_cfg.get("root", "data/processed"))
    blocks = [("main", matrix.get("main", {}))]
    optional = matrix.get("optional", {})
    if include_optional or bool(optional.get("enabled", False)):
        blocks.append(("optional", optional))

    runs: list[FactorialRun] = []
    for block_name, block in blocks:
        resolutions = [int(value) for value in block.get("resolutions", [])]
        models = [str(value) for value in block.get("models", [])]
        coarse_factor = int(block.get("coarse_factor", 4))
        for resolution in resolutions:
            _assert_coarse_factor_legal(resolution, coarse_factor, n_samples)
            for model_name in models:
                for seed in seeds:
                    config = _compose_resolution_model_config(
                        root_path=root_path,
                        dataset=dataset,
                        model_name=model_name,
                        experiment=experiment,
                        overrides=overrides,
                    )
                    config.pop("matrix", None)
                    config.setdefault("dataset", {})["grid_size"] = resolution
                    config["dataset"]["processed_path"] = str(
                        dataset_cache_path(
                            root=cache_root,
                            dataset_name=str(config["dataset"].get("name", dataset)),
                            resolution=resolution,
                            n_samples=n_samples,
                            seed=seed,
                        )
                    )
                    config.setdefault("experiment", {})["seed"] = seed
                    config["experiment"]["name"] = (
                        f"resolution_sweep_{model_name}_seed{seed}_r{resolution}"
                    )
                    config.setdefault("pca", {}).setdefault("two_scale", {})["coarse_factor"] = coarse_factor
                    if model_name == "full":
                        config.setdefault("mechanisms", {}).update(
                            {"two_scale": True, "coupling": True, "in_loop_loss": True}
                        )
                    else:
                        config.setdefault("mechanisms", {}).update(
                            {"two_scale": False, "coupling": False, "in_loop_loss": False}
                        )
                    run_dir = output_root / f"resolution_{resolution}" / model_name / f"seed_{seed}"
                    config["experiment"]["output_dir"] = str(run_dir)
                    config.setdefault("resolved", {}).update(
                        {
                            "multirun": True,
                            "sweep": "resolution_sweep",
                            "sweep_block": block_name,
                            "sweep_model": model_name,
                            "n_samples": n_samples,
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
                            mechanisms={key: bool(config["mechanisms"][key]) for key in MECHANISM_KEYS},
                            resolution=resolution,
                            run_dir=run_dir,
                            baseline_model=model_name,
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
    """Expand and optionally execute the generated resolution sweep."""
    runs = expand_resolution_sweep_runs(
        root=root,
        dataset=dataset,
        experiment=experiment,
        overrides=overrides,
        output_dir=output_dir,
        include_optional=include_optional,
    )
    result_root = ensure_dir(output_dir or "results/resolution_sweep")
    write_pca_projection_table(
        resolutions=sorted({run.resolution for run in runs}),
        output_dir=result_root,
        n_samples=int(runs[0].config.get("resolved", {}).get("n_samples", 8000)) if runs else 8000,
    )
    if dry_run:
        for run in runs:
            print(_dry_run_line(run))
        return runs
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
            run_config = copy.deepcopy(run.config)
            if epochs is not None:
                run_config.setdefault("training", {})["epochs"] = int(epochs)
            if device is not None:
                run_config.setdefault("training", {})["device"] = device
            resolved_dir = ensure_dir(run.run_dir / "resolved_configs")
            config_path = save_config(run_config, resolved_dir, filename="config.yaml")
            if not overwrite and _run_is_complete(run.run_dir):
                metrics = evaluate_run(run.run_dir)
                record = build_result_record(run, metrics=metrics, run_dir=run.run_dir)
                records.append(record)
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                continue
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
            record = build_result_record(run, metrics=metrics, run_dir=trained_dir)
            records.append(record)
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
    write_rollup(records, result_root)
    return records


def build_result_record(
    run: FactorialRun,
    *,
    metrics: Mapping[str, Any],
    run_dir: str | Path,
) -> dict[str, Any]:
    """Build one stable result record with metrics and timings."""
    run_path = Path(run_dir)
    runtime = _load_runtime(run_path / "runtime.json")
    invocation_timings = dict(runtime.get("stages", {}))
    cumulative_timings = dict(invocation_timings)
    warm_start = run.config.get("warm_start", {})
    warm_start_dir: Path | None = None
    if isinstance(warm_start, Mapping) and bool(warm_start.get("enabled", False)):
        raw_source = warm_start.get("run_dir")
        if raw_source:
            warm_start_dir = Path(str(raw_source))
            source_runtime = _load_runtime(warm_start_dir / "runtime.json")
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
        "baseline_model": run.baseline_model,
        "mechanisms": dict(run.mechanisms),
        "metrics": {key: _jsonable(value) for key, value in metrics.items()},
        "timings": cumulative_timings,
        "invocation_timings": invocation_timings,
        "timing_accounting": {
            "includes_warm_start": warm_start_dir is not None,
            "warm_start_run_dir": None if warm_start_dir is None else str(warm_start_dir),
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
    stages = ("pca_fit", "latent_transform", "nn_train", "inference", "end_to_end")
    for record in records:
        timing = record.get("timings", {})
        invocation_timing = record.get("invocation_timings", timing)
        accounting = record.get("timing_accounting", {})
        row = {
            "method": record.get("baseline_model", ""),
            "resolution": record.get("resolution", ""),
            "seed": record.get("seed", ""),
            "config_hash": record.get("config_hash", ""),
            "includes_warm_start": (
                bool(accounting.get("includes_warm_start", False))
                if isinstance(accounting, Mapping)
                else False
            ),
        }
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


def _compose_resolution_model_config(
    *,
    root_path: Path,
    dataset: str,
    model_name: str,
    experiment: str,
    overrides: list[str] | None,
) -> Config:
    base_model = "l2l" if model_name == "full" else model_name
    config = compose_config(
        root=root_path,
        dataset=dataset,
        model=base_model,
        experiment=experiment,
        overrides=overrides,
    )
    if model_name == "full":
        config.setdefault("model", {}).setdefault("coupling", {})
        config["model"]["coupling"].setdefault("backend", "gnn")
    return config


def _assert_coarse_factor_legal(resolution: int, coarse_factor: int, n_samples: int) -> None:
    if resolution % coarse_factor != 0:
        raise ValueError(
            f"resolution={resolution} must be divisible by coarse_factor={coarse_factor}."
        )
    coarse_features = (resolution // coarse_factor) ** 2
    if coarse_features >= n_samples:
        raise ValueError(
            "Coarse factor invariant violated for resolution sweep: "
            f"(D/c)^2={coarse_features} must be < m={n_samples}; "
            f"got D={resolution}, c={coarse_factor}."
        )


def _dry_run_line(run: FactorialRun) -> str:
    flags = " ".join(f"{key}={str(value).lower()}" for key, value in run.mechanisms.items())
    baseline = f" baseline_model={run.baseline_model}" if run.baseline_model else ""
    return (
        f"{run.run_id} seed={run.seed} resolution={run.resolution}{baseline} "
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
    required = (
        "config.yaml",
        "metrics.json",
        "runtime.json",
        "predictions_test.npz",
    )
    return all((run_path / filename).is_file() for filename in required)


def _flatten_record(record: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_id": record["run_id"],
        "config_hash": record["config_hash"],
        "run_dir": record["run_dir"],
        "seed": record["seed"],
        "resolution": record["resolution"],
        "baseline_model": record.get("baseline_model", ""),
    }
    for key, value in record["mechanisms"].items():
        row[f"mechanism_{key}"] = value
    for key, value in record["metrics"].items():
        row[f"metric_{key}"] = value
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


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value
