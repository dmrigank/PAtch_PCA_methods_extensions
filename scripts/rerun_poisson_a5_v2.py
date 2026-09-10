#!/usr/bin/env python
"""Rerun the six PDE-active Poisson A5 cells with residual convention v2."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lpcanet.metrics.evaluate import evaluate_run
from lpcanet.metrics.residuals import (
    PDE_METRICS_VERSION,
    POISSON_RESIDUAL_CONVENTION,
)
from lpcanet.train.checkpointing import save_json
from lpcanet.train.experiment import train_from_config
from lpcanet.train.factorial import (
    FactorialRun,
    build_result_record,
    config_hash_for,
    write_rollup,
    write_stage_cost_table,
)
from lpcanet.utils.config import load_config, save_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("paper_results/physics_loss_ablation"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("paper_results/physics_loss_ablation_v2"),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Train the six corrected cells. The default only lists them.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    source_records = _read_jsonl(source_root / "results.jsonl")
    a5_records = [
        record
        for record in source_records
        if str(record.get("baseline_model", "")) == "a5_full_pde"
    ]
    if len(a5_records) != 6:
        raise ValueError(f"Expected six A5 records, found {len(a5_records)}.")

    generated_dir = output_root / "generated_configs"
    plans: list[tuple[dict[str, Any], Path, Path]] = []
    for record in sorted(a5_records, key=lambda row: (int(row["resolution"]), int(row["seed"]))):
        source_run = Path(str(record["run_dir"])).resolve()
        relative = source_run.relative_to(source_root)
        target_run = output_root / relative
        config = _corrected_config(source_run, target_run)
        config_path = generated_dir / f"a5_r{record['resolution']}_seed{record['seed']}.yaml"
        plans.append((config, config_path, target_run))
        print(
            f"r={record['resolution']} seed={record['seed']}: "
            f"{source_run} -> {target_run}"
        )

    if not args.apply:
        print("Dry run complete. Pass --apply to train the six version-2 A5 cells.")
        return

    generated_dir.mkdir(parents=True, exist_ok=True)
    for config, config_path, target_run in plans:
        if (target_run / "metrics.json").is_file() and not args.overwrite:
            metrics = evaluate_run(target_run)
            if int(metrics.get("pde_metrics_version", 0)) != PDE_METRICS_VERSION:
                raise AssertionError(f"A5 run lacks PDE metric provenance: {target_run}")
            print(f"Reusing completed corrected run: {target_run}")
            continue
        save_config(config, config_path.parent, filename=config_path.name)
        trained = train_from_config(
            config_path,
            output_dir=target_run,
            device=args.device,
            overwrite=args.overwrite,
        )
        metrics = evaluate_run(trained)
        if int(metrics.get("pde_metrics_version", 0)) != PDE_METRICS_VERSION:
            raise AssertionError(f"A5 run lacks PDE metric provenance: {trained}")

    _write_combined_study(source_root, output_root)
    print(f"Corrected A5 study complete: {output_root}")


def _corrected_config(source_run: Path, target_run: Path) -> dict[str, Any]:
    config = load_config(source_run / "config.yaml")
    config.setdefault("experiment", {})["output_dir"] = str(target_run)
    config["experiment"]["name"] = f"{config['experiment']['name']}_pde_v2"
    config.setdefault("dataset", {})[
        "poisson_convention"
    ] = POISSON_RESIDUAL_CONVENTION
    config.setdefault("resolved", {}).update(
        {
            "source_pre_v2_run_dir": str(source_run),
            "pde_metrics_version": PDE_METRICS_VERSION,
            "poisson_residual_convention": POISSON_RESIDUAL_CONVENTION,
        }
    )
    return config


def _write_combined_study(source_root: Path, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    main = _replace_a5_records(
        _read_jsonl(source_root / "results.jsonl"),
        source_root,
        output_root,
    )
    screen = _replace_a5_records(
        _read_jsonl(source_root / "screen" / "results.jsonl"),
        source_root,
        output_root,
    )
    _write_jsonl(main, output_root / "results.jsonl")
    _write_jsonl(screen, output_root / "screen" / "results.jsonl")
    write_rollup(main, output_root)
    write_rollup(screen, output_root / "screen")
    write_stage_cost_table(main, output_root)
    write_stage_cost_table(screen, output_root / "screen")

    selection = _load_json(source_root / "selection.json")
    selection["pde_metrics_version"] = PDE_METRICS_VERSION
    selection["poisson_residual_convention"] = POISSON_RESIDUAL_CONVENTION
    selection["source_pre_v2_study"] = str(source_root)
    pde_record = next(
        record
        for record in screen
        if "a5_full_pde" in str(record.get("baseline_model", ""))
    )
    pde_validation = pde_record.get("validation_metrics", {})
    reference_mre = float(selection["reconstruction_validation_mre"])
    pde_mre = float(pde_validation["mre"])
    selection["terms"]["pde_residual"].update(
        {
            "validation_mre": pde_mre,
            "validation_target_metric": float(
                pde_validation["poisson_residual_rms"]
            ),
            "guardrail_passed": pde_mre
            <= reference_mre * (1.0 + float(selection["mre_guardrail_fraction"])),
        }
    )
    save_json(selection, output_root / "selection.json")


def _replace_a5_records(
    records: list[dict[str, Any]],
    source_root: Path,
    output_root: Path,
) -> list[dict[str, Any]]:
    replaced: list[dict[str, Any]] = []
    for original in records:
        baseline = str(original.get("baseline_model", ""))
        if "a5_full_pde" not in baseline:
            replaced.append(copy.deepcopy(original))
            continue
        source_run = Path(str(original["run_dir"])).resolve()
        target_run = output_root / source_run.relative_to(source_root)
        config = load_config(target_run / "config.yaml")
        run = FactorialRun(
            run_id=str(config.get("experiment", {}).get("name", original["run_id"])),
            config=config,
            config_hash=config_hash_for(config),
            seed=int(original["seed"]),
            mechanisms={
                str(key): bool(value)
                for key, value in config.get("mechanisms", {}).items()
            },
            resolution=int(original["resolution"]),
            run_dir=target_run,
            baseline_model=baseline,
        )
        metrics = evaluate_run(target_run)
        replaced.append(build_result_record(run, metrics=metrics, run_dir=target_run))
    return replaced


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return dict(json.load(handle))


if __name__ == "__main__":
    main()
