#!/usr/bin/env python
"""Compose a YAML config and run one PCA-Net experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lpcanet.metrics.evaluate import evaluate_run
from lpcanet.train.experiment import train_from_config
from lpcanet.utils.config import compose_config, save_config
from lpcanet.utils.paths import ensure_dir


def run_experiment(
    *,
    dataset: str = "poisson",
    model: str = "l2l",
    experiment: str = "single",
    overrides: list[str] | None = None,
    limit_samples: int | None = None,
    epochs: int | None = None,
    output_dir: str | None = None,
    device: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> Path:
    """Compose, save, train, and evaluate a single experiment."""
    normalized_overrides = list(overrides or [])
    config = compose_config(
        root=ROOT,
        dataset=dataset,
        model=model,
        experiment=experiment,
        overrides=normalized_overrides,
    )
    if output_dir is not None:
        config.setdefault("experiment", {})["output_dir"] = output_dir
    if epochs is not None:
        config.setdefault("training", {})["epochs"] = int(epochs)
    if device is not None:
        config.setdefault("training", {})["device"] = device

    run_dir = Path(config.get("experiment", {}).get("output_dir", "results/run")).expanduser()
    generated_dir = ensure_dir(run_dir / "resolved_configs")
    resolved_config = save_config(config, generated_dir, filename="config.yaml")
    if dry_run:
        print(resolved_config)
        return resolved_config

    start = perf_counter()
    trained_run_dir = train_from_config(
        resolved_config,
        limit_samples=limit_samples,
        epochs=epochs,
        output_dir=output_dir,
        device=device,
        overwrite=overwrite,
    )
    metrics = evaluate_run(trained_run_dir)
    elapsed = perf_counter() - start
    print(f"Run complete: {trained_run_dir} ({elapsed:.2f}s)")
    for key in sorted(metrics):
        print(f"{key}: {metrics[key]}")
    return trained_run_dir


def _split_selectors(tokens: list[str]) -> tuple[dict[str, str], list[str]]:
    selectors: dict[str, str] = {}
    overrides: list[str] = []
    for token in tokens:
        if "=" not in token:
            raise ValueError(f"Expected key=value override, got {token!r}.")
        key, value = token.split("=", 1)
        if key in {"dataset", "model", "experiment"}:
            selectors[key] = value
        elif key == "seed":
            overrides.append(f"experiment.seed={value}")
        elif key == "resolution":
            overrides.append(f"dataset.grid_size={value}")
        else:
            overrides.append(token)
    return selectors, overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("overrides", nargs="*", help="Config selectors/overrides as key=value.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selectors, overrides = _split_selectors(args.overrides)
    run_experiment(
        dataset=args.dataset or selectors.get("dataset", "poisson"),
        model=args.model or selectors.get("model", "l2l"),
        experiment=args.experiment or selectors.get("experiment", "single"),
        overrides=overrides,
        limit_samples=args.limit_samples,
        epochs=args.epochs,
        output_dir=args.output_dir,
        device=args.device,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
