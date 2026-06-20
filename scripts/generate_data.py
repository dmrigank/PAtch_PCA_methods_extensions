#!/usr/bin/env python
"""Generate deterministic Poisson or Darcy datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lpcanet.data.generation import generate_dataset


def _parse_overrides(tokens: list[str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for token in tokens:
        if "=" not in token:
            raise ValueError(f"Expected key=value, got {token!r}.")
        key, raw_value = token.split("=", 1)
        values[key.replace("-", "_")] = yaml.safe_load(raw_value)
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("overrides", nargs="*", help="Values such as dataset=poisson resolution=256.")
    parser.add_argument("--dataset", choices=["poisson", "darcy"])
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--n-samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--tau", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--output")
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = _parse_overrides(args.overrides)
    dataset = str(args.dataset or values.get("dataset", "poisson"))
    resolution = int(args.resolution or values.get("resolution", 128))
    n_samples = int(args.n_samples or values.get("n_samples", 10_000))
    seed = int(args.seed if args.seed is not None else values.get("seed", 0))
    alpha = float(args.alpha if args.alpha is not None else values.get("alpha", 2.0))
    tau = float(args.tau if args.tau is not None else values.get("tau", 3.0))
    batch_size = int(args.batch_size or values.get("batch_size", 32))
    workers = int(args.workers or values.get("workers", 1))
    output = args.output or values.get(
        "output",
        f"data/processed/{dataset}_{resolution}.npz",
    )
    compress = bool(args.compress or values.get("compress", False))
    overwrite = bool(args.overwrite or values.get("overwrite", False))
    path = generate_dataset(
        dataset=dataset,
        output=output,
        resolution=resolution,
        n_samples=n_samples,
        seed=seed,
        alpha=alpha,
        tau=tau,
        batch_size=batch_size,
        workers=workers,
        compress=compress,
        overwrite=overwrite,
    )
    print(f"Wrote dataset: {path}")
    print(f"Metadata: {path.with_suffix(path.suffix + '.json')}")


if __name__ == "__main__":
    main()
