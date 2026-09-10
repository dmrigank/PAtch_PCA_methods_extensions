#!/usr/bin/env python
"""Audit or repair Poisson residual metrics in archived result studies."""

from __future__ import annotations

import argparse
from pathlib import Path

from lpcanet.metrics.migration import (
    migrate_poisson_residuals,
    summarize_actions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("paper_results"),
        help="A study directory or a root containing multiple studies.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write corrected metrics and rollups. The default is a dry run.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Do not regenerate validation residuals.",
    )
    parser.add_argument("--device", default="auto", help="Inference device.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    actions = migrate_poisson_residuals(
        args.results_root,
        apply=args.apply,
        include_validation=not args.skip_validation,
        device=args.device,
    )
    for key, count in sorted(summarize_actions(actions).items()):
        print(f"{key}: {count}")
    mode = "applied" if args.apply else "dry-run"
    print(f"Poisson residual migration {mode}: {len(actions)} records")
    for action in actions:
        if action["recovery"] in {"retrain", "blocked"}:
            print(f"{action['recovery']}: {action['run_dir']} ({action['reason']})")


if __name__ == "__main__":
    main()
