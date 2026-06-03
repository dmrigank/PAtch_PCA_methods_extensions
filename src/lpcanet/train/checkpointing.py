"""Checkpoint and run-artifact helpers."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from lpcanet.utils.paths import normalize_path


def save_model(model: torch.nn.Module, path: str | Path) -> Path:
    """Save a model state dict."""
    out_path = normalize_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    return out_path


def save_json(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Save JSON payload."""
    out_path = normalize_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return out_path


def save_metrics(metrics: Mapping[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    """Save metrics as JSON and CSV."""
    out_dir = normalize_path(output_dir)
    json_path = save_json(metrics, out_dir / "metrics.json")
    csv_path = out_dir / "metrics.csv"
    pd.DataFrame([dict(metrics)]).to_csv(csv_path, index=False)
    return json_path, csv_path


def save_environment(path: str | Path) -> Path:
    """Save minimal Python/package environment information."""
    out_path = normalize_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"python: {sys.version.replace(chr(10), ' ')}",
        f"executable: {sys.executable}",
        f"platform: {platform.platform()}",
    ]
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.stdout:
            lines.append("")
            lines.append("pip freeze:")
            lines.extend(result.stdout.strip().splitlines())
    except Exception as exc:
        lines.append(f"pip freeze failed: {exc}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
