"""Configuration helpers for YAML-driven experiments."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

import yaml

Config = dict[str, Any]


def load_config(path: str | Path) -> Config:
    """Load a YAML configuration file into a dictionary."""
    config_path = Path(path).expanduser()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML mapping in {config_path}, got {type(data).__name__}.")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Config:
    """Recursively merge config dictionaries without mutating either input."""
    merged: Config = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def compose_config(
    *,
    root: str | Path = ".",
    dataset: str = "poisson",
    model: str = "l2l",
    experiment: str = "single",
    overrides: Sequence[str] | None = None,
) -> Config:
    """Compose base + dataset + model + experiment YAML configs."""
    root_path = Path(root).expanduser()
    config_root = root_path / "configs"
    pieces = [
        config_root / "base.yaml",
        config_root / "datasets" / f"{dataset}.yaml",
        config_root / "models" / f"{model}.yaml",
        config_root / "experiments" / f"{experiment}.yaml",
    ]
    composed: Config = {}
    for piece in pieces:
        if not piece.exists():
            raise FileNotFoundError(f"Missing config piece: {piece}")
        composed = deep_merge(composed, load_config(piece))
    apply_overrides(composed, overrides)
    composed.setdefault("resolved", {})
    composed["resolved"].update(
        {
            "dataset_config": dataset,
            "model_config": model,
            "experiment_config": experiment,
        }
    )
    return composed


def save_config(config: Mapping[str, Any], output_dir: str | Path, filename: str = "config.yaml") -> Path:
    """Save a resolved configuration dictionary in an output directory."""
    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    with out_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False)
    return out_path


def apply_overrides(config: Config, overrides: Sequence[str] | None) -> Config:
    """Apply simple ``key.subkey=value`` command-line overrides in place.

    Values are parsed with ``yaml.safe_load`` so strings such as ``true``, ``2``,
    ``1.0e-3``, and ``[a, b]`` become native Python values.
    """
    if not overrides:
        return config

    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must have the form key.subkey=value: {override!r}")
        key_path, raw_value = override.split("=", 1)
        if not key_path:
            raise ValueError(f"Override key cannot be empty: {override!r}")
        set_nested(config, key_path.split("."), yaml.safe_load(raw_value))
    return config


def set_nested(config: MutableMapping[str, Any], keys: Sequence[str], value: Any) -> None:
    """Assign ``value`` under a nested path, creating intermediate dicts."""
    if not keys:
        raise ValueError("Nested key path cannot be empty.")

    current: MutableMapping[str, Any] = config
    for key in keys[:-1]:
        if not key:
            raise ValueError(f"Nested key path contains an empty component: {keys!r}")
        next_value = current.get(key)
        if next_value is None:
            next_value = {}
            current[key] = next_value
        if not isinstance(next_value, MutableMapping):
            raise TypeError(f"Cannot set nested key through non-mapping value at {key!r}.")
        current = next_value

    final_key = keys[-1]
    if not final_key:
        raise ValueError(f"Nested key path contains an empty final component: {keys!r}")
    current[final_key] = value
