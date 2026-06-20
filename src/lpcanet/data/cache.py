"""Deterministic content-hash helpers for generated dataset caches."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def content_hash(payload: Mapping[str, Any], *, length: int = 16) -> str:
    """Hash JSON-serializable generation metadata deterministically."""
    serialized = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:length]


def dataset_cache_path(
    *,
    root: str | Path,
    dataset_name: str,
    resolution: int,
    n_samples: int,
    seed: int,
    suffix: str = ".npz",
) -> Path:
    """Return deterministic generated-data cache path keyed by content hash."""
    payload = {
        "dataset_name": dataset_name,
        "resolution": int(resolution),
        "n_samples": int(n_samples),
        "seed": int(seed),
    }
    digest = content_hash(payload)
    return Path(root) / f"{dataset_name}_{resolution}_{digest}{suffix}"
