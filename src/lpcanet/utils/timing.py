"""Timing helpers for experiment runtime accounting."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from types import TracebackType
from typing import Any

PCA_FIT = "pca_fit"
LATENT_TRANSFORM = "latent_transform"
NN_TRAIN = "nn_train"
INFERENCE = "inference"
END_TO_END = "end_to_end"

STAGE_KEYS = (PCA_FIT, LATENT_TRANSFORM, NN_TRAIN, INFERENCE, END_TO_END)


class Timer:
    """Context manager that records elapsed wall-clock seconds."""

    def __init__(self, name: str | None = None, collector: TimingCollector | None = None) -> None:
        self.name = name
        self.collector = collector
        self.elapsed: float | None = None
        self._start: float | None = None

    def __enter__(self) -> Timer:
        self._start = perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        if self._start is None:
            raise RuntimeError("Timer exited before it was started.")
        self.elapsed = perf_counter() - self._start
        if self.name is not None and self.collector is not None:
            self.collector.times[self.name] = self.elapsed


class TimingCollector:
    """Collect named elapsed times and save them as JSON."""

    def __init__(self) -> None:
        self.times: dict[str, float] = {}

    def time(self, name: str) -> Timer:
        """Create a named timer context."""
        return Timer(name=name, collector=self)

    def add(self, name: str, elapsed: float) -> None:
        """Record an elapsed duration in seconds."""
        self.times[name] = float(elapsed)

    def canonical(self) -> dict[str, float]:
        """Return canonical stage timings with missing stages filled as zero."""
        return {stage: float(self.times.get(stage, 0.0)) for stage in STAGE_KEYS}

    def save(self, path: str | Path) -> Path:
        """Save collected times to ``runtime.json`` or the provided JSON path."""
        runtime_path = Path(path).expanduser()
        if runtime_path.suffix.lower() != ".json":
            runtime_path = runtime_path / "runtime.json"
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "stages": self.canonical(),
            "times": self.times,
        }
        with runtime_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return runtime_path
