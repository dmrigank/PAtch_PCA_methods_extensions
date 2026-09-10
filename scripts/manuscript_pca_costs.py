"""Shared PCA/SVD substage extraction for manuscript cost figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PCA_STAGE_SPECS = (
    ("pca_cost_input", "Input PCA", "#4C78A8"),
    ("pca_cost_output", "Output PCA", "#72B7B2"),
    ("pca_cost_coarse", "Coarse global SVD", "#F58518"),
    ("pca_cost_residual", "Residual output PCA", "#E45756"),
    ("pca_cost_other", "PCA fit overhead", "#B9B9B9"),
)

_SOURCE_COLUMNS = {
    "input": "pca_timings_fit_input_pca",
    "output": "pca_timings_fit_output_pca",
    "coarse": "pca_timings_fit_coarse_svd",
    "residual": "pca_timings_fit_output_residual_pca",
}


def add_pca_cost_substages(
    records: pd.DataFrame,
    *,
    results_path: Path,
) -> pd.DataFrame:
    """Add nonoverlapping PCA-fit substages to each result record.

    Newer roll-ups expose timings directly. Older manuscript studies retain
    them on the serialized PCA encoder, which is used as a compatibility
    fallback.
    """
    enriched = records.copy()
    for target in (*[spec[0] for spec in PCA_STAGE_SPECS], "pca_cost_total"):
        enriched[target] = np.nan

    for index, row in enriched.iterrows():
        timings = _row_timings(row)
        if not timings:
            timings = _encoder_timings(row, results_path=results_path)
        if not timings:
            continue

        input_time = _nonnegative(timings.get("fit_input_pca"))
        output_aggregate = _nonnegative(timings.get("fit_output_pca"))
        coarse_time = _nonnegative(timings.get("fit_coarse_svd"))
        residual_time = _nonnegative(timings.get("fit_output_residual_pca"))
        is_two_scale = coarse_time > 0.0 or residual_time > 0.0
        output_time = 0.0 if is_two_scale else output_aggregate
        total = _nonnegative(row.get("timing_pca_fit"))
        if total == 0.0:
            total = input_time + output_time + coarse_time + residual_time
        accounted = input_time + output_time + coarse_time + residual_time

        enriched.at[index, "pca_cost_input"] = input_time
        enriched.at[index, "pca_cost_output"] = output_time
        enriched.at[index, "pca_cost_coarse"] = coarse_time
        enriched.at[index, "pca_cost_residual"] = residual_time
        enriched.at[index, "pca_cost_other"] = max(total - accounted, 0.0)
        enriched.at[index, "pca_cost_total"] = total
    return enriched


def pca_cost_table(
    records: pd.DataFrame,
    *,
    group_columns: list[str],
    label_columns: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Aggregate PCA substages as mean and sample standard deviation."""
    cost_columns = [spec[0] for spec in PCA_STAGE_SPECS] + ["pca_cost_total"]
    available = records.dropna(subset=["pca_cost_total"])
    grouped = available.groupby(group_columns, as_index=False)[cost_columns].agg(
        ["mean", "std"]
    )
    grouped.columns = [
        "_".join(str(piece) for piece in column if piece)
        if isinstance(column, tuple)
        else str(column)
        for column in grouped.columns
    ]
    if label_columns:
        grouped = grouped.rename(columns=label_columns)
    return grouped


def _row_timings(row: pd.Series) -> dict[str, float]:
    timings: dict[str, float] = {}
    for key, column in _SOURCE_COLUMNS.items():
        value = row.get(column)
        if value is not None and not pd.isna(value):
            timings[f"fit_{key}_pca" if key != "coarse" else "fit_coarse_svd"] = (
                float(value)
            )
    if "fit_residual_pca" in timings:
        timings["fit_output_residual_pca"] = timings.pop("fit_residual_pca")
    return timings


def _encoder_timings(row: pd.Series, *, results_path: Path) -> dict[str, float]:
    path = _resolve_run_dir(row, results_path=results_path) / "pca_encoder.joblib"
    if not path.is_file():
        return {}
    import joblib

    encoder: Any = joblib.load(path)
    raw = getattr(encoder, "timings", {})
    if not isinstance(raw, dict):
        return {}
    return {
        str(key): float(value)
        for key, value in raw.items()
        if isinstance(value, (int, float))
    }


def _resolve_run_dir(row: pd.Series, *, results_path: Path) -> Path:
    run_dir = Path(str(row.get("run_dir", "")))
    candidates = (run_dir, results_path / run_dir, results_path.parents[1] / run_dir)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return run_dir


def _nonnegative(value: Any) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return max(float(value), 0.0)
