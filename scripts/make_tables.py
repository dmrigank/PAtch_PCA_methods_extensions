#!/usr/bin/env python
"""Render paper-ready tables from LPCANet result records."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lpcanet.utils.paths import ensure_dir

MECHANISM_KEYS = ("two_scale", "coupling", "in_loop_loss")
STAGE_COLUMNS = ("pca_fit", "latent_transform", "nn_train", "inference", "end_to_end")
PREFERRED_METRICS = (
    "mre",
    "mse",
    "mae",
    "ssim_mean",
    "relative_spectrum_error",
    "interface_jump",
    "interface_flux_jump",
    "poisson_residual_rms",
    "darcy_residual_rms",
)
METHOD_LABELS = {
    "global_pca": "Global PCA-Net",
    "l2g": "L2G PCA-Net",
    "l2l": "Plain L2L",
    "l2l_overlap": "L2L + Overlap",
    "l2l_refinement": "L2L + RefinementNet",
    "fno": "FNO",
    "full": "Full model",
}
METHOD_ORDER = [
    "Plain L2L",
    "two_scale",
    "coupling",
    "in_loop_loss",
    "two_scale+coupling",
    "two_scale+in_loop_loss",
    "coupling+in_loop_loss",
    "Full model",
    "L2L + Overlap",
    "L2L + RefinementNet",
]


def make_tables(
    *,
    results_dir: str | Path = "results",
    out_dir: str | Path = "results/tables",
    dataset: str = "poisson",
) -> list[Path]:
    """Create the headline factorial table and resolution cost table."""
    results_path = Path(results_dir)
    output_dir = ensure_dir(out_dir)
    records = load_result_records(results_path)
    if records.empty:
        raise FileNotFoundError(f"No results.jsonl records found below {results_path}.")

    written: list[Path] = []
    headline_numeric, headline_formatted = headline_factorial_table(records, dataset=dataset)
    numeric_path = output_dir / "headline_factorial_summary.csv"
    formatted_path = output_dir / "headline_factorial_table.csv"
    tex_path = output_dir / "headline_factorial_table.tex"
    headline_numeric.to_csv(numeric_path, index=False)
    headline_formatted.to_csv(formatted_path, index=False)
    tex_path.write_text(headline_formatted.to_latex(index=False, escape=True), encoding="utf-8")
    written.extend([numeric_path, formatted_path, tex_path])

    cost_numeric, cost_formatted = resolution_cost_table(results_path, records)
    if not cost_numeric.empty:
        cost_numeric_path = output_dir / "resolution_stage_costs_summary.csv"
        cost_formatted_path = output_dir / "resolution_stage_costs_table.csv"
        cost_tex_path = output_dir / "resolution_stage_costs_table.tex"
        cost_numeric.to_csv(cost_numeric_path, index=False)
        cost_formatted.to_csv(cost_formatted_path, index=False)
        cost_tex_path.write_text(cost_formatted.to_latex(index=False, escape=True), encoding="utf-8")
        written.extend([cost_numeric_path, cost_formatted_path, cost_tex_path])
    return written


def load_result_records(results_dir: str | Path) -> pd.DataFrame:
    """Load nested JSONL result records and enrich them with config metadata."""
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(results_dir).rglob("results.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                rows.append(_flatten_record(record, source_path=path))
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["method"] = frame.apply(_method_label, axis=1)
    frame["method_order"] = frame["method"].map({name: idx for idx, name in enumerate(METHOD_ORDER)})
    frame["method_order"] = frame["method_order"].fillna(10_000).astype(int)
    return frame


def headline_factorial_table(records: pd.DataFrame, *, dataset: str = "poisson") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize the 2x2x2 mechanism factorial plus comparison baselines."""
    frame = records.copy()
    if "dataset" in frame.columns:
        frame = frame[frame["dataset"].fillna("").str.lower().eq(dataset.lower())]
    if frame.empty:
        raise ValueError(f"No result records found for dataset={dataset!r}.")

    metric_cols = _ordered_existing_columns(frame, [f"metric_{name}" for name in PREFERRED_METRICS])
    timing_cols = _ordered_existing_columns(frame, [f"timing_{name}" for name in STAGE_COLUMNS])
    value_cols = metric_cols + timing_cols
    group_cols = ["method", "method_order"] + [f"mechanism_{key}" for key in MECHANISM_KEYS]
    summary = _summarize(frame, group_cols, value_cols).sort_values(["method_order", "method"])
    summary.insert(1, "n_seeds", frame.groupby("method")["seed"].nunique().reindex(summary["method"]).to_numpy())

    full_row = summary[summary["method"].eq("Full model")]
    if not full_row.empty:
        full = full_row.iloc[0]
        for col in [
            "metric_mre_mean",
            "metric_relative_spectrum_error_mean",
            "metric_interface_jump_mean",
            "metric_interface_flux_jump_mean",
            "timing_end_to_end_mean",
        ]:
            if col in summary.columns:
                name = col.replace("metric_", "").replace("timing_", "").replace("_mean", "")
                summary[f"{name}_delta_vs_full"] = summary[col] - float(full[col])

    formatted = _format_summary_table(summary)
    return summary, formatted


def resolution_cost_table(
    results_dir: str | Path,
    records: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize resolution-sweep stage timings."""
    stage_csvs = sorted(Path(results_dir).rglob("stage_costs.csv"))
    rows: list[pd.DataFrame] = []
    for path in stage_csvs:
        frame = pd.read_csv(path)
        if not frame.empty:
            rows.append(frame)
    if rows:
        costs = pd.concat(rows, ignore_index=True)
    elif records is not None and not records.empty:
        stage_cols = _ordered_existing_columns(records, [f"timing_{name}" for name in STAGE_COLUMNS])
        if not stage_cols:
            return pd.DataFrame(), pd.DataFrame()
        costs = records[["method", "resolution", "seed", *stage_cols]].rename(
            columns={f"timing_{name}": name for name in STAGE_COLUMNS}
        )
    else:
        return pd.DataFrame(), pd.DataFrame()

    if "baseline_model" in costs.columns and "method" not in costs.columns:
        costs["method"] = costs["baseline_model"].map(METHOD_LABELS).fillna(costs["baseline_model"])
    if "method" not in costs.columns:
        costs["method"] = ""
    value_cols = _ordered_existing_columns(costs, list(STAGE_COLUMNS))
    if not value_cols:
        return pd.DataFrame(), pd.DataFrame()
    summary = _summarize(costs, ["resolution", "method"], value_cols).sort_values(["resolution", "method"])
    formatted = _format_summary_table(summary)
    return summary, formatted


def _flatten_record(record: Mapping[str, Any], *, source_path: Path) -> dict[str, Any]:
    run_dir = Path(str(record.get("run_dir", "")))
    config = _load_run_config(run_dir)
    row: dict[str, Any] = {
        "source_path": str(source_path),
        "run_id": record.get("run_id", ""),
        "config_hash": record.get("config_hash", ""),
        "run_dir": str(run_dir),
        "seed": record.get("seed", np.nan),
        "resolution": record.get("resolution", np.nan),
        "baseline_model": record.get("baseline_model", ""),
        "dataset": _dataset_name(config, run_dir),
    }
    mechanisms = dict(record.get("mechanisms", {}))
    for key in MECHANISM_KEYS:
        row[f"mechanism_{key}"] = bool(mechanisms.get(key, False))
    for key, value in dict(record.get("metrics", {})).items():
        row[f"metric_{key}"] = _float_or_nan(value)
    for key, value in dict(record.get("timings", {})).items():
        row[f"timing_{key}"] = _float_or_nan(value)
    return row


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    for path in [run_dir / "config.yaml", run_dir / "resolved_configs" / "config.yaml"]:
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
            return loaded if isinstance(loaded, dict) else {}
    return {}


def _dataset_name(config: Mapping[str, Any], run_dir: Path) -> str:
    dataset = config.get("dataset", {})
    if isinstance(dataset, Mapping) and dataset.get("name"):
        return str(dataset["name"]).lower()
    lowered = str(run_dir).lower()
    if "darcy" in lowered:
        return "darcy"
    if "poisson" in lowered:
        return "poisson"
    return ""


def _method_label(row: pd.Series) -> str:
    baseline = str(row.get("baseline_model", "") or "")
    if baseline in METHOD_LABELS:
        return METHOD_LABELS[baseline]
    flags = {key: bool(row.get(f"mechanism_{key}", False)) for key in MECHANISM_KEYS}
    if all(flags.values()):
        return "Full model"
    if not any(flags.values()):
        return "Plain L2L"
    return "+".join(key for key in MECHANISM_KEYS if flags[key])


def _summarize(frame: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    numeric = frame.copy()
    for col in value_cols:
        numeric[col] = pd.to_numeric(numeric[col], errors="coerce")
    grouped = numeric.groupby(group_cols, dropna=False, as_index=False)
    summary = grouped[value_cols].agg(["mean", "std"])
    summary.columns = [
        "_".join(piece for piece in col if piece)
        if isinstance(col, tuple)
        else str(col)
        for col in summary.columns
    ]
    return summary.fillna({col: 0.0 for col in summary.columns if col.endswith("_std")})


def _format_summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    formatted = summary.copy()
    mean_cols = [col for col in summary.columns if col.endswith("_mean")]
    for mean_col in mean_cols:
        std_col = f"{mean_col[:-5]}_std"
        display_col = _display_name(mean_col[:-5])
        if std_col in summary.columns:
            formatted[display_col] = [
                _format_mean_std(mean, std)
                for mean, std in zip(summary[mean_col], summary[std_col])
            ]
            formatted = formatted.drop(columns=[mean_col, std_col])
    formatted = formatted.rename(
        columns={
            "method": "Method",
            "resolution": "Resolution",
            "n_seeds": "Seeds",
            "mechanism_two_scale": "Two-scale",
            "mechanism_coupling": "Coupling",
            "mechanism_in_loop_loss": "In-loop loss",
        }
    )
    hidden = [col for col in formatted.columns if col == "method_order"]
    return formatted.drop(columns=hidden)


def _format_mean_std(mean: Any, std: Any) -> str:
    if pd.isna(mean):
        return ""
    mean_value = float(mean)
    std_value = 0.0 if pd.isna(std) else float(std)
    return f"{mean_value:.4g} +/- {std_value:.2g}"


def _display_name(column: str) -> str:
    column = column.removeprefix("metric_").removeprefix("timing_")
    labels = {
        "mre": "MRE",
        "mse": "MSE",
        "mae": "MAE",
        "ssim_mean": "SSIM",
        "relative_spectrum_error": "Spectral error",
        "interface_jump": "Interface value jump",
        "interface_flux_jump": "Interface flux jump",
        "poisson_residual_rms": "Poisson residual RMS",
        "darcy_residual_rms": "Darcy residual RMS",
        "pca_fit": "PCA-fit (s)",
        "latent_transform": "Latent-transform (s)",
        "nn_train": "NN-train (s)",
        "inference": "Inference (s)",
        "end_to_end": "End-to-end (s)",
    }
    if column.endswith("_delta_vs_full"):
        base = column.removesuffix("_delta_vs_full")
        return f"{_display_name(base)} delta vs full"
    return labels.get(column, column.replace("_", " ").title())


def _ordered_existing_columns(frame: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [col for col in columns if col in frame.columns]


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="results/tables")
    parser.add_argument("--dataset", default="poisson")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = make_tables(results_dir=args.results_dir, out_dir=args.out_dir, dataset=args.dataset)
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
