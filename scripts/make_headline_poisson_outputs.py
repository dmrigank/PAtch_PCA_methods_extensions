#!/usr/bin/env python
"""Render manuscript outputs for the five-seed Poisson 128 headline study."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from make_resolution_sweep_outputs import (  # noqa: E402
    _apply_style,
    _format_mean_std,
    _plot_sample_pdf,
    _plot_sample_spectrum,
    _prediction_path,
    _save_figure,
    _write_table_pair,
)
from manuscript_pca_costs import (  # noqa: E402
    PCA_STAGE_SPECS,
    add_pca_cost_substages,
)

from lpcanet.metrics.interface import (  # noqa: E402
    interface_flux_jump,
    interface_flux_trace_error,
    interface_value_jump,
    interface_value_trace_error,
)

METHOD_ORDER = (
    "global_pca",
    "plain_l2l",
    "overlap_l2l",
    "l2l_refinement",
    "fno",
    "two_scale",
    "two_scale_interface",
)
METHOD_LABELS = {
    "global_pca": "Global PCA-Net",
    "plain_l2l": "Plain L2L",
    "overlap_l2l": "L2L + overlap",
    "l2l_refinement": "L2L + RefinementNet",
    "fno": "FNO",
    "two_scale": "Two-scale",
    "two_scale_interface": "Two-scale + interface",
}
METHOD_COLORS = {
    "global_pca": "#9C755F",
    "plain_l2l": "#6F6F6F",
    "overlap_l2l": "#4C78A8",
    "l2l_refinement": "#B279A2",
    "fno": "#E45756",
    "two_scale": "#F58518",
    "two_scale_interface": "#54A24B",
}
STAGE_SPECS = (
    ("timing_pca_fit", "PCA fit", "#4C78A8"),
    ("timing_latent_transform", "Latent transform", "#72B7B2"),
    ("timing_nn_train", "NN training", "#F58518"),
    ("timing_inference", "Inference", "#E45756"),
)
NUMERIC_COLUMNS = (
    "metric_mre",
    "metric_ssim_mean",
    "metric_mse",
    "metric_mae",
    "metric_pca_oracle_mre",
    "metric_pca_oracle_ssim_mean",
    "metric_relative_spectrum_error",
    "metric_interface_jump",
    "metric_interface_flux_jump",
    "metric_interface_value_trace_error",
    "metric_interface_flux_trace_error",
    "metric_poisson_residual_mean_abs",
    "metric_poisson_residual_rms",
    "timing_pca_fit",
    "timing_latent_transform",
    "timing_nn_train",
    "timing_inference",
    "timing_end_to_end",
    "invocation_timing_nn_train",
    "invocation_timing_end_to_end",
)
FULL_TABLE_SPECS = (
    ("metric_mre", "MRE", "percent"),
    ("metric_pca_oracle_mre", "PCA-oracle MRE", "percent"),
    ("metric_ssim_mean", "SSIM", "ssim"),
    ("metric_mse", "MSE", "scientific"),
    ("metric_mae", "MAE", "scientific"),
    ("metric_interface_jump", "Value jump", "scientific"),
    ("metric_interface_flux_jump", "Flux jump", "scientific"),
    ("metric_interface_value_trace_error", "Value trace error", "scientific"),
    ("metric_interface_flux_trace_error", "Flux trace error", "scientific"),
    ("metric_relative_spectrum_error", "Spectral error", "scientific"),
    ("metric_poisson_residual_rms", "PDE residual RMS", "metric"),
    ("timing_pca_fit", "PCA fit (s)", "time"),
    ("timing_latent_transform", "Latent transform (s)", "time"),
    ("timing_nn_train", "Cumulative NN train (s)", "time"),
    ("timing_inference", "Cumulative inference (s)", "time"),
    ("invocation_timing_end_to_end", "Invocation total (s)", "time"),
    ("timing_end_to_end", "Cumulative total (s)", "time"),
)


def make_headline_outputs(
    *,
    results_dir: str | Path,
    formats: Sequence[str] = ("png", "pdf"),
    expected_seeds: int = 5,
    qualitative: bool = True,
) -> list[Path]:
    """Generate main-text and appendix assets from the completed study."""
    results_path = Path(results_dir)
    main_figures = results_path / "figures" / "main"
    appendix_figures = results_path / "figures" / "appendix"
    main_tables = results_path / "tables" / "main"
    appendix_tables = results_path / "tables" / "appendix"
    for path in (main_figures, appendix_figures, main_tables, appendix_tables):
        path.mkdir(parents=True, exist_ok=True)
    _apply_style()

    records = load_records(results_path / "results.csv")
    validate_records(records, expected_seeds=expected_seeds)
    summary = summarize_records(records)
    common_seam = common_seam_table(records, results_path=results_path)

    written: list[Path] = []
    written.extend(write_tables(records, summary, common_seam, main_tables, appendix_tables))
    written.extend(
        plot_accuracy(
            records,
            output_base=main_figures / "headline_accuracy",
            formats=formats,
        )
    )
    written.extend(
        plot_interfaces(
            records,
            common_seam=common_seam,
            output_base=main_figures / "headline_interfaces",
            formats=formats,
        )
    )
    written.extend(
        plot_quality_cost(
            records,
            output_base=main_figures / "headline_quality_cost",
            formats=formats,
        )
    )
    written.extend(
        plot_representation_gap(
            records,
            output_base=appendix_figures / "headline_representation_gap",
            formats=formats,
        )
    )

    pca_records = add_pca_cost_substages(records, results_path=results_path)
    written.extend(
        plot_pca_breakdown(
            pca_records,
            output_base=appendix_figures / "headline_pca_breakdown",
            formats=formats,
        )
    )
    pca_table = make_pca_table(pca_records)
    pca_path = appendix_tables / "headline_pca_breakdown.csv"
    pca_table.to_csv(pca_path, index=False)
    written.append(pca_path)
    written.extend(
        _write_table_pair(
            format_pca_table(pca_table),
            appendix_tables / "headline_pca_breakdown",
            csv=False,
        )
    )

    sample_id: int | None = None
    if qualitative:
        qualitative_paths, sample_id = plot_qualitative(
            records,
            results_path=results_path,
            output_base=main_figures / "headline_qualitative",
            formats=formats,
        )
        written.extend(qualitative_paths)

    manifest = {
        "results_dir": str(results_path),
        "records": int(len(records)),
        "methods": list(METHOD_ORDER),
        "expected_seeds": expected_seeds,
        "common_seam_seed": 0,
        "common_seam_stride": 32,
        "qualitative_selection": "seed-0 sample closest to median plain-L2L MRE",
        "qualitative_sample_id": sample_id,
        "outputs": [str(path) for path in written],
    }
    manifest_path = results_path / "headline_poisson_128_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    written.append(manifest_path)
    return written


def load_records(path: Path) -> pd.DataFrame:
    """Load and normalize the stable headline roll-up."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing headline roll-up: {path}")
    records = pd.read_csv(path)
    for column in records.columns:
        if column.startswith(("metric_", "timing_", "invocation_timing_")) or column in {
            "seed",
            "resolution",
        }:
            records[column] = pd.to_numeric(records[column], errors="coerce")
    return records


def validate_records(records: pd.DataFrame, *, expected_seeds: int) -> None:
    """Reject incomplete or malformed headline matrices."""
    required = {"baseline_model", "seed", "run_dir", *NUMERIC_COLUMNS}
    missing = sorted(required.difference(records.columns))
    if missing:
        raise ValueError(f"Missing required headline columns: {', '.join(missing)}")
    observed = set(records["baseline_model"])
    if observed != set(METHOD_ORDER):
        raise ValueError(
            f"Expected methods {list(METHOD_ORDER)}, got {sorted(observed)}."
        )
    duplicates = records.duplicated(["baseline_model", "seed"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate headline method/seed records detected.")
    counts = records.groupby("baseline_model")["seed"].nunique()
    if expected_seeds > 0 and not (counts == expected_seeds).all():
        raise ValueError(
            f"Expected {expected_seeds} seeds per method, got {counts.to_dict()}."
        )


def summarize_records(records: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and sample standard deviation over paired seeds."""
    numeric = [column for column in NUMERIC_COLUMNS if column in records]
    summary = records.groupby("baseline_model", as_index=False)[numeric].agg(["mean", "std"])
    summary.columns = [
        "_".join(str(piece) for piece in column if piece)
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    summary["method_order"] = summary["baseline_model"].map(
        {method: index for index, method in enumerate(METHOD_ORDER)}
    )
    return summary.sort_values("method_order").reset_index(drop=True)


def write_tables(
    records: pd.DataFrame,
    summary: pd.DataFrame,
    common_seam: pd.DataFrame,
    main_dir: Path,
    appendix_dir: Path,
) -> list[Path]:
    """Write headline, complete, paired, and per-seed manuscript tables."""
    written: list[Path] = []
    numeric_path = appendix_dir / "headline_summary_numeric.csv"
    summary.to_csv(numeric_path, index=False)
    written.append(numeric_path)

    main = formatted_summary(
        summary,
        (
            ("metric_mre", "MRE", "percent"),
            ("metric_ssim_mean", "SSIM", "ssim"),
            ("metric_interface_value_trace_error", "Value trace", "scientific"),
            ("metric_interface_flux_trace_error", "Flux trace", "scientific"),
            ("metric_relative_spectrum_error", "Spectral", "scientific"),
            ("timing_end_to_end", "Total time (s)", "time"),
        ),
    )
    written.extend(_write_table_pair(main, main_dir / "headline_poisson_128"))
    written.extend(
        _write_table_pair(
            formatted_summary(summary, FULL_TABLE_SPECS),
            appendix_dir / "headline_poisson_128_full",
        )
    )

    paired = paired_comparisons(records)
    paired_path = appendix_dir / "headline_paired_comparisons.csv"
    paired.to_csv(paired_path, index=False)
    written.append(paired_path)

    per_seed_columns = [
        column
        for column in (
            "baseline_model",
            "seed",
            "config_hash",
            *NUMERIC_COLUMNS,
        )
        if column in records
    ]
    per_seed_path = appendix_dir / "headline_per_seed.csv"
    records[per_seed_columns].sort_values(["baseline_model", "seed"]).to_csv(
        per_seed_path,
        index=False,
    )
    written.append(per_seed_path)

    common_path = appendix_dir / "headline_common_seam_seed0.csv"
    common_seam.to_csv(common_path, index=False)
    written.append(common_path)
    formatted_common = common_seam.copy()
    for column in ("Value jump", "Flux jump", "Value trace error", "Flux trace error"):
        formatted_common[column] = formatted_common[column].map(
            lambda value: f"{float(value):.3e}"
        )
    written.extend(
        _write_table_pair(
            formatted_common,
            appendix_dir / "headline_common_seam_seed0",
            csv=False,
        )
    )
    return written


def formatted_summary(
    summary: pd.DataFrame,
    specs: Sequence[tuple[str, str, str]],
) -> pd.DataFrame:
    """Format a mean-plus-standard-deviation manuscript table."""
    rows: list[dict[str, str]] = []
    for method in METHOD_ORDER:
        selected = summary[summary["baseline_model"] == method].iloc[0]
        row = {"Method": METHOD_LABELS[method]}
        for metric, label, style in specs:
            row[label] = _format_mean_std(
                selected.get(f"{metric}_mean"),
                selected.get(f"{metric}_std"),
                style=style,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def paired_comparisons(records: pd.DataFrame) -> pd.DataFrame:
    """Return paired relative changes and seed wins for central comparisons."""
    comparisons = (
        ("two_scale", "plain_l2l"),
        ("two_scale", "overlap_l2l"),
        ("two_scale", "l2l_refinement"),
        ("two_scale_interface", "two_scale"),
        ("two_scale_interface", "overlap_l2l"),
        ("fno", "two_scale"),
    )
    metrics = {
        "mre": "metric_mre",
        "value_trace": "metric_interface_value_trace_error",
        "flux_trace": "metric_interface_flux_trace_error",
        "spectral": "metric_relative_spectrum_error",
        "pde_rms": "metric_poisson_residual_rms",
        "total_time": "timing_end_to_end",
    }
    indexed = records.set_index(["baseline_model", "seed"])
    rows: list[dict[str, Any]] = []
    for target, reference in comparisons:
        row: dict[str, Any] = {
            "target": target,
            "reference": reference,
        }
        for short, metric in metrics.items():
            target_values = indexed.loc[target][metric]
            reference_values = indexed.loc[reference][metric]
            changes = 100.0 * (target_values / reference_values - 1.0)
            row[f"{short}_paired_change_mean_pct"] = float(changes.mean())
            row[f"{short}_wins_lower_is_better"] = int(
                (target_values < reference_values).sum()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def common_seam_table(records: pd.DataFrame, *, results_path: Path) -> pd.DataFrame:
    """Re-evaluate every retained seed-zero prediction on stride-32 seams."""
    rows: list[dict[str, Any]] = []
    selected = records[records["seed"] == 0]
    for method in METHOD_ORDER:
        row = selected[selected["baseline_model"] == method].iloc[0]
        path = _prediction_path(row, results_path)
        if not path.is_file():
            return pd.DataFrame()
        with np.load(path, allow_pickle=False) as archive:
            prediction = np.asarray(archive["y_pred"], dtype=np.float64)
            truth = np.asarray(archive["y_true"], dtype=np.float64)
        dx = 1.0 / 127.0
        rows.append(
            {
                "Method": METHOD_LABELS[method],
                "Method key": method,
                "Seed": 0,
                "Diagnostic stride": 32,
                "Value jump": interface_value_jump(
                    prediction, patch_size=32, stride=32
                ),
                "Flux jump": interface_flux_jump(
                    prediction, patch_size=32, stride=32, dx=dx
                ),
                "Value trace error": interface_value_trace_error(
                    prediction, truth, patch_size=32, stride=32
                ),
                "Flux trace error": interface_flux_trace_error(
                    prediction, truth, patch_size=32, stride=32, dx=dx
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_accuracy(
    records: pd.DataFrame,
    *,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Plot reconstruction, structure, spectrum, and residual outcomes."""
    specs = (
        ("metric_mre", "Mean relative error (%)", 100.0, False),
        ("metric_ssim_mean", "$1-\mathrm{SSIM}$", 1.0, True),
        ("metric_relative_spectrum_error", "Relative spectral error", 1.0, False),
        ("metric_poisson_residual_rms", "Poisson residual RMS", 1.0, False),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.1), squeeze=False)
    for panel, (metric, ylabel, scale, complement) in enumerate(specs):
        axis = axes.flat[panel]
        values = records.copy()
        plot_column = metric
        if complement:
            plot_column = "structural_error"
            values[plot_column] = 1.0 - values[metric]
        _bar_with_seed_points(axis, values, metric=plot_column, scale=scale)
        axis.set_ylabel(ylabel)
        axis.set_yscale("log")
        axis.grid(True, axis="y", alpha=0.22)
        _panel_label(axis, panel)
    fig.tight_layout(h_pad=1.5, w_pad=1.4)
    return _save_figure(fig, output_base, formats)


def plot_interfaces(
    records: pd.DataFrame,
    *,
    common_seam: pd.DataFrame,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Plot native multi-seed and common-seam interface diagnostics."""
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.1), squeeze=False)
    native = (
        ("metric_interface_value_trace_error", "Native value trace error"),
        ("metric_interface_flux_trace_error", "Native flux trace error"),
    )
    for panel, (metric, ylabel) in enumerate(native):
        axis = axes.flat[panel]
        _bar_with_seed_points(axis, records, metric=metric, scale=1.0)
        axis.set_ylabel(ylabel)
        axis.set_yscale("log")
        axis.grid(True, axis="y", alpha=0.22)
        _panel_label(axis, panel)

    common_specs = (
        ("Value trace error", "Common stride-32 value trace error"),
        ("Flux trace error", "Common stride-32 flux trace error"),
    )
    for offset, (metric, ylabel) in enumerate(common_specs, start=2):
        axis = axes.flat[offset]
        values = np.array(
            [
                common_seam.loc[common_seam["Method key"] == method, metric].iloc[0]
                for method in METHOD_ORDER
            ]
        )
        axis.bar(
            np.arange(len(METHOD_ORDER)),
            values,
            color=[METHOD_COLORS[method] for method in METHOD_ORDER],
            width=0.7,
        )
        _method_ticks(axis)
        axis.set_ylabel(ylabel)
        axis.set_yscale("log")
        axis.grid(True, axis="y", alpha=0.22)
        axis.text(
            0.98,
            0.96,
            "seed 0",
            transform=axis.transAxes,
            ha="right",
            va="top",
            color="#666666",
        )
        _panel_label(axis, offset)
    fig.tight_layout(h_pad=1.5, w_pad=1.4)
    return _save_figure(fig, output_base, formats)


def plot_quality_cost(
    records: pd.DataFrame,
    *,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Plot the quality-cost plane and cumulative stage breakdown."""
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), squeeze=False)
    scatter = axes[0, 0]
    grouped = records.groupby("baseline_model").mean(numeric_only=True)
    std = records.groupby("baseline_model").std(numeric_only=True)
    for method in METHOD_ORDER:
        scatter.errorbar(
            grouped.loc[method, "timing_end_to_end"],
            100.0 * grouped.loc[method, "metric_mre"],
            xerr=std.loc[method, "timing_end_to_end"],
            yerr=100.0 * std.loc[method, "metric_mre"],
            fmt="o",
            color=METHOD_COLORS[method],
            capsize=2.5,
            markersize=6,
        )
        scatter.annotate(
            METHOD_LABELS[method],
            (
                grouped.loc[method, "timing_end_to_end"],
                100.0 * grouped.loc[method, "metric_mre"],
            ),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7.5,
        )
    scatter.set_xscale("log")
    scatter.set_yscale("log")
    scatter.set_xlabel("Cumulative end-to-end time (s)")
    scatter.set_ylabel("Mean relative error (%)")
    scatter.grid(True, alpha=0.22)
    _panel_label(scatter, 0)

    stage_axis = axes[0, 1]
    x = np.arange(len(METHOD_ORDER))
    bottom = np.zeros(len(METHOD_ORDER), dtype=float)
    total = np.array(
        [grouped.loc[method, "timing_end_to_end"] for method in METHOD_ORDER]
    )
    for metric, label, color in STAGE_SPECS:
        values = np.array([grouped.loc[method, metric] for method in METHOD_ORDER])
        stage_axis.bar(x, values, bottom=bottom, color=color, width=0.68, label=label)
        bottom += values
    other = np.maximum(total - bottom, 0.0)
    stage_axis.bar(
        x,
        other,
        bottom=bottom,
        color="#B9B9B9",
        width=0.68,
        label="Other pipeline",
    )
    for index, value in enumerate(total):
        stage_axis.text(index, value + 8.0, f"{value:.0f}", ha="center", fontsize=7.5)
    _method_ticks(stage_axis)
    stage_axis.set_ylabel("Cumulative time (s)")
    stage_axis.grid(True, axis="y", alpha=0.22)
    stage_axis.legend(frameon=False, fontsize=7, ncol=2)
    _panel_label(stage_axis, 1)
    fig.tight_layout(w_pad=1.6)
    return _save_figure(fig, output_base, formats)


def plot_representation_gap(
    records: pd.DataFrame,
    *,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Contrast learned errors with PCA representation floors."""
    methods = (
        "global_pca",
        "plain_l2l",
        "overlap_l2l",
        "two_scale",
        "two_scale_interface",
    )
    grouped = records.groupby("baseline_model").mean(numeric_only=True)
    x = np.arange(len(methods))
    width = 0.36
    learned = np.array([100.0 * grouped.loc[method, "metric_mre"] for method in methods])
    oracle = np.array(
        [100.0 * grouped.loc[method, "metric_pca_oracle_mre"] for method in methods]
    )
    fig, axis = plt.subplots(figsize=(8.8, 4.2))
    axis.bar(x - width / 2, oracle, width=width, color="#B9B9B9", label="PCA oracle")
    for index, method in enumerate(methods):
        axis.bar(
            x[index] + width / 2,
            learned[index],
            width=width,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    axis.set_xticks(x, [METHOD_LABELS[method] for method in methods], rotation=18)
    axis.set_ylabel("Mean relative error (%)")
    axis.set_yscale("log")
    axis.grid(True, axis="y", alpha=0.22)
    axis.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8,
    )
    fig.tight_layout()
    return _save_figure(fig, output_base, formats)


def plot_pca_breakdown(
    records: pd.DataFrame,
    *,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Plot PCA/SVD fitting substages for unique representations."""
    methods = ("global_pca", "plain_l2l", "overlap_l2l", "two_scale")
    selected = records[records["baseline_model"].isin(methods)]
    grouped = selected.groupby("baseline_model").mean(numeric_only=True)
    x = np.arange(len(methods))
    bottom = np.zeros(len(methods), dtype=float)
    fig, axis = plt.subplots(figsize=(7.4, 4.2))
    for metric, label, color in PCA_STAGE_SPECS:
        values = np.array([grouped.loc[method, metric] for method in methods])
        axis.bar(x, values, bottom=bottom, color=color, width=0.66, label=label)
        bottom += values
    for index, value in enumerate(bottom):
        axis.text(index, value + 0.2, f"{value:.1f}", ha="center", fontsize=8)
    axis.set_xticks(x, [METHOD_LABELS[method] for method in methods], rotation=15)
    axis.set_ylabel("PCA/SVD fit time (s)")
    axis.grid(True, axis="y", alpha=0.22)
    axis.legend(frameon=False, ncol=2, fontsize=7.5)
    fig.tight_layout()
    return _save_figure(fig, output_base, formats)


def make_pca_table(records: pd.DataFrame) -> pd.DataFrame:
    """Aggregate PCA substages for unique output representations."""
    methods = ("global_pca", "plain_l2l", "overlap_l2l", "two_scale")
    rows: list[dict[str, Any]] = []
    for method in methods:
        selected = records[records["baseline_model"] == method]
        row: dict[str, Any] = {"Method": METHOD_LABELS[method]}
        for metric, label, _color in PCA_STAGE_SPECS:
            row[f"{label} mean (s)"] = selected[metric].mean()
            row[f"{label} std (s)"] = selected[metric].std(ddof=1)
        row["Total PCA fit mean (s)"] = selected["pca_cost_total"].mean()
        row["Total PCA fit std (s)"] = selected["pca_cost_total"].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def format_pca_table(table: pd.DataFrame) -> pd.DataFrame:
    """Format the PCA breakdown for LaTeX output."""
    formatted = table.copy()
    for column in formatted.columns[1:]:
        formatted[column] = formatted[column].map(lambda value: f"{float(value):.2f}")
    return formatted


def plot_qualitative(
    records: pd.DataFrame,
    *,
    results_path: Path,
    output_base: Path,
    formats: Sequence[str],
) -> tuple[list[Path], int]:
    """Plot truth, prediction, error, spectrum, and PDF for every method."""
    selected = records[records["seed"] == 0]
    paths = {
        method: _prediction_path(
            selected[selected["baseline_model"] == method].iloc[0], results_path
        )
        for method in METHOD_ORDER
    }
    with np.load(paths["plain_l2l"], allow_pickle=False) as archive:
        truth_all = np.asarray(archive["y_true"], dtype=np.float32)
        prediction_all = np.asarray(archive["y_pred"], dtype=np.float32)
        sample_indices = np.asarray(archive["sample_indices"], dtype=np.int64)
        sample_mre = np.linalg.norm(
            (prediction_all - truth_all).reshape(len(truth_all), -1), axis=1
        ) / np.maximum(
            np.linalg.norm(truth_all.reshape(len(truth_all), -1), axis=1), 1.0e-12
        )
        median = float(np.median(sample_mre))
        chosen = int(np.argmin(np.abs(sample_mre - median)))
        sample_id = int(sample_indices[chosen])
        truth = truth_all[chosen].copy()

    predictions: dict[str, np.ndarray] = {}
    for method, path in paths.items():
        with np.load(path, allow_pickle=False) as archive:
            indices = np.asarray(archive["sample_indices"], dtype=np.int64)
            matches = np.flatnonzero(indices == sample_id)
            if len(matches) != 1:
                raise ValueError(f"Sample {sample_id} is not uniquely aligned in {path}.")
            index = int(matches[0])
            method_truth = np.asarray(archive["y_true"][index], dtype=np.float32)
            if not np.allclose(method_truth, truth, rtol=0.0, atol=1.0e-7):
                raise ValueError(f"Ground truth mismatch for sample {sample_id} in {path}.")
            predictions[method] = np.asarray(archive["y_pred"][index], dtype=np.float32)

    all_fields = np.stack([truth, *predictions.values()])
    solution_min = float(np.min(all_fields))
    solution_max = float(np.max(all_fields))
    errors = {method: np.abs(prediction - truth) for method, prediction in predictions.items()}
    error_max = max(float(np.percentile(np.stack(list(errors.values())), 99.5)), 1.0e-12)
    bins = np.linspace(solution_min, solution_max, 40)

    fig, axes = plt.subplots(
        len(METHOD_ORDER),
        5,
        figsize=(13.2, 2.05 * len(METHOD_ORDER)),
        squeeze=False,
    )
    for column, title in enumerate(
        ("Ground truth", "Prediction", "Absolute error", "Energy spectrum", "Value PDF")
    ):
        axes[0, column].set_title(title)
    for row_index, method in enumerate(METHOD_ORDER):
        prediction = predictions[method]
        error = errors[method]
        axes[row_index, 0].imshow(
            truth, origin="lower", cmap="RdBu_r", vmin=solution_min, vmax=solution_max
        )
        axes[row_index, 1].imshow(
            prediction,
            origin="lower",
            cmap="RdBu_r",
            vmin=solution_min,
            vmax=solution_max,
        )
        axes[row_index, 2].imshow(
            error, origin="lower", cmap="magma", vmin=0.0, vmax=error_max
        )
        mre = float(np.linalg.norm(prediction - truth)) / max(
            float(np.linalg.norm(truth)), 1.0e-12
        )
        axes[row_index, 1].text(
            0.03,
            0.97,
            f"MRE {mre:.4f}",
            transform=axes[row_index, 1].transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.8},
        )
        axes[row_index, 0].set_ylabel(METHOD_LABELS[method], fontweight="bold")
        for column in range(3):
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])
        _plot_sample_spectrum(axes[row_index, 3], truth, prediction, k_max=32)
        _plot_sample_pdf(axes[row_index, 4], truth, prediction, bins=bins)
        if row_index == len(METHOD_ORDER) - 1:
            axes[row_index, 3].set_xlabel("Wavenumber $k$")
            axes[row_index, 4].set_xlabel("Field value")
    fig.suptitle(
        f"Poisson 128, seed 0, median plain-L2L sample {sample_id}",
        y=0.997,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985), h_pad=0.75, w_pad=0.7)
    return _save_figure(fig, output_base, formats), sample_id


def _bar_with_seed_points(
    axis: plt.Axes,
    records: pd.DataFrame,
    *,
    metric: str,
    scale: float,
) -> None:
    x = np.arange(len(METHOD_ORDER))
    means = []
    stds = []
    for index, method in enumerate(METHOD_ORDER):
        values = (
            records.loc[records["baseline_model"] == method, metric].to_numpy(dtype=float)
            * scale
        )
        means.append(float(np.mean(values)))
        stds.append(float(np.std(values, ddof=1)))
        jitter = np.linspace(-0.12, 0.12, len(values))
        axis.scatter(
            index + jitter,
            values,
            color="white",
            edgecolor="#333333",
            linewidth=0.45,
            s=12,
            zorder=4,
        )
    axis.bar(
        x,
        means,
        yerr=stds,
        color=[METHOD_COLORS[method] for method in METHOD_ORDER],
        width=0.7,
        capsize=2.5,
        zorder=2,
    )
    _method_ticks(axis)


def _method_ticks(axis: plt.Axes) -> None:
    axis.set_xticks(
        np.arange(len(METHOD_ORDER)),
        [METHOD_LABELS[method] for method in METHOD_ORDER],
        rotation=22,
        ha="right",
    )


def _panel_label(axis: plt.Axes, index: int) -> None:
    axis.text(
        0.02,
        0.96,
        f"({chr(97 + index)})",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default="paper_results/headline_poisson_128",
    )
    parser.add_argument(
        "--formats",
        default="png,pdf",
        help="Comma-separated output formats.",
    )
    parser.add_argument("--expected-seeds", type=int, default=5)
    parser.add_argument("--skip-qualitative", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    formats = tuple(piece.strip() for piece in args.formats.split(",") if piece.strip())
    written = make_headline_outputs(
        results_dir=args.results_dir,
        formats=formats,
        expected_seeds=args.expected_seeds,
        qualitative=not args.skip_qualitative,
    )
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
