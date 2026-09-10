#!/usr/bin/env python
"""Render manuscript figures and tables for the Poisson resolution sweep."""

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
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from manuscript_pca_costs import PCA_STAGE_SPECS, add_pca_cost_substages

from lpcanet.metrics.interface import (
    interface_flux_jump,
    interface_flux_trace_error,
    interface_value_jump,
    interface_value_trace_error,
)
from lpcanet.metrics.spectral import energy_spectrum_2d

METHOD_ORDER = (
    "global_pca",
    "plain_l2l",
    "overlap_l2l",
    "two_scale",
    "two_scale_interface",
    "fno",
)
PCA_METHODS = METHOD_ORDER[:-1]
METHOD_LABELS = {
    "global_pca": "Global PCA-Net",
    "plain_l2l": "Plain L2L",
    "overlap_l2l": "L2L + overlap",
    "two_scale": "Two-scale",
    "two_scale_interface": "Two-scale + interface",
    "fno": "FNO",
}
METHOD_COLORS = {
    "global_pca": "#9C755F",
    "plain_l2l": "#6F6F6F",
    "overlap_l2l": "#4C78A8",
    "two_scale": "#F58518",
    "two_scale_interface": "#54A24B",
    "fno": "#E45756",
}
METHOD_MARKERS = {
    "global_pca": "P",
    "plain_l2l": "o",
    "overlap_l2l": "s",
    "two_scale": "^",
    "two_scale_interface": "D",
    "fno": "X",
}
STAGE_SPECS = (
    ("timing_pca_fit", "PCA fit", "#4C78A8"),
    ("timing_latent_transform", "Latent transform", "#72B7B2"),
    ("timing_nn_train", "NN training", "#F58518"),
    ("timing_inference", "Inference", "#E45756"),
)
STYLE = {
    "dpi": 300,
    "font_size": 9,
    "title_size": 10,
    "label_size": 9,
    "tick_size": 8,
    "legend_size": 8,
    "line_width": 1.8,
    "marker_size": 5,
    "capsize": 3,
}
QUALITY_METRICS = (
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
)
TIMING_METRICS = (
    "timing_pca_fit",
    "timing_latent_transform",
    "timing_nn_train",
    "timing_inference",
    "timing_end_to_end",
    "invocation_timing_nn_train",
    "invocation_timing_end_to_end",
)
FULL_METRICS = (
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
    ("timing_nn_train", "NN train, cumulative (s)", "time"),
    ("timing_inference", "Inference (s)", "time"),
    ("invocation_timing_nn_train", "Invocation NN train (s)", "time"),
    ("invocation_timing_end_to_end", "Invocation total (s)", "time"),
    ("timing_end_to_end", "Cumulative total (s)", "time"),
)


def make_resolution_sweep_outputs(
    *,
    results_dir: str | Path,
    formats: Sequence[str] = ("png", "pdf"),
    expected_seeds: int = 3,
    qualitative: bool = True,
) -> list[Path]:
    """Generate main-text and appendix assets from a completed sweep."""
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

    written: list[Path] = []
    written.extend(write_tables(records, summary, main_tables, appendix_tables))
    written.extend(
        plot_accuracy_scaling(
            records,
            output_base=main_figures / "resolution_accuracy",
            formats=formats,
        )
    )
    written.extend(
        plot_artifact_scaling(
            records,
            output_base=main_figures / "resolution_artifacts",
            formats=formats,
        )
    )
    written.extend(
        plot_quality_cost(
            records,
            output_base=main_figures / "resolution_quality_cost",
            formats=formats,
        )
    )
    time_projection = time_scaling_projection_table(records)
    written.extend(
        plot_time_scaling_projection(
            records,
            output_base=main_figures / "resolution_time_projection",
            formats=formats,
        )
    )
    projection_path = appendix_tables / "resolution_time_projection.csv"
    time_projection.to_csv(projection_path, index=False)
    written.append(projection_path)
    written.extend(
        _write_table_pair(
            _format_time_projection_table(time_projection),
            appendix_tables / "resolution_time_projection",
            csv=False,
        )
    )
    written.extend(
        plot_representation_gap(
            records,
            output_base=appendix_figures / "resolution_representation_gap",
            formats=formats,
        )
    )
    written.extend(
        plot_stage_costs(
            records,
            output_base=appendix_figures / "resolution_stage_costs",
            formats=formats,
        )
    )
    pca_costs = add_pca_cost_substages(records, results_path=results_path)
    written.extend(
        plot_pca_cost_breakdown(
            pca_costs,
            output_base=appendix_figures / "resolution_pca_breakdown",
            formats=formats,
        )
    )
    pca_table = pca_cost_breakdown_table(pca_costs)
    if not pca_table.empty:
        pca_path = appendix_tables / "resolution_pca_breakdown.csv"
        pca_table.to_csv(pca_path, index=False)
        written.append(pca_path)
        written.extend(
            _write_table_pair(
                pca_table,
                appendix_tables / "resolution_pca_breakdown",
                csv=False,
            )
        )

    common_seam = common_seam_table(records, results_path=results_path)
    if not common_seam.empty:
        common_path = appendix_tables / "resolution_common_seam_seed0.csv"
        common_seam.to_csv(common_path, index=False)
        written.append(common_path)
        written.extend(
            _write_table_pair(
                _format_common_seam_table(common_seam),
                appendix_tables / "resolution_common_seam_seed0",
                csv=False,
            )
        )

    qualitative_resolutions: list[int] = []
    if qualitative:
        for resolution in sorted(int(value) for value in records["resolution"].unique()):
            paths = plot_qualitative_diagnostics(
                records,
                results_path=results_path,
                resolution=resolution,
                output_base=appendix_figures
                / f"resolution_qualitative_{resolution}",
                formats=formats,
            )
            if paths:
                qualitative_resolutions.append(resolution)
                written.extend(paths)

    manifest = {
        "results_dir": str(results_path),
        "records": int(len(records)),
        "resolutions": sorted(
            int(value) for value in records["resolution"].unique()
        ),
        "methods": list(METHOD_ORDER),
        "expected_seeds": expected_seeds,
        "common_seam_rows": int(len(common_seam)),
        "qualitative_resolutions": qualitative_resolutions,
        "outputs": [str(path) for path in written],
    }
    manifest_path = results_path / "resolution_sweep_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    written.append(manifest_path)
    return written


def load_records(path: Path) -> pd.DataFrame:
    """Load the stable study roll-up and coerce metric fields to numeric."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing resolution-sweep roll-up: {path}")
    frame = pd.read_csv(path)
    for column in frame.columns:
        if column.startswith(
            (
                "metric_",
                "validation_metric_",
                "timing_",
                "invocation_timing_",
                "pca_component_counts_",
            )
        ) or column in {"seed", "resolution"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def validate_records(records: pd.DataFrame, *, expected_seeds: int) -> None:
    """Reject incomplete, duplicated, or malformed sweep matrices."""
    required = {
        "seed",
        "resolution",
        "ablation_method",
        "run_dir",
        *QUALITY_METRICS,
        *TIMING_METRICS,
    }
    missing = sorted(required.difference(records.columns))
    if missing:
        raise ValueError(f"Missing required resolution columns: {', '.join(missing)}")
    unexpected = sorted(set(records["ablation_method"]) - set(METHOD_ORDER))
    if unexpected:
        raise ValueError(f"Unexpected resolution methods: {', '.join(unexpected)}")
    missing_methods = sorted(set(METHOD_ORDER) - set(records["ablation_method"]))
    if missing_methods:
        raise ValueError(f"Missing resolution methods: {', '.join(missing_methods)}")
    keys = ["resolution", "ablation_method", "seed"]
    duplicates = records.duplicated(keys, keep=False)
    if duplicates.any():
        details = records.loc[duplicates, keys]
        raise ValueError(f"Duplicate resolution records:\n{details.to_string(index=False)}")
    counts = records.groupby(["resolution", "ablation_method"])["seed"].nunique()
    if expected_seeds > 0:
        incomplete = counts[counts != expected_seeds]
        if not incomplete.empty:
            details = ", ".join(
                f"{int(resolution)}/{method}: {int(count)}"
                for (resolution, method), count in incomplete.items()
            )
            raise ValueError(
                f"Expected {expected_seeds} seeds per resolution cell; found {details}."
            )
    expected_cells = pd.MultiIndex.from_product(
        [
            sorted(records["resolution"].unique()),
            METHOD_ORDER,
        ],
        names=["resolution", "ablation_method"],
    )
    missing_cells = expected_cells.difference(counts.index)
    if len(missing_cells):
        raise ValueError(f"Missing resolution cells: {list(missing_cells)}")


def summarize_records(records: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and sample standard deviation over seeds."""
    numeric = [
        column
        for column in (*QUALITY_METRICS, *TIMING_METRICS)
        if column in records
    ]
    summary = (
        records.groupby(["resolution", "ablation_method"], as_index=False)[numeric]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(piece) for piece in column if piece)
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    counts = records.groupby(["resolution", "ablation_method"])["seed"].nunique()
    summary["n_seeds"] = counts.reindex(
        pd.MultiIndex.from_frame(summary[["resolution", "ablation_method"]])
    ).to_numpy()
    summary["method_order"] = summary["ablation_method"].map(
        {name: index for index, name in enumerate(METHOD_ORDER)}
    )
    return summary.sort_values(
        ["resolution", "method_order"],
        kind="stable",
    ).reset_index(drop=True)


def write_tables(
    records: pd.DataFrame,
    summary: pd.DataFrame,
    main_dir: Path,
    appendix_dir: Path,
) -> list[Path]:
    """Write formatted manuscript tables and complete numeric roll-ups."""
    written: list[Path] = []
    numeric_path = appendix_dir / "resolution_summary_numeric.csv"
    summary.to_csv(numeric_path, index=False)
    written.append(numeric_path)

    main = _main_table(summary)
    written.extend(_write_table_pair(main, main_dir / "resolution_sweep_main"))

    full = _formatted_summary(summary, FULL_METRICS)
    written.extend(
        _write_table_pair(full, appendix_dir / "resolution_sweep_full")
    )

    for resolution in sorted(int(value) for value in records["resolution"].unique()):
        selected = full[full["Resolution"] == resolution].drop(
            columns=["Resolution"]
        )
        written.extend(
            _write_table_pair(
                selected,
                appendix_dir / f"resolution_sweep_full_{resolution}",
            )
        )

    stage = _formatted_summary(
        summary,
        tuple(spec for spec in FULL_METRICS if spec[2] == "time"),
    )
    written.extend(
        _write_table_pair(stage, appendix_dir / "resolution_stage_costs")
    )

    relative = relative_change_table(records)
    relative_path = appendix_dir / "resolution_relative_changes.csv"
    relative.to_csv(relative_path, index=False)
    written.append(relative_path)
    formatted_relative = relative.copy()
    for column in formatted_relative.columns:
        if column.endswith("_pct"):
            formatted_relative[column] = formatted_relative[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):+.1f}%"
            )
    written.extend(
        _write_table_pair(
            formatted_relative,
            appendix_dir / "resolution_relative_changes",
            csv=False,
        )
    )

    capacity = capacity_table(records)
    capacity_path = appendix_dir / "resolution_pca_capacity.csv"
    capacity.to_csv(capacity_path, index=False)
    written.append(capacity_path)
    written.extend(
        _write_table_pair(
            _format_capacity_table(capacity),
            appendix_dir / "resolution_pca_capacity",
            csv=False,
        )
    )

    projection_path = Path(records.attrs.get("results_dir", "")) / "pca_projection.csv"
    if not projection_path.is_file():
        candidate = appendix_dir.parents[1] / "pca_projection.csv"
        projection_path = candidate
    if projection_path.is_file():
        projection = pd.read_csv(projection_path)
        output_path = appendix_dir / "resolution_pca_projection.csv"
        projection.to_csv(output_path, index=False)
        written.append(output_path)
        written.extend(
            _write_table_pair(
                projection,
                appendix_dir / "resolution_pca_projection",
                csv=False,
            )
        )

    per_seed_columns = [
        column
        for column in (
            "resolution",
            "seed",
            "ablation_method",
            "config_hash",
            *QUALITY_METRICS,
            *TIMING_METRICS,
        )
        if column in records
    ]
    per_seed = records[per_seed_columns].sort_values(
        ["resolution", "ablation_method", "seed"]
    )
    per_seed_path = appendix_dir / "resolution_per_seed.csv"
    per_seed.to_csv(per_seed_path, index=False)
    written.append(per_seed_path)
    return written


def relative_change_table(records: pd.DataFrame) -> pd.DataFrame:
    """Compute paired headline comparisons from mean metrics."""
    metrics = {
        "mre": "metric_mre",
        "ssim": "metric_ssim_mean",
        "value_trace": "metric_interface_value_trace_error",
        "flux_trace": "metric_interface_flux_trace_error",
        "spectral": "metric_relative_spectrum_error",
        "pde_rms": "metric_poisson_residual_rms",
        "pca_fit": "timing_pca_fit",
        "total_time": "timing_end_to_end",
    }
    grouped = records.groupby(["resolution", "ablation_method"])[
        list(metrics.values())
    ].mean()
    rows: list[dict[str, float | int]] = []
    for resolution in sorted(int(value) for value in records["resolution"].unique()):
        values = grouped.loc[resolution]
        row: dict[str, float | int] = {"Resolution": resolution}
        for short, metric in metrics.items():
            overlap = float(values.loc["overlap_l2l", metric])
            two_scale = float(values.loc["two_scale", metric])
            interface = float(values.loc["two_scale_interface", metric])
            direction = 1.0 if short == "ssim" else -1.0
            row[f"two_scale_vs_overlap_{short}_pct"] = (
                direction * 100.0 * (two_scale - overlap) / overlap
            )
            row[f"interface_vs_two_scale_{short}_pct"] = (
                direction * 100.0 * (interface - two_scale) / two_scale
            )
        row["fno_vs_two_scale_total_time_ratio"] = (
            float(values.loc["fno", "timing_end_to_end"])
            / float(values.loc["two_scale", "timing_end_to_end"])
        )
        rows.append(row)
    return pd.DataFrame(rows)


def capacity_table(records: pd.DataFrame) -> pd.DataFrame:
    """Summarize retained dimensions and learned-to-oracle gaps."""
    columns = [
        column
        for column in (
            "metric_mre",
            "metric_pca_oracle_mre",
            "pca_component_counts_input_total",
            "pca_component_counts_output_total",
            "pca_component_counts_coarse_output",
            "pca_component_counts_output_residual_total",
        )
        if column in records
    ]
    grouped = records.groupby(["resolution", "ablation_method"])[columns].mean()
    rows: list[dict[str, Any]] = []
    for (resolution, method), row in grouped.iterrows():
        if method == "fno":
            continue
        oracle = float(row["metric_pca_oracle_mre"])
        rows.append(
            {
                "Resolution": int(resolution),
                "Method": METHOD_LABELS[str(method)],
                "Input dimensions": row.get("pca_component_counts_input_total"),
                "Output dimensions": row.get("pca_component_counts_output_total"),
                "Coarse dimensions": row.get(
                    "pca_component_counts_coarse_output"
                ),
                "Residual dimensions": row.get(
                    "pca_component_counts_output_residual_total"
                ),
                "Learned MRE (%)": 100.0 * float(row["metric_mre"]),
                "Oracle MRE (%)": 100.0 * oracle,
                "Learned / oracle": float(row["metric_mre"]) / oracle,
            }
        )
    return pd.DataFrame(rows)


def plot_accuracy_scaling(
    records: pd.DataFrame,
    *,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Plot reconstruction accuracy across grid resolutions."""
    specs = (
        ("metric_mre", "Mean relative error (%)", 100.0),
        ("metric_ssim_mean", "SSIM", 1.0),
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), squeeze=False)
    for index, (metric, label, scale) in enumerate(specs):
        ax = axes[0, index]
        _plot_metric_curves(ax, records, metric=metric, scale=scale)
        ax.set_xlabel("Grid resolution")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.22)
        _panel_label(ax, index)
    fig.legend(
        handles=_method_handles(),
        loc="upper center",
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84), w_pad=1.8)
    return _save_figure(fig, output_base, formats)


def plot_artifact_scaling(
    records: pd.DataFrame,
    *,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Plot truth-referenced seam and radial-spectrum errors."""
    specs = (
        ("metric_interface_value_trace_error", "Value trace error"),
        ("metric_interface_flux_trace_error", "Flux trace error"),
        ("metric_relative_spectrum_error", "Relative spectral error"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.75), squeeze=False)
    for index, (metric, label) in enumerate(specs):
        ax = axes[0, index]
        _plot_metric_curves(ax, records, metric=metric, scale=1.0)
        ax.set_xlabel("Grid resolution")
        ax.set_ylabel(label)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.22)
        _panel_label(ax, index)
    fig.legend(
        handles=_method_handles(),
        loc="upper center",
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84), w_pad=1.4)
    return _save_figure(fig, output_base, formats)


def plot_quality_cost(
    records: pd.DataFrame,
    *,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Plot total cost scaling and the observed quality-cost frontier."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9), squeeze=False)
    _plot_metric_curves(
        axes[0, 0],
        records,
        metric="timing_end_to_end",
        scale=1.0,
    )
    axes[0, 0].set_xlabel("Grid resolution")
    axes[0, 0].set_ylabel("Cumulative end-to-end time (s)")
    axes[0, 0].set_yscale("log")
    axes[0, 0].grid(True, alpha=0.22)
    _panel_label(axes[0, 0], 0)

    for method in METHOD_ORDER:
        selected = records[records["ablation_method"] == method]
        grouped = selected.groupby("resolution").agg(
            time=("timing_end_to_end", "mean"),
            mre=("metric_mre", "mean"),
        )
        axes[0, 1].plot(
            grouped["time"],
            100.0 * grouped["mre"],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linewidth=STYLE["line_width"],
            markersize=STYLE["marker_size"],
        )
        for resolution, row in grouped.iterrows():
            axes[0, 1].annotate(
                str(int(resolution)),
                (float(row["time"]), 100.0 * float(row["mre"])),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7,
                color=METHOD_COLORS[method],
            )
    axes[0, 1].set_xlabel("Cumulative end-to-end time (s)")
    axes[0, 1].set_ylabel("Mean relative error (%)")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].grid(True, alpha=0.22)
    _panel_label(axes[0, 1], 1)
    fig.legend(
        handles=_method_handles(),
        loc="upper center",
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84), w_pad=1.7)
    return _save_figure(fig, output_base, formats)


def time_scaling_projection_table(
    records: pd.DataFrame,
    *,
    projected_resolutions: Sequence[int] = (512, 1024),
) -> pd.DataFrame:
    """Fit observed log-log time slopes and project beyond measured grids.

    The exponent is least-squares fit on the mean cumulative wall times at the
    measured resolutions.  Extrapolated values are then anchored at the
    observed largest grid so the dashed continuation meets the measured curve.
    They are trend projections, not unexecuted experimental measurements.
    """
    measured_resolutions = sorted(int(value) for value in records["resolution"].unique())
    if len(measured_resolutions) < 2:
        raise ValueError("At least two resolutions are required for time projection.")
    projection_resolutions = sorted(
        int(value) for value in projected_resolutions if int(value) > measured_resolutions[-1]
    )
    rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        selected = records[records["ablation_method"] == method]
        grouped = selected.groupby("resolution")["timing_end_to_end"].mean()
        times = np.asarray(
            [float(grouped.loc[resolution]) for resolution in measured_resolutions],
            dtype=float,
        )
        exponent, _intercept = np.polyfit(
            np.log2(np.asarray(measured_resolutions, dtype=float)),
            np.log2(times),
            deg=1,
        )
        anchor_resolution = float(measured_resolutions[-1])
        anchor_time = float(times[-1])
        row: dict[str, Any] = {
            "Method": METHOD_LABELS[method],
            "Fitted log-log exponent": float(exponent),
        }
        for resolution, time in zip(measured_resolutions, times):
            row[f"Measured {resolution} (s)"] = float(time)
        for resolution in projection_resolutions:
            row[f"Projected {resolution} (s)"] = anchor_time * (
                float(resolution) / anchor_resolution
            ) ** float(exponent)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_time_scaling_projection(
    records: pd.DataFrame,
    *,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Plot measured cumulative times and log-log trend projections."""
    table = time_scaling_projection_table(records)
    measured_resolutions = sorted(int(value) for value in records["resolution"].unique())
    projected_resolutions = (512, 1024)
    fig, axis = plt.subplots(figsize=(7.7, 4.25))

    for method in METHOD_ORDER:
        row = table.loc[table["Method"] == METHOD_LABELS[method]].iloc[0]
        measured_times = np.asarray(
            [float(row[f"Measured {resolution} (s)"]) for resolution in measured_resolutions],
            dtype=float,
        )
        projected_times = np.asarray(
            [float(row[f"Projected {resolution} (s)"]) for resolution in projected_resolutions],
            dtype=float,
        )
        color = METHOD_COLORS[method]
        marker = METHOD_MARKERS[method]
        axis.plot(
            measured_resolutions,
            measured_times,
            color=color,
            marker=marker,
            linewidth=STYLE["line_width"],
            markersize=STYLE["marker_size"],
        )
        axis.plot(
            (measured_resolutions[-1], *projected_resolutions),
            (measured_times[-1], *projected_times),
            color=color,
            linestyle="--",
            linewidth=STYLE["line_width"],
        )
        axis.scatter(
            projected_resolutions,
            projected_times,
            marker=marker,
            s=38,
            facecolors="white",
            edgecolors=color,
            linewidths=1.25,
            zorder=3,
        )

    axis.axvline(
        measured_resolutions[-1],
        color="#777777",
        linestyle=":",
        linewidth=1.0,
        zorder=0,
    )
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    all_resolutions = (*measured_resolutions, *projected_resolutions)
    axis.set_xticks(all_resolutions, [str(value) for value in all_resolutions])
    axis.set_xlabel("Grid resolution")
    axis.set_ylabel("Cumulative end-to-end time (s)")
    axis.grid(True, which="both", alpha=0.22)
    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linewidth=STYLE["line_width"],
            label=(
                f"{METHOD_LABELS[method]} "
                f"($\\alpha={float(table.loc[table['Method'] == METHOD_LABELS[method], 'Fitted log-log exponent'].iloc[0]):.2f}$)"
            ),
        )
        for method in METHOD_ORDER
    ]
    axis.legend(handles=handles, loc="upper left", ncol=2, frameon=False)
    fig.tight_layout()
    return _save_figure(fig, output_base, formats)


def plot_representation_gap(
    records: pd.DataFrame,
    *,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Contrast learned errors with each PCA representation floor."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), squeeze=False)
    for method in PCA_METHODS:
        selected = records[records["ablation_method"] == method].copy()
        selected["learned_oracle_ratio"] = (
            selected["metric_mre"] / selected["metric_pca_oracle_mre"]
        )
        _plot_one_curve(
            axes[0, 0],
            selected,
            metric="metric_pca_oracle_mre",
            method=method,
            scale=100.0,
        )
        _plot_one_curve(
            axes[0, 1],
            selected,
            metric="learned_oracle_ratio",
            method=method,
            scale=1.0,
        )
    axes[0, 0].set_xlabel("Grid resolution")
    axes[0, 0].set_ylabel("PCA-oracle MRE (%)")
    axes[0, 0].set_yscale("log")
    axes[0, 0].grid(True, alpha=0.22)
    _panel_label(axes[0, 0], 0)
    axes[0, 1].axhline(1.0, color="#777777", linestyle=":", linewidth=1.1)
    axes[0, 1].set_xlabel("Grid resolution")
    axes[0, 1].set_ylabel("Learned MRE / PCA-oracle MRE")
    axes[0, 1].set_yscale("log")
    axes[0, 1].grid(True, alpha=0.22)
    _panel_label(axes[0, 1], 1)
    fig.legend(
        handles=_method_handles(PCA_METHODS),
        loc="upper center",
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84), w_pad=1.8)
    return _save_figure(fig, output_base, formats)


def plot_stage_costs(
    records: pd.DataFrame,
    *,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Plot cumulative end-to-end time split into measured stages."""
    resolutions = sorted(int(value) for value in records["resolution"].unique())
    fig, axes = plt.subplots(
        len(resolutions),
        1,
        figsize=(9.4, 3.0 * len(resolutions)),
        squeeze=False,
    )
    for panel, resolution in enumerate(resolutions):
        ax = axes[panel, 0]
        selected = records[records["resolution"] == resolution]
        grouped = selected.groupby("ablation_method").mean(numeric_only=True)
        x = np.arange(len(METHOD_ORDER))
        total = np.array(
            [grouped.loc[method, "timing_end_to_end"] for method in METHOD_ORDER]
        )
        bottom = np.zeros(len(METHOD_ORDER), dtype=float)
        for metric, label, color in STAGE_SPECS:
            values = np.array([grouped.loc[method, metric] for method in METHOD_ORDER])
            ax.bar(x, values, bottom=bottom, width=0.68, color=color, label=label)
            bottom += values
        other = np.maximum(total - bottom, 0.0)
        ax.bar(
            x,
            other,
            bottom=bottom,
            width=0.68,
            color="#B9B9B9",
            label="Other pipeline",
        )
        for index, value in enumerate(total):
            ax.text(
                index,
                value + 0.018 * max(total),
                f"{value:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_xticks(x, [METHOD_LABELS[method] for method in METHOD_ORDER])
        ax.tick_params(axis="x", rotation=18)
        ax.set_ylabel("Cumulative time (s)")
        ax.set_title(f"Poisson {resolution} x {resolution}")
        ax.grid(True, axis="y", alpha=0.22)
        _panel_label(ax, panel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=1.4)
    return _save_figure(fig, output_base, formats)


def plot_pca_cost_breakdown(
    records: pd.DataFrame,
    *,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Plot only PCA/SVD fitting, split into its measured substages."""
    methods = ("global_pca", "plain_l2l", "overlap_l2l", "two_scale")
    resolutions = sorted(int(value) for value in records["resolution"].unique())
    fig, axes = plt.subplots(
        1,
        len(resolutions),
        figsize=(4.25 * len(resolutions), 4.0),
        squeeze=False,
    )
    for panel, resolution in enumerate(resolutions):
        ax = axes[0, panel]
        selected = records[
            (records["resolution"] == resolution)
            & records["ablation_method"].isin(methods)
        ]
        grouped = selected.groupby("ablation_method").mean(numeric_only=True)
        x = np.arange(len(methods))
        bottom = np.zeros(len(methods), dtype=float)
        for metric, label, color in PCA_STAGE_SPECS:
            values = np.array([grouped.loc[method, metric] for method in methods])
            ax.bar(x, values, bottom=bottom, color=color, width=0.68, label=label)
            bottom += values
        for index, total in enumerate(bottom):
            ax.text(index, total + 0.025 * max(bottom), f"{total:.1f}", ha="center")
        ax.set_xticks(x, [METHOD_LABELS[method] for method in methods])
        ax.tick_params(axis="x", rotation=20)
        ax.set_ylabel("PCA/SVD fit time (s)")
        ax.set_title(f"Poisson {resolution} x {resolution}")
        ax.grid(True, axis="y", alpha=0.22)
        _panel_label(ax, panel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=1.3)
    return _save_figure(fig, output_base, formats)


def pca_cost_breakdown_table(records: pd.DataFrame) -> pd.DataFrame:
    """Return mean PCA/SVD substages for each unique representation."""
    methods = ("global_pca", "plain_l2l", "overlap_l2l", "two_scale")
    selected = records[records["ablation_method"].isin(methods)]
    columns = [spec[0] for spec in PCA_STAGE_SPECS] + ["pca_cost_total"]
    grouped = selected.groupby(["resolution", "ablation_method"])[columns].agg(
        ["mean", "std"]
    )
    rows: list[dict[str, Any]] = []
    for (resolution, method), values in grouped.iterrows():
        row: dict[str, Any] = {
            "Resolution": int(resolution),
            "Method": METHOD_LABELS[str(method)],
        }
        for metric, label, _color in PCA_STAGE_SPECS:
            row[f"{label} mean (s)"] = values[(metric, "mean")]
            row[f"{label} std (s)"] = values[(metric, "std")]
        row["Total PCA fit mean (s)"] = values[("pca_cost_total", "mean")]
        row["Total PCA fit std (s)"] = values[("pca_cost_total", "std")]
        rows.append(row)
    return pd.DataFrame(rows)


def common_seam_table(
    records: pd.DataFrame,
    *,
    results_path: Path,
) -> pd.DataFrame:
    """Re-evaluate retained seed-zero predictions on equal seam locations."""
    rows: list[dict[str, Any]] = []
    seed_zero = records[records["seed"] == 0]
    for resolution in sorted(int(value) for value in records["resolution"].unique()):
        stride = resolution // 4
        for method in METHOD_ORDER:
            selected = seed_zero[
                (seed_zero["resolution"] == resolution)
                & (seed_zero["ablation_method"] == method)
            ]
            if selected.empty:
                return pd.DataFrame()
            path = _prediction_path(selected.iloc[0], results_path)
            if not path.is_file():
                return pd.DataFrame()
            with np.load(path, allow_pickle=False) as archive:
                prediction = np.asarray(archive["y_pred"], dtype=np.float64)
                truth = np.asarray(archive["y_true"], dtype=np.float64)
            dx = 1.0 / float(resolution - 1)
            rows.append(
                {
                    "Resolution": resolution,
                    "Method": METHOD_LABELS[method],
                    "Seed": 0,
                    "Diagnostic stride": stride,
                    "Value jump": interface_value_jump(
                        prediction,
                        patch_size=stride,
                        stride=stride,
                    ),
                    "Flux jump": interface_flux_jump(
                        prediction,
                        patch_size=stride,
                        stride=stride,
                        dx=dx,
                    ),
                    "Value trace error": interface_value_trace_error(
                        prediction,
                        truth,
                        patch_size=stride,
                        stride=stride,
                    ),
                    "Flux trace error": interface_flux_trace_error(
                        prediction,
                        truth,
                        patch_size=stride,
                        stride=stride,
                        dx=dx,
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_qualitative_diagnostics(
    records: pd.DataFrame,
    *,
    results_path: Path,
    resolution: int,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Plot truth, prediction, error, spectrum, and PDF for every method."""
    selected = records[(records["resolution"] == resolution) & (records["seed"] == 0)]
    paths: dict[str, Path] = {}
    for method in METHOD_ORDER:
        row = selected[selected["ablation_method"] == method]
        if row.empty:
            return []
        path = _prediction_path(row.iloc[0], results_path)
        if not path.is_file():
            return []
        paths[method] = path

    with np.load(paths["plain_l2l"], allow_pickle=False) as archive:
        truth_all = np.asarray(archive["y_true"], dtype=np.float32)
        prediction_all = np.asarray(archive["y_pred"], dtype=np.float32)
        sample_indices = np.asarray(archive["sample_indices"], dtype=np.int64)
        sample_mre = np.linalg.norm(
            (prediction_all - truth_all).reshape(len(truth_all), -1),
            axis=1,
        ) / np.maximum(
            np.linalg.norm(truth_all.reshape(len(truth_all), -1), axis=1),
            1.0e-12,
        )
        chosen = int(np.argsort(sample_mre)[len(sample_mre) // 2])
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
            predictions[method] = np.asarray(
                archive["y_pred"][index],
                dtype=np.float32,
            )

    all_fields = np.stack([truth, *predictions.values()])
    solution_min = float(np.min(all_fields))
    solution_max = float(np.max(all_fields))
    errors = {
        method: np.abs(prediction - truth)
        for method, prediction in predictions.items()
    }
    error_max = max(
        float(np.percentile(np.stack(list(errors.values())), 99.5)),
        1.0e-12,
    )
    bins = np.linspace(solution_min, solution_max, 40)
    k_max = resolution // 2

    fig, axes = plt.subplots(
        len(METHOD_ORDER),
        5,
        figsize=(13.6, 2.35 * len(METHOD_ORDER)),
        squeeze=False,
    )
    titles = (
        "Ground truth",
        "Prediction",
        "Absolute error",
        "Energy spectrum",
        "Value PDF",
    )
    for column, title in enumerate(titles):
        axes[0, column].set_title(title)

    for row_index, method in enumerate(METHOD_ORDER):
        prediction = predictions[method]
        error = errors[method]
        axes[row_index, 0].imshow(
            truth,
            origin="lower",
            cmap="RdBu_r",
            vmin=solution_min,
            vmax=solution_max,
        )
        axes[row_index, 1].imshow(
            prediction,
            origin="lower",
            cmap="RdBu_r",
            vmin=solution_min,
            vmax=solution_max,
        )
        axes[row_index, 2].imshow(
            error,
            origin="lower",
            cmap="magma",
            vmin=0.0,
            vmax=error_max,
        )
        mre = float(np.linalg.norm(prediction - truth)) / max(
            float(np.linalg.norm(truth)),
            1.0e-12,
        )
        axes[row_index, 1].text(
            0.03,
            0.97,
            f"MRE {mre:.4f}",
            transform=axes[row_index, 1].transAxes,
            ha="left",
            va="top",
            color="black",
            fontsize=8,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78,
                "pad": 2.0,
            },
        )
        axes[row_index, 0].set_ylabel(
            METHOD_LABELS[method],
            fontweight="bold",
        )
        for column in range(3):
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])
        _plot_sample_spectrum(
            axes[row_index, 3],
            truth,
            prediction,
            k_max=k_max,
        )
        _plot_sample_pdf(axes[row_index, 4], truth, prediction, bins=bins)
        if row_index == len(METHOD_ORDER) - 1:
            axes[row_index, 3].set_xlabel("Wavenumber $k$")
            axes[row_index, 4].set_xlabel("Field value")
    fig.suptitle(
        f"Poisson {resolution}, seed 0, median plain-L2L sample {sample_id}",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98), h_pad=0.9, w_pad=0.8)
    return _save_figure(fig, output_base, formats)


def _plot_sample_spectrum(
    axis: plt.Axes,
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    k_max: int,
) -> None:
    truth_energy = energy_spectrum_2d(truth)
    prediction_energy = energy_spectrum_2d(prediction)
    stop = min(k_max, len(truth_energy) - 1, len(prediction_energy) - 1)
    k = np.arange(1, stop + 1)
    axis.plot(
        k,
        truth_energy[1 : stop + 1] + 1.0e-30,
        color="black",
        linewidth=1.5,
        label="Truth",
    )
    axis.plot(
        k,
        prediction_energy[1 : stop + 1] + 1.0e-30,
        color="tab:red",
        linewidth=1.5,
        label="Prediction",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(1, max(stop, 2))
    axis.grid(True, alpha=0.22)
    axis.legend(frameon=False, fontsize=7)


def _plot_sample_pdf(
    axis: plt.Axes,
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    bins: np.ndarray,
) -> None:
    axis.hist(
        truth.ravel(),
        bins=bins.tolist(),
        density=True,
        histtype="step",
        color="black",
        linewidth=1.5,
        label="Truth",
    )
    axis.hist(
        prediction.ravel(),
        bins=bins.tolist(),
        density=True,
        histtype="step",
        color="tab:red",
        linewidth=1.5,
        label="Prediction",
    )
    axis.grid(True, alpha=0.22)


def _plot_metric_curves(
    ax: plt.Axes,
    records: pd.DataFrame,
    *,
    metric: str,
    scale: float,
) -> None:
    for method in METHOD_ORDER:
        _plot_one_curve(
            ax,
            records[records["ablation_method"] == method],
            metric=metric,
            method=method,
            scale=scale,
        )
    resolutions = sorted(int(value) for value in records["resolution"].unique())
    ax.set_xticks(resolutions, [str(value) for value in resolutions])


def _plot_one_curve(
    ax: plt.Axes,
    selected: pd.DataFrame,
    *,
    metric: str,
    method: str,
    scale: float,
) -> None:
    grouped = selected.groupby("resolution")[metric].agg(["mean", "std"])
    resolutions = sorted(int(value) for value in grouped.index)
    means = grouped.loc[resolutions, "mean"].to_numpy(dtype=float) * scale
    stds = grouped.loc[resolutions, "std"].fillna(0.0).to_numpy(dtype=float) * scale
    ax.errorbar(
        resolutions,
        means,
        yerr=stds,
        color=METHOD_COLORS[method],
        marker=METHOD_MARKERS[method],
        linewidth=STYLE["line_width"],
        markersize=STYLE["marker_size"],
        capsize=STYLE["capsize"],
        label=METHOD_LABELS[method],
    )
    for resolution in resolutions:
        values = (
            selected.loc[selected["resolution"] == resolution, metric].to_numpy(
                dtype=float
            )
            * scale
        )
        jitter = np.linspace(-1.5, 1.5, len(values))
        ax.scatter(
            resolution + jitter,
            values,
            color=METHOD_COLORS[method],
            s=8,
            alpha=0.28,
            linewidths=0,
        )
    ax.set_xticks(resolutions, [str(value) for value in resolutions])


def _main_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        row: dict[str, Any] = {"Method": METHOD_LABELS[method]}
        for resolution in sorted(int(value) for value in summary["resolution"].unique()):
            selected = summary[
                (summary["resolution"] == resolution)
                & (summary["ablation_method"] == method)
            ].iloc[0]
            row[f"MRE {resolution} (%)"] = _format_mean_std(
                selected["metric_mre_mean"],
                selected["metric_mre_std"],
                style="percent",
            )
            row[f"SSIM {resolution}"] = _format_mean_std(
                selected["metric_ssim_mean_mean"],
                selected["metric_ssim_mean_std"],
                style="ssim",
            )
            row[f"Time {resolution} (s)"] = _format_mean_std(
                selected["timing_end_to_end_mean"],
                selected["timing_end_to_end_std"],
                style="time",
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _formatted_summary(
    summary: pd.DataFrame,
    specs: Sequence[tuple[str, str, str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        formatted: dict[str, Any] = {
            "Resolution": int(row["resolution"]),
            "Method": METHOD_LABELS[str(row["ablation_method"])],
        }
        for metric, label, style in specs:
            formatted[label] = _format_mean_std(
                row.get(f"{metric}_mean"),
                row.get(f"{metric}_std"),
                style=style,
            )
        rows.append(formatted)
    return pd.DataFrame(rows)


def _format_common_seam_table(table: pd.DataFrame) -> pd.DataFrame:
    formatted = table.copy()
    for column in (
        "Value jump",
        "Flux jump",
        "Value trace error",
        "Flux trace error",
    ):
        formatted[column] = formatted[column].map(lambda value: f"{float(value):.3e}")
    return formatted


def _format_capacity_table(table: pd.DataFrame) -> pd.DataFrame:
    formatted = table.copy()
    for column in (
        "Input dimensions",
        "Output dimensions",
        "Coarse dimensions",
        "Residual dimensions",
    ):
        formatted[column] = formatted[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.1f}"
        )
    for column in ("Learned MRE (%)", "Oracle MRE (%)", "Learned / oracle"):
        formatted[column] = formatted[column].map(lambda value: f"{float(value):.3f}")
    return formatted


def _format_time_projection_table(table: pd.DataFrame) -> pd.DataFrame:
    """Format log-log trend projections for the appendix LaTex table."""
    formatted = table.copy()
    formatted["Fitted log-log exponent"] = formatted[
        "Fitted log-log exponent"
    ].map(lambda value: f"{float(value):.3f}")
    for column in formatted.columns:
        if column.endswith("(s)"):
            formatted[column] = formatted[column].map(
                lambda value: f"{float(value):.1f}"
            )
    return formatted


def _prediction_path(row: pd.Series, results_path: Path) -> Path:
    run_dir = Path(str(row["run_dir"]))
    candidates = (run_dir, ROOT / run_dir, results_path / run_dir)
    for candidate in candidates:
        path = candidate / "predictions_test.npz"
        if path.is_file():
            return path
    return run_dir / "predictions_test.npz"


def _method_handles(
    methods: Sequence[str] = METHOD_ORDER,
) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linewidth=STYLE["line_width"],
            label=METHOD_LABELS[method],
        )
        for method in methods
    ]


def _write_table_pair(
    table: pd.DataFrame,
    base_path: Path,
    *,
    csv: bool = True,
) -> list[Path]:
    written: list[Path] = []
    if csv:
        csv_path = base_path.with_suffix(".csv")
        table.to_csv(csv_path, index=False)
        written.append(csv_path)
    tex_path = base_path.with_suffix(".tex")
    tex_path.write_text(_to_latex(table), encoding="utf-8")
    written.append(tex_path)
    return written


def _format_mean_std(mean: Any, std: Any, *, style: str) -> str:
    if pd.isna(mean):
        return ""
    mean_value = float(mean)
    std_value = 0.0 if pd.isna(std) else float(std)
    if style == "scientific":
        return f"{mean_value:.3e} +/- {std_value:.1e}"
    if style == "ssim":
        return f"{mean_value:.4f} +/- {std_value:.4f}"
    if style == "time":
        return f"{mean_value:.1f} +/- {std_value:.1f}"
    if style == "percent":
        return f"{100.0 * mean_value:.3f} +/- {100.0 * std_value:.3f}"
    return f"{mean_value:.5f} +/- {std_value:.5f}"


def _to_latex(table: pd.DataFrame) -> str:
    return table.to_latex(index=False, escape=True).replace(
        "+/-",
        r"\ensuremath{\pm}",
    )


def _panel_label(ax: plt.Axes, index: int) -> None:
    ax.text(
        0.02,
        0.96,
        f"({chr(97 + index)})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": STYLE["dpi"],
            "savefig.dpi": STYLE["dpi"],
            "font.size": STYLE["font_size"],
            "axes.titlesize": STYLE["title_size"],
            "axes.labelsize": STYLE["label_size"],
            "xtick.labelsize": STYLE["tick_size"],
            "ytick.labelsize": STYLE["tick_size"],
            "legend.fontsize": STYLE["legend_size"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.formatter.use_mathtext": True,
        }
    )


def _save_figure(
    fig: plt.Figure,
    base_path: Path,
    formats: Sequence[str],
) -> list[Path]:
    written: list[Path] = []
    for suffix in formats:
        path = base_path.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default="paper_results/resolution_sweep",
    )
    parser.add_argument(
        "--formats",
        default="png,pdf",
        help="Comma-separated output formats.",
    )
    parser.add_argument(
        "--expected-seeds",
        type=int,
        default=3,
        help="Required unique seeds per cell; use 0 to disable.",
    )
    parser.add_argument(
        "--skip-qualitative",
        action="store_true",
        help="Skip the prediction-archive diagnostic figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    formats = tuple(piece.strip() for piece in args.formats.split(",") if piece.strip())
    written = make_resolution_sweep_outputs(
        results_dir=args.results_dir,
        formats=formats,
        expected_seeds=args.expected_seeds,
        qualitative=not args.skip_qualitative,
    )
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
