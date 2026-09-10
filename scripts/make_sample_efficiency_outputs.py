#!/usr/bin/env python
"""Render manuscript figures and tables for the sample-efficiency study."""

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

METHOD_ORDER = (
    "plain_l2l",
    "overlap_l2l",
    "two_scale",
    "two_scale_interface",
)
METHOD_LABELS = {
    "plain_l2l": "Plain L2L",
    "overlap_l2l": "L2L + overlap",
    "two_scale": "Two-scale",
    "two_scale_interface": "Two-scale + interface",
}
METHOD_COLORS = {
    "plain_l2l": "#6F6F6F",
    "overlap_l2l": "#4C78A8",
    "two_scale": "#F58518",
    "two_scale_interface": "#54A24B",
}
METHOD_MARKERS = {
    "plain_l2l": "o",
    "overlap_l2l": "s",
    "two_scale": "^",
    "two_scale_interface": "D",
}
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
MAIN_METRICS = (
    ("metric_mre", "MRE (%)", "percent"),
    ("metric_ssim_mean", "SSIM", "ssim"),
    ("metric_interface_value_trace_error", "Value trace error", "scientific"),
    ("metric_interface_flux_trace_error", "Flux trace error", "scientific"),
    ("metric_relative_spectrum_error", "Spectral error", "metric"),
    ("timing_end_to_end", "Total time (s)", "time"),
)
FULL_METRICS = (
    ("metric_mre", "MRE", "metric"),
    ("metric_pca_oracle_mre", "PCA-oracle MRE", "metric"),
    ("metric_ssim_mean", "SSIM", "ssim"),
    ("metric_mse", "MSE", "scientific"),
    ("metric_mae", "MAE", "scientific"),
    ("metric_interface_jump", "Value jump", "scientific"),
    ("metric_interface_flux_jump", "Flux jump", "scientific"),
    ("metric_interface_value_trace_error", "Value trace error", "scientific"),
    ("metric_interface_flux_trace_error", "Flux trace error", "scientific"),
    ("metric_relative_spectrum_error", "Spectral error", "metric"),
    ("metric_poisson_residual_rms", "PDE residual RMS", "metric"),
    ("timing_pca_fit", "PCA fit (s)", "time"),
    ("timing_latent_transform", "Latent transform (s)", "time"),
    ("timing_nn_train", "NN train (s)", "time"),
    ("timing_inference", "Inference (s)", "time"),
    ("invocation_timing_nn_train", "Invocation NN train (s)", "time"),
    ("invocation_timing_end_to_end", "Invocation total (s)", "time"),
    ("timing_end_to_end", "Cumulative total (s)", "time"),
)


def make_sample_efficiency_outputs(
    *,
    results_dir: str | Path,
    formats: Sequence[str] = ("png", "pdf"),
    expected_seeds: int = 3,
    qualitative: bool = True,
    common_seam_stride: int = 32,
) -> list[Path]:
    """Generate main-text and appendix assets from a completed study."""
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
            output_base=main_figures / "sample_efficiency_accuracy",
            formats=formats,
        )
    )
    written.extend(
        plot_representation_gap(
            records,
            output_base=main_figures / "sample_efficiency_representation_gap",
            formats=formats,
        )
    )
    written.extend(
        plot_artifact_scaling(
            records,
            output_base=main_figures / "sample_efficiency_artifacts",
            formats=formats,
        )
    )
    written.extend(
        plot_cost_scaling(
            records,
            output_base=appendix_figures / "sample_efficiency_cost",
            formats=formats,
        )
    )
    pca_costs = add_pca_cost_substages(records, results_path=results_path)
    written.extend(
        plot_pca_cost_breakdown(
            pca_costs,
            output_base=appendix_figures / "sample_efficiency_pca_breakdown",
            formats=formats,
        )
    )
    pca_table = pca_cost_breakdown_table(pca_costs)
    if not pca_table.empty:
        pca_path = appendix_tables / "sample_efficiency_pca_breakdown.csv"
        pca_table.to_csv(pca_path, index=False)
        written.append(pca_path)
        written.extend(
            _write_table_pair(
                pca_table,
                appendix_tables / "sample_efficiency_pca_breakdown",
                csv=False,
            )
        )

    common_seam = common_seam_endpoint_table(
        records,
        results_path=results_path,
        stride=common_seam_stride,
    )
    if not common_seam.empty:
        common_path = appendix_tables / "sample_efficiency_common_seam_seed0.csv"
        common_seam.to_csv(common_path, index=False)
        written.append(common_path)
        written.extend(
            _write_table_pair(
                _format_common_seam_table(common_seam),
                appendix_tables / "sample_efficiency_common_seam_seed0",
                csv=False,
            )
        )
    if qualitative:
        written.extend(
            plot_endpoint_qualitative(
                records,
                results_path=results_path,
                output_base=appendix_figures / "sample_efficiency_qualitative",
                formats=formats,
            )
        )

    manifest = {
        "results_dir": str(results_path),
        "records": int(len(records)),
        "sample_sizes": sorted(
            int(value) for value in records["ablation_sample_size"].unique()
        ),
        "methods": list(METHOD_ORDER),
        "expected_seeds": expected_seeds,
        "common_seam_stride": common_seam_stride,
        "common_seam_rows": int(len(common_seam)),
        "qualitative": qualitative,
        "outputs": [str(path) for path in written],
    }
    manifest_path = results_path / "sample_efficiency_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    written.append(manifest_path)
    return written


def load_records(path: Path) -> pd.DataFrame:
    """Load the stable study roll-up and coerce metric fields to numeric."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing sample-efficiency roll-up: {path}")
    frame = pd.read_csv(path)
    for column in frame.columns:
        if column.startswith(
            ("metric_", "validation_metric_", "timing_", "invocation_timing_")
        ) or column in {"seed", "resolution", "ablation_sample_size"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def validate_records(records: pd.DataFrame, *, expected_seeds: int) -> None:
    """Reject incomplete, duplicated, or malformed study matrices."""
    required = {
        "seed",
        "resolution",
        "baseline_model",
        "run_dir",
        "ablation_sample_size",
        *QUALITY_METRICS,
        *TIMING_METRICS,
    }
    missing = sorted(required.difference(records.columns))
    if missing:
        raise ValueError(
            f"Missing required sample-efficiency columns: {', '.join(missing)}"
        )
    unexpected = sorted(set(records["baseline_model"]) - set(METHOD_ORDER))
    if unexpected:
        raise ValueError(f"Unexpected sample-efficiency methods: {', '.join(unexpected)}")
    missing_methods = sorted(set(METHOD_ORDER) - set(records["baseline_model"]))
    if missing_methods:
        raise ValueError(f"Missing sample-efficiency methods: {', '.join(missing_methods)}")
    keys = ["ablation_sample_size", "baseline_model", "seed"]
    duplicates = records.duplicated(keys, keep=False)
    if duplicates.any():
        details = records.loc[duplicates, keys]
        raise ValueError(f"Duplicate sample-efficiency records:\n{details.to_string(index=False)}")
    counts = records.groupby(["ablation_sample_size", "baseline_model"])["seed"].nunique()
    if expected_seeds > 0:
        incomplete = counts[counts != expected_seeds]
        if not incomplete.empty:
            details = ", ".join(
                f"{int(size)}/{method}: {int(count)}"
                for (size, method), count in incomplete.items()
            )
            raise ValueError(
                f"Expected {expected_seeds} seeds per sample-efficiency cell; "
                f"found {details}."
            )
    expected_cells = pd.MultiIndex.from_product(
        [
            sorted(records["ablation_sample_size"].unique()),
            METHOD_ORDER,
        ],
        names=["ablation_sample_size", "baseline_model"],
    )
    missing_cells = expected_cells.difference(counts.index)
    if len(missing_cells):
        raise ValueError(f"Missing sample-efficiency cells: {list(missing_cells)}")


def summarize_records(records: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and sample standard deviation over seeds."""
    numeric = [
        column
        for column in (*QUALITY_METRICS, *TIMING_METRICS)
        if column in records
    ]
    summary = (
        records.groupby(["ablation_sample_size", "baseline_model"], as_index=False)[
            numeric
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(piece) for piece in column if piece)
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    counts = records.groupby(["ablation_sample_size", "baseline_model"])["seed"].nunique()
    summary["n_seeds"] = counts.reindex(
        pd.MultiIndex.from_frame(
            summary[["ablation_sample_size", "baseline_model"]]
        )
    ).to_numpy()
    summary["method_order"] = summary["baseline_model"].map(
        {name: index for index, name in enumerate(METHOD_ORDER)}
    )
    return summary.sort_values(
        ["ablation_sample_size", "method_order"],
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
    numeric_path = appendix_dir / "sample_efficiency_summary_numeric.csv"
    summary.to_csv(numeric_path, index=False)
    written.append(numeric_path)

    main = _formatted_summary(summary, MAIN_METRICS)
    full = _formatted_summary(summary, FULL_METRICS)
    written.extend(_write_table_pair(main, main_dir / "sample_efficiency_main"))
    written.extend(_write_table_pair(full, appendix_dir / "sample_efficiency_full"))

    cost_specs = tuple(spec for spec in FULL_METRICS if spec[2] == "time")
    cost = _formatted_summary(summary, cost_specs)
    written.extend(_write_table_pair(cost, appendix_dir / "sample_efficiency_cost"))

    relative = relative_change_table(records)
    relative_path = appendix_dir / "sample_efficiency_relative_changes.csv"
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
            appendix_dir / "sample_efficiency_relative_changes",
            csv=False,
        )
    )

    per_seed_columns = [
        column
        for column in (
            "ablation_sample_size",
            "seed",
            "baseline_model",
            "config_hash",
            *QUALITY_METRICS,
            *TIMING_METRICS,
        )
        if column in records
    ]
    per_seed = records[per_seed_columns].sort_values(
        ["ablation_sample_size", "baseline_model", "seed"]
    )
    per_seed_path = appendix_dir / "sample_efficiency_per_seed.csv"
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
        "total_time": "timing_end_to_end",
    }
    grouped = records.groupby(["ablation_sample_size", "baseline_model"])[
        list(metrics.values())
    ].mean()
    rows: list[dict[str, float | int]] = []
    for size in sorted(int(value) for value in records["ablation_sample_size"].unique()):
        values = grouped.loc[size]
        row: dict[str, float | int] = {"Training samples": size}
        for short, metric in metrics.items():
            overlap = float(values.loc["overlap_l2l", metric])
            two_scale = float(values.loc["two_scale", metric])
            interface = float(values.loc["two_scale_interface", metric])
            if short == "ssim":
                row[f"two_scale_vs_overlap_{short}_pct"] = (
                    100.0 * (two_scale - overlap) / overlap
                )
                row[f"interface_vs_two_scale_{short}_pct"] = (
                    100.0 * (interface - two_scale) / two_scale
                )
            else:
                row[f"two_scale_vs_overlap_{short}_pct"] = (
                    100.0 * (overlap - two_scale) / overlap
                )
                row[f"interface_vs_two_scale_{short}_pct"] = (
                    100.0 * (two_scale - interface) / two_scale
                )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_accuracy_scaling(
    records: pd.DataFrame,
    *,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Plot reconstruction accuracy against the number of training samples."""
    specs = (
        ("metric_mre", "Mean relative error (%)", 100.0),
        ("metric_ssim_mean", "SSIM", 1.0),
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.8), squeeze=False)
    for index, (metric, label, scale) in enumerate(specs):
        ax = axes[0, index]
        _plot_metric_curves(ax, records, metric=metric, scale=scale)
        ax.set_xlabel("Training samples")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.22)
        ax.text(
            0.02,
            0.96,
            f"({chr(97 + index)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
    handles = _method_handles()
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=1.8)
    return _save_figure(fig, output_base, formats)


def plot_representation_gap(
    records: pd.DataFrame,
    *,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Contrast PCA representation floors with learned latent-map error."""
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.8), squeeze=False)
    oracle = records.groupby("baseline_model")["metric_pca_oracle_mre"].agg(
        ["mean", "std"]
    )
    x = np.arange(len(METHOD_ORDER))
    means = np.array([oracle.loc[method, "mean"] for method in METHOD_ORDER]) * 100.0
    stds = np.array([oracle.loc[method, "std"] for method in METHOD_ORDER]) * 100.0
    axes[0, 0].bar(
        x,
        means,
        yerr=stds,
        color=[METHOD_COLORS[method] for method in METHOD_ORDER],
        width=0.68,
        capsize=STYLE["capsize"],
    )
    for index, value in enumerate(means):
        axes[0, 0].text(
            index,
            value + 0.025 * means.max(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[0, 0].set_xticks(x, [METHOD_LABELS[method] for method in METHOD_ORDER])
    axes[0, 0].tick_params(axis="x", rotation=18)
    axes[0, 0].set_ylabel("PCA-oracle MRE (%)")
    axes[0, 0].grid(True, axis="y", alpha=0.22)
    axes[0, 0].set_title("Representation error floor")

    ratio_records = records.copy()
    ratio_records["learned_oracle_ratio"] = (
        ratio_records["metric_mre"] / ratio_records["metric_pca_oracle_mre"]
    )
    _plot_metric_curves(
        axes[0, 1],
        ratio_records,
        metric="learned_oracle_ratio",
        scale=1.0,
    )
    axes[0, 1].axhline(1.0, color="#777777", linestyle=":", linewidth=1.1)
    axes[0, 1].set_xlabel("Training samples")
    axes[0, 1].set_ylabel("Learned MRE / PCA-oracle MRE")
    axes[0, 1].set_yscale("log")
    axes[0, 1].grid(True, alpha=0.22)
    axes[0, 1].set_title("Latent-map trainability gap")
    for index, ax in enumerate(axes.flat):
        ax.text(
            0.02,
            0.96,
            f"({chr(97 + index)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
    fig.tight_layout(w_pad=1.9)
    return _save_figure(fig, output_base, formats)


def plot_artifact_scaling(
    records: pd.DataFrame,
    *,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Plot truth-referenced seam and radial-spectrum errors."""
    specs = (
        ("metric_interface_value_trace_error", "Value trace error\n(native seam grid)"),
        ("metric_interface_flux_trace_error", "Flux trace error\n(native seam grid)"),
        ("metric_relative_spectrum_error", "Relative spectral error"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.75), squeeze=False)
    for index, (metric, label) in enumerate(specs):
        ax = axes[0, index]
        _plot_metric_curves(ax, records, metric=metric, scale=1.0)
        ax.set_xlabel("Training samples")
        ax.set_ylabel(label)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.22)
        ax.text(
            0.02,
            0.96,
            f"({chr(97 + index)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
    fig.legend(handles=_method_handles(), loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=1.4)
    return _save_figure(fig, output_base, formats)


def plot_cost_scaling(
    records: pd.DataFrame,
    *,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Plot PCA, neural-network, and cumulative end-to-end costs."""
    specs = (
        ("timing_pca_fit", "PCA fit (s)"),
        ("timing_nn_train", "NN training, cumulative (s)"),
        ("timing_end_to_end", "End-to-end, cumulative (s)"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.75), squeeze=False)
    for index, (metric, label) in enumerate(specs):
        ax = axes[0, index]
        _plot_metric_curves(ax, records, metric=metric, scale=1.0)
        ax.set_xlabel("Training samples")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.22)
        ax.text(
            0.02,
            0.96,
            f"({chr(97 + index)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
    fig.legend(handles=_method_handles(), loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=1.4)
    return _save_figure(fig, output_base, formats)


def plot_pca_cost_breakdown(
    records: pd.DataFrame,
    *,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Plot only PCA/SVD fit time, decomposed into nonoverlapping substages."""
    methods = ("plain_l2l", "overlap_l2l", "two_scale")
    sizes = sorted(int(value) for value in records["ablation_sample_size"].unique())
    columns = 2
    rows = int(np.ceil(len(sizes) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(9.4, 3.45 * rows),
        squeeze=False,
    )
    for panel, size in enumerate(sizes):
        ax = axes.flat[panel]
        selected = records[
            (records["ablation_sample_size"] == size)
            & records["baseline_model"].isin(methods)
        ]
        grouped = selected.groupby("baseline_model").mean(numeric_only=True)
        x = np.arange(len(methods))
        bottom = np.zeros(len(methods), dtype=float)
        for metric, label, color in PCA_STAGE_SPECS:
            values = np.array([grouped.loc[method, metric] for method in methods])
            ax.bar(x, values, bottom=bottom, color=color, width=0.68, label=label)
            bottom += values
        for index, total in enumerate(bottom):
            ax.text(index, total + 0.025 * max(bottom), f"{total:.2f}", ha="center")
        ax.set_xticks(x, [METHOD_LABELS[method] for method in methods])
        ax.tick_params(axis="x", rotation=15)
        ax.set_ylabel("PCA/SVD fit time (s)")
        ax.set_title(f"m = {size:,}")
        ax.grid(True, axis="y", alpha=0.22)
        ax.text(
            0.02,
            0.96,
            f"({chr(97 + panel)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
    for ax in axes.flat[len(sizes) :]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=1.4, w_pad=1.3)
    return _save_figure(fig, output_base, formats)


def pca_cost_breakdown_table(records: pd.DataFrame) -> pd.DataFrame:
    """Return per-cell mean PCA/SVD substages for the manuscript appendix."""
    methods = ("plain_l2l", "overlap_l2l", "two_scale")
    selected = records[records["baseline_model"].isin(methods)]
    columns = [spec[0] for spec in PCA_STAGE_SPECS] + ["pca_cost_total"]
    if selected["pca_cost_total"].isna().all():
        return pd.DataFrame()
    grouped = selected.groupby(["ablation_sample_size", "baseline_model"])[
        columns
    ].agg(["mean", "std"])
    rows: list[dict[str, Any]] = []
    for (size, method), values in grouped.iterrows():
        row: dict[str, Any] = {
            "Training samples": int(size),
            "Method": METHOD_LABELS[str(method)],
        }
        for metric, label, _color in PCA_STAGE_SPECS:
            row[f"{label} mean (s)"] = values[(metric, "mean")]
            row[f"{label} std (s)"] = values[(metric, "std")]
        row["Total PCA fit mean (s)"] = values[("pca_cost_total", "mean")]
        row["Total PCA fit std (s)"] = values[("pca_cost_total", "std")]
        rows.append(row)
    return pd.DataFrame(rows)


def common_seam_endpoint_table(
    records: pd.DataFrame,
    *,
    results_path: Path,
    stride: int,
) -> pd.DataFrame:
    """Re-evaluate retained seed-0 endpoint predictions on one seam grid."""
    seed_zero = records[records["seed"] == 0]
    rows: list[dict[str, float | int | str]] = []
    for size in sorted(int(value) for value in records["ablation_sample_size"].unique()):
        for method in METHOD_ORDER:
            selected = seed_zero[
                (seed_zero["ablation_sample_size"] == size)
                & (seed_zero["baseline_model"] == method)
            ]
            if selected.empty:
                continue
            prediction_path = _prediction_path(selected.iloc[0], results_path)
            if not prediction_path.is_file():
                continue
            with np.load(prediction_path) as archive:
                prediction = archive["y_pred"]
                truth = archive["y_true"]
            resolution = int(prediction.shape[-1])
            if stride >= resolution:
                continue
            dx = 1.0 / float(resolution - 1)
            rows.append(
                {
                    "Training samples": size,
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


def plot_endpoint_qualitative(
    records: pd.DataFrame,
    *,
    results_path: Path,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Compare low- and full-data predictions for one aligned median sample."""
    sizes = sorted(int(value) for value in records["ablation_sample_size"].unique())
    if len(sizes) < 2:
        return []
    low_size, high_size = sizes[0], sizes[-1]
    seed_zero = records[records["seed"] == 0]
    paths: dict[tuple[int, str], Path] = {}
    for size in (low_size, high_size):
        for method in METHOD_ORDER:
            selected = seed_zero[
                (seed_zero["ablation_sample_size"] == size)
                & (seed_zero["baseline_model"] == method)
            ]
            if selected.empty:
                return []
            path = _prediction_path(selected.iloc[0], results_path)
            if not path.is_file():
                return []
            paths[(size, method)] = path

    reference_path = paths[(high_size, "plain_l2l")]
    with np.load(reference_path) as archive:
        truth_all = archive["y_true"]
        prediction_all = archive["y_pred"]
        sample_indices = archive["sample_indices"]
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

    predictions: dict[tuple[int, str], np.ndarray] = {}
    for key, path in paths.items():
        with np.load(path) as archive:
            matches = np.flatnonzero(archive["sample_indices"] == sample_id)
            if len(matches) != 1:
                raise ValueError(f"Sample {sample_id} is not uniquely aligned in {path}.")
            predictions[key] = archive["y_pred"][int(matches[0])].copy()

    all_fields = np.stack([truth, *predictions.values()])
    field_lo, field_hi = np.percentile(all_fields, [0.5, 99.5])
    errors = {key: np.abs(value - truth) for key, value in predictions.items()}
    error_hi = float(np.percentile(np.stack(list(errors.values())), 99.5))
    resolution = int(truth.shape[-1])
    patch_size = max(resolution // 4, 1)
    fig, axes = plt.subplots(
        len(METHOD_ORDER),
        5,
        figsize=(12.4, 2.45 * len(METHOD_ORDER)),
        squeeze=False,
    )
    titles = (
        "Ground truth",
        f"Prediction, m = {low_size:,}",
        f"Absolute error, m = {low_size:,}",
        f"Prediction, m = {high_size:,}",
        f"Absolute error, m = {high_size:,}",
    )
    for column, title in enumerate(titles):
        axes[0, column].set_title(title)
    for row_index, method in enumerate(METHOD_ORDER):
        axes[row_index, 0].imshow(
            truth,
            origin="lower",
            cmap="RdBu_r",
            vmin=field_lo,
            vmax=field_hi,
        )
        axes[row_index, 0].set_ylabel(METHOD_LABELS[method], fontweight="bold")
        _draw_patch_grid(axes[row_index, 0], resolution, patch_size)
        for pair_index, size in enumerate((low_size, high_size)):
            prediction = predictions[(size, method)]
            error = errors[(size, method)]
            pred_column = 1 + 2 * pair_index
            error_column = pred_column + 1
            axes[row_index, pred_column].imshow(
                prediction,
                origin="lower",
                cmap="RdBu_r",
                vmin=field_lo,
                vmax=field_hi,
            )
            axes[row_index, error_column].imshow(
                error,
                origin="lower",
                cmap="magma",
                vmin=0.0,
                vmax=error_hi,
            )
            for column in (pred_column, error_column):
                _draw_patch_grid(axes[row_index, column], resolution, patch_size)
            mre = float(np.linalg.norm(prediction - truth)) / max(
                float(np.linalg.norm(truth)),
                1.0e-12,
            )
            axes[row_index, pred_column].text(
                0.03,
                0.97,
                f"MRE = {mre:.4f}",
                transform=axes[row_index, pred_column].transAxes,
                va="top",
                color="white",
                fontsize=8,
                bbox={
                    "facecolor": "black",
                    "alpha": 0.5,
                    "pad": 2,
                    "edgecolor": "none",
                },
            )
        for ax in axes[row_index]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(
        f"Poisson {resolution}, seed 0, median full-data plain-L2L sample {sample_id}",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98), h_pad=0.8, w_pad=0.6)
    return _save_figure(fig, output_base, formats)


def _plot_metric_curves(
    ax: plt.Axes,
    records: pd.DataFrame,
    *,
    metric: str,
    scale: float,
) -> None:
    sizes = sorted(int(value) for value in records["ablation_sample_size"].unique())
    for method in METHOD_ORDER:
        selected = records[records["baseline_model"] == method]
        grouped = selected.groupby("ablation_sample_size")[metric].agg(["mean", "std"])
        means = grouped.loc[sizes, "mean"].to_numpy(dtype=float) * scale
        stds = grouped.loc[sizes, "std"].fillna(0.0).to_numpy(dtype=float) * scale
        ax.errorbar(
            sizes,
            means,
            yerr=stds,
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linewidth=STYLE["line_width"],
            markersize=STYLE["marker_size"],
            capsize=STYLE["capsize"],
            label=METHOD_LABELS[method],
        )
        for size in sizes:
            values = (
                selected.loc[selected["ablation_sample_size"] == size, metric]
                .to_numpy(dtype=float)
                * scale
            )
            jitter = np.linspace(-0.018, 0.018, len(values)) * size
            ax.scatter(
                size + jitter,
                values,
                color=METHOD_COLORS[method],
                s=8,
                alpha=0.28,
                linewidths=0,
            )
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes, [f"{size:,}" for size in sizes])


def _formatted_summary(
    summary: pd.DataFrame,
    specs: Sequence[tuple[str, str, str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        formatted: dict[str, Any] = {
            "Training samples": int(row["ablation_sample_size"]),
            "Method": METHOD_LABELS[str(row["baseline_model"])],
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


def _prediction_path(row: pd.Series, results_path: Path) -> Path:
    run_dir = Path(str(row["run_dir"]))
    candidates = (run_dir, ROOT / run_dir, results_path / run_dir)
    for candidate in candidates:
        path = candidate / "predictions_test.npz"
        if path.is_file():
            return path
    return run_dir / "predictions_test.npz"


def _draw_patch_grid(ax: plt.Axes, resolution: int, patch_size: int) -> None:
    for boundary in range(patch_size, resolution, patch_size):
        location = boundary - 0.5
        ax.axvline(location, color="white", linewidth=0.35, alpha=0.6)
        ax.axhline(location, color="white", linewidth=0.35, alpha=0.6)


def _method_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linewidth=STYLE["line_width"],
            label=METHOD_LABELS[method],
        )
        for method in METHOD_ORDER
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
        default="paper_results/sample_efficiency",
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
        help="Skip the endpoint prediction-archive figure.",
    )
    parser.add_argument(
        "--common-seam-stride",
        type=int,
        default=32,
        help="Shared stride used to audit retained endpoint predictions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    formats = tuple(piece.strip() for piece in args.formats.split(",") if piece.strip())
    written = make_sample_efficiency_outputs(
        results_dir=args.results_dir,
        formats=formats,
        expected_seeds=args.expected_seeds,
        qualitative=not args.skip_qualitative,
        common_seam_stride=args.common_seam_stride,
    )
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
