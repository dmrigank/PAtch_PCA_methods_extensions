#!/usr/bin/env python
"""Render manuscript outputs for the Poisson 128 mechanism ablation."""

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
    "plain_l2l",
    "plain_l2l_interface",
    "two_scale",
    "two_scale_interface",
    "overlap_l2l",
)
METHOD_LABELS = {
    "plain_l2l": "Plain L2L",
    "plain_l2l_interface": "Plain L2L + interface",
    "two_scale": "Two-scale",
    "two_scale_interface": "Two-scale + interface",
    "overlap_l2l": "L2L + overlap",
}
SHORT_LABELS = {
    "plain_l2l": "L2L",
    "plain_l2l_interface": "L2L + int.",
    "two_scale": "Two-scale",
    "two_scale_interface": "Two-scale + int.",
    "overlap_l2l": "Overlap",
}
METHOD_COLORS = {
    "plain_l2l": "#6F6F6F",
    "plain_l2l_interface": "#72B7B2",
    "two_scale": "#F58518",
    "two_scale_interface": "#54A24B",
    "overlap_l2l": "#4C78A8",
}
STAGE_SPECS = (
    ("timing_pca_fit", "PCA fit", "#4C78A8"),
    ("timing_latent_transform", "Latent transform", "#72B7B2"),
    ("timing_nn_train", "NN training", "#F58518"),
    ("timing_inference", "Inference", "#E45756"),
)
QUALITY_METRICS = (
    "metric_mre",
    "metric_ssim_mean",
    "metric_mse",
    "metric_mae",
    "metric_pca_oracle_mre",
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
MAIN_TABLE_SPECS = (
    ("metric_mre", "MRE", "percent"),
    ("metric_ssim_mean", "SSIM", "ssim"),
    ("metric_interface_value_trace_error", "Value trace", "scientific"),
    ("metric_interface_flux_trace_error", "Flux trace", "scientific"),
    ("metric_relative_spectrum_error", "Spectral error", "scientific"),
    ("timing_end_to_end", "Total time (s)", "time"),
)
FULL_TABLE_SPECS = (
    *MAIN_TABLE_SPECS[:2],
    ("metric_mse", "MSE", "scientific"),
    ("metric_mae", "MAE", "scientific"),
    ("metric_pca_oracle_mre", "PCA-oracle MRE", "percent"),
    ("metric_interface_jump", "Value jump", "scientific"),
    ("metric_interface_flux_jump", "Flux jump", "scientific"),
    *MAIN_TABLE_SPECS[2:5],
    ("metric_poisson_residual_rms", "PDE residual RMS", "metric"),
    ("timing_pca_fit", "PCA fit (s)", "time"),
    ("timing_latent_transform", "Latent transform (s)", "time"),
    ("timing_nn_train", "Cumulative NN train (s)", "time"),
    ("invocation_timing_nn_train", "Fine-tune NN (s)", "time"),
    ("invocation_timing_end_to_end", "Fine-tune total (s)", "time"),
    ("timing_end_to_end", "Cumulative total (s)", "time"),
)
PAIR_SPECS = (
    ("plain_l2l_interface", "plain_l2l"),
    ("two_scale", "plain_l2l"),
    ("two_scale_interface", "two_scale"),
    ("overlap_l2l", "plain_l2l"),
    ("two_scale_interface", "overlap_l2l"),
)
PAIR_METRICS = {
    "mre": "metric_mre",
    "structural_error": "metric_ssim_mean",
    "value_jump": "metric_interface_jump",
    "flux_jump": "metric_interface_flux_jump",
    "value_trace": "metric_interface_value_trace_error",
    "flux_trace": "metric_interface_flux_trace_error",
    "spectral": "metric_relative_spectrum_error",
    "pde_rms": "metric_poisson_residual_rms",
    "total_time": "timing_end_to_end",
}


def make_mechanism_ablation_outputs(
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
    paired = paired_effects(records)
    common_seams = common_seam_table(records, results_path=results_path)

    written: list[Path] = []
    written.extend(
        write_tables(
            records,
            summary,
            paired,
            common_seams,
            main_dir=main_tables,
            appendix_dir=appendix_tables,
        )
    )
    written.extend(
        plot_outcomes(
            records,
            output_base=main_figures / "mechanism_outcomes",
            formats=formats,
        )
    )
    written.extend(
        plot_interaction(
            records,
            output_base=main_figures / "mechanism_interaction",
            formats=formats,
        )
    )
    written.extend(
        plot_interfaces(
            records,
            output_base=main_figures / "mechanism_interfaces",
            formats=formats,
        )
    )
    written.extend(
        plot_quality_cost(
            records,
            output_base=appendix_figures / "mechanism_quality_cost",
            formats=formats,
        )
    )
    if not common_seams.empty:
        written.extend(
            plot_common_seams(
                common_seams,
                output_base=appendix_figures / "mechanism_common_seams_seed0",
                formats=formats,
            )
        )

    pca_records = add_pca_cost_substages(records, results_path=results_path)
    written.extend(
        plot_pca_breakdown(
            pca_records,
            output_base=appendix_figures / "mechanism_pca_breakdown",
            formats=formats,
        )
    )
    pca_table = make_pca_table(pca_records)
    pca_path = appendix_tables / "mechanism_pca_breakdown.csv"
    pca_table.to_csv(pca_path, index=False)
    written.append(pca_path)
    written.extend(
        _write_table_pair(
            format_pca_table(pca_table),
            appendix_tables / "mechanism_pca_breakdown",
            csv=False,
        )
    )

    sample_id: int | None = None
    if qualitative:
        qualitative_paths, sample_id = plot_qualitative(
            records,
            results_path=results_path,
            output_base=main_figures / "mechanism_qualitative",
            formats=formats,
        )
        written.extend(qualitative_paths)

    summary_path = results_path / "summary.md"
    summary_path.write_text(
        build_summary(summary, paired, common_seams),
        encoding="utf-8",
    )
    written.append(summary_path)

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
    manifest_path = results_path / "mechanism_ablation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    written.append(manifest_path)
    return written


def load_records(path: Path) -> pd.DataFrame:
    """Load and normalize the stable mechanism-ablation roll-up."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing mechanism-ablation roll-up: {path}")
    records = pd.read_csv(path)
    for column in records.columns:
        if column.startswith(("metric_", "timing_", "invocation_timing_")) or column in {
            "seed",
            "resolution",
        }:
            records[column] = pd.to_numeric(records[column], errors="coerce")
    return records


def validate_records(records: pd.DataFrame, *, expected_seeds: int) -> None:
    """Reject incomplete or malformed mechanism matrices."""
    required = {
        "baseline_model",
        "seed",
        "resolution",
        "run_dir",
        *QUALITY_METRICS,
        *TIMING_METRICS,
    }
    missing = sorted(required.difference(records.columns))
    if missing:
        raise ValueError(f"Missing required mechanism columns: {', '.join(missing)}")
    observed = set(records["baseline_model"])
    if observed != set(METHOD_ORDER):
        raise ValueError(f"Expected methods {list(METHOD_ORDER)}, got {sorted(observed)}.")
    if set(records["resolution"]) != {128}:
        raise ValueError("The mechanism ablation must contain only resolution 128.")
    duplicates = records.duplicated(["baseline_model", "seed"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate mechanism method/seed records detected.")
    counts = records.groupby("baseline_model")["seed"].nunique()
    if expected_seeds > 0 and not (counts == expected_seeds).all():
        raise ValueError(
            f"Expected {expected_seeds} seeds per method, got {counts.to_dict()}."
        )


def summarize_records(records: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and sample standard deviation over paired seeds."""
    numeric = [
        column for column in (*QUALITY_METRICS, *TIMING_METRICS) if column in records
    ]
    summary = records.groupby("baseline_model", as_index=False)[numeric].agg(
        ["mean", "std"]
    )
    summary.columns = [
        "_".join(str(piece) for piece in column if piece)
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    summary["method_order"] = summary["baseline_model"].map(
        {method: index for index, method in enumerate(METHOD_ORDER)}
    )
    counts = records.groupby("baseline_model")["seed"].nunique()
    summary["n_seeds"] = summary["baseline_model"].map(counts)
    return summary.sort_values("method_order").reset_index(drop=True)


def paired_effects(records: pd.DataFrame) -> pd.DataFrame:
    """Compute paired relative changes and seed wins for central contrasts."""
    indexed = records.set_index(["baseline_model", "seed"])
    rows: list[dict[str, Any]] = []
    for target, reference in PAIR_SPECS:
        row: dict[str, Any] = {
            "target": target,
            "reference": reference,
            "comparison": (
                f"{METHOD_LABELS[target]} vs {METHOD_LABELS[reference]}"
            ),
        }
        for short_name, metric in PAIR_METRICS.items():
            target_values = indexed.loc[target][metric].sort_index()
            reference_values = indexed.loc[reference][metric].sort_index()
            if short_name == "structural_error":
                target_values = 1.0 - target_values
                reference_values = 1.0 - reference_values
            changes = 100.0 * (target_values / reference_values - 1.0)
            row[f"{short_name}_target_mean"] = float(target_values.mean())
            row[f"{short_name}_reference_mean"] = float(reference_values.mean())
            row[f"{short_name}_paired_change_mean_pct"] = float(changes.mean())
            row[f"{short_name}_wins_lower_is_better"] = int(
                (target_values < reference_values).sum()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def common_seam_table(records: pd.DataFrame, *, results_path: Path) -> pd.DataFrame:
    """Re-evaluate retained seed-zero predictions on common stride-32 seams."""
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


def write_tables(
    records: pd.DataFrame,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    common_seams: pd.DataFrame,
    *,
    main_dir: Path,
    appendix_dir: Path,
) -> list[Path]:
    """Write formatted main and complete appendix tables."""
    written: list[Path] = []
    numeric_path = appendix_dir / "mechanism_summary_numeric.csv"
    summary.to_csv(numeric_path, index=False)
    written.append(numeric_path)
    written.extend(
        _write_table_pair(
            formatted_summary(summary, MAIN_TABLE_SPECS),
            main_dir / "mechanism_ablation_main",
        )
    )
    written.extend(
        _write_table_pair(
            formatted_summary(summary, FULL_TABLE_SPECS),
            appendix_dir / "mechanism_ablation_full",
        )
    )

    paired_path = appendix_dir / "mechanism_paired_effects.csv"
    paired.to_csv(paired_path, index=False)
    written.append(paired_path)
    paired_formatted = paired[
        [
            "comparison",
            "mre_paired_change_mean_pct",
            "value_trace_paired_change_mean_pct",
            "flux_trace_paired_change_mean_pct",
            "spectral_paired_change_mean_pct",
            "total_time_paired_change_mean_pct",
        ]
    ].copy()
    paired_formatted.columns = (
        "Comparison",
        "MRE change",
        "Value-trace change",
        "Flux-trace change",
        "Spectral change",
        "Time change",
    )
    for column in paired_formatted.columns[1:]:
        paired_formatted[column] = paired_formatted[column].map(
            lambda value: f"{float(value):+.1f}%"
        )
    written.extend(
        _write_table_pair(
            paired_formatted,
            appendix_dir / "mechanism_paired_effects",
            csv=False,
        )
    )

    per_seed_columns = [
        column
        for column in (
            "baseline_model",
            "seed",
            "config_hash",
            *QUALITY_METRICS,
            *TIMING_METRICS,
        )
        if column in records
    ]
    per_seed_path = appendix_dir / "mechanism_per_seed.csv"
    records[per_seed_columns].sort_values(["baseline_model", "seed"]).to_csv(
        per_seed_path,
        index=False,
    )
    written.append(per_seed_path)

    if not common_seams.empty:
        common_path = appendix_dir / "mechanism_common_seams_seed0.csv"
        common_seams.to_csv(common_path, index=False)
        written.append(common_path)
        common_formatted = common_seams.drop(columns=["Method key"]).copy()
        for column in (
            "Value jump",
            "Flux jump",
            "Value trace error",
            "Flux trace error",
        ):
            common_formatted[column] = common_formatted[column].map(
                lambda value: f"{float(value):.3e}"
            )
        written.extend(
            _write_table_pair(
                common_formatted,
                appendix_dir / "mechanism_common_seams_seed0",
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


def plot_outcomes(
    records: pd.DataFrame,
    *,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Plot accuracy, continuity, and spectral outcomes over paired seeds."""
    specs = (
        ("metric_mre", "Mean relative error (%)", 100.0),
        ("metric_interface_value_trace_error", "Value trace error", 1.0),
        ("metric_interface_flux_trace_error", "Flux trace error", 1.0),
        ("metric_relative_spectrum_error", "Relative spectral error", 1.0),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), squeeze=False)
    for panel, (metric, ylabel, scale) in enumerate(specs):
        axis = axes.flat[panel]
        _bar_with_seed_points(axis, records, metric=metric, scale=scale)
        axis.set_ylabel(ylabel)
        axis.set_yscale("log")
        axis.grid(True, axis="y", alpha=0.22)
        _panel_label(axis, panel)
    fig.tight_layout(h_pad=1.5, w_pad=1.4)
    return _save_figure(fig, output_base, formats)


def plot_interaction(
    records: pd.DataFrame,
    *,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Show how interface fine-tuning interacts with representation choice."""
    specs = (
        ("metric_mre", "Mean relative error (%)", 100.0),
        ("metric_interface_value_trace_error", "Value trace error", 1.0),
        ("metric_interface_flux_trace_error", "Flux trace error", 1.0),
    )
    representations = ("Plain L2L", "Two-scale")
    methods = (
        ("plain_l2l", "two_scale"),
        ("plain_l2l_interface", "two_scale_interface"),
    )
    conditions = (
        ("Latent objective", "#6F6F6F", "o"),
        ("+ interface objective", "#54A24B", "s"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.7), squeeze=False)
    for panel, (metric, ylabel, scale) in enumerate(specs):
        axis = axes[0, panel]
        for condition, method_pair in zip(conditions, methods):
            label, color, marker = condition
            means = []
            errors = []
            for method in method_pair:
                values = records.loc[
                    records["baseline_model"] == method, metric
                ].to_numpy(dtype=float)
                means.append(scale * float(np.mean(values)))
                errors.append(scale * float(np.std(values, ddof=1)))
            axis.errorbar(
                np.arange(2),
                means,
                yerr=errors,
                color=color,
                marker=marker,
                linewidth=1.8,
                markersize=5,
                capsize=3,
                label=label,
            )
        axis.set_xticks(np.arange(2), representations)
        axis.set_ylabel(ylabel)
        axis.set_yscale("log")
        axis.grid(True, axis="y", alpha=0.22)
        _panel_label(axis, panel)
    axes[0, 0].legend(frameon=False, loc="best")
    fig.tight_layout(w_pad=1.35)
    return _save_figure(fig, output_base, formats)


def plot_interfaces(
    records: pd.DataFrame,
    *,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Plot raw jumps and truth-referenced traces over paired seeds."""
    specs = (
        ("metric_interface_jump", "Value jump"),
        ("metric_interface_flux_jump", "Flux jump"),
        ("metric_interface_value_trace_error", "Value trace error"),
        ("metric_interface_flux_trace_error", "Flux trace error"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), squeeze=False)
    for panel, (metric, ylabel) in enumerate(specs):
        axis = axes.flat[panel]
        _bar_with_seed_points(axis, records, metric=metric, scale=1.0)
        axis.set_ylabel(ylabel)
        axis.set_yscale("log")
        axis.grid(True, axis="y", alpha=0.22)
        _panel_label(axis, panel)
    fig.tight_layout(h_pad=1.5, w_pad=1.4)
    return _save_figure(fig, output_base, formats)


def plot_quality_cost(
    records: pd.DataFrame,
    *,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Plot the accuracy-cost plane and cumulative stage costs."""
    grouped = records.groupby("baseline_model").mean(numeric_only=True)
    spread = records.groupby("baseline_model").std(numeric_only=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), squeeze=False)
    scatter = axes[0, 0]
    for method in METHOD_ORDER:
        x_value = grouped.loc[method, "timing_end_to_end"]
        y_value = 100.0 * grouped.loc[method, "metric_mre"]
        scatter.errorbar(
            x_value,
            y_value,
            xerr=spread.loc[method, "timing_end_to_end"],
            yerr=100.0 * spread.loc[method, "metric_mre"],
            fmt="o",
            color=METHOD_COLORS[method],
            capsize=2.5,
            markersize=6,
        )
        scatter.annotate(
            SHORT_LABELS[method],
            (x_value, y_value),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7.5,
        )
    scatter.set_xlabel("Cumulative end-to-end time (s)")
    scatter.set_ylabel("Mean relative error (%)")
    scatter.set_yscale("log")
    scatter.grid(True, alpha=0.22)
    _panel_label(scatter, 0)

    stage_axis = axes[0, 1]
    x = np.arange(len(METHOD_ORDER))
    bottom = np.zeros(len(METHOD_ORDER), dtype=float)
    totals = np.array(
        [grouped.loc[method, "timing_end_to_end"] for method in METHOD_ORDER]
    )
    for metric, label, color in STAGE_SPECS:
        values = np.array([grouped.loc[method, metric] for method in METHOD_ORDER])
        stage_axis.bar(x, values, bottom=bottom, color=color, width=0.68, label=label)
        bottom += values
    other = np.maximum(totals - bottom, 0.0)
    stage_axis.bar(
        x,
        other,
        bottom=bottom,
        color="#B9B9B9",
        width=0.68,
        label="Other pipeline",
    )
    for index, value in enumerate(totals):
        stage_axis.text(index, value + 7.0, f"{value:.0f}", ha="center", fontsize=7.5)
    _method_ticks(stage_axis)
    stage_axis.set_ylabel("Cumulative time (s)")
    stage_axis.grid(True, axis="y", alpha=0.22)
    stage_axis.legend(frameon=False, fontsize=7, ncol=2)
    _panel_label(stage_axis, 1)
    fig.tight_layout(w_pad=1.5)
    return _save_figure(fig, output_base, formats)


def plot_common_seams(
    table: pd.DataFrame,
    *,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Plot common-stride continuity diagnostics for retained predictions."""
    specs = (
        ("Value trace error", "Stride-32 value trace error"),
        ("Flux trace error", "Stride-32 flux trace error"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), squeeze=False)
    for panel, (metric, ylabel) in enumerate(specs):
        axis = axes[0, panel]
        values = np.array(
            [
                table.loc[table["Method key"] == method, metric].iloc[0]
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
        _panel_label(axis, panel)
    fig.tight_layout(w_pad=1.5)
    return _save_figure(fig, output_base, formats)


def plot_pca_breakdown(
    records: pd.DataFrame,
    *,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Plot PCA/SVD fitting substages for the three unique representations."""
    methods = ("plain_l2l", "two_scale", "overlap_l2l")
    selected = records[records["baseline_model"].isin(methods)]
    grouped = selected.groupby("baseline_model").mean(numeric_only=True)
    x = np.arange(len(methods))
    bottom = np.zeros(len(methods), dtype=float)
    fig, axis = plt.subplots(figsize=(6.7, 4.0))
    for metric, label, color in PCA_STAGE_SPECS:
        values = np.array([grouped.loc[method, metric] for method in methods])
        axis.bar(x, values, bottom=bottom, color=color, width=0.66, label=label)
        bottom += values
    for index, value in enumerate(bottom):
        axis.text(index, value + 0.18, f"{value:.1f}", ha="center", fontsize=8)
    axis.set_xticks(x, [METHOD_LABELS[method] for method in methods])
    axis.set_ylabel("PCA/SVD fit time (s)")
    axis.grid(True, axis="y", alpha=0.22)
    axis.legend(frameon=False, ncol=2, fontsize=7.5)
    fig.tight_layout()
    return _save_figure(fig, output_base, formats)


def make_pca_table(records: pd.DataFrame) -> pd.DataFrame:
    """Aggregate PCA substages for the three unique representations."""
    methods = ("plain_l2l", "two_scale", "overlap_l2l")
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
    """Format PCA substage values for LaTeX output."""
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


def build_summary(
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    common_seams: pd.DataFrame,
) -> str:
    """Build a manuscript-oriented interpretation from measured effects."""
    def mean(method: str, metric: str) -> float:
        row = summary[summary["baseline_model"] == method].iloc[0]
        return float(row[f"{metric}_mean"])

    def effect(target: str, reference: str, metric: str) -> float:
        row = paired[
            (paired["target"] == target) & (paired["reference"] == reference)
        ].iloc[0]
        return float(row[f"{metric}_paired_change_mean_pct"])

    n_seeds = int(summary.iloc[0]["n_seeds"])

    lines = [
        "# Mechanism Ablation Summary",
        "",
        "## Study Design",
        "",
        (
            "This study isolates output representation and interface-aware "
            "fine-tuning on Poisson 128. It compares plain L2L, plain L2L with "
            "the selected A3 reconstruction/value/flux objective, two-scale, "
            "two-scale with the same A3 objective, and L2L with overlap over "
            f"{n_seeds} paired seeds. All PCA fits use randomized SVD."
        ),
        "",
        "## Aggregate Results",
        "",
        "| Method | MRE (%) | SSIM | Value trace | Flux trace | Spectral error | Total time (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        lines.append(
            "| "
            f"{METHOD_LABELS[method]} | "
            f"{100.0 * mean(method, 'metric_mre'):.3f} | "
            f"{mean(method, 'metric_ssim_mean'):.4f} | "
            f"{mean(method, 'metric_interface_value_trace_error'):.3e} | "
            f"{mean(method, 'metric_interface_flux_trace_error'):.3e} | "
            f"{mean(method, 'metric_relative_spectrum_error'):.3e} | "
            f"{mean(method, 'timing_end_to_end'):.1f} |"
        )

    plain_mre = effect("plain_l2l_interface", "plain_l2l", "mre")
    plain_value = effect("plain_l2l_interface", "plain_l2l", "value_trace")
    plain_flux = effect("plain_l2l_interface", "plain_l2l", "flux_trace")
    plain_value_jump = effect("plain_l2l_interface", "plain_l2l", "value_jump")
    plain_flux_jump = effect("plain_l2l_interface", "plain_l2l", "flux_jump")
    plain_spectral = effect("plain_l2l_interface", "plain_l2l", "spectral")
    plain_time = effect("plain_l2l_interface", "plain_l2l", "total_time")
    scale_mre = effect("two_scale", "plain_l2l", "mre")
    scale_value = effect("two_scale", "plain_l2l", "value_trace")
    scale_flux = effect("two_scale", "plain_l2l", "flux_trace")
    interface_mre = effect("two_scale_interface", "two_scale", "mre")
    interface_value = effect("two_scale_interface", "two_scale", "value_trace")
    interface_flux = effect("two_scale_interface", "two_scale", "flux_trace")
    interface_value_jump = effect("two_scale_interface", "two_scale", "value_jump")
    interface_flux_jump_change = effect(
        "two_scale_interface", "two_scale", "flux_jump"
    )
    interface_spectral = effect("two_scale_interface", "two_scale", "spectral")
    interface_time = effect("two_scale_interface", "two_scale", "total_time")
    full_overlap_mre = effect("two_scale_interface", "overlap_l2l", "mre")
    full_overlap_value = effect("two_scale_interface", "overlap_l2l", "value_trace")
    full_overlap_flux = effect("two_scale_interface", "overlap_l2l", "flux_trace")

    lines.extend(
        [
            "",
            "## Main Findings",
            "",
            (
                "1. **Two-scale is the dominant accuracy mechanism.** Relative "
                f"to plain L2L, two-scale changes MRE by {scale_mre:+.1f}%, "
                f"value-trace error by {scale_value:+.1f}%, and flux-trace "
                f"error by {scale_flux:+.1f}%. The coarse global component "
                "removes the low-frequency patch mismatch that a local-only "
                "representation cannot express consistently."
            ),
            "",
            (
                "2. **The A3 objective is not a substitute for representation.** "
                "On plain L2L, interface fine-tuning changes MRE by "
                f"{plain_mre:+.1f}%, value-trace error by {plain_value:+.1f}%, "
                f"and flux-trace error by {plain_flux:+.1f}%, while spectral "
                f"error changes by {plain_spectral:+.1f}% and total time by "
                f"{plain_time:+.1f}%. It improves seams modestly but remains "
                "constrained by the local PCA output space."
            ),
            "",
            (
                "3. **Interface fine-tuning is complementary to two-scale.** "
                f"Starting from two-scale, A3 changes MRE by {interface_mre:+.1f}%, "
                f"value-trace error by {interface_value:+.1f}%, flux-trace "
                f"error by {interface_flux:+.1f}%, and spectral error by "
                f"{interface_spectral:+.1f}%. This is the intended role of the "
                "loss: clean up the residual interface error after the global "
                "representation has already removed the dominant artifact."
            ),
            "",
            (
                "   The raw discontinuity metrics tell the same story with an "
                "important nuance. A3 changes plain-L2L value/flux jumps by "
                f"{plain_value_jump:+.1f}%/{plain_flux_jump:+.1f}%, and changes "
                "two-scale value/flux jumps by "
                f"{interface_value_jump:+.1f}%/{interface_flux_jump_change:+.1f}%. "
                "Thus its strongest direct continuity effect is on normal flux; "
                "the larger value-trace reduction also reflects better agreement "
                "with the true interface profile."
            ),
            "",
            (
                "4. **The best-accuracy model is not the cheapest continuity "
                "solution.** Two-scale + interface changes MRE by "
                f"{full_overlap_mre:+.1f}% relative to overlap, but its native "
                f"value- and flux-trace errors change by {full_overlap_value:+.1f}% "
                f"and {full_overlap_flux:+.1f}%, respectively. Its cumulative "
                f"time is {mean('two_scale_interface', 'timing_end_to_end'):.1f} s "
                f"versus {mean('overlap_l2l', 'timing_end_to_end'):.1f} s for "
                "overlap. Overlap remains a strong continuity reference, while "
                "two-scale + interface prioritizes global field accuracy."
            ),
            "",
            (
                "5. **Fine-tuning cost must be reported cumulatively.** A3 adds "
                f"{interface_time:+.1f}% to the two-scale end-to-end cost. The "
                "timing columns include both the warm-start run and the fine-tune "
                "invocation, so the improved result is not presented as a free "
                "post-processing step."
            ),
            "",
            "## Interpretation for the Manuscript",
            "",
            (
                "The ablation supports a representation-first story. Two-scale "
                "PCA supplies the large, robust reduction in reconstruction and "
                "interface error. The interface-aware physical-space objective "
                "then acts as a targeted refinement: it is weak when the decoder "
                "basis cannot represent a globally compatible field, but becomes "
                "effective after the coarse global channel is present. The main "
                "method should therefore be described as two-scale representation "
                "with optional interface-aware fine-tuning, not as loss design "
                "alone removing tiling artifacts."
            ),
            "",
            "## Reporting Notes",
            "",
            (
                "- Native interface metrics use each method's configured patch "
                "geometry. Because overlap uses stride 16 while the other methods "
                "use stride 32, the seed-0 common-stride table and figure should "
                "accompany direct overlap comparisons."
            ),
            (
                "- The selected A3 objective contains reconstruction, interface "
                "value, and interface flux terms. Spectral and PDE-residual terms "
                "are not active in this mechanism study."
            ),
            (
                "- Error bars and table uncertainties are sample standard "
                f"deviations over {n_seeds} paired seeds."
            ),
        ]
    )
    if common_seams.empty:
        lines.append("- Common-stride diagnostics were unavailable because predictions were not retained.")
    return "\n".join(lines) + "\n"


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
        [SHORT_LABELS[method] for method in METHOD_ORDER],
        rotation=20,
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
        default="paper_results/mechanism_ablation",
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
    written = make_mechanism_ablation_outputs(
        results_dir=args.results_dir,
        formats=formats,
        expected_seeds=args.expected_seeds,
        qualitative=not args.skip_qualitative,
    )
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
