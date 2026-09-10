#!/usr/bin/env python
"""Render manuscript outputs for the five-seed Darcy 256 study."""

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
    "fno",
    "two_scale",
    "two_scale_interface",
)
METHOD_LABELS = {
    "global_pca": "Global PCA-Net",
    "plain_l2l": "Plain L2L",
    "overlap_l2l": "L2L + overlap",
    "fno": "FNO",
    "two_scale": "Two-scale",
    "two_scale_interface": "Two-scale + interface",
}
SHORT_LABELS = {
    "global_pca": "Global",
    "plain_l2l": "L2L",
    "overlap_l2l": "Overlap",
    "fno": "FNO",
    "two_scale": "Two-scale",
    "two_scale_interface": "Two-scale + int.",
}
METHOD_COLORS = {
    "global_pca": "#9C755F",
    "plain_l2l": "#6F6F6F",
    "overlap_l2l": "#4C78A8",
    "fno": "#E45756",
    "two_scale": "#F58518",
    "two_scale_interface": "#54A24B",
}
QUALITY_COST_LABEL_OFFSETS = {
    "global_pca": (5, 9),
    "plain_l2l": (-24, -18),
    "overlap_l2l": (5, -14),
    "fno": (5, 8),
    "two_scale": (-2, -30),
    "two_scale_interface": (5, 8),
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
    "metric_darcy_residual_mean_abs",
    "metric_darcy_residual_rms",
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
    ("metric_mre", "MRE (%)", "percent"),
    ("metric_ssim_mean", "SSIM", "ssim"),
    ("metric_interface_flux_jump", "Flux jump", "scientific"),
    ("metric_relative_spectrum_error", "Spectral error", "scientific"),
    ("metric_darcy_residual_rms", "Darcy residual RMS", "metric"),
    ("timing_end_to_end", "Total time (s)", "time"),
)
FULL_TABLE_SPECS = (
    *MAIN_TABLE_SPECS[:2],
    ("metric_mse", "MSE", "scientific"),
    ("metric_mae", "MAE", "scientific"),
    ("metric_pca_oracle_mre", "PCA-oracle MRE (%)", "percent"),
    ("metric_interface_jump", "Value jump", "scientific"),
    ("metric_interface_flux_jump", "Flux jump", "scientific"),
    ("metric_interface_value_trace_error", "Value trace", "scientific"),
    ("metric_interface_flux_trace_error", "Flux trace", "scientific"),
    ("metric_relative_spectrum_error", "Spectral error", "scientific"),
    ("metric_darcy_residual_mean_abs", "Darcy residual MAE", "metric"),
    ("metric_darcy_residual_rms", "Darcy residual RMS", "metric"),
    ("timing_pca_fit", "PCA fit (s)", "time"),
    ("timing_latent_transform", "Latent transform (s)", "time"),
    ("timing_nn_train", "Cumulative NN train (s)", "time"),
    ("invocation_timing_nn_train", "Fine-tune NN (s)", "time"),
    ("invocation_timing_end_to_end", "Fine-tune total (s)", "time"),
    ("timing_end_to_end", "Cumulative total (s)", "time"),
)
PAIR_SPECS = (
    ("global_pca", "plain_l2l"),
    ("overlap_l2l", "plain_l2l"),
    ("two_scale", "plain_l2l"),
    ("two_scale_interface", "two_scale"),
    ("two_scale_interface", "overlap_l2l"),
    ("fno", "two_scale_interface"),
)
PAIR_METRICS = {
    "mre": "metric_mre",
    "structural_error": "metric_ssim_mean",
    "value_jump": "metric_interface_jump",
    "flux_jump": "metric_interface_flux_jump",
    "value_trace": "metric_interface_value_trace_error",
    "flux_trace": "metric_interface_flux_trace_error",
    "spectral": "metric_relative_spectrum_error",
    "darcy_rms": "metric_darcy_residual_rms",
    "total_time": "timing_end_to_end",
}


def make_darcy_generalization_outputs(
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
        plot_accuracy(
            records,
            output_base=main_figures / "darcy_accuracy_physics",
            formats=formats,
        )
    )
    written.extend(
        plot_interfaces(
            records,
            output_base=main_figures / "darcy_interfaces",
            formats=formats,
        )
    )
    written.extend(
        plot_quality_cost(
            records,
            output_base=appendix_figures / "darcy_quality_cost",
            formats=formats,
        )
    )
    written.extend(
        plot_representation_gap(
            records,
            output_base=main_figures / "darcy_representation_gap",
            formats=formats,
        )
    )
    if not common_seams.empty:
        written.extend(
            plot_common_seams(
                common_seams,
                output_base=appendix_figures / "darcy_common_seams_seed0",
                formats=formats,
            )
        )

    pca_records = add_pca_cost_substages(records, results_path=results_path)
    written.extend(
        plot_pca_breakdown(
            pca_records,
            output_base=appendix_figures / "darcy_pca_breakdown",
            formats=formats,
        )
    )
    pca_table = make_pca_table(pca_records)
    pca_path = appendix_tables / "darcy_pca_breakdown.csv"
    pca_table.to_csv(pca_path, index=False)
    written.append(pca_path)
    written.extend(
        _write_table_pair(
            format_pca_table(pca_table),
            appendix_tables / "darcy_pca_breakdown",
            csv=False,
        )
    )

    sample_id: int | None = None
    if qualitative:
        qualitative_paths, sample_id = plot_qualitative(
            records,
            results_path=results_path,
            output_base=main_figures / "darcy_qualitative",
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
        "common_seam_stride": 64,
        "qualitative_selection": "seed-0 sample closest to median plain-L2L MRE",
        "qualitative_sample_id": sample_id,
        "outputs": [str(path) for path in written],
    }
    manifest_path = results_path / "darcy_generalization_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    written.append(manifest_path)
    return written


def load_records(path: Path) -> pd.DataFrame:
    """Load and normalize the stable Darcy roll-up."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing Darcy roll-up: {path}")
    records = pd.read_csv(path)
    for column in records.columns:
        if column.startswith(("metric_", "timing_", "invocation_timing_")) or column in {
            "seed",
            "resolution",
        }:
            records[column] = pd.to_numeric(records[column], errors="coerce")
    return records


def validate_records(records: pd.DataFrame, *, expected_seeds: int) -> None:
    """Reject incomplete or malformed Darcy matrices."""
    required = {
        "baseline_model",
        "dataset",
        "seed",
        "resolution",
        "run_dir",
        *QUALITY_METRICS,
        *TIMING_METRICS,
    }
    missing = sorted(required.difference(records.columns))
    if missing:
        raise ValueError(f"Missing required Darcy columns: {', '.join(missing)}")
    observed = set(records["baseline_model"])
    if observed != set(METHOD_ORDER):
        raise ValueError(f"Expected methods {list(METHOD_ORDER)}, got {sorted(observed)}.")
    if set(records["dataset"]) != {"darcy"}:
        raise ValueError("The generalization study must contain only Darcy records.")
    if set(records["resolution"]) != {256}:
        raise ValueError("The generalization study must contain only resolution 256.")
    duplicates = records.duplicated(["baseline_model", "seed"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate Darcy method/seed records detected.")
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
            "comparison": f"{METHOD_LABELS[target]} vs {METHOD_LABELS[reference]}",
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
    """Compare seed-zero methods using a common stride-64 seam set."""
    selected = records[records["seed"] == 0]
    rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        row = selected[selected["baseline_model"] == method].iloc[0]
        if method == "overlap_l2l":
            path = _prediction_path(row, results_path)
            if not path.is_file():
                return pd.DataFrame()
            with np.load(path, allow_pickle=False) as archive:
                prediction = np.asarray(archive["y_pred"], dtype=np.float32)
                truth = np.asarray(archive["y_true"], dtype=np.float32)
            value_jump = interface_value_jump(
                prediction, patch_size=64, stride=64
            )
            flux_jump = interface_flux_jump(
                prediction, patch_size=64, stride=64, dx=1.0 / 255.0
            )
            value_trace = interface_value_trace_error(
                prediction, truth, patch_size=64, stride=64
            )
            flux_trace = interface_flux_trace_error(
                prediction,
                truth,
                patch_size=64,
                stride=64,
                dx=1.0 / 255.0,
            )
        else:
            value_jump = float(row["metric_interface_jump"])
            flux_jump = float(row["metric_interface_flux_jump"])
            value_trace = float(row["metric_interface_value_trace_error"])
            flux_trace = float(row["metric_interface_flux_trace_error"])
        rows.append(
            {
                "Method": METHOD_LABELS[method],
                "Method key": method,
                "Seed": 0,
                "Diagnostic stride": 64,
                "Value jump": value_jump,
                "Flux jump": flux_jump,
                "Value trace error": value_trace,
                "Flux trace error": flux_trace,
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
    numeric_path = appendix_dir / "darcy_summary_numeric.csv"
    summary.to_csv(numeric_path, index=False)
    written.append(numeric_path)
    written.extend(
        _write_table_pair(
            formatted_summary(summary, MAIN_TABLE_SPECS),
            main_dir / "darcy_generalization_main",
        )
    )
    written.extend(
        _write_table_pair(
            formatted_summary(summary, FULL_TABLE_SPECS),
            appendix_dir / "darcy_generalization_full",
        )
    )

    paired_path = appendix_dir / "darcy_paired_effects.csv"
    paired.to_csv(paired_path, index=False)
    written.append(paired_path)
    paired_formatted = paired[
        [
            "comparison",
            "mre_paired_change_mean_pct",
            "flux_jump_paired_change_mean_pct",
            "spectral_paired_change_mean_pct",
            "darcy_rms_paired_change_mean_pct",
            "total_time_paired_change_mean_pct",
        ]
    ].copy()
    paired_formatted.columns = (
        "Comparison",
        "MRE change",
        "Flux-jump change",
        "Spectral change",
        "Darcy-RMS change",
        "Time change",
    )
    for column in paired_formatted.columns[1:]:
        paired_formatted[column] = paired_formatted[column].map(
            lambda value: f"{float(value):+.1f}%"
        )
    written.extend(
        _write_table_pair(
            paired_formatted,
            appendix_dir / "darcy_paired_effects",
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
    per_seed_path = appendix_dir / "darcy_per_seed.csv"
    records[per_seed_columns].sort_values(["baseline_model", "seed"]).to_csv(
        per_seed_path,
        index=False,
    )
    written.append(per_seed_path)

    if not common_seams.empty:
        common_path = appendix_dir / "darcy_common_seams_seed0.csv"
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
                appendix_dir / "darcy_common_seams_seed0",
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


def plot_accuracy(
    records: pd.DataFrame,
    *,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Plot reconstruction, structure, spectrum, and Darcy residual outcomes."""
    values = records.copy()
    values["metric_structural_error"] = 1.0 - values["metric_ssim_mean"]
    specs = (
        ("metric_mre", "Mean relative error (%)", 100.0),
        ("metric_structural_error", "$1-\\mathrm{SSIM}$", 1.0),
        ("metric_relative_spectrum_error", "Relative spectral error", 1.0),
        ("metric_darcy_residual_rms", "Darcy residual RMS", 1.0),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), squeeze=False)
    for panel, (metric, ylabel, scale) in enumerate(specs):
        axis = axes.flat[panel]
        _bar_with_seed_points(axis, values, metric=metric, scale=scale)
        axis.set_ylabel(ylabel)
        axis.set_yscale("log")
        axis.grid(True, axis="y", alpha=0.22)
        _panel_label(axis, panel)
    fig.tight_layout(h_pad=1.5, w_pad=1.4)
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
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), squeeze=False)
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
            xytext=QUALITY_COST_LABEL_OFFSETS[method],
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
        stage_axis.text(index, value + 45.0, f"{value:.0f}", ha="center", fontsize=7.5)
    _method_ticks(stage_axis)
    stage_axis.set_ylabel("Cumulative time (s)")
    stage_axis.grid(True, axis="y", alpha=0.22)
    stage_axis.legend(frameon=False, fontsize=7, ncol=2)
    _panel_label(stage_axis, 1)
    fig.tight_layout(w_pad=1.5)
    return _save_figure(fig, output_base, formats)


def plot_representation_gap(
    records: pd.DataFrame,
    *,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Contrast learned errors with output-representation oracle floors."""
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
    for index, (learned_value, oracle_value) in enumerate(zip(learned, oracle)):
        ratio = learned_value / oracle_value
        axis.text(
            index,
            max(learned_value, oracle_value) * 1.12,
            f"{ratio:.1f}x",
            ha="center",
            fontsize=7.5,
        )
    axis.set_xticks(x, [SHORT_LABELS[method] for method in methods], rotation=18)
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


def plot_common_seams(
    table: pd.DataFrame,
    *,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Plot common-stride continuity diagnostics for retained predictions."""
    specs = (
        ("Value trace error", "Stride-64 value trace error"),
        ("Flux trace error", "Stride-64 flux trace error"),
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
        axis.text(index, value + 4.0, f"{value:.0f}", ha="center", fontsize=8)
    axis.set_xticks(x, [SHORT_LABELS[method] for method in methods])
    axis.set_ylabel("PCA/SVD fit time (s)")
    axis.grid(True, axis="y", alpha=0.22)
    axis.legend(frameon=False, loc="upper right", fontsize=7.5)
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
    """Plot coefficient, truth, prediction, error, spectrum, and PDF."""
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
        coefficient_key = "a" if "a" in archive.files else "x_input"
        coefficient = np.asarray(archive[coefficient_key][chosen], dtype=np.float32)
    del truth_all, prediction_all

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
    coefficient_min = float(np.min(coefficient))
    coefficient_max = float(np.max(coefficient))
    errors = {method: np.abs(prediction - truth) for method, prediction in predictions.items()}
    error_max = max(float(np.percentile(np.stack(list(errors.values())), 99.5)), 1.0e-12)
    bins = np.linspace(solution_min, solution_max, 40)

    fig, axes = plt.subplots(
        len(METHOD_ORDER),
        6,
        figsize=(15.4, 2.05 * len(METHOD_ORDER)),
        squeeze=False,
    )
    titles = (
        "Coefficient $a$",
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
            coefficient,
            origin="lower",
            cmap="cividis",
            vmin=coefficient_min,
            vmax=coefficient_max,
        )
        axes[row_index, 1].imshow(
            truth,
            origin="lower",
            cmap="RdBu_r",
            vmin=solution_min,
            vmax=solution_max,
        )
        axes[row_index, 2].imshow(
            prediction,
            origin="lower",
            cmap="RdBu_r",
            vmin=solution_min,
            vmax=solution_max,
        )
        axes[row_index, 3].imshow(
            error,
            origin="lower",
            cmap="magma",
            vmin=0.0,
            vmax=error_max,
        )
        mre = float(np.linalg.norm(prediction - truth)) / max(
            float(np.linalg.norm(truth)), 1.0e-12
        )
        axes[row_index, 2].text(
            0.03,
            0.97,
            f"MRE {mre:.4f}",
            transform=axes[row_index, 2].transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.8},
        )
        axes[row_index, 0].set_ylabel(METHOD_LABELS[method], fontweight="bold")
        for column in range(4):
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])
        _plot_sample_spectrum(axes[row_index, 4], truth, prediction, k_max=64)
        _plot_sample_pdf(axes[row_index, 5], truth, prediction, bins=bins)
        if row_index == len(METHOD_ORDER) - 1:
            axes[row_index, 4].set_xlabel("Wavenumber $k$")
            axes[row_index, 5].set_xlabel("Field value")
    fig.suptitle(
        f"Darcy 256, seed 0, median plain-L2L sample {sample_id}",
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

    def aggregate_change(target: str, reference: str, metric: str) -> float:
        return 100.0 * (mean(target, metric) / mean(reference, metric) - 1.0)

    n_seeds = int(summary.iloc[0]["n_seeds"])
    lines = [
        "# Darcy Generalization Summary",
        "",
        "## Study Design",
        "",
        (
            "This study tests whether the Poisson conclusions transfer to the "
            "heterogeneous Darcy operator at 256 x 256 resolution. Global PCA-Net, "
            "plain L2L, L2L + overlap, FNO, two-scale, and two-scale with the "
            "selected A3 reconstruction/value/flux objective are compared over "
            f"{n_seeds} paired seeds. Every PCA method uses randomized SVD."
        ),
        "",
        "## Aggregate Results",
        "",
        "| Method | MRE (%) | SSIM | Flux jump | Spectral error | Darcy RMS | Total time (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        lines.append(
            "| "
            f"{METHOD_LABELS[method]} | "
            f"{100.0 * mean(method, 'metric_mre'):.3f} | "
            f"{mean(method, 'metric_ssim_mean'):.4f} | "
            f"{mean(method, 'metric_interface_flux_jump'):.3e} | "
            f"{mean(method, 'metric_relative_spectrum_error'):.3e} | "
            f"{mean(method, 'metric_darcy_residual_rms'):.3f} | "
            f"{mean(method, 'timing_end_to_end'):.1f} |"
        )

    scale_mre = effect("two_scale", "plain_l2l", "mre")
    scale_flux = effect("two_scale", "plain_l2l", "flux_jump")
    scale_residual = effect("two_scale", "plain_l2l", "darcy_rms")
    scale_spectral = effect("two_scale", "plain_l2l", "spectral")
    scale_spectral_aggregate = aggregate_change(
        "two_scale", "plain_l2l", "metric_relative_spectrum_error"
    )
    scale_time = effect("two_scale", "plain_l2l", "total_time")
    interface_mre = effect("two_scale_interface", "two_scale", "mre")
    interface_flux = effect("two_scale_interface", "two_scale", "flux_jump")
    interface_residual = effect("two_scale_interface", "two_scale", "darcy_rms")
    interface_spectral = effect("two_scale_interface", "two_scale", "spectral")
    interface_spectral_aggregate = aggregate_change(
        "two_scale_interface", "two_scale", "metric_relative_spectrum_error"
    )
    interface_time = effect("two_scale_interface", "two_scale", "total_time")
    overlap_mre = effect("overlap_l2l", "plain_l2l", "mre")
    overlap_flux = effect("overlap_l2l", "plain_l2l", "flux_jump")
    overlap_time = effect("overlap_l2l", "plain_l2l", "total_time")
    fno_mre = effect("fno", "two_scale_interface", "mre")
    fno_time = effect("fno", "two_scale_interface", "total_time")
    full_overlap_mre = effect("two_scale_interface", "overlap_l2l", "mre")

    learned_two_scale = mean("two_scale", "metric_mre")
    oracle_two_scale = mean("two_scale", "metric_pca_oracle_mre")
    oracle_ratio = learned_two_scale / oracle_two_scale

    lines.extend(
        [
            "",
            "## Main Findings",
            "",
            (
                "1. **The two-scale representation generalizes physically, but "
                "not as a large Darcy MRE win.** Relative to plain L2L, two-scale "
                f"changes MRE by {scale_mre:+.1f}%, flux jump by {scale_flux:+.1f}%, "
                f"and Darcy residual RMS by {scale_residual:+.1f}%, at a "
                f"{scale_time:+.1f}% time increase. The aggregate spectral mean "
                f"changes by {scale_spectral_aggregate:+.1f}%, while the mean "
                f"paired percentage is {scale_spectral:+.1f}% because the metric "
                "varies strongly by seed. The coarse field strongly suppresses "
                "patch-scale derivative defects, but the global accuracy gain is "
                "far smaller than on Poisson."
            ),
            "",
            (
                "2. **The dominant Darcy limitation is latent prediction, not "
                "output compression.** The two-scale PCA oracle has "
                f"{100.0 * oracle_two_scale:.3f}% MRE, while the learned operator "
                f"has {100.0 * learned_two_scale:.3f}% MRE, a {oracle_ratio:.1f}x "
                "learned/oracle gap. The output basis can represent Darcy solutions "
                "very accurately; mapping a rough, discontinuous permeability field "
                "to its coarse and residual scores is the unresolved bottleneck."
            ),
            "",
            (
                "3. **A3 fine-tuning improves physical consistency but barely "
                "moves MRE.** Starting from two-scale, it changes MRE by "
                f"{interface_mre:+.1f}%, flux jump by {interface_flux:+.1f}%, "
                f"Darcy residual RMS by {interface_residual:+.1f}%, and spectral "
                f"mean by {interface_spectral_aggregate:+.1f}%. The mean paired "
                f"spectral percentage is {interface_spectral:+.1f}%, so no stable "
                "spectral direction should be claimed. A3 increases cumulative "
                f"time by {interface_time:+.1f}%. This is useful as a continuity "
                "and residual refinement, not as an accuracy mechanism."
            ),
            "",
            (
                "4. **Overlap is the strongest PCA-Net quality baseline on Darcy.** "
                f"Relative to plain L2L, overlap changes MRE by {overlap_mre:+.1f}% "
                f"and flux jump by {overlap_flux:+.1f}%, at a {overlap_time:+.1f}% "
                "time increase. Two-scale is much cheaper than overlap, but "
                f"two-scale + interface remains {full_overlap_mre:+.1f}% worse in "
                "MRE and is slightly more expensive."
            ),
            "",
            (
                "5. **Two-scale retains the intended PCA-cost advantage over "
                "global and overlap representations.** Mean PCA fit time is "
                f"{mean('two_scale', 'timing_pca_fit'):.1f} s, versus "
                f"{mean('plain_l2l', 'timing_pca_fit'):.1f} s for plain L2L, "
                f"{mean('overlap_l2l', 'timing_pca_fit'):.1f} s for overlap, and "
                f"{mean('global_pca', 'timing_pca_fit'):.1f} s for global PCA. "
                "The extra coarse/residual fit is modest compared with the much "
                "larger input/global or overlapping-patch SVD costs."
            ),
            "",
            (
                "6. **FNO sets the Darcy accuracy ceiling in this comparison.** "
                f"Relative to two-scale + interface, FNO changes MRE by {fno_mre:+.1f}% "
                f"and total time by {fno_time:+.1f}%. Its mean MRE nearly matches "
                "the two-scale PCA-oracle floor, showing that end-to-end spatial "
                "operator learning can exploit the permeability field more fully, "
                "but at a large training-cost premium."
            ),
            "",
            "## Interpretation for the Manuscript",
            "",
            (
                "The Darcy result should be framed as a qualified generalization. "
                "Two-scale decoding reliably improves continuity and discrete PDE "
                "residuals at modest cost, confirming that the coarse-global plus "
                "local-residual representation is not Poisson-specific. It does "
                "not, however, dominate Darcy reconstruction accuracy. The large "
                "learned/oracle gap identifies the next methodological target: a "
                "more expressive map from heterogeneous coefficient fields to the "
                "two-scale latent scores, while retaining the efficient decoder."
            ),
            "",
            "## Reporting Notes",
            "",
            (
                "- Native interface metrics use stride 32 for overlap and stride "
                "64 for the other methods. Use the seed-0 common-stride table for "
                "direct overlap seam comparisons."
            ),
            (
                "- The A3 variant uses reconstruction, truth-referenced interface "
                "value, and interface flux terms. Explicit spectral and Darcy-PDE "
                "loss weights are off; both quantities are evaluation metrics."
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
        default="paper_results/darcy_generalization",
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
    written = make_darcy_generalization_outputs(
        results_dir=args.results_dir,
        formats=formats,
        expected_seeds=args.expected_seeds,
        qualitative=not args.skip_qualitative,
    )
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
