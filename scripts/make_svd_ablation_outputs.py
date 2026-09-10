#!/usr/bin/env python
"""Render manuscript figures and tables for the SVD-solver ablation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from manuscript_pca_costs import PCA_STAGE_SPECS, add_pca_cost_substages
from matplotlib.lines import Line2D
from scipy.stats import t

CELL_ORDER = (
    "plain_full",
    "plain_randomized",
    "overlap_full",
    "overlap_randomized",
    "two_scale_full",
    "two_scale_hybrid",
    "two_scale_randomized",
)
CELL_LABELS = {
    "plain_full": "Plain L2L\nFull",
    "plain_randomized": "Plain L2L\nRandomized",
    "overlap_full": "L2L + overlap\nFull",
    "overlap_randomized": "L2L + overlap\nRandomized",
    "two_scale_full": "Two-scale\nFull",
    "two_scale_hybrid": "Two-scale\nHybrid",
    "two_scale_randomized": "Two-scale\nRandomized",
}
METHOD_LABELS = {
    "plain_l2l": "Plain L2L",
    "overlap_l2l": "L2L + overlap",
    "two_scale": "Two-scale",
}
METHOD_COLORS = {
    "plain_l2l": "#6F6F6F",
    "overlap_l2l": "#4C78A8",
    "two_scale": "#F58518",
}
SOLVER_COLORS = {
    "full": "#6F6F6F",
    "hybrid": "#4C78A8",
    "randomized": "#F58518",
}
FULL_RANDOMIZED_PAIRS = (
    ("plain_full", "plain_randomized", "plain_l2l"),
    ("overlap_full", "overlap_randomized", "overlap_l2l"),
    ("two_scale_full", "two_scale_randomized", "two_scale"),
)
ALL_PAIRS = (
    ("plain_full", "plain_randomized", "Plain: randomized vs full"),
    ("overlap_full", "overlap_randomized", "Overlap: randomized vs full"),
    ("two_scale_full", "two_scale_hybrid", "Two-scale: hybrid vs full"),
    ("two_scale_full", "two_scale_randomized", "Two-scale: randomized vs full"),
    (
        "two_scale_hybrid",
        "two_scale_randomized",
        "Two-scale: randomized vs hybrid",
    ),
)
QUALITY_METRICS = (
    "metric_mre",
    "metric_pca_oracle_mre",
    "metric_ssim_mean",
    "metric_mse",
    "metric_mae",
    "metric_relative_spectrum_error",
    "metric_interface_value_trace_error",
    "metric_interface_flux_trace_error",
    "metric_poisson_residual_rms",
)
TIMING_METRICS = (
    "timing_pca_fit",
    "timing_latent_transform",
    "timing_nn_train",
    "timing_inference",
    "timing_end_to_end",
    "timing_coarse_svd",
)
MAIN_QUALITY_SPECS = (
    ("metric_mre", "Learned MRE", "metric"),
    ("metric_pca_oracle_mre", "PCA-oracle MRE", "metric"),
    ("metric_ssim_mean", "SSIM", "ssim"),
    ("metric_relative_spectrum_error", "Spectral error", "metric"),
)
FULL_SPECS = (
    *MAIN_QUALITY_SPECS,
    ("metric_mse", "MSE", "scientific"),
    ("metric_mae", "MAE", "scientific"),
    ("metric_interface_value_trace_error", "Value trace error", "scientific"),
    ("metric_interface_flux_trace_error", "Flux trace error", "scientific"),
    ("metric_poisson_residual_rms", "PDE residual RMS", "metric"),
    ("timing_pca_fit", "PCA fit (s)", "time"),
    ("timing_latent_transform", "Latent transform (s)", "time"),
    ("timing_nn_train", "NN train (s)", "time"),
    ("timing_inference", "Inference (s)", "time"),
    ("timing_end_to_end", "End-to-end (s)", "time"),
)
GATE_SPECS = (
    ("metric_mre", "Learned MRE", "relative", 0.05, "lower"),
    ("metric_pca_oracle_mre", "PCA-oracle MRE", "relative", 0.02, "lower"),
    ("metric_ssim_mean", "SSIM", "absolute", 0.001, "higher"),
    ("metric_mse", "MSE", "relative", 0.10, "lower"),
    (
        "metric_relative_spectrum_error",
        "Spectral error",
        "relative",
        0.10,
        "lower",
    ),
    (
        "metric_interface_value_trace_error",
        "Value trace error",
        "relative",
        0.10,
        "lower",
    ),
    (
        "metric_interface_flux_trace_error",
        "Flux trace error",
        "relative",
        0.10,
        "lower",
    ),
    (
        "metric_poisson_residual_rms",
        "PDE residual RMS",
        "relative",
        0.10,
        "lower",
    ),
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


def make_svd_ablation_outputs(
    *,
    results_dir: str | Path,
    formats: Sequence[str] = ("png", "pdf"),
    expected_seeds: int = 5,
) -> list[Path]:
    """Generate main-text and appendix assets from a completed SVD study."""
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
        plot_quality_equivalence(
            records,
            output_base=main_figures / "svd_quality_equivalence",
            formats=formats,
        )
    )
    written.extend(
        plot_runtime_speedup(
            records,
            output_base=main_figures / "svd_runtime_speedup",
            formats=formats,
        )
    )
    written.extend(
        plot_paired_mre(
            records,
            output_base=appendix_figures / "svd_paired_mre",
            formats=formats,
        )
    )
    written.extend(
        plot_stage_costs(
            records,
            output_base=appendix_figures / "svd_stage_costs",
            formats=formats,
        )
    )
    pca_costs = add_pca_cost_substages(records, results_path=results_path)
    written.extend(
        plot_pca_cost_breakdown(
            pca_costs,
            output_base=appendix_figures / "svd_pca_breakdown",
            formats=formats,
        )
    )
    pca_table = pca_cost_breakdown_table(pca_costs)
    if not pca_table.empty:
        pca_path = appendix_tables / "svd_pca_breakdown.csv"
        pca_table.to_csv(pca_path, index=False)
        written.append(pca_path)
        written.extend(
            _write_table_pair(
                pca_table,
                appendix_tables / "svd_pca_breakdown",
                csv=False,
            )
        )
    written.extend(
        plot_trainability_gap(
            records,
            output_base=appendix_figures / "svd_trainability_gap",
            formats=formats,
        )
    )

    manifest = {
        "results_dir": str(results_path),
        "records": int(len(records)),
        "cells": list(CELL_ORDER),
        "expected_seeds": expected_seeds,
        "outputs": [str(path) for path in written],
    }
    manifest_path = results_path / "svd_ablation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    written.append(manifest_path)
    return written


def load_records(path: Path) -> pd.DataFrame:
    """Load the stable results roll-up and coerce measured fields to numeric."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing SVD-ablation roll-up: {path}")
    records = pd.read_csv(path)
    for column in records.columns:
        if column.startswith(
            (
                "metric_",
                "timing_",
                "pca_component_counts_",
                "pca_groups_",
                "pca_timings_",
            )
        ) or column in {"seed", "resolution"}:
            converted = pd.to_numeric(records[column], errors="coerce")
            if converted.notna().sum() == records[column].notna().sum():
                records[column] = converted
    return records


def validate_records(records: pd.DataFrame, *, expected_seeds: int) -> None:
    """Reject incomplete, duplicated, or malformed solver matrices."""
    required = {
        "seed",
        "resolution",
        "ablation_cell",
        "ablation_method",
        "ablation_solver_class",
        "config_hash",
        *QUALITY_METRICS,
        *TIMING_METRICS,
    }
    missing = sorted(required.difference(records.columns))
    if missing:
        raise ValueError(f"Missing required SVD-ablation columns: {', '.join(missing)}")
    unexpected = sorted(set(records["ablation_cell"]) - set(CELL_ORDER))
    if unexpected:
        raise ValueError(f"Unexpected SVD-ablation cells: {', '.join(unexpected)}")
    missing_cells = sorted(set(CELL_ORDER) - set(records["ablation_cell"]))
    if missing_cells:
        raise ValueError(f"Missing SVD-ablation cells: {', '.join(missing_cells)}")
    duplicates = records.duplicated(["ablation_cell", "seed"], keep=False)
    if duplicates.any():
        keys = records.loc[duplicates, ["ablation_cell", "seed"]]
        raise ValueError(f"Duplicate SVD-ablation records:\n{keys.to_string(index=False)}")
    counts = records.groupby("ablation_cell")["seed"].nunique()
    if expected_seeds > 0:
        incomplete = counts[counts != expected_seeds]
        if not incomplete.empty:
            details = ", ".join(f"{cell}: {count}" for cell, count in incomplete.items())
            raise ValueError(
                f"Expected {expected_seeds} seeds per SVD cell; found {details}."
            )
    for _, _, method in FULL_RANDOMIZED_PAIRS:
        method_rows = records[records["ablation_method"] == method]
        seed_sets = [
            set(method_rows.loc[method_rows["ablation_cell"] == cell, "seed"])
            for cell in method_rows["ablation_cell"].unique()
        ]
        if seed_sets and any(seed_set != seed_sets[0] for seed_set in seed_sets[1:]):
            raise ValueError(f"Solver cells for {method} do not use paired seeds.")


def summarize_records(records: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and sample standard deviation over seeds."""
    numeric = [
        column
        for column in (*QUALITY_METRICS, *TIMING_METRICS, "metric_pca_fit_seconds")
        if column in records
    ]
    summary = (
        records.groupby(
            ["ablation_cell", "ablation_method", "ablation_solver_class"],
            as_index=False,
        )[numeric]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(piece) for piece in column if piece)
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    summary["cell_order"] = summary["ablation_cell"].map(
        {cell: index for index, cell in enumerate(CELL_ORDER)}
    )
    return summary.sort_values("cell_order", kind="stable").reset_index(drop=True)


def write_tables(
    records: pd.DataFrame,
    summary: pd.DataFrame,
    main_dir: Path,
    appendix_dir: Path,
) -> list[Path]:
    """Write formatted manuscript tables and complete numeric audits."""
    written: list[Path] = []
    numeric_path = appendix_dir / "svd_solver_summary_numeric.csv"
    summary.to_csv(numeric_path, index=False)
    written.append(numeric_path)

    main = _formatted_summary(summary, MAIN_QUALITY_SPECS)
    written.extend(_write_table_pair(main, main_dir / "svd_solver_quality"))

    timing = _timing_table(records, summary)
    written.extend(_write_table_pair(timing, main_dir / "svd_solver_timing"))

    full = _formatted_summary(summary, FULL_SPECS)
    written.extend(_write_table_pair(full, appendix_dir / "svd_solver_full_metrics"))

    equivalence = paired_equivalence_table(records)
    equivalence_path = appendix_dir / "svd_solver_equivalence.csv"
    equivalence.to_csv(equivalence_path, index=False)
    written.append(equivalence_path)
    formatted_equivalence = equivalence.copy()
    for column in ("Mean change", "90% CI lower", "90% CI upper", "Margin"):
        formatted_equivalence[column] = formatted_equivalence.apply(
            lambda row, value_column=column: _format_gate_value(
                row[value_column],
                mode=str(row["Scale"]),
            ),
            axis=1,
        )
    written.extend(
        _write_table_pair(
            formatted_equivalence.drop(columns=["Scale"]),
            appendix_dir / "svd_solver_equivalence",
            csv=False,
        )
    )

    diagnostics = pca_diagnostics_table(records)
    diagnostics_path = appendix_dir / "svd_solver_pca_diagnostics.csv"
    diagnostics.to_csv(diagnostics_path, index=False)
    written.append(diagnostics_path)
    written.extend(
        _write_table_pair(
            _format_pca_diagnostics(diagnostics),
            appendix_dir / "svd_solver_pca_diagnostics",
            csv=False,
        )
    )

    per_seed_columns = [
        column
        for column in (
            "seed",
            "ablation_cell",
            "ablation_method",
            "ablation_solver_class",
            "config_hash",
            *QUALITY_METRICS,
            *TIMING_METRICS,
        )
        if column in records
    ]
    per_seed = records[per_seed_columns].sort_values(["ablation_cell", "seed"])
    per_seed_path = appendix_dir / "svd_solver_per_seed.csv"
    per_seed.to_csv(per_seed_path, index=False)
    written.append(per_seed_path)
    return written


def paired_equivalence_table(records: pd.DataFrame) -> pd.DataFrame:
    """Evaluate paired 90% confidence intervals against predeclared margins."""
    rows: list[dict[str, Any]] = []
    for reference, candidate, comparison in ALL_PAIRS:
        reference_rows, candidate_rows = _paired_rows(records, reference, candidate)
        for metric, label, mode, margin, direction in GATE_SPECS:
            reference_values = reference_rows[metric].to_numpy(dtype=float)
            candidate_values = candidate_rows[metric].to_numpy(dtype=float)
            if mode == "relative":
                changes = candidate_values / reference_values - 1.0
            else:
                changes = candidate_values - reference_values
            mean, lower, upper = _mean_confidence_interval(changes)
            rows.append(
                {
                    "Comparison": comparison,
                    "Metric": label,
                    "Scale": mode,
                    "Mean change": mean,
                    "90% CI lower": lower,
                    "90% CI upper": upper,
                    "Margin": margin,
                    "Decision": _equivalence_decision(
                        lower,
                        upper,
                        margin=margin,
                        direction=direction,
                    ),
                }
            )

        speedup = (
            reference_rows["timing_pca_fit"].to_numpy(dtype=float)
            / candidate_rows["timing_pca_fit"].to_numpy(dtype=float)
        )
        mean, lower, upper = _mean_confidence_interval(speedup)
        minimum = float(
            candidate_rows["equivalence_gate_minimum_pca_speedup"].iloc[0]
            if "equivalence_gate_minimum_pca_speedup" in candidate_rows
            else 1.2
        )
        rows.append(
            {
                "Comparison": comparison,
                "Metric": "PCA-fit speedup",
                "Scale": "ratio",
                "Mean change": mean,
                "90% CI lower": lower,
                "90% CI upper": upper,
                "Margin": minimum,
                "Decision": "Pass" if lower >= minimum else "Inconclusive",
            }
        )
    return pd.DataFrame(rows)


def pca_diagnostics_table(records: pd.DataFrame) -> pd.DataFrame:
    """Collect retained dimensions, fitted solvers, and variance diagnostics."""
    specs = (
        ("pca_component_counts_input_total", "Input latent dimension"),
        ("pca_component_counts_output_total", "Output latent dimension"),
        ("pca_component_counts_coarse_output", "Coarse dimension"),
        (
            "pca_component_counts_output_residual_total",
            "Residual latent dimension",
        ),
        (
            "pca_groups_input_explained_variance_ratio_mean",
            "Input explained variance",
        ),
        (
            "pca_groups_output_explained_variance_ratio_mean",
            "Output explained variance",
        ),
        (
            "pca_groups_coarse_output_explained_variance_ratio_mean",
            "Coarse explained variance",
        ),
        (
            "pca_groups_output_residual_explained_variance_ratio_mean",
            "Residual explained variance",
        ),
        ("metric_pca_fit_seconds", "Internal PCA fit (s)"),
        ("timing_coarse_svd", "Coarse SVD (s)"),
    )
    rows: list[dict[str, Any]] = []
    for cell in CELL_ORDER:
        selected = records[records["ablation_cell"] == cell]
        row: dict[str, Any] = {
            "Cell": CELL_LABELS[cell].replace("\n", " "),
            "Method": METHOD_LABELS[str(selected["ablation_method"].iloc[0])],
            "Solver": str(selected["ablation_solver_class"].iloc[0]).title(),
        }
        for column, label in specs:
            row[label] = (
                float(pd.to_numeric(selected[column], errors="coerce").mean())
                if column in selected
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_quality_equivalence(
    records: pd.DataFrame,
    *,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Plot paired randomized-minus-full changes against quality margins."""
    specs = (
        ("metric_mre", "Learned MRE change (%)", "relative", 5.0),
        ("metric_pca_oracle_mre", "PCA-oracle MRE change (%)", "relative", 2.0),
        ("metric_ssim_mean", r"$\Delta$SSIM ($\times 10^3$)", "absolute", 1.0),
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.6), squeeze=False)
    for index, (metric, ylabel, mode, margin) in enumerate(specs):
        ax = axes[0, index]
        ax.axhspan(-margin, margin, color="#D9EAD3", alpha=0.65, zorder=0)
        for method_index, (reference, candidate, method) in enumerate(
            FULL_RANDOMIZED_PAIRS
        ):
            reference_rows, candidate_rows = _paired_rows(records, reference, candidate)
            reference_values = reference_rows[metric].to_numpy(dtype=float)
            candidate_values = candidate_rows[metric].to_numpy(dtype=float)
            if mode == "relative":
                changes = 100.0 * (candidate_values / reference_values - 1.0)
            else:
                changes = 1000.0 * (candidate_values - reference_values)
            mean, lower, upper = _mean_confidence_interval(changes)
            jitter = np.linspace(-0.07, 0.07, len(changes))
            ax.scatter(
                method_index + jitter,
                changes,
                color=METHOD_COLORS[method],
                s=14,
                alpha=0.38,
                linewidths=0,
                zorder=2,
            )
            ax.errorbar(
                method_index,
                mean,
                yerr=np.array([[mean - lower], [upper - mean]]),
                color=METHOD_COLORS[method],
                marker="o",
                markersize=6,
                capsize=STYLE["capsize"],
                linewidth=STYLE["line_width"],
                zorder=3,
            )
        ax.axhline(0.0, color="#555555", linewidth=1.0)
        ax.set_xticks(
            np.arange(len(FULL_RANDOMIZED_PAIRS)),
            [METHOD_LABELS[method] for _, _, method in FULL_RANDOMIZED_PAIRS],
            rotation=14,
            ha="right",
        )
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.22)
        ax.text(
            0.02,
            0.96,
            f"({chr(97 + index)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
    fig.text(
        0.5,
        0.995,
        "Randomized SVD relative to paired full-SVD runs; shaded region is the equivalence band",
        ha="center",
        va="top",
        fontsize=STYLE["title_size"],
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=1.6)
    return _save_figure(fig, output_base, formats)


def plot_runtime_speedup(
    records: pd.DataFrame,
    *,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Compare paired PCA-fit and end-to-end speedups."""
    fig, ax = plt.subplots(figsize=(6.7, 4.0))
    x = np.arange(len(FULL_RANDOMIZED_PAIRS))
    width = 0.34
    timing_specs = (
        ("timing_pca_fit", "PCA fit", "#4C78A8"),
        ("timing_end_to_end", "End-to-end", "#F58518"),
    )
    for offset_index, (metric, label, color) in enumerate(timing_specs):
        means: list[float] = []
        errors: list[list[float]] = []
        for reference, candidate, _ in FULL_RANDOMIZED_PAIRS:
            reference_rows, candidate_rows = _paired_rows(records, reference, candidate)
            ratios = (
                reference_rows[metric].to_numpy(dtype=float)
                / candidate_rows[metric].to_numpy(dtype=float)
            )
            mean, lower, upper = _mean_confidence_interval(ratios)
            means.append(mean)
            errors.append([mean - lower, upper - mean])
        positions = x + (offset_index - 0.5) * width
        bars = ax.bar(
            positions,
            means,
            width=width,
            color=color,
            label=label,
            yerr=np.asarray(errors).T,
            capsize=STYLE["capsize"],
        )
        for bar, value in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.08,
                f"{value:.2f}x",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.axhline(1.0, color="#555555", linewidth=1.0)
    ax.axhline(
        1.2,
        color="#777777",
        linestyle=":",
        linewidth=1.2,
        label="Minimum PCA speedup gate",
    )
    ax.set_xticks(
        x,
        [METHOD_LABELS[method] for _, _, method in FULL_RANDOMIZED_PAIRS],
    )
    ax.set_ylabel("Full-SVD time / randomized-SVD time")
    ax.set_ylim(0.8, 3.25)
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=3, loc="upper center")
    fig.tight_layout()
    return _save_figure(fig, output_base, formats)


def plot_paired_mre(
    records: pd.DataFrame,
    *,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Show the downstream-training variation behind aggregate MRE values."""
    groups = (
        ("Plain L2L", ("plain_full", "plain_randomized")),
        ("L2L + overlap", ("overlap_full", "overlap_randomized")),
        (
            "Two-scale",
            ("two_scale_full", "two_scale_hybrid", "two_scale_randomized"),
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.6), squeeze=False)
    for index, (title, cells) in enumerate(groups):
        ax = axes[0, index]
        pivot = records[records["ablation_cell"].isin(cells)].pivot(
            index="seed",
            columns="ablation_cell",
            values="metric_mre",
        )
        positions = np.arange(len(cells))
        for seed, row in pivot.iterrows():
            values = np.array([row[cell] for cell in cells], dtype=float)
            ax.plot(
                positions,
                100.0 * values,
                color="#9A9A9A",
                linewidth=0.9,
                alpha=0.7,
            )
            ax.scatter(
                positions,
                100.0 * values,
                color=[SOLVER_COLORS[_solver_from_cell(cell)] for cell in cells],
                s=18,
                zorder=3,
                label=f"Seed {seed}" if index == 0 else None,
            )
        means = np.array([pivot[cell].mean() for cell in cells], dtype=float)
        ax.plot(
            positions,
            100.0 * means,
            color="#222222",
            marker="D",
            markersize=5,
            linewidth=STYLE["line_width"],
            zorder=4,
        )
        ax.set_xticks(
            positions,
            [_solver_from_cell(cell).title() for cell in cells],
            rotation=15,
            ha="right",
        )
        ax.set_title(title)
        if index == 0:
            ax.set_ylabel("Learned MRE (%)")
        ax.grid(True, axis="y", alpha=0.22)
        ax.text(
            0.02,
            0.96,
            f"({chr(97 + index)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
    legend = [
        Line2D(
            [0],
            [0],
            color="#222222",
            marker="D",
            linewidth=STYLE["line_width"],
            label="Five-seed mean",
        ),
        Line2D(
            [0],
            [0],
            color="#9A9A9A",
            marker="o",
            linewidth=0.9,
            label="Paired seed",
        ),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=1.6)
    return _save_figure(fig, output_base, formats)


def plot_stage_costs(
    records: pd.DataFrame,
    *,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Render end-to-end time as additive measured stages for all solver cells."""
    grouped = records.groupby("ablation_cell").mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10.2, 4.6))
    x = np.arange(len(CELL_ORDER))
    stack_specs = (
        ("timing_pca_fit", "PCA fit", "#4C78A8"),
        ("timing_latent_transform", "Latent transform", "#72B7B2"),
        ("timing_nn_train", "NN train", "#F2CF5B"),
        ("timing_inference", "Inference", "#54A24B"),
    )
    bottom = np.zeros(len(CELL_ORDER))
    for metric, label, color in stack_specs:
        values = np.array([grouped.loc[cell, metric] for cell in CELL_ORDER])
        ax.bar(x, values, bottom=bottom, color=color, label=label, width=0.72)
        bottom += values
    totals = np.array([grouped.loc[cell, "timing_end_to_end"] for cell in CELL_ORDER])
    other = np.maximum(totals - bottom, 0.0)
    ax.bar(
        x,
        other,
        bottom=bottom,
        color="#B9B9B9",
        label="Assembly, I/O, evaluation",
        width=0.72,
    )
    for index, total in enumerate(totals):
        ax.text(
            index,
            total + 0.012 * totals.max(),
            f"{total:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(x, [CELL_LABELS[cell] for cell in CELL_ORDER])
    ax.set_ylabel("End-to-end time (s)")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(loc="upper center", ncol=5, frameon=False)
    fig.tight_layout()
    return _save_figure(fig, output_base, formats)


def plot_pca_cost_breakdown(
    records: pd.DataFrame,
    *,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Render PCA/SVD fit time alone, decomposed by fitting substage."""
    grouped = records.groupby("ablation_cell").mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    x = np.arange(len(CELL_ORDER))
    bottom = np.zeros(len(CELL_ORDER), dtype=float)
    for metric, label, color in PCA_STAGE_SPECS:
        values = np.array([grouped.loc[cell, metric] for cell in CELL_ORDER])
        ax.bar(x, values, bottom=bottom, color=color, label=label, width=0.72)
        bottom += values
    for index, total in enumerate(bottom):
        ax.text(
            index,
            total + 0.025 * max(bottom),
            f"{total:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(x, [CELL_LABELS[cell] for cell in CELL_ORDER])
    ax.set_ylabel("PCA/SVD fit time (s)")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(loc="upper center", ncol=5, frameon=False)
    fig.tight_layout()
    return _save_figure(fig, output_base, formats)


def pca_cost_breakdown_table(records: pd.DataFrame) -> pd.DataFrame:
    """Return mean PCA/SVD substages for each solver cell."""
    columns = [spec[0] for spec in PCA_STAGE_SPECS] + ["pca_cost_total"]
    grouped = records.groupby("ablation_cell")[columns].agg(["mean", "std"])
    rows: list[dict[str, Any]] = []
    for cell in CELL_ORDER:
        values = grouped.loc[cell]
        row: dict[str, Any] = {"Cell": CELL_LABELS[cell].replace("\n", " ")}
        for metric, label, _color in PCA_STAGE_SPECS:
            row[f"{label} mean (s)"] = values[(metric, "mean")]
            row[f"{label} std (s)"] = values[(metric, "std")]
        row["Total PCA fit mean (s)"] = values[("pca_cost_total", "mean")]
        row["Total PCA fit std (s)"] = values[("pca_cost_total", "std")]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_trainability_gap(
    records: pd.DataFrame,
    *,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Contrast learned-model error with each PCA representation's oracle floor."""
    grouped = records.groupby("ablation_cell")
    x = np.arange(len(CELL_ORDER))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.3, 4.3))
    for offset_index, (metric, label, color) in enumerate(
        (
            ("metric_mre", "Learned model", "#4C78A8"),
            ("metric_pca_oracle_mre", "PCA oracle", "#F58518"),
        )
    ):
        means = np.array([grouped.get_group(cell)[metric].mean() for cell in CELL_ORDER])
        stds = np.array(
            [grouped.get_group(cell)[metric].std(ddof=1) for cell in CELL_ORDER]
        )
        ax.bar(
            x + (offset_index - 0.5) * width,
            means,
            yerr=stds,
            width=width,
            color=color,
            label=label,
            capsize=STYLE["capsize"],
        )
    ax.set_yscale("log")
    ax.set_xticks(x, [CELL_LABELS[cell] for cell in CELL_ORDER])
    ax.set_ylabel("Mean relative error")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    return _save_figure(fig, output_base, formats)


def _timing_table(records: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    full_reference = {
        "plain_l2l": "plain_full",
        "overlap_l2l": "overlap_full",
        "two_scale": "two_scale_full",
    }
    for _, row in summary.iterrows():
        cell = str(row["ablation_cell"])
        method = str(row["ablation_method"])
        reference = full_reference[method]
        reference_rows, candidate_rows = _paired_rows(records, reference, cell)
        pca_speedup = (
            reference_rows["timing_pca_fit"].to_numpy(dtype=float)
            / candidate_rows["timing_pca_fit"].to_numpy(dtype=float)
        ).mean()
        rows.append(
            {
                "Method": METHOD_LABELS[method],
                "Solver": str(row["ablation_solver_class"]).title(),
                "PCA fit (s)": _format_mean_std(
                    row["timing_pca_fit_mean"],
                    row["timing_pca_fit_std"],
                    style="time",
                ),
                "PCA speedup": f"{pca_speedup:.2f}x",
                "Latent transform (s)": _format_mean_std(
                    row["timing_latent_transform_mean"],
                    row["timing_latent_transform_std"],
                    style="time",
                ),
                "NN train (s)": _format_mean_std(
                    row["timing_nn_train_mean"],
                    row["timing_nn_train_std"],
                    style="time",
                ),
                "End-to-end (s)": _format_mean_std(
                    row["timing_end_to_end_mean"],
                    row["timing_end_to_end_std"],
                    style="time",
                ),
            }
        )
    return pd.DataFrame(rows)


def _formatted_summary(
    summary: pd.DataFrame,
    specs: Sequence[tuple[str, str, str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        formatted: dict[str, Any] = {
            "Method": METHOD_LABELS[str(row["ablation_method"])],
            "Solver": str(row["ablation_solver_class"]).title(),
        }
        for metric, label, style in specs:
            formatted[label] = _format_mean_std(
                row.get(f"{metric}_mean"),
                row.get(f"{metric}_std"),
                style=style,
            )
        rows.append(formatted)
    return pd.DataFrame(rows)


def _format_pca_diagnostics(table: pd.DataFrame) -> pd.DataFrame:
    formatted = table.copy()
    for column in formatted.columns:
        if column in {"Cell", "Method", "Solver"}:
            continue
        if "dimension" in column.lower():
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.1f}"
            )
        elif "variance" in column.lower():
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.6f}"
            )
        else:
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.3f}"
            )
    return formatted


def _paired_rows(
    records: pd.DataFrame,
    reference: str,
    candidate: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference_rows = records[records["ablation_cell"] == reference].sort_values("seed")
    candidate_rows = records[records["ablation_cell"] == candidate].sort_values("seed")
    if list(reference_rows["seed"]) != list(candidate_rows["seed"]):
        raise ValueError(f"Unpaired seeds for {candidate} versus {reference}.")
    return reference_rows, candidate_rows


def _mean_confidence_interval(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, mean, mean
    half_width = float(
        t.ppf(0.95, len(values) - 1)
        * np.std(values, ddof=1)
        / np.sqrt(len(values))
    )
    return mean, mean - half_width, mean + half_width


def _equivalence_decision(
    lower: float,
    upper: float,
    *,
    margin: float,
    direction: str,
) -> str:
    if lower >= -margin and upper <= margin:
        return "Equivalent"
    if direction == "lower" and upper < -margin:
        return "Improved beyond band"
    if direction == "higher" and lower > margin:
        return "Improved beyond band"
    if direction == "lower" and lower > margin:
        return "Degraded beyond band"
    if direction == "higher" and upper < -margin:
        return "Degraded beyond band"
    return "Inconclusive"


def _format_gate_value(value: Any, *, mode: str) -> str:
    if pd.isna(value):
        return ""
    numeric = float(value)
    if mode == "relative":
        return f"{100.0 * numeric:+.2f}%"
    if mode == "absolute":
        return f"{numeric:+.5f}"
    return f"{numeric:.2f}x"


def _solver_from_cell(cell: str) -> str:
    if cell.endswith("_randomized"):
        return "randomized"
    if cell.endswith("_hybrid"):
        return "hybrid"
    return "full"


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
        return f"{mean_value:.5f} +/- {std_value:.5f}"
    if style == "time":
        return f"{mean_value:.1f} +/- {std_value:.1f}"
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
        default="paper_results/svd_solver_ablation",
    )
    parser.add_argument(
        "--formats",
        default="png,pdf",
        help="Comma-separated output formats.",
    )
    parser.add_argument(
        "--expected-seeds",
        type=int,
        default=5,
        help="Required unique seeds per cell; use 0 to disable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    formats = tuple(piece.strip() for piece in args.formats.split(",") if piece.strip())
    written = make_svd_ablation_outputs(
        results_dir=args.results_dir,
        formats=formats,
        expected_seeds=args.expected_seeds,
    )
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
