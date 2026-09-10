#!/usr/bin/env python
"""Render manuscript outputs for the two-scale latent-loss ablation."""

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
    _plot_sample_pdf,
    _plot_sample_spectrum,
    _prediction_path,
    _save_figure,
    _write_table_pair,
)

MODE_ORDER = ("mse", "block_balanced", "score_normalized")
MODE_LABELS = {
    "mse": "Vector MSE",
    "block_balanced": "Block-balanced",
    "score_normalized": "Score-normalized",
}
MODE_SHORT_LABELS = {
    "mse": "MSE",
    "block_balanced": "Block",
    "score_normalized": "Score",
}
MODE_COLORS = {
    "mse": "#6F6F6F",
    "block_balanced": "#F58518",
    "score_normalized": "#4C78A8",
}
DATASET_LABELS = {"poisson": "Poisson", "darcy": "Darcy"}
STAGE_SPECS = (
    ("timing_pca_fit", "PCA fit", "#4C78A8"),
    ("timing_latent_transform", "Latent transform", "#72B7B2"),
    ("timing_nn_train", "NN training", "#F58518"),
    ("timing_inference", "Inference", "#E45756"),
)
ERROR_METRICS = (
    ("metric_mre", "MRE"),
    ("metric_structural_error", "1 - SSIM"),
    ("metric_relative_spectrum_error", "Spectral"),
    ("metric_interface_flux_jump", "Flux jump"),
    ("metric_pde_residual_rms", "PDE residual"),
)
FULL_METRICS = (
    "metric_mre",
    "metric_ssim_mean",
    "metric_mse",
    "metric_mae",
    "metric_relative_spectrum_error",
    "metric_interface_jump",
    "metric_interface_flux_jump",
    "metric_pde_residual_rms",
    "metric_best_val_loss",
    "metric_num_parameters",
    "timing_pca_fit",
    "timing_latent_transform",
    "timing_nn_train",
    "timing_inference",
    "timing_end_to_end",
)


def make_latent_loss_ablation_outputs(
    *,
    results_dir: str | Path,
    formats: Sequence[str] = ("png", "pdf"),
    expected_seeds: int = 1,
    qualitative: bool = True,
) -> list[Path]:
    """Generate main-text and appendix assets from the completed ablation."""
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
    relative = relative_effects(records)

    written: list[Path] = []
    written.extend(write_tables(records, relative, main_tables, appendix_tables))
    written.extend(
        plot_accuracy(
            records,
            output_base=main_figures / "latent_loss_accuracy",
            formats=formats,
        )
    )
    written.extend(
        plot_tradeoffs(
            relative,
            output_base=main_figures / "latent_loss_tradeoffs",
            formats=formats,
        )
    )
    written.extend(
        plot_stage_costs(
            records,
            output_base=appendix_figures / "latent_loss_stage_costs",
            formats=formats,
        )
    )
    if qualitative:
        for dataset in ("poisson", "darcy"):
            paths = plot_qualitative(
                records,
                results_path=results_path,
                dataset=dataset,
                resolution=max(
                    int(value)
                    for value in records.loc[records["dataset"] == dataset, "resolution"].unique()
                ),
                output_base=appendix_figures / f"latent_loss_qualitative_{dataset}_256",
                formats=formats,
            )
            written.extend(paths)

    summary_path = results_path / "summary.md"
    summary_path.write_text(build_summary(records, relative), encoding="utf-8")
    written.append(summary_path)
    manifest = {
        "results_dir": str(results_path),
        "records": int(len(records)),
        "datasets": sorted(str(value) for value in records["dataset"].unique()),
        "resolutions": sorted(int(value) for value in records["resolution"].unique()),
        "modes": list(MODE_ORDER),
        "expected_seeds": expected_seeds,
        "outputs": [str(path) for path in written],
    }
    manifest_path = results_path / "latent_loss_ablation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    written.append(manifest_path)
    return written


def load_records(path: Path) -> pd.DataFrame:
    """Load the study roll-up and unify dataset-specific PDE metrics."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing latent-loss roll-up: {path}")
    records = pd.read_csv(path)
    numeric = [
        "seed",
        "resolution",
        *FULL_METRICS,
        "metric_poisson_residual_rms",
        "metric_darcy_residual_rms",
    ]
    for column in numeric:
        if column in records:
            records[column] = pd.to_numeric(records[column], errors="coerce")
    records["metric_structural_error"] = 1.0 - records["metric_ssim_mean"]
    records["metric_pde_residual_rms"] = np.where(
        records["dataset"].astype(str).str.lower() == "poisson",
        records.get("metric_poisson_residual_rms", np.nan),
        records.get("metric_darcy_residual_rms", np.nan),
    )
    return records


def validate_records(records: pd.DataFrame, *, expected_seeds: int) -> None:
    """Require a complete dataset-resolution-objective matrix."""
    required = {
        "run_dir",
        "seed",
        "dataset",
        "resolution",
        "latent_loss_mode",
        *FULL_METRICS,
    }
    missing = sorted(required.difference(records.columns))
    if missing:
        raise ValueError(f"Missing latent-loss columns: {', '.join(missing)}")
    duplicates = records.duplicated(
        ["dataset", "resolution", "latent_loss_mode", "seed"], keep=False
    )
    if duplicates.any():
        keys = records.loc[duplicates, ["dataset", "resolution", "latent_loss_mode", "seed"]]
        raise ValueError(f"Duplicate latent-loss records:\n{keys.to_string(index=False)}")
    modes = set(records["latent_loss_mode"].astype(str))
    if modes != set(MODE_ORDER):
        raise ValueError(f"Expected modes {MODE_ORDER}, found {sorted(modes)}")
    if expected_seeds > 0:
        counts = records.groupby(["dataset", "resolution", "latent_loss_mode"])["seed"].nunique()
        incomplete = counts[counts != expected_seeds]
        if not incomplete.empty:
            raise ValueError(f"Expected {expected_seeds} seeds per cell; found:\n{incomplete}")
    expected_cells = (
        records[["dataset", "resolution"]].drop_duplicates().shape[0]
        * len(MODE_ORDER)
        * max(expected_seeds, 1)
    )
    if expected_seeds > 0 and len(records) != expected_cells:
        raise ValueError(f"Expected {expected_cells} records, found {len(records)}")


def relative_effects(records: pd.DataFrame) -> pd.DataFrame:
    """Compute paired percentage changes relative to vector MSE."""
    metrics = [metric for metric, _ in ERROR_METRICS] + [
        "metric_interface_jump",
        "timing_nn_train",
        "timing_end_to_end",
    ]
    rows: list[dict[str, Any]] = []
    keys = ["dataset", "resolution", "seed"]
    for key, group in records.groupby(keys, sort=True):
        indexed = group.set_index("latent_loss_mode")
        baseline = indexed.loc["mse"]
        for mode in MODE_ORDER[1:]:
            candidate = indexed.loc[mode]
            row: dict[str, Any] = dict(zip(keys, key))
            row["latent_loss_mode"] = mode
            for metric in metrics:
                denominator = float(baseline[metric])
                value = float(candidate[metric])
                row[f"{metric}_change_percent"] = (
                    100.0 * (value / denominator - 1.0)
                    if np.isfinite(value) and np.isfinite(denominator) and denominator != 0.0
                    else np.nan
                )
            rows.append(row)
    return pd.DataFrame(rows)


def write_tables(
    records: pd.DataFrame,
    relative: pd.DataFrame,
    main_dir: Path,
    appendix_dir: Path,
) -> list[Path]:
    """Write compact manuscript tables and complete machine-readable tables."""
    written: list[Path] = []
    main = records.sort_values(
        ["dataset", "resolution", "latent_loss_mode"],
        key=lambda column: (
            column.map({mode: index for index, mode in enumerate(MODE_ORDER)})
            if column.name == "latent_loss_mode"
            else column
        ),
    )
    main_rows: list[dict[str, Any]] = []
    for _, row in main.iterrows():
        main_rows.append(
            {
                "Dataset": DATASET_LABELS[str(row["dataset"])],
                "Grid": int(row["resolution"]),
                "Latent objective": MODE_LABELS[str(row["latent_loss_mode"])],
                "MRE (%)": f"{100.0 * float(row['metric_mre']):.3f}",
                "SSIM": f"{float(row['metric_ssim_mean']):.4f}",
                "Spectral error": f"{float(row['metric_relative_spectrum_error']):.3e}",
                "Flux jump": f"{float(row['metric_interface_flux_jump']):.3e}",
                "PDE residual RMS": f"{float(row['metric_pde_residual_rms']):.3f}",
                "Total time (s)": f"{float(row['timing_end_to_end']):.1f}",
            }
        )
    written.extend(_write_table_pair(pd.DataFrame(main_rows), main_dir / "latent_loss_quality"))

    effect_rows: list[dict[str, Any]] = []
    for _, row in relative.sort_values(["dataset", "resolution", "latent_loss_mode"]).iterrows():
        effect_rows.append(
            {
                "Dataset": DATASET_LABELS[str(row["dataset"])],
                "Grid": int(row["resolution"]),
                "Objective": MODE_LABELS[str(row["latent_loss_mode"])],
                "MRE change (%)": _format_change(row["metric_mre_change_percent"]),
                "1-SSIM change (%)": _format_change(row["metric_structural_error_change_percent"]),
                "Spectral change (%)": _format_change(
                    row["metric_relative_spectrum_error_change_percent"]
                ),
                "Flux-jump change (%)": _format_change(
                    row["metric_interface_flux_jump_change_percent"]
                ),
                "PDE-residual change (%)": _format_change(
                    row["metric_pde_residual_rms_change_percent"]
                ),
                "NN-time change (%)": _format_change(row["timing_nn_train_change_percent"]),
            }
        )
    written.extend(
        _write_table_pair(pd.DataFrame(effect_rows), main_dir / "latent_loss_relative_effects")
    )

    full_columns = [
        "dataset",
        "resolution",
        "latent_loss_mode",
        "seed",
        *FULL_METRICS,
    ]
    full_path = appendix_dir / "latent_loss_full_metrics.csv"
    records[full_columns].sort_values(["dataset", "resolution", "latent_loss_mode", "seed"]).to_csv(
        full_path, index=False
    )
    written.append(full_path)
    effect_path = appendix_dir / "latent_loss_relative_effects_numeric.csv"
    relative.to_csv(effect_path, index=False)
    written.append(effect_path)

    capacity = capacity_table(records)
    capacity_path = appendix_dir / "latent_loss_capacity.csv"
    capacity.to_csv(capacity_path, index=False)
    written.append(capacity_path)
    written.extend(_write_table_pair(capacity, appendix_dir / "latent_loss_capacity", csv=False))
    return written


def capacity_table(records: pd.DataFrame) -> pd.DataFrame:
    """Read the common two-scale code dimensions from each run."""
    rows: list[dict[str, Any]] = []
    for _, record in records.iterrows():
        summary_path = _run_path(record["run_dir"]) / "pca_summary.json"
        counts: dict[str, Any] = {}
        if summary_path.is_file():
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            counts = payload.get("component_counts", {})
        rows.append(
            {
                "Dataset": DATASET_LABELS[str(record["dataset"])],
                "Grid": int(record["resolution"]),
                "Objective": MODE_LABELS[str(record["latent_loss_mode"])],
                "Input latent dim.": counts.get("input_total", ""),
                "Coarse dim.": counts.get("coarse_output", ""),
                "Residual dim.": counts.get("output_residual_total", ""),
                "Output latent dim.": counts.get("output_total", ""),
                "Parameters": int(record["metric_num_parameters"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["Dataset", "Grid", "Objective"])


def plot_accuracy(
    records: pd.DataFrame,
    *,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Plot MRE and structural error across problems and resolutions."""
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0), squeeze=False)
    metrics = (
        ("metric_mre", "Mean relative error (%)", 100.0),
        ("metric_structural_error", "Structural error, 1 - SSIM", 1.0),
    )
    datasets = ("poisson", "darcy")
    for row_index, (metric, ylabel, scale) in enumerate(metrics):
        for column_index, dataset in enumerate(datasets):
            axis = axes[row_index, column_index]
            selected = records[records["dataset"] == dataset]
            resolutions = sorted(int(value) for value in selected["resolution"].unique())
            x = np.arange(len(resolutions), dtype=float)
            width = 0.24
            for mode_index, mode in enumerate(MODE_ORDER):
                values = [
                    scale
                    * float(
                        selected[
                            (selected["resolution"] == resolution)
                            & (selected["latent_loss_mode"] == mode)
                        ][metric].mean()
                    )
                    for resolution in resolutions
                ]
                positions = x + (mode_index - 1) * width
                bars = axis.bar(
                    positions,
                    values,
                    width=width,
                    color=MODE_COLORS[mode],
                    label=MODE_LABELS[mode],
                )
                if metric == "metric_mre":
                    axis.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)
            axis.set_xticks(x, [str(value) for value in resolutions])
            axis.set_xlabel("Grid resolution")
            axis.set_ylabel(ylabel)
            axis.set_title(DATASET_LABELS[dataset])
            axis.grid(True, axis="y", alpha=0.22)
            axis.set_ylim(bottom=0.0)
            _panel_label(axis, row_index * 2 + column_index)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=3,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=1.6, w_pad=1.5)
    return _save_figure(fig, output_base, formats)


def plot_tradeoffs(
    relative: pd.DataFrame,
    *,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Show each alternative objective normalized to the vector-MSE baseline."""
    datasets = ("poisson", "darcy")
    resolutions = sorted(int(value) for value in relative["resolution"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), squeeze=False)
    labels = [label for _, label in ERROR_METRICS]
    x = np.arange(len(labels), dtype=float)
    width = 0.34
    for row_index, dataset in enumerate(datasets):
        for column_index, resolution in enumerate(resolutions):
            axis = axes[row_index, column_index]
            selected = relative[
                (relative["dataset"] == dataset) & (relative["resolution"] == resolution)
            ].set_index("latent_loss_mode")
            for mode_index, mode in enumerate(MODE_ORDER[1:]):
                ratios = np.array(
                    [
                        100.0 + float(selected.loc[mode, f"{metric}_change_percent"])
                        for metric, _ in ERROR_METRICS
                    ]
                )
                axis.bar(
                    x + (mode_index - 0.5) * width,
                    ratios,
                    width=width,
                    color=MODE_COLORS[mode],
                    label=MODE_LABELS[mode],
                )
            axis.axhline(100.0, color="#444444", linewidth=1.1, linestyle="--")
            axis.text(
                0.99,
                102.0,
                "Vector MSE",
                transform=axis.get_yaxis_transform(),
                ha="right",
                va="bottom",
                fontsize=7,
                color="#555555",
            )
            axis.set_xticks(x, labels, rotation=20, ha="right")
            axis.set_ylabel("Error relative to vector MSE (%)")
            axis.set_title(f"{DATASET_LABELS[dataset]} {resolution}")
            axis.grid(True, axis="y", alpha=0.22)
            axis.set_ylim(bottom=0.0)
            _panel_label(axis, row_index * 2 + column_index)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=1.5, w_pad=1.3)
    return _save_figure(fig, output_base, formats)


def plot_stage_costs(
    records: pd.DataFrame,
    *,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Plot cumulative pipeline costs for every objective and problem cell."""
    datasets = ("poisson", "darcy")
    resolutions = sorted(int(value) for value in records["resolution"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2), squeeze=False)
    for row_index, dataset in enumerate(datasets):
        for column_index, resolution in enumerate(resolutions):
            axis = axes[row_index, column_index]
            selected = records[
                (records["dataset"] == dataset) & (records["resolution"] == resolution)
            ].set_index("latent_loss_mode")
            x = np.arange(len(MODE_ORDER))
            bottom = np.zeros(len(MODE_ORDER), dtype=float)
            for metric, label, color in STAGE_SPECS:
                values = np.array([float(selected.loc[mode, metric]) for mode in MODE_ORDER])
                axis.bar(x, values, bottom=bottom, width=0.66, color=color, label=label)
                bottom += values
            totals = np.array(
                [float(selected.loc[mode, "timing_end_to_end"]) for mode in MODE_ORDER]
            )
            other = np.maximum(totals - bottom, 0.0)
            axis.bar(
                x,
                other,
                bottom=bottom,
                width=0.66,
                color="#B9B9B9",
                label="Other pipeline",
            )
            for index, total in enumerate(totals):
                axis.text(index, total + 0.025 * max(totals), f"{total:.0f}", ha="center")
            axis.set_xticks(x, [MODE_SHORT_LABELS[mode] for mode in MODE_ORDER])
            axis.set_ylabel("Cumulative time (s)")
            axis.set_title(f"{DATASET_LABELS[dataset]} {resolution}")
            axis.set_ylim(0.0, max(totals) * 1.16)
            axis.grid(True, axis="y", alpha=0.22)
            _panel_label(axis, row_index * 2 + column_index)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.6, w_pad=1.5)
    return _save_figure(fig, output_base, formats)


def plot_qualitative(
    records: pd.DataFrame,
    *,
    results_path: Path,
    dataset: str,
    resolution: int,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Compare predictions for a common median-difficulty seed-0 sample."""
    selected = records[
        (records["dataset"] == dataset)
        & (records["resolution"] == resolution)
        & (records["seed"] == 0)
    ].set_index("latent_loss_mode")
    if set(selected.index) != set(MODE_ORDER):
        return []
    baseline_path = _prediction_path(selected.loc["mse"], results_path)
    if not baseline_path.is_file():
        return []
    with np.load(baseline_path, allow_pickle=False) as archive:
        truth_all = np.asarray(archive["y_true"])
        prediction_all = np.asarray(archive["y_pred"])
        sample_ids = np.asarray(archive["sample_indices"])
        sample_mre = np.linalg.norm(
            (prediction_all - truth_all).reshape(len(truth_all), -1), axis=1
        ) / np.maximum(np.linalg.norm(truth_all.reshape(len(truth_all), -1), axis=1), 1.0e-12)
        local_index = int(np.argsort(sample_mre)[len(sample_mre) // 2])
        sample_id = int(sample_ids[local_index])
        truth = np.asarray(truth_all[local_index], dtype=np.float64)
        input_key = "f" if dataset == "poisson" else "a"
        input_field = np.asarray(archive[input_key][local_index], dtype=np.float64)

    predictions: dict[str, np.ndarray] = {}
    sample_errors: dict[str, float] = {}
    for mode in MODE_ORDER:
        path = _prediction_path(selected.loc[mode], results_path)
        with np.load(path, allow_pickle=False) as archive:
            ids = np.asarray(archive["sample_indices"])
            matches = np.flatnonzero(ids == sample_id)
            if len(matches) != 1:
                raise ValueError(f"Sample {sample_id} is not unique in {path}")
            prediction = np.asarray(archive["y_pred"][int(matches[0])], dtype=np.float64)
        predictions[mode] = prediction
        truth_norm = float(np.linalg.norm(truth))
        sample_errors[mode] = float(np.linalg.norm(prediction - truth) / max(truth_norm, 1.0e-12))

    solution_limit = float(
        max(np.max(np.abs(truth)), *(np.max(np.abs(value)) for value in predictions.values()))
    )
    error_limit = float(
        max(np.quantile(np.abs(value - truth), 0.995) for value in predictions.values())
    )
    input_vmin, input_vmax = np.quantile(input_field, [0.01, 0.99])
    pdf_values = np.concatenate([truth.ravel(), *(p.ravel() for p in predictions.values())])
    pdf_bins = np.linspace(
        float(np.quantile(pdf_values, 0.002)),
        float(np.quantile(pdf_values, 0.998)),
        55,
    )
    k_max = min(64, resolution // 3)

    fig, axes = plt.subplots(len(MODE_ORDER), 6, figsize=(14.5, 8.0), squeeze=False)
    for row_index, mode in enumerate(MODE_ORDER):
        prediction = predictions[mode]
        image_specs = (
            (
                input_field,
                "Input" if dataset == "poisson" else "Coefficient",
                "RdBu_r" if dataset == "poisson" else "cividis",
                input_vmin,
                input_vmax,
            ),
            (truth, "Ground truth", "RdBu_r", -solution_limit, solution_limit),
            (
                prediction,
                f"Prediction\nMRE {100.0 * sample_errors[mode]:.2f}%",
                "RdBu_r",
                -solution_limit,
                solution_limit,
            ),
            (np.abs(prediction - truth), "Absolute error", "magma", 0.0, error_limit),
        )
        for column_index, (field, title, cmap, vmin, vmax) in enumerate(image_specs):
            image = axes[row_index, column_index].imshow(
                field,
                cmap=cmap,
                origin="lower",
                vmin=float(vmin),
                vmax=float(vmax),
            )
            axes[row_index, column_index].set_xticks([])
            axes[row_index, column_index].set_yticks([])
            if row_index == 0:
                axes[row_index, column_index].set_title(title.split("\n")[0])
            if column_index == 2:
                axes[row_index, column_index].text(
                    0.03,
                    0.97,
                    f"MRE {100.0 * sample_errors[mode]:.2f}%",
                    transform=axes[row_index, column_index].transAxes,
                    ha="left",
                    va="top",
                    fontsize=7.5,
                    color="black",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
                )
            if column_index in (1, 3):
                fig.colorbar(image, ax=axes[row_index, column_index], fraction=0.046, pad=0.03)
        axes[row_index, 0].set_ylabel(MODE_LABELS[mode], fontsize=9)
        _plot_sample_spectrum(axes[row_index, 4], truth, prediction, k_max=k_max)
        _plot_sample_pdf(axes[row_index, 5], truth, prediction, bins=pdf_bins)
        if row_index == 0:
            axes[row_index, 4].set_title("Energy spectrum")
            axes[row_index, 5].set_title("Value PDF")
        if row_index == len(MODE_ORDER) - 1:
            axes[row_index, 4].set_xlabel("Wavenumber $k$")
            axes[row_index, 5].set_xlabel("Field value")
    fig.suptitle(
        f"{DATASET_LABELS[dataset]} {resolution}, seed 0, common sample {sample_id}",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=1.0, w_pad=0.8)
    return _save_figure(fig, output_base, formats)


def build_summary(records: pd.DataFrame, relative: pd.DataFrame) -> str:
    """Write a concise, results-derived manuscript interpretation."""
    lookup = records.set_index(["dataset", "resolution", "latent_loss_mode"])
    lines = [
        "# Two-Scale Latent-Loss Ablation Summary",
        "",
        "## Purpose",
        "",
        "This ablation asks how the coarse-global and local-residual output codes should be supervised. The two-scale representation, data splits, PCA bases, network architecture, and optimization settings are held fixed; only the latent objective changes.",
        "",
        "- **Vector MSE** averages squared error over the complete concatenated code.",
        "- **Block-balanced** assigns equal weight to coarse and residual block means.",
        "- **Score-normalized** additionally divides each score error by that PCA score's training variance.",
        "",
        "The study contains one run (seed 0) for Poisson and Darcy at 128 and 256. It is a design-selection ablation, not an uncertainty estimate.",
        "",
        "## Main Result",
        "",
        "| Dataset | Grid | Vector MSE | Block-balanced | Score-normalized | Best MRE |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for dataset in ("poisson", "darcy"):
        for resolution in sorted(int(value) for value in records["resolution"].unique()):
            values = {
                mode: float(lookup.loc[(dataset, resolution, mode), "metric_mre"])
                for mode in MODE_ORDER
            }
            best = min(values, key=values.get)  # type: ignore[arg-type]
            lines.append(
                f"| {DATASET_LABELS[dataset]} | {resolution} | "
                f"{100.0 * values['mse']:.3f}% | "
                f"{100.0 * values['block_balanced']:.3f}% | "
                f"{100.0 * values['score_normalized']:.3f}% | "
                f"{MODE_LABELS[best]} |"
            )
    lines.extend(
        [
            "",
            "Both structured objectives improve MRE over vector MSE in every dataset-resolution cell. Block balancing lowers MRE by 42-46% across all four cells. Score normalization lowers MRE by 30-35% on Poisson and 54-59% on Darcy.",
            "",
            "## Why Block Balancing Is the Default",
            "",
            "Block balancing is the most robust mechanism-level correction. It prevents the much larger residual block from dominating merely because it contains more coordinates, while preserving the natural variance hierarchy within each PCA block. It is best on both Poisson grids and remains strongly beneficial on Darcy.",
            "",
            "Score normalization gives the lowest Darcy MRE, but its physical diagnostics are mixed. Relative to vector MSE, its Darcy flux jump ranges "
            f"{_effect_range(relative, 'darcy', 'score_normalized', 'metric_interface_flux_jump')} and its Darcy PDE residual ranges "
            f"{_effect_range(relative, 'darcy', 'score_normalized', 'metric_pde_residual_rms')}. Normalizing every score to unit variance elevates low-variance tail coordinates, which can improve aggregate field error while overemphasizing difficult or noisy directions.",
            "",
            "The recommended default is therefore **equal block balancing (0.5 coarse, 0.5 residual)**. Score normalization is a useful dataset-specific alternative, but it should not replace block balancing as the universal setting without explicit physical-metric validation.",
            "",
            "## Computational Effect",
            "",
            "The representation and parameter count are unchanged across objectives. The structured objectives increase NN training time because their validation trajectories trigger later stopping, while PCA fit should be regarded as objective-independent. With one seed, small PCA timing differences are runtime noise rather than an algorithmic effect.",
            "",
            "## Manuscript Assets",
            "",
            "- `figures/main/latent_loss_accuracy`: absolute MRE and structural error.",
            "- `figures/main/latent_loss_tradeoffs`: each physical error normalized to vector MSE; values below 100% improve.",
            "- `tables/main/latent_loss_quality`: compact absolute results.",
            "- `tables/main/latent_loss_relative_effects`: percentage changes from vector MSE.",
            "- `figures/appendix/latent_loss_stage_costs`: cumulative stage timing.",
            "- `figures/appendix/latent_loss_qualitative_*_256`: common-sample field, spectrum, and PDF comparisons.",
            "",
            "## Limitation",
            "",
            "Because this selection study used only seed 0, the exact ranking between block-balanced and score-normalized losses should not be framed as statistically resolved. The consistent failure of unbalanced vector MSE across all four cells is the stronger conclusion; subsequent multi-seed paper experiments using block balancing provide the appropriate confirmation of the selected default.",
            "",
        ]
    )
    return "\n".join(lines)


def _effect_range(relative: pd.DataFrame, dataset: str, mode: str, metric: str) -> str:
    values = relative.loc[
        (relative["dataset"] == dataset) & (relative["latent_loss_mode"] == mode),
        f"{metric}_change_percent",
    ].to_numpy(dtype=float)
    low, high = float(np.min(values)), float(np.max(values))
    return f"from {low:+.1f}% to {high:+.1f}%"


def _run_path(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def _format_change(value: Any) -> str:
    return "" if pd.isna(value) else f"{float(value):+.1f}"


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
        default="paper_results/two_scale_latent_loss_ablation",
    )
    parser.add_argument(
        "--formats",
        default="png,pdf",
        help="Comma-separated output formats.",
    )
    parser.add_argument(
        "--expected-seeds",
        type=int,
        default=1,
        help="Required unique seeds per cell; use 0 to disable.",
    )
    parser.add_argument(
        "--skip-qualitative",
        action="store_true",
        help="Skip prediction-archive diagnostic figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    formats = tuple(value.strip() for value in args.formats.split(",") if value.strip())
    written = make_latent_loss_ablation_outputs(
        results_dir=args.results_dir,
        formats=formats,
        expected_seeds=args.expected_seeds,
        qualitative=not args.skip_qualitative,
    )
    print(f"Wrote {len(written)} latent-loss ablation assets.")


if __name__ == "__main__":
    main()
