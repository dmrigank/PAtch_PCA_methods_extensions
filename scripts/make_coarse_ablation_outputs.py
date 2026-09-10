#!/usr/bin/env python
"""Render manuscript figures and tables for the coarse-representation ablation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

SWEEP_ORDER = ("coarse_rank", "coarse_factor", "residual_variance")
SWEEP_LABELS = {
    "coarse_rank": "Coarse rank",
    "coarse_factor": "Coarse factor",
    "residual_variance": "Residual variance",
    "adaptive_reference": "Adaptive reference",
}
SWEEP_X_LABELS = {
    "coarse_rank": "Retained coarse modes",
    "coarse_factor": "Coarse factor c",
    "residual_variance": "Residual retained variance (%)",
}
SWEEP_DEFAULTS = {
    "coarse_rank": 10.0,
    "coarse_factor": 4.0,
    "residual_variance": 99.0,
}
RESOLUTION_COLORS = {128: "#4C78A8", 256: "#F58518"}
RESOLUTION_MARKERS = {128: "o", 256: "s"}
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
QUALITY_COLUMNS = (
    "metric_mre",
    "metric_pca_oracle_mre",
    "metric_ssim_mean",
    "metric_interface_jump",
    "metric_interface_flux_jump",
    "metric_relative_spectrum_error",
)
COST_COLUMNS = (
    "pca_coarse_output",
    "pca_output_residual_total",
    "pca_output_total",
    "timing_coarse_svd",
    "timing_pca_fit",
    "timing_latent_transform",
    "timing_nn_train",
    "timing_inference",
    "timing_end_to_end",
)


def make_coarse_ablation_outputs(
    *,
    results_dir: str | Path,
    figures_dir: str | Path | None = None,
    tables_dir: str | Path | None = None,
    formats: Sequence[str] = ("png", "pdf"),
    expected_seeds: int = 3,
) -> list[Path]:
    """Generate all coarse-ablation manuscript assets."""
    results_path = Path(results_dir)
    figure_path = Path(figures_dir) if figures_dir else results_path / "figures"
    table_path = Path(tables_dir) if tables_dir else results_path / "tables"
    figure_path.mkdir(parents=True, exist_ok=True)
    table_path.mkdir(parents=True, exist_ok=True)
    _apply_style()

    records = load_coarse_ablation_records(results_path)
    validate_ablation_records(records, expected_seeds=expected_seeds)
    sweep_records = expand_shared_baseline(records)
    summary = summarize_ablation(sweep_records)

    written: list[Path] = []
    written.extend(write_ablation_tables(summary, table_path))
    written.extend(
        plot_sensitivity_grid(
            sweep_records,
            metrics=(
                ("metric_mre", "Learned reconstruction MRE"),
                ("metric_pca_oracle_mre", "PCA-oracle MRE"),
            ),
            output_base=figure_path / "coarse_ablation_reconstruction",
            formats=formats,
            log_scale=True,
        )
    )
    written.extend(
        plot_sensitivity_grid(
            sweep_records,
            metrics=(
                ("metric_interface_jump", "Interface value jump"),
                ("metric_interface_flux_jump", "Interface flux jump"),
                ("metric_relative_spectrum_error", "Relative spectral error"),
            ),
            output_base=figure_path / "coarse_ablation_artifacts",
            formats=formats,
            log_scale=True,
        )
    )
    written.extend(
        plot_sensitivity_grid(
            sweep_records,
            metrics=(
                ("pca_output_total", "Output latent dimension"),
                ("timing_pca_fit", "PCA fit (s)"),
                ("timing_end_to_end", "End-to-end time (s)"),
            ),
            output_base=figure_path / "coarse_ablation_capacity_cost",
            formats=formats,
            log_scale=False,
        )
    )
    written.extend(
        plot_rank_trainability_gap(
            sweep_records,
            output_base=figure_path / "coarse_rank_trainability_gap",
            formats=formats,
        )
    )

    manifest = {
        "results_dir": str(results_path),
        "expected_seeds": expected_seeds,
        "records": int(len(records)),
        "unique_cells": int(records[["resolution", "cell"]].drop_duplicates().shape[0]),
        "outputs": [str(path) for path in written],
    }
    manifest_path = results_path / "coarse_ablation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    written.append(manifest_path)
    return written


def load_coarse_ablation_records(results_dir: str | Path) -> pd.DataFrame:
    """Load coarse-ablation JSONL records and PCA capacity metadata."""
    results_path = Path(results_dir)
    paths = sorted(results_path.rglob("results.jsonl"))
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                ablation = record.get("ablation", {})
                if not isinstance(ablation, Mapping) or not ablation:
                    continue
                rows.append(_flatten_record(record, source_path=path))
    if not rows:
        raise FileNotFoundError(
            f"No coarse-ablation records found below {results_path}."
        )
    frame = pd.DataFrame(rows)
    numeric_columns = [
        "seed",
        "resolution",
        "coarse_factor",
        "coarse_components",
        "coarse_variance",
        "residual_variance",
        *QUALITY_COLUMNS,
        *COST_COLUMNS,
    ]
    for column in numeric_columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def validate_ablation_records(records: pd.DataFrame, *, expected_seeds: int) -> None:
    """Fail early when a manuscript roll-up would silently mix incomplete cells."""
    required = {
        "resolution",
        "seed",
        "family",
        "cell",
        "coarse_factor",
        "coarse_components",
        "residual_variance",
        *QUALITY_COLUMNS,
        "timing_pca_fit",
        "timing_end_to_end",
    }
    missing = sorted(required.difference(records.columns))
    if missing:
        raise ValueError(f"Missing required coarse-ablation columns: {', '.join(missing)}")
    duplicates = records.duplicated(["resolution", "cell", "seed"], keep=False)
    if duplicates.any():
        keys = records.loc[duplicates, ["resolution", "cell", "seed"]]
        raise ValueError(f"Duplicate coarse-ablation records:\n{keys.to_string(index=False)}")
    if expected_seeds > 0:
        counts = records.groupby(["resolution", "cell"])["seed"].nunique()
        incomplete = counts[counts != expected_seeds]
        if not incomplete.empty:
            details = ", ".join(
                f"{resolution}/{cell}: {count}"
                for (resolution, cell), count in incomplete.items()
            )
            raise ValueError(
                f"Expected {expected_seeds} seeds per ablation cell; found {details}."
            )
    for resolution in sorted(records["resolution"].unique()):
        baseline = records[
            (records["resolution"] == resolution)
            & (records["family"] == "shared_baseline")
        ]
        if baseline.empty:
            raise ValueError(f"Resolution {resolution} has no shared baseline.")


def expand_shared_baseline(records: pd.DataFrame) -> pd.DataFrame:
    """Insert the shared default into each one-factor sweep for fair plotting."""
    pieces: list[pd.DataFrame] = []
    baseline = records[records["family"] == "shared_baseline"]
    for sweep in SWEEP_ORDER:
        family_rows = records[records["family"] == sweep].copy()
        default_rows = baseline.copy()
        family_rows["sweep"] = sweep
        default_rows["sweep"] = sweep
        family_rows["sweep_value"] = _sweep_values(family_rows, sweep)
        default_rows["sweep_value"] = SWEEP_DEFAULTS[sweep]
        family_rows["setting"] = [
            _setting_label(sweep, value, is_default=False)
            for value in family_rows["sweep_value"]
        ]
        default_rows["setting"] = _setting_label(
            sweep,
            SWEEP_DEFAULTS[sweep],
            is_default=True,
        )
        pieces.extend([family_rows, default_rows])

    adaptive = records[records["family"] == "adaptive_reference"].copy()
    if not adaptive.empty:
        adaptive["sweep"] = "adaptive_reference"
        adaptive["sweep_value"] = adaptive["pca_coarse_output"]
        adaptive["setting"] = "99% variance (cap 20)"
        pieces.append(adaptive)
    return pd.concat(pieces, ignore_index=True)


def summarize_ablation(records: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and sample standard deviation over seeds."""
    value_columns = [
        column
        for column in (*QUALITY_COLUMNS, *COST_COLUMNS)
        if column in records.columns
    ]
    group_columns = ["resolution", "sweep", "setting", "sweep_value"]
    grouped = records.groupby(group_columns, as_index=False, dropna=False)
    summary = grouped[value_columns].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(str(piece) for piece in column if piece)
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    summary["n_seeds"] = (
        records.groupby(group_columns, dropna=False)["seed"]
        .nunique()
        .reindex(pd.MultiIndex.from_frame(summary[group_columns]))
        .to_numpy()
    )
    summary["sweep_order"] = summary["sweep"].map(
        {name: index for index, name in enumerate((*SWEEP_ORDER, "adaptive_reference"))}
    )
    return summary.sort_values(
        ["resolution", "sweep_order", "sweep_value"],
        kind="stable",
    ).reset_index(drop=True)


def write_ablation_tables(summary: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Write numeric, formatted quality, and formatted cost tables."""
    written: list[Path] = []
    numeric_path = output_dir / "coarse_ablation_summary_numeric.csv"
    summary.to_csv(numeric_path, index=False)
    written.append(numeric_path)

    quality_specs = (
        ("metric_mre", "MRE", "metric"),
        ("metric_pca_oracle_mre", "Oracle MRE", "metric"),
        ("metric_ssim_mean", "SSIM", "ssim"),
        ("metric_interface_jump", "Value jump", "scientific"),
        ("metric_interface_flux_jump", "Flux jump", "scientific"),
        ("metric_relative_spectrum_error", "Spectral error", "metric"),
        ("pca_output_total", "Output dim.", "dimension"),
    )
    cost_specs = (
        ("pca_coarse_output", "Coarse modes", "dimension"),
        ("pca_output_residual_total", "Residual dim.", "dimension"),
        ("timing_coarse_svd", "Coarse SVD (s)", "time"),
        ("timing_pca_fit", "PCA fit (s)", "time"),
        ("timing_latent_transform", "Latent transform (s)", "time"),
        ("timing_nn_train", "NN train (s)", "time"),
        ("timing_inference", "Inference (s)", "time"),
        ("timing_end_to_end", "End-to-end (s)", "time"),
    )
    quality = _formatted_table(summary, quality_specs)
    cost = _formatted_table(summary, cost_specs)
    for stem, table in (
        ("coarse_ablation_quality_table", quality),
        ("coarse_ablation_cost_table", cost),
    ):
        csv_path = output_dir / f"{stem}.csv"
        tex_path = output_dir / f"{stem}.tex"
        table.to_csv(csv_path, index=False)
        tex_path.write_text(
            _to_latex(table),
            encoding="utf-8",
        )
        written.extend([csv_path, tex_path])

    for resolution in sorted(summary["resolution"].unique()):
        compact = quality[quality["Resolution"] == int(resolution)].drop(
            columns=["Resolution"]
        )
        tex_path = output_dir / f"coarse_ablation_quality_{int(resolution)}.tex"
        tex_path.write_text(
            _to_latex(compact),
            encoding="utf-8",
        )
        written.append(tex_path)
    return written


def plot_sensitivity_grid(
    records: pd.DataFrame,
    *,
    metrics: Sequence[tuple[str, str]],
    output_base: Path,
    formats: Sequence[str],
    log_scale: bool,
) -> list[Path]:
    """Plot each response against all three one-factor sweeps."""
    fig, axes = plt.subplots(
        len(metrics),
        len(SWEEP_ORDER),
        figsize=(10.8, 2.65 * len(metrics)),
        constrained_layout=False,
        squeeze=False,
        sharex=False,
    )
    resolutions = sorted(int(value) for value in records["resolution"].unique())
    for column, sweep in enumerate(SWEEP_ORDER):
        sweep_rows = records[records["sweep"] == sweep]
        for row, (metric, label) in enumerate(metrics):
            ax = axes[row, column]
            for resolution in resolutions:
                selected = sweep_rows[sweep_rows["resolution"] == resolution]
                if selected.empty or metric not in selected:
                    continue
                grouped = (
                    selected.groupby("sweep_value")[metric]
                    .agg(["mean", "std"])
                    .sort_index()
                )
                x = grouped.index.to_numpy(dtype=float)
                y = grouped["mean"].to_numpy(dtype=float)
                yerr = grouped["std"].fillna(0.0).to_numpy(dtype=float)
                ax.errorbar(
                    x,
                    y,
                    yerr=yerr,
                    color=RESOLUTION_COLORS.get(resolution, None),
                    marker=RESOLUTION_MARKERS.get(resolution, "o"),
                    markersize=STYLE["marker_size"],
                    linewidth=STYLE["line_width"],
                    capsize=STYLE["capsize"],
                )
            ax.axvline(
                SWEEP_DEFAULTS[sweep],
                color="#6F6F6F",
                linestyle=":",
                linewidth=1.2,
            )
            if log_scale:
                ax.set_yscale("log")
            ax.grid(True, alpha=0.22)
            if row == 0:
                ax.set_title(SWEEP_LABELS[sweep])
            if column == 0:
                ax.set_ylabel(label)
            if row == len(metrics) - 1:
                ax.set_xlabel(SWEEP_X_LABELS[sweep])
            _set_sweep_ticks(ax, sweep, sweep_rows)
            ax.text(
                0.02,
                0.96,
                f"({chr(97 + row * len(SWEEP_ORDER) + column)})",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontweight="bold",
            )
    handles = [
        Line2D(
            [0],
            [0],
            color=RESOLUTION_COLORS.get(resolution, "black"),
            marker=RESOLUTION_MARKERS.get(resolution, "o"),
            linewidth=STYLE["line_width"],
            markersize=STYLE["marker_size"],
            label=f"{resolution} x {resolution}",
        )
        for resolution in resolutions
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color="#6F6F6F",
            linestyle=":",
            linewidth=1.2,
            label="Default",
        )
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=max(len(handles), 1),
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93), h_pad=1.5, w_pad=1.3)
    return _save_figure(fig, output_base, formats)


def plot_rank_trainability_gap(
    records: pd.DataFrame,
    *,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Expose when richer PCA representations become harder for the regressor."""
    rank_rows = records[records["sweep"] == "coarse_rank"]
    resolutions = sorted(int(value) for value in rank_rows["resolution"].unique())
    fig, axes = plt.subplots(
        1,
        len(resolutions),
        figsize=(4.7 * len(resolutions), 3.7),
        constrained_layout=True,
        squeeze=False,
        sharey=True,
    )
    for index, resolution in enumerate(resolutions):
        ax = axes[0, index]
        selected = rank_rows[rank_rows["resolution"] == resolution]
        grouped = (
            selected.groupby("sweep_value")[
                ["metric_mre", "metric_pca_oracle_mre"]
            ]
            .agg(["mean", "std"])
            .sort_index()
        )
        x = grouped.index.to_numpy(dtype=float)
        for metric, label, color, marker in (
            ("metric_mre", "Learned model", "#E45756", "o"),
            ("metric_pca_oracle_mre", "PCA oracle", "#4C78A8", "s"),
        ):
            ax.errorbar(
                x,
                grouped[(metric, "mean")].to_numpy(dtype=float),
                yerr=grouped[(metric, "std")].fillna(0.0).to_numpy(dtype=float),
                color=color,
                marker=marker,
                linewidth=STYLE["line_width"],
                capsize=STYLE["capsize"],
                label=label,
            )
        ax.axvline(10, color="#6F6F6F", linestyle=":", linewidth=1.2)
        ax.set_yscale("log")
        ax.set_title(f"{resolution} x {resolution}")
        ax.set_xlabel("Retained coarse modes")
        ax.set_xticks(x)
        ax.grid(True, alpha=0.22)
        if index == 0:
            ax.set_ylabel("Mean relative error")
        ax.text(
            0.02,
            0.96,
            f"({chr(97 + index)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
    axes[0, 0].legend(frameon=False)
    return _save_figure(fig, output_base, formats)


def _flatten_record(record: Mapping[str, Any], *, source_path: Path) -> dict[str, Any]:
    ablation = dict(record["ablation"])
    run_dir = _resolve_run_dir(
        Path(str(record.get("run_dir", ""))),
        source_path=source_path,
    )
    row: dict[str, Any] = {
        "run_id": record.get("run_id", ""),
        "run_dir": str(run_dir),
        "seed": record.get("seed", np.nan),
        "resolution": record.get("resolution", np.nan),
        "family": ablation.get("family", ""),
        "cell": ablation.get("cell", ""),
        "coarse_factor": ablation.get("coarse_factor", np.nan),
        "coarse_components": ablation.get("coarse_components", np.nan),
        "coarse_variance": ablation.get("coarse_variance", np.nan),
        "residual_variance": ablation.get("residual_variance", np.nan),
    }
    for key, value in dict(record.get("metrics", {})).items():
        row[f"metric_{key}"] = _float_or_nan(value)
    for key, value in dict(record.get("timings", {})).items():
        row[f"timing_{key}"] = _float_or_nan(value)
    pca_summary = _load_json(run_dir / "pca_summary.json")
    component_counts = pca_summary.get("component_counts", {})
    if isinstance(component_counts, Mapping):
        for key in (
            "input_total",
            "coarse_output",
            "output_residual_total",
            "output_total",
        ):
            row[f"pca_{key}"] = _float_or_nan(component_counts.get(key))
    return row


def _resolve_run_dir(run_dir: Path, *, source_path: Path) -> Path:
    if run_dir.is_dir():
        return run_dir
    parts = run_dir.parts
    for parent in source_path.parents:
        if parent.name and parent.name in parts:
            index = parts.index(parent.name)
            candidate = parent.joinpath(*parts[index + 1 :])
            if candidate.is_dir():
                return candidate
    return run_dir


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _sweep_values(frame: pd.DataFrame, sweep: str) -> pd.Series:
    if sweep == "coarse_rank":
        return frame["coarse_components"].astype(float)
    if sweep == "coarse_factor":
        return frame["coarse_factor"].astype(float)
    if sweep == "residual_variance":
        return 100.0 * frame["residual_variance"].astype(float)
    raise ValueError(f"Unknown sweep: {sweep}")


def _setting_label(sweep: str, value: float, *, is_default: bool) -> str:
    suffix = " (default)" if is_default else ""
    if sweep == "coarse_rank":
        return f"rank {int(value)}{suffix}"
    if sweep == "coarse_factor":
        return f"c = {int(value)}{suffix}"
    if sweep == "residual_variance":
        return f"{value:g}%{suffix}"
    return f"{value:g}{suffix}"


def _formatted_table(
    summary: pd.DataFrame,
    specs: Sequence[tuple[str, str, str]],
) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "Resolution": summary["resolution"].astype(int),
            "Ablation": summary["sweep"].map(SWEEP_LABELS),
            "Setting": summary["setting"],
            "Seeds": summary["n_seeds"].astype(int),
        }
    )
    for base, label, style in specs:
        mean_column = f"{base}_mean"
        std_column = f"{base}_std"
        if mean_column not in summary:
            continue
        table[label] = [
            _format_mean_std(mean, std, style=style)
            for mean, std in zip(summary[mean_column], summary[std_column])
        ]
    return table


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
    if style == "dimension":
        return f"{mean_value:.1f} +/- {std_value:.1f}"
    return f"{mean_value:.4f} +/- {std_value:.4f}"


def _to_latex(table: pd.DataFrame) -> str:
    return table.to_latex(index=False, escape=True).replace(
        "+/-",
        r"\ensuremath{\pm}",
    )


def _set_sweep_ticks(ax: plt.Axes, sweep: str, rows: pd.DataFrame) -> None:
    values = sorted(float(value) for value in rows["sweep_value"].dropna().unique())
    ax.set_xticks(values)
    if sweep == "residual_variance":
        ax.set_xticklabels([f"{value:g}" for value in values])
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


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
        default="paper_results/coarse_representation_ablation",
    )
    parser.add_argument("--figures-dir", default=None)
    parser.add_argument("--tables-dir", default=None)
    parser.add_argument(
        "--formats",
        default="png,pdf",
        help="Comma-separated output formats.",
    )
    parser.add_argument(
        "--expected-seeds",
        type=int,
        default=3,
        help="Required unique seeds per resolution/configuration cell; use 0 to disable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    formats = tuple(piece.strip() for piece in args.formats.split(",") if piece.strip())
    written = make_coarse_ablation_outputs(
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        tables_dir=args.tables_dir,
        formats=formats,
        expected_seeds=args.expected_seeds,
    )
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
