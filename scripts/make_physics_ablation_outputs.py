#!/usr/bin/env python
"""Render manuscript figures and tables for the physics-loss ablation."""

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

from lpcanet.metrics.spectral import energy_spectrum_2d

STAGE_ORDER = (
    "a0_latent_baseline",
    "a1_reconstruction",
    "a2_value",
    "a3_value_flux",
    "a4_value_flux_spectral",
    "a5_full_pde",
)
STAGE_LABELS = {
    "a0_latent_baseline": "A0\nLatent",
    "a1_reconstruction": "A1\nRecon.",
    "a2_value": "A2\n+ value",
    "a3_value_flux": "A3\n+ flux",
    "a4_value_flux_spectral": "A4\n+ spectral",
    "a5_full_pde": "A5\n+ PDE",
}
TABLE_LABELS = {
    "a0_latent_baseline": "A0: latent baseline",
    "a1_reconstruction": "A1: reconstruction",
    "a2_value": "A2: + interface value",
    "a3_value_flux": "A3: + interface flux",
    "a4_value_flux_spectral": "A4: + spectral",
    "a5_full_pde": "A5: + PDE residual",
}
STAGE_COLORS = {
    "a0_latent_baseline": "#6F6F6F",
    "a1_reconstruction": "#4C78A8",
    "a2_value": "#72B7B2",
    "a3_value_flux": "#54A24B",
    "a4_value_flux_spectral": "#F58518",
    "a5_full_pde": "#E45756",
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
QUALITY_METRICS = (
    "metric_mre",
    "metric_ssim_mean",
    "metric_mse",
    "metric_mae",
    "metric_interface_jump",
    "metric_interface_flux_jump",
    "metric_interface_value_trace_error",
    "metric_interface_flux_trace_error",
    "metric_relative_spectrum_error",
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
    ("metric_mre", "MRE", "metric"),
    ("metric_ssim_mean", "SSIM", "ssim"),
    ("metric_interface_jump", "Value jump", "scientific"),
    ("metric_interface_flux_jump", "Flux jump", "scientific"),
    ("metric_relative_spectrum_error", "Spectral error", "metric"),
    ("metric_poisson_residual_rms", "PDE residual RMS", "metric"),
    ("invocation_timing_end_to_end", "Fine-tune time (s)", "time"),
    ("timing_end_to_end", "Total time (s)", "time"),
)
FULL_METRICS = (
    ("metric_mre", "MRE", "metric"),
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
    ("timing_nn_train", "NN train, cumulative (s)", "time"),
    ("timing_inference", "Inference, cumulative (s)", "time"),
    ("invocation_timing_nn_train", "Fine-tune NN (s)", "time"),
    ("invocation_timing_end_to_end", "Fine-tune total (s)", "time"),
    ("timing_end_to_end", "End-to-end, cumulative (s)", "time"),
)


def make_physics_ablation_outputs(
    *,
    results_dir: str | Path,
    formats: Sequence[str] = ("png", "pdf"),
    expected_seeds: int = 3,
    qualitative: bool = True,
) -> list[Path]:
    """Generate main-text and appendix assets from a completed ablation."""
    results_path = Path(results_dir)
    main_figures = results_path / "figures" / "main"
    appendix_figures = results_path / "figures" / "appendix"
    main_tables = results_path / "tables" / "main"
    appendix_tables = results_path / "tables" / "appendix"
    for path in (main_figures, appendix_figures, main_tables, appendix_tables):
        path.mkdir(parents=True, exist_ok=True)
    _apply_style()

    records = load_records(results_path / "results.csv")
    screen = load_records(results_path / "screen" / "results.csv")
    selection = json.loads(
        (results_path / "selection.json").read_text(encoding="utf-8")
    )
    validate_records(records, expected_seeds=expected_seeds)
    summary = summarize_records(records)

    written: list[Path] = []
    written.extend(
        write_tables(
            records,
            screen,
            summary,
            selection,
            main_tables,
            appendix_tables,
        )
    )
    written.extend(
        plot_loss_ladder(
            records,
            output_base=main_figures / "physics_loss_ladder",
            formats=formats,
        )
    )
    written.extend(
        plot_accuracy_continuity_pareto(
            records,
            output_base=main_figures / "physics_accuracy_continuity_pareto",
            formats=formats,
        )
    )
    written.extend(
        plot_cost_breakdown(
            records,
            output_base=appendix_figures / "physics_cost_breakdown",
            formats=formats,
        )
    )
    written.extend(
        plot_weight_screen(
            screen,
            selection_path=results_path / "selection.json",
            output_base=appendix_figures / "physics_weight_screen",
            formats=formats,
        )
    )
    written.extend(
        plot_training_curves(
            records,
            output_base=appendix_figures / "physics_training_curves",
            formats=formats,
        )
    )
    if qualitative:
        for resolution, output_dir in (
            (128, main_figures),
            (256, appendix_figures),
        ):
            written.extend(
                plot_qualitative(
                    records,
                    resolution=resolution,
                    output_base=output_dir / f"physics_qualitative_{resolution}",
                    formats=formats,
                )
            )

    manifest = {
        "results_dir": str(results_path),
        "records": int(len(records)),
        "screen_records": int(len(screen)),
        "expected_seeds": expected_seeds,
        "qualitative": qualitative,
        "outputs": [str(path) for path in written],
    }
    manifest_path = results_path / "physics_ablation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    written.append(manifest_path)
    return written


def load_records(path: Path) -> pd.DataFrame:
    """Load a stable results roll-up."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing physics-ablation roll-up: {path}")
    frame = pd.read_csv(path)
    for column in frame.columns:
        if column.startswith(
            (
                "metric_",
                "validation_metric_",
                "timing_",
                "invocation_timing_",
                "ablation_target_ratio_",
            )
        ) or column in {"seed", "resolution"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def validate_records(records: pd.DataFrame, *, expected_seeds: int) -> None:
    """Reject incomplete or duplicated confirmation matrices."""
    required = {
        "resolution",
        "seed",
        "baseline_model",
        "run_dir",
        *QUALITY_METRICS[:6],
        "metric_poisson_residual_rms",
        *TIMING_METRICS,
    }
    missing = sorted(required.difference(records.columns))
    if missing:
        raise ValueError(f"Missing required physics-ablation columns: {', '.join(missing)}")
    unexpected = sorted(set(records["baseline_model"]) - set(STAGE_ORDER))
    if unexpected:
        raise ValueError(f"Unexpected physics-ablation variants: {', '.join(unexpected)}")
    duplicates = records.duplicated(["resolution", "baseline_model", "seed"], keep=False)
    if duplicates.any():
        keys = records.loc[duplicates, ["resolution", "baseline_model", "seed"]]
        raise ValueError(f"Duplicate physics-ablation records:\n{keys.to_string(index=False)}")
    if expected_seeds <= 0:
        return
    counts = records.groupby(["resolution", "baseline_model"])["seed"].nunique()
    incomplete = counts[counts != expected_seeds]
    if not incomplete.empty:
        details = ", ".join(
            f"{resolution}/{variant}: {count}"
            for (resolution, variant), count in incomplete.items()
        )
        raise ValueError(f"Expected {expected_seeds} seeds per cell; found {details}.")


def summarize_records(records: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and sample standard deviation over seeds."""
    numeric = [
        column
        for column in (*QUALITY_METRICS, *TIMING_METRICS)
        if column in records
    ]
    summary = (
        records.groupby(["resolution", "baseline_model"], as_index=False)[numeric]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(piece) for piece in column if piece)
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    counts = records.groupby(["resolution", "baseline_model"])["seed"].nunique()
    summary["n_seeds"] = counts.reindex(
        pd.MultiIndex.from_frame(summary[["resolution", "baseline_model"]])
    ).to_numpy()
    summary["stage_order"] = summary["baseline_model"].map(
        {name: index for index, name in enumerate(STAGE_ORDER)}
    )
    return summary.sort_values(["resolution", "stage_order"]).reset_index(drop=True)


def write_tables(
    records: pd.DataFrame,
    screen: pd.DataFrame,
    summary: pd.DataFrame,
    selection: dict[str, Any],
    main_dir: Path,
    appendix_dir: Path,
) -> list[Path]:
    """Write formatted main tables and complete appendix roll-ups."""
    written: list[Path] = []
    numeric_path = appendix_dir / "physics_loss_ablation_summary_numeric.csv"
    summary.to_csv(numeric_path, index=False)
    written.append(numeric_path)

    main_table = _formatted_summary(summary, MAIN_METRICS)
    full_table = _formatted_summary(summary, FULL_METRICS)
    written.extend(_write_table_pair(main_table, main_dir / "physics_loss_ablation_main"))
    for resolution in sorted(int(value) for value in records["resolution"].unique()):
        selected = main_table[main_table["Resolution"] == resolution].drop(
            columns=["Resolution"]
        )
        written.extend(
            _write_table_pair(
                selected,
                main_dir / f"physics_loss_ablation_main_{resolution}",
            )
        )
    written.extend(
        _write_table_pair(full_table, appendix_dir / "physics_loss_ablation_full")
    )

    relative = _relative_change_table(records)
    relative_path = appendix_dir / "physics_loss_ablation_relative_changes.csv"
    relative.to_csv(relative_path, index=False)
    relative_tex = relative.copy()
    for column in relative_tex.columns:
        if column.endswith("_pct"):
            relative_tex[column] = relative_tex[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):+.1f}%"
            )
    written.append(relative_path)
    written.extend(
        _write_table_pair(
            relative_tex,
            appendix_dir / "physics_loss_ablation_relative_changes_formatted",
            csv=False,
        )
    )

    screen_table = _screen_table(screen, selection)
    written.extend(
        _write_table_pair(screen_table, appendix_dir / "physics_weight_screen")
    )
    calibration = _calibration_table(records)
    written.extend(
        _write_table_pair(calibration, appendix_dir / "physics_calibrated_weights")
    )
    cost = _formatted_summary(
        summary,
        tuple(spec for spec in FULL_METRICS if "time" in spec[2]),
    )
    written.extend(_write_table_pair(cost, appendix_dir / "physics_stage_costs"))

    per_seed_columns = [
        column
        for column in (
            "resolution",
            "seed",
            "baseline_model",
            "config_hash",
            *QUALITY_METRICS,
            *TIMING_METRICS,
        )
        if column in records
    ]
    per_seed = records[per_seed_columns].sort_values(
        ["resolution", "baseline_model", "seed"]
    )
    per_seed_path = appendix_dir / "physics_loss_ablation_per_seed.csv"
    per_seed.to_csv(per_seed_path, index=False)
    written.append(per_seed_path)
    return written


def plot_loss_ladder(
    records: pd.DataFrame,
    *,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Plot paired metrics normalized by each seed's A0 result."""
    metrics = (
        ("metric_mre", "MRE / A0"),
        ("metric_interface_jump", "Value jump / A0"),
        ("metric_interface_flux_jump", "Flux jump / A0"),
        ("metric_relative_spectrum_error", "Spectral error / A0"),
        ("metric_poisson_residual_rms", "PDE residual / A0"),
        ("timing_end_to_end", "Total time / A0"),
    )
    normalized = _normalize_to_reference(records, reference="a0_latent_baseline")
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.0), squeeze=False)
    x = np.arange(len(STAGE_ORDER))
    resolutions = sorted(int(value) for value in records["resolution"].unique())
    for index, (metric, label) in enumerate(metrics):
        ax = axes.flat[index]
        for resolution in resolutions:
            selected = normalized[normalized["resolution"] == resolution]
            grouped = selected.groupby("baseline_model")[metric].agg(["mean", "std"])
            means = np.array([grouped.loc[stage, "mean"] for stage in STAGE_ORDER])
            stds = np.array([grouped.loc[stage, "std"] for stage in STAGE_ORDER])
            ax.errorbar(
                x,
                means,
                yerr=np.nan_to_num(stds),
                color=RESOLUTION_COLORS[resolution],
                marker=RESOLUTION_MARKERS[resolution],
                linewidth=STYLE["line_width"],
                markersize=STYLE["marker_size"],
                capsize=STYLE["capsize"],
                label=f"{resolution} x {resolution}",
            )
            for stage_index, stage in enumerate(STAGE_ORDER):
                values = selected.loc[
                    selected["baseline_model"] == stage,
                    metric,
                ].to_numpy(dtype=float)
                jitter = np.linspace(-0.055, 0.055, len(values))
                ax.scatter(
                    stage_index + jitter,
                    values,
                    color=RESOLUTION_COLORS[resolution],
                    s=9,
                    alpha=0.35,
                    linewidths=0,
                )
        ax.axhline(1.0, color="#777777", linestyle=":", linewidth=1.1)
        ax.set_ylabel(label)
        ax.set_xticks(x, [STAGE_LABELS[stage] for stage in STAGE_ORDER])
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
    handles = [
        Line2D(
            [0],
            [0],
            color=RESOLUTION_COLORS[resolution],
            marker=RESOLUTION_MARKERS[resolution],
            linewidth=STYLE["line_width"],
            label=f"{resolution} x {resolution}",
        )
        for resolution in resolutions
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.6, w_pad=1.4)
    return _save_figure(fig, output_base, formats)


def plot_accuracy_continuity_pareto(
    records: pd.DataFrame,
    *,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Show the reconstruction/interface tradeoff and dominated loss variants."""
    resolutions = sorted(int(value) for value in records["resolution"].unique())
    fig, axes = plt.subplots(
        1,
        len(resolutions),
        figsize=(4.8 * len(resolutions), 3.8),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    for column, resolution in enumerate(resolutions):
        ax = axes[0, column]
        selected = records[records["resolution"] == resolution]
        for stage in STAGE_ORDER:
            stage_rows = selected[selected["baseline_model"] == stage]
            x = stage_rows["metric_mre"].to_numpy(dtype=float)
            y = stage_rows["metric_interface_flux_jump"].to_numpy(dtype=float)
            ax.scatter(x, y, color=STAGE_COLORS[stage], s=15, alpha=0.35)
            ax.errorbar(
                float(np.mean(x)),
                float(np.mean(y)),
                xerr=float(np.std(x, ddof=1)),
                yerr=float(np.std(y, ddof=1)),
                color=STAGE_COLORS[stage],
                marker="o",
                markersize=6,
                capsize=STYLE["capsize"],
                linewidth=1.3,
            )
            ax.annotate(
                stage.split("_", 1)[0].upper(),
                (float(np.mean(x)), float(np.mean(y))),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=8,
                color=STAGE_COLORS[stage],
                fontweight="bold",
            )
        ax.set_yscale("log")
        ax.set_xlabel("Mean relative error")
        if column == 0:
            ax.set_ylabel("Interface flux jump")
        ax.set_title(f"{resolution} x {resolution}")
        ax.grid(True, alpha=0.22)
        ax.text(
            0.02,
            0.96,
            f"({chr(97 + column)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
    fig.tight_layout(w_pad=1.6)
    return _save_figure(fig, output_base, formats)


def plot_cost_breakdown(
    records: pd.DataFrame,
    *,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Render cumulative end-to-end time as additive measured stages."""
    resolutions = sorted(int(value) for value in records["resolution"].unique())
    fig, axes = plt.subplots(
        1,
        len(resolutions),
        figsize=(5.2 * len(resolutions), 4.0),
        squeeze=False,
        sharey=False,
    )
    stack_specs = (
        ("timing_pca_fit", "PCA fit", "#4C78A8"),
        ("timing_latent_transform", "Latent transform", "#72B7B2"),
        ("timing_nn_train", "NN train", "#F2CF5B"),
        ("timing_inference", "Inference", "#54A24B"),
    )
    for column, resolution in enumerate(resolutions):
        ax = axes[0, column]
        selected = records[records["resolution"] == resolution]
        grouped = selected.groupby("baseline_model").mean(numeric_only=True)
        x = np.arange(len(STAGE_ORDER))
        bottom = np.zeros(len(STAGE_ORDER))
        for metric, label, color in stack_specs:
            values = np.array([grouped.loc[stage, metric] for stage in STAGE_ORDER])
            ax.bar(x, values, bottom=bottom, color=color, label=label, width=0.72)
            bottom += values
        totals = np.array(
            [grouped.loc[stage, "timing_end_to_end"] for stage in STAGE_ORDER]
        )
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
            ax.text(index, total + 0.015 * totals.max(), f"{total:.0f}", ha="center", va="bottom")
        ax.set_xticks(x, [STAGE_LABELS[stage] for stage in STAGE_ORDER])
        ax.set_title(f"{resolution} x {resolution}")
        ax.set_ylabel("Cumulative end-to-end time (s)")
        ax.grid(True, axis="y", alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=1.6)
    return _save_figure(fig, output_base, formats)


def plot_weight_screen(
    screen: pd.DataFrame,
    *,
    selection_path: Path,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Plot validation target response against the seed-0 MRE guardrail."""
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    reference_mre = float(selection["reconstruction_validation_mre"])
    threshold = reference_mre * (1.0 + float(selection["mre_guardrail_fraction"]))
    specs = (
        (
            "interface_value",
            "a2_value",
            "validation_metric_interface_value_trace_error",
            "Value trace error",
        ),
        (
            "interface_flux",
            "a3_value_flux",
            "validation_metric_interface_flux_trace_error",
            "Flux trace error",
        ),
        (
            "spectral",
            "a4_value_flux_spectral",
            "validation_metric_relative_spectrum_error",
            "Relative spectral error",
        ),
        (
            "pde_residual",
            "a5_full_pde",
            "validation_metric_poisson_residual_rms",
            "PDE residual RMS",
        ),
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), squeeze=False)
    legend_handles: list[Line2D] = []
    for index, (term, prefix, target_metric, target_label) in enumerate(specs):
        ax = axes.flat[index]
        candidates = screen[
            screen["baseline_model"].astype(str).str.startswith(prefix)
        ].copy()
        ratio_column = f"ablation_target_ratio_{term}"
        candidates = candidates.dropna(subset=[ratio_column]).sort_values(ratio_column)
        x = candidates[ratio_column].to_numpy(dtype=float)
        y = candidates[target_metric].to_numpy(dtype=float)
        target_line = ax.plot(
            x,
            y,
            color="#4C78A8",
            marker="o",
            linewidth=STYLE["line_width"],
            label="Target metric",
        )[0]
        ax.set_xscale("log")
        ax.set_xlabel("Calibrated contribution ratio")
        ax.set_ylabel(target_label, color="#4C78A8")
        ax.tick_params(axis="y", colors="#4C78A8")
        ax.grid(True, alpha=0.2)
        mre_ax = ax.twinx()
        mre_line = mre_ax.plot(
            x,
            candidates["validation_metric_mre"].to_numpy(dtype=float),
            color="#E45756",
            marker="s",
            linestyle="--",
            linewidth=1.5,
            label="Validation MRE",
        )[0]
        guard_line = mre_ax.axhline(
            threshold,
            color="#777777",
            linestyle=":",
            linewidth=1.2,
            label="5% MRE guard",
        )
        mre_ax.set_ylabel("Validation MRE", color="#E45756")
        mre_ax.tick_params(axis="y", colors="#E45756")
        report = selection.get("terms", {}).get(term, {})
        selected_ratio = report.get("selected_target_ratio")
        if selected_ratio is not None:
            ax.axvline(float(selected_ratio), color="#222222", linewidth=0.9, alpha=0.5)
        status = "guard passed" if report.get("guardrail_passed", report.get("guardrail_had_eligible_candidate", False)) else "no eligible weight"
        ax.set_title(f"{target_label} ({status})")
        ax.text(
            0.02,
            0.96,
            f"({chr(97 + index)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
        if not legend_handles:
            legend_handles = [target_line, mre_line, guard_line]
    fig.legend(
        legend_handles,
        [str(handle.get_label()) for handle in legend_handles],
        loc="upper center",
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.5, w_pad=1.8)
    return _save_figure(fig, output_base, formats)


def plot_training_curves(
    records: pd.DataFrame,
    *,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Plot normalized validation components for seed 0 at 128."""
    selected = records[
        (records["resolution"] == 128)
        & (records["seed"] == 0)
        & (records["baseline_model"] != "a0_latent_baseline")
    ]
    curves: list[tuple[str, dict[str, np.ndarray]]] = []
    for stage in STAGE_ORDER[1:]:
        row = selected[selected["baseline_model"] == stage]
        if row.empty:
            continue
        path = _resolve_run_dir(Path(str(row.iloc[0]["run_dir"])), output_base)
        curve_path = path / "loss_curves.npz"
        if not curve_path.is_file():
            continue
        with np.load(curve_path) as archive:
            curves.append((stage, {name: archive[name].copy() for name in archive.files}))
    if not curves:
        return []

    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.0), squeeze=False)
    component_specs = (
        ("val_component_recon", "Reconstruction", "#4C78A8"),
        ("val_component_interface_value", "Value", "#72B7B2"),
        ("val_component_interface_flux", "Flux", "#54A24B"),
        ("val_component_spectral", "Spectral", "#F58518"),
        ("val_component_pde_residual", "PDE", "#E45756"),
    )
    for index, (stage, curve) in enumerate(curves):
        ax = axes.flat[index]
        active_terms = set(
            str(
                selected.loc[selected["baseline_model"] == stage, "ablation_active_terms"].iloc[0]
            ).split("+")
        )
        for key, label, color in component_specs:
            term = key.removeprefix("val_component_")
            if term not in active_terms or key not in curve:
                continue
            values = np.asarray(curve[key], dtype=float)
            scale = values[0] if abs(values[0]) > 1.0e-15 else 1.0
            ax.plot(
                np.arange(1, len(values) + 1),
                values / scale,
                color=color,
                linewidth=1.5,
                label=label,
            )
        ax.axvline(5, color="#777777", linestyle=":", linewidth=1.0)
        ax.axvline(15, color="#777777", linestyle="--", linewidth=1.0)
        ax.set_title(TABLE_LABELS[stage])
        ax.set_xlabel("Fine-tuning epoch")
        ax.set_ylabel("Validation component / epoch 1")
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False, ncol=2)
        ax.text(
            0.02,
            0.96,
            f"({chr(97 + index)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
    for ax in axes.flat[len(curves) :]:
        ax.axis("off")
    fig.tight_layout(h_pad=1.5, w_pad=1.3)
    return _save_figure(fig, output_base, formats)


def plot_qualitative(
    records: pd.DataFrame,
    *,
    resolution: int,
    output_base: Path,
    formats: Sequence[str],
) -> list[Path]:
    """Plot aligned fields, errors, spectra, and PDFs for representative variants."""
    stages = (
        "a0_latent_baseline",
        "a1_reconstruction",
        "a3_value_flux",
        "a4_value_flux_spectral",
    )
    selected = records[
        (records["resolution"] == resolution) & (records["seed"] == 0)
    ]
    rows = {stage: selected[selected["baseline_model"] == stage] for stage in stages}
    if any(row.empty for row in rows.values()):
        return []

    reference_path = _prediction_path(rows["a0_latent_baseline"].iloc[0], output_base)
    if not reference_path.is_file():
        return []
    with np.load(reference_path) as archive:
        truth_all = archive["y_true"]
        pred_all = archive["y_pred"]
        sample_indices = archive["sample_indices"]
        sample_mre = np.linalg.norm(
            (pred_all - truth_all).reshape(len(truth_all), -1),
            axis=1,
        ) / np.maximum(
            np.linalg.norm(truth_all.reshape(len(truth_all), -1), axis=1),
            1.0e-12,
        )
        chosen = int(np.argsort(sample_mre)[len(sample_mre) // 2])
        sample_id = int(sample_indices[chosen])
        truth = truth_all[chosen].copy()

    predictions: dict[str, np.ndarray] = {}
    for stage in stages:
        path = _prediction_path(rows[stage].iloc[0], output_base)
        with np.load(path) as archive:
            indices = archive["sample_indices"]
            matches = np.flatnonzero(indices == sample_id)
            if len(matches) != 1:
                raise ValueError(f"Sample {sample_id} is not uniquely aligned in {path}.")
            predictions[stage] = archive["y_pred"][int(matches[0])].copy()

    all_fields = np.stack([truth, *predictions.values()])
    field_lo, field_hi = np.percentile(all_fields, [0.5, 99.5])
    errors = {stage: np.abs(pred - truth) for stage, pred in predictions.items()}
    error_hi = float(np.percentile(np.stack(list(errors.values())), 99.5))
    patch_size = 32 if resolution == 128 else 64
    fig, axes = plt.subplots(
        len(stages),
        5,
        figsize=(12.6, 2.5 * len(stages)),
        squeeze=False,
    )
    column_titles = ("Ground truth", "Prediction", "Absolute error", "Energy spectrum", "Value PDF")
    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title)
    for row_index, stage in enumerate(stages):
        prediction = predictions[stage]
        error_norm = float(np.linalg.norm(prediction - truth))
        truth_norm = max(float(np.linalg.norm(truth)), 1.0e-12)
        mre = error_norm / truth_norm
        for column, field in enumerate((truth, prediction)):
            ax = axes[row_index, column]
            image = ax.imshow(
                field,
                origin="lower",
                cmap="RdBu_r",
                vmin=field_lo,
                vmax=field_hi,
            )
            _draw_patch_grid(ax, resolution, patch_size)
            ax.set_xticks([])
            ax.set_yticks([])
            if column == 1:
                ax.text(
                    0.03,
                    0.97,
                    f"MRE = {mre:.4f}",
                    transform=ax.transAxes,
                    va="top",
                    color="white",
                    fontsize=8,
                    bbox={"facecolor": "black", "alpha": 0.5, "pad": 2, "edgecolor": "none"},
                )
            if row_index == len(stages) - 1:
                fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        error_ax = axes[row_index, 2]
        error_image = error_ax.imshow(
            errors[stage],
            origin="lower",
            cmap="magma",
            vmin=0.0,
            vmax=error_hi,
        )
        _draw_patch_grid(error_ax, resolution, patch_size)
        error_ax.set_xticks([])
        error_ax.set_yticks([])
        if row_index == len(stages) - 1:
            fig.colorbar(error_image, ax=error_ax, fraction=0.046, pad=0.03)
        _plot_spectrum(axes[row_index, 3], truth, prediction)
        _plot_pdf(axes[row_index, 4], truth, prediction)
        axes[row_index, 0].set_ylabel(TABLE_LABELS[stage], fontweight="bold")
        axes[row_index, 3].grid(True, alpha=0.2)
        axes[row_index, 4].grid(True, alpha=0.2)
    axes[-1, 3].set_xlabel("Radial wavenumber k")
    axes[-1, 4].set_xlabel("Field value")
    fig.suptitle(
        f"Poisson {resolution} physics-loss ablation, seed 0, median A0 sample {sample_id}",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98), h_pad=1.1, w_pad=1.0)
    return _save_figure(fig, output_base, formats)


def _formatted_summary(
    summary: pd.DataFrame,
    specs: Sequence[tuple[str, str, str]],
) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "Resolution": summary["resolution"].astype(int),
            "Variant": summary["baseline_model"].map(TABLE_LABELS),
            "Seeds": summary["n_seeds"].astype(int),
        }
    )
    for base, label, style in specs:
        mean_column = f"{base}_mean"
        std_column = f"{base}_std"
        if mean_column not in summary:
            continue
        values = [
            _format_mean_std(mean, std, style=style)
            for mean, std in zip(summary[mean_column], summary[std_column])
        ]
        if label.startswith("Fine-tune"):
            values = [
                "" if stage == "a0_latent_baseline" else value
                for stage, value in zip(summary["baseline_model"], values)
            ]
        table[label] = values
    return table


def _relative_change_table(records: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "metric_mre",
        "metric_interface_jump",
        "metric_interface_flux_jump",
        "metric_relative_spectrum_error",
        "metric_poisson_residual_rms",
        "timing_end_to_end",
    )
    rows: list[dict[str, Any]] = []
    for resolution in sorted(int(value) for value in records["resolution"].unique()):
        selected = records[records["resolution"] == resolution]
        means = selected.groupby("baseline_model")[list(metrics)].mean()
        for stage in STAGE_ORDER:
            row: dict[str, Any] = {
                "Resolution": resolution,
                "Variant": TABLE_LABELS[stage],
            }
            for reference in ("a0_latent_baseline", "a1_reconstruction"):
                suffix = "A0" if reference.startswith("a0") else "A1"
                for metric in metrics:
                    short = metric.removeprefix("metric_").removeprefix("timing_")
                    row[f"{short}_vs_{suffix}_pct"] = (
                        100.0 * (means.loc[stage, metric] / means.loc[reference, metric] - 1.0)
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def _screen_table(
    screen: pd.DataFrame,
    selection: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    term_specs = (
        ("interface_value", "a2_value", "validation_metric_interface_value_trace_error"),
        ("interface_flux", "a3_value_flux", "validation_metric_interface_flux_trace_error"),
        ("spectral", "a4_value_flux_spectral", "validation_metric_relative_spectrum_error"),
        ("pde_residual", "a5_full_pde", "validation_metric_poisson_residual_rms"),
    )
    for term, prefix, target_metric in term_specs:
        report = selection.get("terms", {}).get(term, {})
        candidate_reports = {
            str(candidate.get("cell")): bool(candidate.get("guardrail_passed", False))
            for candidate in report.get("candidates", [])
        }
        selected_cell = report.get("selected_cell")
        if selected_cell is None and report.get("selected_level") is not None:
            selected_cell = f"{prefix}_{report['selected_level']}"
        ratio_column = f"ablation_target_ratio_{term}"
        weight_column = f"metric_calibrated_weight_{term}"
        candidates = screen[screen["baseline_model"].astype(str).str.startswith(prefix)]
        for _, candidate in candidates.iterrows():
            candidate_name = str(candidate["baseline_model"])
            guard_passed = candidate_reports.get(
                candidate_name,
                bool(report.get("guardrail_passed", False)),
            )
            rows.append(
                {
                    "Term": term.replace("_", " "),
                    "Candidate": candidate_name,
                    "Target ratio": _format_scalar(candidate.get(ratio_column), "ratio"),
                    "Calibrated weight": _format_scalar(candidate.get(weight_column), "scientific"),
                    "Validation target": _format_scalar(candidate.get(target_metric), "scientific"),
                    "Validation MRE": _format_scalar(candidate.get("validation_metric_mre"), "metric"),
                    "Guard passed": "yes" if guard_passed else "no",
                    "Selected": "yes" if candidate_name == selected_cell else "no",
                }
            )
    return pd.DataFrame(rows)


def _calibration_table(records: pd.DataFrame) -> pd.DataFrame:
    weight_columns = [
        f"metric_calibrated_weight_{term}"
        for term in (
            "interface_value",
            "interface_flux",
            "spectral",
            "pde_residual",
        )
        if f"metric_calibrated_weight_{term}" in records
    ]
    rows: list[dict[str, Any]] = []
    for (resolution, stage), selected in records.groupby(
        ["resolution", "baseline_model"]
    ):
        if stage == "a0_latent_baseline":
            continue
        row: dict[str, Any] = {
            "Resolution": int(resolution),
            "Variant": TABLE_LABELS[str(stage)],
        }
        for column in weight_columns:
            label = column.removeprefix("metric_calibrated_weight_").replace("_", " ")
            active = selected[column].dropna()
            active = active[active != 0.0]
            row[label] = (
                ""
                if active.empty
                else _format_mean_std(
                    active.mean(),
                    active.std(ddof=1),
                    style="scientific",
                )
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _normalize_to_reference(records: pd.DataFrame, *, reference: str) -> pd.DataFrame:
    normalized = records.copy()
    metric_columns = [
        "metric_mre",
        "metric_interface_jump",
        "metric_interface_flux_jump",
        "metric_relative_spectrum_error",
        "metric_poisson_residual_rms",
        "timing_end_to_end",
    ]
    references = records[records["baseline_model"] == reference].set_index(
        ["resolution", "seed"]
    )
    for index, row in normalized.iterrows():
        reference_row = references.loc[(row["resolution"], row["seed"])]
        for metric in metric_columns:
            normalized.at[index, metric] = row[metric] / reference_row[metric]
    return normalized


def _prediction_path(row: pd.Series, output_base: Path) -> Path:
    return _resolve_run_dir(Path(str(row["run_dir"])), output_base) / "predictions_test.npz"


def _resolve_run_dir(run_dir: Path, output_base: Path) -> Path:
    if run_dir.is_dir():
        return run_dir
    for parent in output_base.parents:
        candidate = parent / run_dir
        if candidate.is_dir():
            return candidate
    return run_dir


def _plot_spectrum(ax: plt.Axes, truth: np.ndarray, prediction: np.ndarray) -> None:
    truth_energy = energy_spectrum_2d(truth)
    prediction_energy = energy_spectrum_2d(prediction)
    k_max = min(len(truth_energy), len(prediction_energy), min(truth.shape) // 2)
    k = np.arange(1, k_max)
    ax.loglog(
        k,
        truth_energy[1:k_max] + 1.0e-30,
        color="#222222",
        linewidth=1.5,
        label="Truth",
    )
    ax.loglog(
        k,
        prediction_energy[1:k_max] + 1.0e-30,
        color="#E45756",
        linewidth=1.3,
        linestyle="--",
        label="Prediction",
    )
    if not ax.get_legend():
        ax.legend(frameon=False, fontsize=7)
    ax.set_ylabel("Energy")


def _plot_pdf(ax: plt.Axes, truth: np.ndarray, prediction: np.ndarray) -> None:
    lo, hi = np.percentile(np.concatenate([truth.ravel(), prediction.ravel()]), [0.25, 99.75])
    bins = np.linspace(lo, hi, 70)
    ax.hist(
        truth.ravel(),
        bins=bins,
        density=True,
        histtype="step",
        color="#222222",
        linewidth=1.5,
        label="Truth",
    )
    ax.hist(
        prediction.ravel(),
        bins=bins,
        density=True,
        histtype="step",
        color="#E45756",
        linewidth=1.3,
        linestyle="--",
        label="Prediction",
    )
    if not ax.get_legend():
        ax.legend(frameon=False, fontsize=7)
    ax.set_ylabel("Density")


def _draw_patch_grid(ax: plt.Axes, resolution: int, patch_size: int) -> None:
    for boundary in range(patch_size, resolution, patch_size):
        location = boundary - 0.5
        ax.axvline(location, color="white", linewidth=0.35, alpha=0.6)
        ax.axhline(location, color="white", linewidth=0.35, alpha=0.6)


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
    return f"{mean_value:.5f} +/- {std_value:.5f}"


def _format_scalar(value: Any, style: str) -> str:
    if pd.isna(value):
        return ""
    number = float(value)
    if style == "scientific":
        return f"{number:.4e}"
    if style == "ratio":
        return f"{number:g}"
    return f"{number:.5f}"


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
        default="paper_results/physics_loss_ablation",
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
        help="Skip prediction-archive figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    formats = tuple(piece.strip() for piece in args.formats.split(",") if piece.strip())
    written = make_physics_ablation_outputs(
        results_dir=args.results_dir,
        formats=formats,
        expected_seeds=args.expected_seeds,
        qualitative=not args.skip_qualitative,
    )
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
