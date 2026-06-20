#!/usr/bin/env python
"""Render paper-ready figures from LPCANet result records."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lpcanet.data.io import load_npz
from lpcanet.metrics.spectral import energy_spectrum_2d
from lpcanet.utils.paths import ensure_dir

MECHANISM_KEYS = ("two_scale", "coupling", "in_loop_loss")
METHOD_LABELS = {
    "l2l": "Plain L2L",
    "l2l_overlap": "L2L + Overlap",
    "l2l_refinement": "L2L + RefinementNet",
    "full": "Full model",
}
FACTORIAL_ORDER = [
    "Plain L2L",
    "two_scale",
    "coupling",
    "in_loop_loss",
    "two_scale+coupling",
    "two_scale+in_loop_loss",
    "coupling+in_loop_loss",
    "Full model",
]
DISPLAY_LABELS = {
    "Plain L2L": "Plain L2L",
    "two_scale": "Two-scale only",
    "coupling": "Coupling only",
    "in_loop_loss": "In-loop loss only",
    "two_scale+coupling": "Two-scale + coupling",
    "two_scale+in_loop_loss": "Two-scale + in-loop",
    "coupling+in_loop_loss": "Coupling + in-loop",
    "Full model": "Full model",
}
STYLE = {
    "dpi": 300,
    "font_size": 10,
    "title_size": 11,
    "label_size": 10,
    "tick_size": 9,
    "legend_size": 9,
    "line_width": 2.0,
    "marker_size": 5,
    "errorbar_capsize": 3,
}


def make_figures(
    *,
    results_dir: str | Path = "results",
    out_dir: str | Path = "results/figures",
    sample_index: int = 0,
    formats: tuple[str, ...] = ("png", "pdf"),
) -> list[Path]:
    """Create qualitative and factorial diagnostic figures."""
    results_path = Path(results_dir)
    output_dir = ensure_dir(out_dir)
    _apply_style()

    records = load_result_records(results_path)
    written: list[Path] = []
    if not records.empty:
        comparison = plot_factorial_interface_spectral(records, output_dir, formats=formats)
        written.extend(comparison)
        timings = plot_factorial_timing_breakdown(records, output_dir, formats=formats)
        written.extend(timings)

    prediction_runs = discover_prediction_runs(results_path, records)
    for dataset in ["poisson", "darcy"]:
        for variant in ["Full model", "two_scale", "coupling", "in_loop_loss"]:
            paths = plot_qualitative_panel(
                prediction_runs,
                dataset=dataset,
                variant=variant,
                output_dir=output_dir,
                sample_index=sample_index,
                formats=formats,
            )
            written.extend(paths)

    manifest_path = output_dir / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps({"figures": [str(path) for path in written]}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    written.append(manifest_path)
    return written


def load_result_records(results_dir: str | Path) -> pd.DataFrame:
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
    frame["method_order"] = frame["method"].map({name: idx for idx, name in enumerate(FACTORIAL_ORDER)})
    frame["method_order"] = frame["method_order"].fillna(10_000).astype(int)
    return frame


def discover_prediction_runs(results_dir: Path, records: pd.DataFrame) -> list[dict[str, Any]]:
    """Find run directories with predictions and enough metadata for plotting."""
    candidates: dict[Path, dict[str, Any]] = {}
    if not records.empty and "run_dir" in records.columns:
        for row in records.to_dict(orient="records"):
            run_dir = Path(str(row["run_dir"]))
            candidates[run_dir] = dict(row)
    for prediction_path in sorted(results_dir.rglob("predictions_test.npz")):
        candidates.setdefault(prediction_path.parent, {"run_dir": str(prediction_path.parent)})

    runs: list[dict[str, Any]] = []
    for run_dir, row in candidates.items():
        prediction_path = run_dir / "predictions_test.npz"
        if not prediction_path.exists():
            continue
        config = _load_run_config(run_dir)
        method = str(row.get("method") or _method_label_from_config(config, run_dir))
        dataset = str(row.get("dataset") or _dataset_name(config, run_dir)).lower()
        runs.append(
            {
                **row,
                "run_dir": str(run_dir),
                "prediction_path": str(prediction_path),
                "config": config,
                "dataset": dataset,
                "method": method,
                "seed": int(row.get("seed", config.get("experiment", {}).get("seed", 0)) or 0),
                "resolution": int(row.get("resolution", config.get("dataset", {}).get("grid_size", 0)) or 0),
            }
        )
    return runs


def plot_qualitative_panel(
    runs: list[dict[str, Any]],
    *,
    dataset: str,
    variant: str = "Full model",
    output_dir: Path,
    sample_index: int,
    formats: tuple[str, ...],
) -> list[Path]:
    selected = {
        "Plain L2L": _select_prediction_run(runs, dataset=dataset, method="Plain L2L"),
        variant: _select_prediction_run(runs, dataset=dataset, method=variant),
    }
    if selected["Plain L2L"] is None or selected[variant] is None:
        return []

    loaded = {label: _load_prediction_triplet(run, sample_index) for label, run in selected.items() if run}
    truth = loaded["Plain L2L"]["truth"]
    input_field = loaded["Plain L2L"]["input"]
    columns = ["Input", "Ground truth", "Prediction", "Absolute error", "Energy spectrum", "Value PDF"]
    fig, axes = plt.subplots(
        2,
        len(columns),
        figsize=(2.25 * len(columns), 4.7),
        constrained_layout=True,
        squeeze=False,
    )
    for col_idx, title in enumerate(columns):
        axes[0, col_idx].set_title(title)

    vmin = float(np.min(truth))
    vmax = float(np.max(truth))
    err_max = max(
        float(np.max(np.abs(loaded["Plain L2L"]["prediction"] - truth))),
        float(np.max(np.abs(loaded[variant]["prediction"] - loaded[variant]["truth"]))),
        1e-12,
    )
    for row_idx, label in enumerate(["Plain L2L", variant]):
        data = loaded[label]
        pred = data["prediction"]
        true = data["truth"]
        inp = input_field if row_idx == 0 else data["input"]
        panels = [
            (inp, "RdBu_r", None, None),
            (true, "RdBu_r", vmin, vmax),
            (pred, "RdBu_r", vmin, vmax),
            (np.abs(pred - true), "magma", 0.0, err_max),
        ]
        for col_idx, (field, cmap, lo, hi) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(field, origin="lower", cmap=cmap, vmin=lo, vmax=hi)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            if col_idx == 0:
                ax.set_ylabel(DISPLAY_LABELS.get(label, label))

        spectrum_ax = axes[row_idx, 4]
        _plot_spectrum(spectrum_ax, true, pred)
        if row_idx == 1:
            spectrum_ax.set_xlabel("Wavenumber k")
        spectrum_ax.set_ylabel("Energy")

        pdf_ax = axes[row_idx, 5]
        _plot_pdf(pdf_ax, true, pred)
        if row_idx == 1:
            pdf_ax.set_xlabel("Field value")
        pdf_ax.set_ylabel("Density")

    variant_slug = "full" if variant == "Full model" else variant
    paths = _save(
        fig,
        output_dir / f"qualitative_{dataset}_plain_l2l_vs_{variant_slug}",
        formats=formats,
    )
    return paths


def plot_factorial_interface_spectral(
    records: pd.DataFrame,
    output_dir: Path,
    *,
    formats: tuple[str, ...],
) -> list[Path]:
    metric_cols = [
        col
        for col in [
            "metric_interface_jump",
            "metric_interface_flux_jump",
            "metric_relative_spectrum_error",
        ]
        if col in records.columns
    ]
    if not metric_cols:
        return []
    frame = records[records["method"].isin(FACTORIAL_ORDER)].copy()
    if frame.empty:
        return []
    grouped = (
        frame.groupby(["method", "method_order"], as_index=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = [
        "_".join(piece for piece in col if piece)
        if isinstance(col, tuple)
        else str(col)
        for col in grouped.columns
    ]
    grouped = grouped.sort_values(["method_order", "method"])
    x = np.arange(len(grouped))
    labels = [str(value).replace("+", "\n+") for value in grouped["method"]]

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.5), constrained_layout=True)
    specs = [
        ("metric_interface_jump", "Interface value jump", "tab:blue", "bar"),
        ("metric_interface_flux_jump", "Interface flux jump", "tab:orange", "bar"),
        ("metric_relative_spectrum_error", "Spectral error", "tab:green", "line"),
    ]
    for ax, (base, title, color, kind) in zip(axes, specs):
        mean_col = f"{base}_mean"
        std_col = f"{base}_std"
        if mean_col not in grouped:
            ax.axis("off")
            continue
        y = grouped[mean_col].to_numpy(dtype=float)
        yerr = grouped[std_col].fillna(0.0).to_numpy(dtype=float) if std_col in grouped else None
        if kind == "bar":
            ax.bar(x, y, yerr=yerr, color=color, alpha=0.82, capsize=STYLE["errorbar_capsize"])
        else:
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                color=color,
                linewidth=STYLE["line_width"],
                capsize=STYLE["errorbar_capsize"],
            )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
    paths = _save(fig, output_dir / "factorial_interface_spectral_comparison", formats=formats)
    return paths


def plot_factorial_timing_breakdown(
    records: pd.DataFrame,
    output_dir: Path,
    *,
    formats: tuple[str, ...],
) -> list[Path]:
    """Plot mean stage timings with each stack summing to measured end-to-end time."""
    stage_columns = [
        "timing_pca_fit",
        "timing_latent_transform",
        "timing_nn_train",
        "timing_inference",
        "timing_end_to_end",
    ]
    if not all(column in records.columns for column in stage_columns):
        return []
    frame = records[records["method"].isin(FACTORIAL_ORDER)].copy()
    if frame.empty:
        return []
    grouped = (
        frame.groupby(["method", "method_order"], as_index=False)[stage_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = [
        "_".join(piece for piece in col if piece)
        if isinstance(col, tuple)
        else str(col)
        for col in grouped.columns
    ]
    grouped = grouped.sort_values(["method_order", "method"])
    component_specs = [
        ("timing_pca_fit_mean", "PCA fit", "#4C78A8"),
        ("timing_latent_transform_mean", "Latent transform", "#72B7B2"),
        ("timing_nn_train_mean", "NN training", "#F58518"),
        ("timing_inference_mean", "Inference", "#E45756"),
    ]
    components = np.column_stack(
        [grouped[column].fillna(0.0).to_numpy(dtype=float) for column, _, _ in component_specs]
    )
    total = grouped["timing_end_to_end_mean"].fillna(0.0).to_numpy(dtype=float)
    overhead = np.maximum(total - np.sum(components, axis=1), 0.0)
    x = np.arange(len(grouped))
    labels = [DISPLAY_LABELS.get(str(value), str(value)) for value in grouped["method"]]

    fig, ax = plt.subplots(figsize=(9.4, 4.8), constrained_layout=True)
    bottom = np.zeros(len(grouped), dtype=float)
    for column, label, color in component_specs:
        values = grouped[column].fillna(0.0).to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, label=label, color=color, width=0.72)
        bottom += values
    ax.bar(x, overhead, bottom=bottom, label="Other pipeline", color="#B9B9B9", width=0.72)
    total_std = grouped["timing_end_to_end_std"].fillna(0.0).to_numpy(dtype=float)
    ax.errorbar(
        x,
        total,
        yerr=total_std,
        fmt="none",
        ecolor="black",
        elinewidth=1.0,
        capsize=STYLE["errorbar_capsize"],
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("End-to-end wall time (s)")
    ax.set_title("Factorial pipeline cost by stage")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    return _save(fig, output_dir / "factorial_stage_timing_breakdown", formats=formats)


def _flatten_record(record: Mapping[str, Any], *, source_path: Path) -> dict[str, Any]:
    run_dir = Path(str(record.get("run_dir", "")))
    config = _load_run_config(run_dir)
    row: dict[str, Any] = {
        "source_path": str(source_path),
        "run_id": record.get("run_id", ""),
        "run_dir": str(run_dir),
        "seed": record.get("seed", 0),
        "resolution": record.get("resolution", 0),
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


def _load_prediction_triplet(run: Mapping[str, Any], sample_index: int) -> dict[str, np.ndarray]:
    arrays = load_npz(run["prediction_path"])
    pred = _first_array(arrays, ["y_pred", "pred", "prediction", "u_pred"])
    truth = _first_array(arrays, ["y_true", "truth", "target", "u"])
    input_field = _first_array(arrays, ["x_input", "input", "f", "a", "coeff", "coefficient"])
    if pred is None or truth is None:
        raise ValueError(f"{run['prediction_path']} must contain prediction and truth arrays.")
    pred = _sample_field(pred, sample_index)
    truth = _sample_field(truth, sample_index)
    if input_field is None:
        input_field = np.zeros_like(truth)
    else:
        input_field = _sample_field(input_field, sample_index)
    return {"input": input_field, "truth": truth, "prediction": pred}


def _first_array(arrays: Mapping[str, np.ndarray], keys: list[str]) -> np.ndarray | None:
    for key in keys:
        if key in arrays:
            return arrays[key]
    return None


def _sample_field(array: np.ndarray, sample_index: int) -> np.ndarray:
    field = np.asarray(array)
    if field.ndim == 4 and field.shape[1] == 1:
        field = field[:, 0]
    if field.ndim == 3:
        index = min(max(sample_index, 0), field.shape[0] - 1)
        field = field[index]
    elif field.ndim == 2:
        pass
    else:
        raise ValueError(f"Expected scalar fields with shape (N,H,W) or (H,W), got {field.shape}.")
    return np.asarray(field, dtype=np.float64)


def _plot_spectrum(ax: plt.Axes, truth: np.ndarray, pred: np.ndarray) -> None:
    truth_energy = energy_spectrum_2d(truth)
    pred_energy = energy_spectrum_2d(pred)
    k = np.arange(1, len(truth_energy))
    ax.plot(k, truth_energy[1:] + 1e-30, color="black", linewidth=STYLE["line_width"], label="Truth")
    ax.plot(k, pred_energy[1:] + 1e-30, color="tab:red", linewidth=STYLE["line_width"], label="Prediction")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=STYLE["legend_size"])


def _plot_pdf(ax: plt.Axes, truth: np.ndarray, pred: np.ndarray) -> None:
    lo = float(min(np.min(truth), np.min(pred)))
    hi = float(max(np.max(truth), np.max(pred)))
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    bins = np.linspace(lo, hi, 40).tolist()
    ax.hist(truth.ravel(), bins=bins, density=True, histtype="step", color="black", linewidth=STYLE["line_width"], label="Truth")
    ax.hist(pred.ravel(), bins=bins, density=True, histtype="step", color="tab:red", linewidth=STYLE["line_width"], label="Prediction")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid(True, alpha=0.25)


def _select_prediction_run(
    runs: list[dict[str, Any]],
    *,
    dataset: str,
    method: str,
) -> dict[str, Any] | None:
    candidates = [
        run
        for run in runs
        if str(run.get("dataset", "")).lower() == dataset.lower() and run.get("method") == method
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda run: (int(run.get("seed", 0)), -int(run.get("resolution", 0))))[0]


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


def _method_label_from_config(config: Mapping[str, Any], run_dir: Path) -> str:
    mechanisms = config.get("mechanisms", {})
    if isinstance(mechanisms, Mapping):
        flags = {key: bool(mechanisms.get(key, False)) for key in MECHANISM_KEYS}
        if all(flags.values()):
            return "Full model"
        if not any(flags.values()):
            lowered = str(run_dir).lower()
            if "overlap" in lowered:
                return "L2L + Overlap"
            return "Plain L2L"
    lowered = str(run_dir).lower()
    for key, label in METHOD_LABELS.items():
        if key in lowered:
            return label
    return ""


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
        }
    )


def _save(fig: plt.Figure, base_path: Path, *, formats: tuple[str, ...]) -> list[Path]:
    paths: list[Path] = []
    for suffix in formats:
        path = base_path.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="results/figures")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--formats", default="png,pdf", help="Comma-separated output formats.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    formats = tuple(piece.strip() for piece in args.formats.split(",") if piece.strip())
    written = make_figures(
        results_dir=args.results_dir,
        out_dir=args.out_dir,
        sample_index=args.sample_index,
        formats=formats,
    )
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
