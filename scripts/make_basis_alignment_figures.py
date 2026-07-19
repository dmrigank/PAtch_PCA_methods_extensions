#!/usr/bin/env python
"""Generate figures for the post-hoc basis alignment experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lpcanet.data.io import load_npz
from lpcanet.metrics.spectral import energy_spectrum_2d
from lpcanet.utils.paths import ensure_dir

SAMPLE_GLOBAL_IDX = 777
BASELINE_DIR = ROOT / "results" / "guard_band_sweep" / "g0"
ALIGN_DIR    = ROOT / "results" / "basis_alignment"
FIGURES_DIR  = ROOT / "results" / "figures"

STYLE = {
    "dpi": 300,
    "font_size": 10,
    "title_size": 11,
    "label_size": 10,
    "tick_size": 9,
    "legend_size": 9,
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": STYLE["font_size"],
            "axes.titlesize": STYLE["title_size"],
            "axes.labelsize": STYLE["label_size"],
            "xtick.labelsize": STYLE["tick_size"],
            "ytick.labelsize": STYLE["tick_size"],
            "legend.fontsize": STYLE["legend_size"],
        }
    )


def load_metrics(run_dir: Path) -> dict:
    with open(run_dir / "metrics.json") as f:
        return json.load(f)


def load_runtime(run_dir: Path) -> dict:
    with open(run_dir / "runtime.json") as f:
        return json.load(f)


def load_sample(
    run_dir: Path, global_sample_idx: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    preds = load_npz(run_dir / "predictions_test.npz")
    sample_indices = preds["sample_indices"]
    pos = int(np.where(sample_indices == global_sample_idx)[0][0])
    x_input = preds["a"][pos].astype(np.float64)
    y_true   = preds["y_true"][pos].astype(np.float64)
    y_pred   = preds["y_pred"][pos].astype(np.float64)
    return x_input, y_true, y_pred


def _plot_field_row(
    axes: np.ndarray,
    x_input: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    row_label: str,
    vmin: float,
    vmax: float,
    err_vmax: float,
    fig: plt.Figure,
    show_col_titles: bool = False,
) -> None:
    col_titles = [
        "Input (a)", "Ground truth (u)", "Prediction",
        "Absolute error", "Energy spectrum E(k)", "Value PDF",
    ]
    panels_field = [
        (x_input,                       "RdBu_r", None,  None),
        (y_true,                         "RdBu_r", vmin,  vmax),
        (y_pred,                         "RdBu_r", vmin,  vmax),
        (np.abs(y_pred - y_true),        "magma",  0.0,   err_vmax),
    ]
    for col, (field, cmap, lo, hi) in enumerate(panels_field):
        ax = axes[col]
        im = ax.imshow(field, origin="lower", cmap=cmap, vmin=lo, vmax=hi)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        if show_col_titles:
            ax.set_title(col_titles[col], fontsize=9)
    axes[0].set_ylabel(row_label, fontsize=9)

    ax_spec = axes[4]
    k_truth = energy_spectrum_2d(y_true)
    k_pred  = energy_spectrum_2d(y_pred)
    k = np.arange(1, len(k_truth))
    ax_spec.plot(k, k_truth[1:] + 1e-30, color="black",   lw=1.4, label="Truth")
    ax_spec.plot(k, k_pred[1:]  + 1e-30, color="tab:red", lw=1.4, label="Pred")
    ax_spec.set_xscale("log")
    ax_spec.set_yscale("log")
    ax_spec.grid(True, alpha=0.25)
    ax_spec.legend(frameon=False, fontsize=7)
    if show_col_titles:
        ax_spec.set_title(col_titles[4], fontsize=9)

    ax_pdf = axes[5]
    lo_v = float(min(np.min(y_true), np.min(y_pred)))
    hi_v = float(max(np.max(y_true), np.max(y_pred)))
    bins = np.linspace(lo_v, hi_v, 40)
    ax_pdf.hist(y_true.ravel(), bins=bins, density=True, histtype="step", color="black",   lw=1.4, label="Truth")
    ax_pdf.hist(y_pred.ravel(), bins=bins, density=True, histtype="step", color="tab:red", lw=1.4, label="Pred")
    ax_pdf.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_pdf.grid(True, alpha=0.25)
    ax_pdf.legend(frameon=False, fontsize=7)
    if show_col_titles:
        ax_pdf.set_title(col_titles[5], fontsize=9)


def make_single_figure(
    run_dir: Path,
    label: str,
    mre: float,
    err_vmax: float,
    vmin: float,
    vmax: float,
    out_path: Path,
) -> None:
    x_in, y_true, y_pred = load_sample(run_dir, SAMPLE_GLOBAL_IDX)
    fig, axes = plt.subplots(1, 6, figsize=(14, 2.8), constrained_layout=True)
    title = f"{label} | Darcy 128 | patch_size=32 | seed=0 | MRE={mre:.4f}"
    fig.suptitle(title, fontsize=9)
    _plot_field_row(axes, x_in, y_true, y_pred, label, vmin, vmax, err_vmax, fig, show_col_titles=True)
    fig.savefig(out_path, dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def make_comparison_figure(
    err_vmax: float,
    vmin: float,
    vmax: float,
    baseline_mre: float,
    align_mre: float,
    out_path: Path,
) -> None:
    x_base, y_true, y_pred_base = load_sample(BASELINE_DIR, SAMPLE_GLOBAL_IDX)
    _,      _,      y_pred_align = load_sample(ALIGN_DIR,    SAMPLE_GLOBAL_IDX)

    fig, axes = plt.subplots(2, 6, figsize=(14, 5.6), constrained_layout=True, squeeze=False)
    fig.suptitle(
        f"Basis alignment comparison | Darcy 128 | patch_size=32 | seed=0 | sample {SAMPLE_GLOBAL_IDX}",
        fontsize=STYLE["title_size"],
    )
    col_titles = ["Input (a)", "Ground truth (u)", "Prediction", "Absolute error", "Energy spectrum", "Value PDF"]
    for col, ct in enumerate(col_titles):
        axes[0, col].set_title(ct, fontsize=9)

    _plot_field_row(
        axes[0], x_base, y_true, y_pred_base,
        f"Plain L2L\nMRE={baseline_mre:.4f}",
        vmin, vmax, err_vmax, fig, show_col_titles=False,
    )
    _plot_field_row(
        axes[1], x_base, y_true, y_pred_align,
        f"L2L + basis alignment\nMRE={align_mre:.4f}",
        vmin, vmax, err_vmax, fig, show_col_titles=False,
    )
    fig.savefig(out_path, dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def make_metrics_table(
    baseline_metrics: dict,
    baseline_runtime: dict,
    align_metrics: dict,
    align_runtime: dict,
    out_path: Path,
) -> None:
    bt = baseline_runtime["times"]
    at = align_runtime["times"]

    rows = [
        ("Plain L2L",            baseline_metrics, bt, None),
        ("L2L + basis alignment", align_metrics,    at, at.get("pca_basis_alignment", 0.0)),
    ]
    col_labels = [
        "Method",
        "MRE", "SSIM", "MAE",
        "interface\njump", "interface\nflux jump", "rel spectrum\nerror",
        "PCA fit (s)", "align (s)",
    ]

    table_data: list[list[str]] = []
    for name, m, t, align_t in rows:
        row = [
            name,
            f"{m['mre']:.5f}",
            f"{m.get('ssim_mean', float('nan')):.4f}",
            f"{m['mae']:.2e}",
            f"{m['interface_jump']:.2e}",
            f"{m['interface_flux_jump']:.5f}",
            f"{m['relative_spectrum_error']:.2e}",
            f"{t['pca_fit']:.2f}",
            f"{align_t:.3f}" if align_t is not None else "—",
        ]
        table_data.append(row)

    fig, ax = plt.subplots(figsize=(13, 1.5), constrained_layout=True)
    ax.axis("off")
    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1565C0")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#E3F2FD")
        cell.set_edgecolor("#CFD8DC")

    fig.suptitle(
        "Basis alignment experiment — Darcy 128 | patch_size=32 | seed=0",
        fontsize=10, fontweight="bold",
    )
    fig.savefig(out_path, dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def print_summary(
    baseline_metrics: dict,
    baseline_runtime: dict,
    align_metrics: dict,
    align_runtime: dict,
) -> None:
    bt = baseline_runtime["times"]
    at = align_runtime["times"]
    print()
    print("=" * 100)
    print("BASIS ALIGNMENT EXPERIMENT — SUMMARY")
    print("=" * 100)
    hdr = (
        f"{'Method':<28}  {'MRE':>8}  {'SSIM':>7}  {'MAE':>10}  "
        f"{'ij':>10}  {'ifj':>10}  {'spec_err':>10}  "
        f"{'pca_fit(s)':>10}  {'align(s)':>9}"
    )
    print(hdr)
    print("-" * 100)
    for name, m, t, align_t in [
        ("Plain L2L",             baseline_metrics, bt, None),
        ("L2L + basis alignment", align_metrics,    at, at.get("pca_basis_alignment", 0.0)),
    ]:
        at_str = f"{align_t:9.3f}" if align_t is not None else "        —"
        print(
            f"{name:<28}  {m['mre']:>8.5f}  {m.get('ssim_mean', float('nan')):>7.4f}  "
            f"{m['mae']:>10.2e}  "
            f"{m['interface_jump']:>10.2e}  {m['interface_flux_jump']:>10.5f}  "
            f"{m['relative_spectrum_error']:>10.2e}  "
            f"{t['pca_fit']:>10.2f}  {at_str}"
        )
    print("=" * 100)


def main() -> None:
    apply_style()
    out_dir = ensure_dir(FIGURES_DIR)

    baseline_metrics = load_metrics(BASELINE_DIR)
    baseline_runtime = load_runtime(BASELINE_DIR)
    align_metrics    = load_metrics(ALIGN_DIR)
    align_runtime    = load_runtime(ALIGN_DIR)

    _, y_true_ref, _ = load_sample(BASELINE_DIR, SAMPLE_GLOBAL_IDX)
    vmin = float(np.min(y_true_ref))
    vmax = float(np.max(y_true_ref))

    _, _, y_pred_base  = load_sample(BASELINE_DIR, SAMPLE_GLOBAL_IDX)
    _, _, y_pred_align = load_sample(ALIGN_DIR,    SAMPLE_GLOBAL_IDX)
    err_base  = np.abs(y_pred_base  - y_true_ref)
    err_align = np.abs(y_pred_align - y_true_ref)
    err_vmax  = max(float(np.max(err_base)), float(np.max(err_align)), 1e-12)
    print(f"Shared absolute error vmax = {err_vmax:.4f}")

    make_single_figure(
        ALIGN_DIR,
        "L2L + basis alignment",
        align_metrics["mre"],
        err_vmax, vmin, vmax,
        out_dir / "basis_alignment_sample777.png",
    )
    make_comparison_figure(
        err_vmax, vmin, vmax,
        baseline_metrics["mre"],
        align_metrics["mre"],
        out_dir / "basis_alignment_vs_baseline_sample777.png",
    )
    make_metrics_table(
        baseline_metrics, baseline_runtime,
        align_metrics,    align_runtime,
        out_dir / "basis_alignment_metrics_table.png",
    )
    print_summary(baseline_metrics, baseline_runtime, align_metrics, align_runtime)


if __name__ == "__main__":
    main()
