#!/usr/bin/env python
"""Generate figures for the context-fit PCA (guard band) sweep."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lpcanet.data.io import load_npz
from lpcanet.metrics.spectral import energy_spectrum_2d
from lpcanet.utils.paths import ensure_dir

G_VALUES = [0, 2, 4, 6, 16]
SAMPLE_GLOBAL_IDX = 777
RESULTS_ROOT = ROOT / "results" / "guard_band_sweep"
FIGURES_DIR = ROOT / "results" / "figures"

STYLE = {
    "dpi": 300,
    "font_size": 10,
    "title_size": 11,
    "label_size": 10,
    "tick_size": 9,
    "legend_size": 9,
    "line_width": 1.8,
}

COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]
MARKERS = ["o", "s", "^", "D", "v"]


def load_metrics(g: int) -> dict:
    path = RESULTS_ROOT / f"g{g}" / "metrics.json"
    with open(path) as f:
        return json.load(f)


def load_runtime(g: int) -> dict:
    path = RESULTS_ROOT / f"g{g}" / "runtime.json"
    with open(path) as f:
        return json.load(f)


def load_sample(g: int, global_sample_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    preds = load_npz(RESULTS_ROOT / f"g{g}" / "predictions_test.npz")
    sample_indices = preds["sample_indices"]
    pos = int(np.where(sample_indices == global_sample_idx)[0][0])
    y_true = preds["y_true"][pos].astype(np.float64)
    y_pred = preds["y_pred"][pos].astype(np.float64)
    x_input = preds["a"][pos].astype(np.float64)
    return x_input, y_true, y_pred


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


def make_time_vs_accuracy_figure(
    all_metrics: dict[int, dict],
    all_runtimes: dict[int, dict],
    out_dir: Path,
) -> Path:
    pca_fit_times = [all_runtimes[g]["times"]["pca_fit"] for g in G_VALUES]
    mres = [all_metrics[g]["mre"] for g in G_VALUES]
    ijumps = [all_metrics[g]["interface_jump"] for g in G_VALUES]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    fig.suptitle(
        "Context-fit PCA (guard band) — time cost vs. accuracy\n"
        "Darcy 128 | patch_size=32 | seed=0",
        fontsize=STYLE["title_size"],
    )

    baseline_mre = mres[0]  # g=0
    baseline_ijump = ijumps[0]  # g=0

    for panel_idx, (ax, y_vals, y_label, baseline) in enumerate(
        [
            (axes[0], mres, "MRE (↓ better)", baseline_mre),
            (axes[1], ijumps, "Interface jump (↓ better)", baseline_ijump),
        ]
    ):
        ax.axhline(baseline, color="gray", linestyle="--", linewidth=1.2, alpha=0.7, label="g=0 baseline")
        for i, g in enumerate(G_VALUES):
            ax.scatter(
                pca_fit_times[i],
                y_vals[i],
                color=COLORS[i],
                marker=MARKERS[i],
                s=80,
                zorder=5,
                label=f"g={g}",
            )
            ax.annotate(
                f"g={g}",
                (pca_fit_times[i], y_vals[i]),
                textcoords="offset points",
                xytext=(6, 3),
                fontsize=8,
            )
        ax.set_xlabel("PCA fit time (s)")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)

    # Panel 3: bar chart of PCA fit time by g
    ax3 = axes[2]
    pca_fit_output = [all_runtimes[g]["times"]["pca_fit_output_pca"] for g in G_VALUES]
    pca_fit_input = [all_runtimes[g]["times"]["pca_fit_input_pca"] for g in G_VALUES]
    x = np.arange(len(G_VALUES))
    bars1 = ax3.bar(x, pca_fit_input, color="#90CAF9", label="Input PCA fit")
    bars2 = ax3.bar(x, pca_fit_output, bottom=pca_fit_input, color="#1565C0", label="Output PCA fit")
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"g={g}" for g in G_VALUES])
    ax3.set_xlabel("Guard band g")
    ax3.set_ylabel("PCA fit time (s)")
    ax3.set_title("PCA fit time breakdown")
    ax3.legend(frameon=False)
    ax3.grid(True, axis="y", alpha=0.25)
    for xi, (inp, out) in zip(x, zip(pca_fit_input, pca_fit_output)):
        ax3.text(xi, inp + out + 2, f"{inp+out:.1f}s", ha="center", fontsize=8)

    out_path = out_dir / "guard_band_time_vs_accuracy.png"
    fig.savefig(out_path, dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


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
    col_titles = ["Input (a)", "Ground truth (u)", "Prediction", "Absolute error", "Energy spectrum E(k)", "Value PDF"]
    panels_field = [
        (x_input, "RdBu_r", None, None),
        (y_true, "RdBu_r", vmin, vmax),
        (y_pred, "RdBu_r", vmin, vmax),
        (np.abs(y_pred - y_true), "magma", 0.0, err_vmax),
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

    # Spectrum
    ax_spec = axes[4]
    k_truth = energy_spectrum_2d(y_true)
    k_pred = energy_spectrum_2d(y_pred)
    k = np.arange(1, len(k_truth))
    ax_spec.plot(k, k_truth[1:] + 1e-30, color="black", lw=1.4, label="Truth")
    ax_spec.plot(k, k_pred[1:] + 1e-30, color="tab:red", lw=1.4, label="Pred")
    ax_spec.set_xscale("log")
    ax_spec.set_yscale("log")
    ax_spec.grid(True, alpha=0.25)
    ax_spec.legend(frameon=False, fontsize=7)
    if show_col_titles:
        ax_spec.set_title(col_titles[4], fontsize=9)

    # PDF
    ax_pdf = axes[5]
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    bins = np.linspace(lo, hi, 40)
    ax_pdf.hist(y_true.ravel(), bins=bins, density=True, histtype="step", color="black", lw=1.4, label="Truth")
    ax_pdf.hist(y_pred.ravel(), bins=bins, density=True, histtype="step", color="tab:red", lw=1.4, label="Pred")
    ax_pdf.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_pdf.grid(True, alpha=0.25)
    ax_pdf.legend(frameon=False, fontsize=7)
    if show_col_titles:
        ax_pdf.set_title(col_titles[5], fontsize=9)


def make_individual_reconstruction_figures(
    all_metrics: dict[int, dict],
    err_vmax: float,
    out_dir: Path,
) -> list[Path]:
    paths = []
    _, y_true_ref, _ = load_sample(0, SAMPLE_GLOBAL_IDX)
    vmin = float(np.min(y_true_ref))
    vmax = float(np.max(y_true_ref))

    for g in G_VALUES:
        mets = all_metrics[g]
        x_in, y_true, y_pred = load_sample(g, SAMPLE_GLOBAL_IDX)
        mre = mets["mre"]
        ijump = mets["interface_jump"]

        fig, axes = plt.subplots(1, 6, figsize=(14, 2.8), constrained_layout=True)
        label = f"g={g}"
        title = (
            f"Context-fit PCA (g={g}) | Darcy 128 | patch_size=32 | seed=0 | "
            f"MRE={mre:.4f} | interface_jump={ijump:.2e}"
        )
        fig.suptitle(title, fontsize=9)
        _plot_field_row(
            axes, x_in, y_true, y_pred, label,
            vmin, vmax, err_vmax, fig, show_col_titles=True
        )

        out_path = out_dir / f"guard_band_g{g}_sample{SAMPLE_GLOBAL_IDX}.png"
        fig.savefig(out_path, dpi=STYLE["dpi"], bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")
        paths.append(out_path)
    return paths


def make_combined_figure(
    all_metrics: dict[int, dict],
    err_vmax: float,
    out_dir: Path,
) -> Path:
    _, y_true_ref, _ = load_sample(0, SAMPLE_GLOBAL_IDX)
    vmin = float(np.min(y_true_ref))
    vmax = float(np.max(y_true_ref))

    n_rows = len(G_VALUES)
    n_cols = 6
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(14, 2.5 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )
    fig.suptitle(
        f"Context-fit PCA sweep | Darcy 128 | patch_size=32 | seed=0 | sample {SAMPLE_GLOBAL_IDX}",
        fontsize=STYLE["title_size"],
    )

    col_titles = ["Input (a)", "Ground truth (u)", "Prediction", "Absolute error", "Energy spectrum", "Value PDF"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=9)

    for row_idx, g in enumerate(G_VALUES):
        mets = all_metrics[g]
        x_in, y_true, y_pred = load_sample(g, SAMPLE_GLOBAL_IDX)
        mre = mets["mre"]
        ijump = mets["interface_jump"]
        row_label = f"g={g}\nMRE={mre:.4f}\nij={ijump:.2e}"
        _plot_field_row(
            axes[row_idx], x_in, y_true, y_pred, row_label,
            vmin, vmax, err_vmax, fig, show_col_titles=False,
        )

    out_path = out_dir / f"guard_band_sweep_combined_sample{SAMPLE_GLOBAL_IDX}.png"
    fig.savefig(out_path, dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def print_summary_table(all_metrics: dict[int, dict], all_runtimes: dict[int, dict]) -> None:
    print()
    print("=" * 120)
    print("GUARD BAND SWEEP — SUMMARY TABLE")
    print("=" * 120)
    hdr = (
        f"{'g':>4}  {'PCA fit(s)':>10}  {'nn_train(s)':>12}  {'infer(s)':>9}  "
        f"{'MRE':>8}  {'SSIM':>7}  {'MAE':>10}  "
        f"{'ij':>10}  {'ifj':>10}  {'spec_err':>10}"
    )
    print(hdr)
    print("-" * 120)
    for g in G_VALUES:
        m = all_metrics[g]
        t = all_runtimes[g]["times"]
        print(
            f"{g:>4}  {t['pca_fit']:>10.2f}  {t['nn_train']:>12.2f}  {t['inference']:>9.3f}  "
            f"{m['mre']:>8.5f}  {m.get('ssim_mean', float('nan')):>7.4f}  {m['mae']:>10.6f}  "
            f"{m['interface_jump']:>10.6f}  {m['interface_flux_jump']:>10.5f}  "
            f"{m['relative_spectrum_error']:>10.6f}"
        )
    print("=" * 120)


def main() -> None:
    apply_style()
    out_dir = ensure_dir(FIGURES_DIR)

    all_metrics = {g: load_metrics(g) for g in G_VALUES}
    all_runtimes = {g: load_runtime(g) for g in G_VALUES}

    # Compute shared vmax for absolute error across all g values on sample 777
    err_vmax = max(
        float(np.max(np.abs(load_sample(g, SAMPLE_GLOBAL_IDX)[2] - load_sample(g, SAMPLE_GLOBAL_IDX)[1])))
        for g in G_VALUES
    )
    err_vmax = max(err_vmax, 1e-12)
    print(f"Shared absolute error vmax = {err_vmax:.4f}")

    make_time_vs_accuracy_figure(all_metrics, all_runtimes, out_dir)
    make_individual_reconstruction_figures(all_metrics, err_vmax, out_dir)
    make_combined_figure(all_metrics, err_vmax, out_dir)
    print_summary_table(all_metrics, all_runtimes)


if __name__ == "__main__":
    main()
