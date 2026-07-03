#!/usr/bin/env python3
"""Generate figures for the boundary-correction GNN interface weight sweep."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lpcanet.metrics.spectral import energy_spectrum_2d

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

SAMPLE_IDX = 777

# All runs: (label for figures, result_dir, interface_value, interface_flux, recon_weight)
RUNS = [
    ("iv=5.0, if=0.05, recon=1.0",  "boundary_correction_gnn_darcy128_seed0_fixed_v2",         5.0,  0.05,  1.0),
    ("iv=10.0, if=0.10, recon=1.0", "boundary_correction_gnn_darcy128_seed0_iv10_if01",         10.0, 0.10,  1.0),
    ("iv=20.0, if=0.20, recon=1.0", "boundary_correction_gnn_darcy128_seed0_iv20_if02",         20.0, 0.20,  1.0),
    ("iv=5.0, if=0.50, recon=1.0",  "boundary_correction_gnn_darcy128_seed0_iv5_if05",          5.0,  0.50,  1.0),
    ("iv=5.0, if=1.00, recon=1.0",  "boundary_correction_gnn_darcy128_seed0_iv5_if1",           5.0,  1.00,  1.0),
    ("iv=5.0, if=1.00, recon=0.5",  "boundary_correction_gnn_darcy128_seed0_iv5_if1_recon05",   5.0,  1.00,  0.5),
]

DPI = 180
LABEL_SIZE = 8
TICK_SIZE = 7

plt.rcParams.update({
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "font.size": LABEL_SIZE,
    "axes.titlesize": LABEL_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": TICK_SIZE,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_run(result_dir: str) -> dict | None:
    path = RESULTS / result_dir
    pred_path = path / "predictions_test.npz"
    metrics_path = path / "metrics.json"
    if not pred_path.exists():
        return None
    arrays = np.load(pred_path)
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    sample_indices = arrays.get("sample_indices", np.arange(arrays["y_pred"].shape[0]))
    pos = int(np.where(sample_indices == SAMPLE_IDX)[0][0])
    return {
        "y_pred": arrays["y_pred"][pos],
        "y_true": arrays["y_true"][pos],
        "x_input": arrays["x_input"][pos] if "x_input" in arrays else np.zeros((128, 128)),
        "metrics": metrics,
    }


def _plot_spectrum(ax, truth, pred):
    t_e = energy_spectrum_2d(truth)
    p_e = energy_spectrum_2d(pred)
    k = np.arange(1, len(t_e))
    ax.plot(k, t_e[1:] + 1e-30, color="black", lw=1.2, label="Truth")
    ax.plot(k, p_e[1:] + 1e-30, color="tab:red", lw=1.2, label="Pred")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=TICK_SIZE)


def _plot_pdf(ax, truth, pred):
    lo = float(min(truth.min(), pred.min()))
    hi = float(max(truth.max(), pred.max()))
    if lo == hi:
        lo -= 0.5; hi += 0.5
    bins = np.linspace(lo, hi, 40)
    ax.hist(truth.ravel(), bins=bins, density=True, histtype="step", color="black", lw=1.2, label="Truth")
    ax.hist(pred.ravel(), bins=bins, density=True, histtype="step", color="tab:red", lw=1.2, label="Pred")
    ax.grid(True, alpha=0.25)


def make_single_figure(run_data: dict, label: str, mre: float, err_vmax: float, out_path: Path):
    pred = run_data["y_pred"]
    true = run_data["y_true"]
    inp  = run_data["x_input"]
    err  = np.abs(pred - true)

    vmin = float(true.min())
    vmax = float(true.max())

    cols = ["Input", "Ground truth", "Prediction", "Absolute error", "Energy spectrum", "Value PDF"]
    fig, axes = plt.subplots(1, 6, figsize=(13.5, 2.4), constrained_layout=True)

    panels = [
        (inp,  "RdBu_r", None,  None),
        (true, "RdBu_r", vmin,  vmax),
        (pred, "RdBu_r", vmin,  vmax),
        (err,  "magma",  0.0,   err_vmax),
    ]
    for col_idx, (field, cmap, lo, hi) in enumerate(panels):
        ax = axes[col_idx]
        im = ax.imshow(field, origin="lower", cmap=cmap, vmin=lo, vmax=hi)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(cols[col_idx])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    _plot_spectrum(axes[4], true, pred)
    axes[4].set_title(cols[4])
    axes[4].set_xlabel("k")
    axes[4].set_ylabel("Energy")

    _plot_pdf(axes[5], true, pred)
    axes[5].set_title(cols[5])
    axes[5].set_xlabel("Field value")
    axes[5].set_ylabel("Density")

    title = (
        f"GNN + boundary-correction (δ=4) | Darcy 128 | {label} | "
        f"patch_size=32 | seed=0 | sample {SAMPLE_IDX} | MRE={mre:.4f}"
    )
    fig.suptitle(title, fontsize=7, y=1.01)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def make_combined_figure(runs_data: list[dict | None], run_labels: list[str], mre_values: list[float], err_vmax: float, out_path: Path):
    valid = [(d, l, m) for d, l, m in zip(runs_data, run_labels, mre_values) if d is not None]
    n_rows = len(valid)
    if n_rows == 0:
        return

    cols = ["Input", "Ground truth", "Prediction", "Absolute error", "Energy spectrum", "Value PDF"]
    fig, axes = plt.subplots(n_rows, 6, figsize=(13.5, 2.2 * n_rows), constrained_layout=True, squeeze=False)

    for row_idx, (run_data, label, mre) in enumerate(valid):
        pred = run_data["y_pred"]
        true = run_data["y_true"]
        inp  = run_data["x_input"]
        err  = np.abs(pred - true)
        vmin = float(true.min())
        vmax_f = float(true.max())

        panels = [
            (inp,  "RdBu_r", None, None),
            (true, "RdBu_r", vmin, vmax_f),
            (pred, "RdBu_r", vmin, vmax_f),
            (err,  "magma",  0.0,  err_vmax),
        ]
        for col_idx, (field, cmap, lo, hi) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(field, origin="lower", cmap=cmap, vmin=lo, vmax=hi)
            ax.set_xticks([]); ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(cols[col_idx])
            if col_idx == 0:
                ax.set_ylabel(f"{label}\nMRE={mre:.4f}", fontsize=6)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

        _plot_spectrum(axes[row_idx, 4], true, pred)
        if row_idx == 0:
            axes[row_idx, 4].set_title(cols[4])
        if row_idx == n_rows - 1:
            axes[row_idx, 4].set_xlabel("k")

        _plot_pdf(axes[row_idx, 5], true, pred)
        if row_idx == 0:
            axes[row_idx, 5].set_title(cols[5])
        if row_idx == n_rows - 1:
            axes[row_idx, 5].set_xlabel("Field value")

    fig.suptitle(
        f"GNN + boundary-correction (δ=4) | Darcy 128 | patch_size=32 | seed=0 | sample {SAMPLE_IDX} | shared error colorbar",
        fontsize=8, y=1.005,
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def make_metrics_table(rows: list[dict], out_png: Path, out_txt: Path):
    headers = ["iv", "if", "recon", "MRE", "SSIM", "MAE", "if_jump", "if_flux_jump", "rel_spec_err", "train_s", "spikes"]
    table_data = []
    for row in rows:
        table_data.append([
            f"{row['iv']:.1f}",
            f"{row['if_w']:.2f}",
            f"{row['recon']:.1f}",
            f"{row['mre']:.4f}",
            f"{row['ssim']:.4f}",
            f"{row['mae']:.2e}",
            f"{row['if_jump']:.2e}",
            f"{row['if_flux_jump']:.4f}",
            f"{row['rel_spec_err']:.2e}",
            f"{row['train_s']:.0f}",
            f"{row['spikes']}",
        ])

    # Sort by interface_flux ascending
    table_data.sort(key=lambda r: float(r[1]))

    fig, ax = plt.subplots(figsize=(14, 0.45 * (len(table_data) + 1.5)))
    ax.axis("off")
    tbl = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.4)

    # Color header
    for j in range(len(headers)):
        tbl[0, j].set_facecolor("#2d5986")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight row with best MRE
    mre_vals = [float(r[3]) for r in table_data]
    best_mre_row = mre_vals.index(min(mre_vals)) + 1
    for j in range(len(headers)):
        tbl[best_mre_row, j].set_facecolor("#d4edda")

    fig.suptitle(
        "BoundaryCorrectionGNN — full interface weight sweep | Darcy 128, seed=0 | sorted by interface_flux",
        fontsize=9, y=0.98,
    )
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")

    # Text version
    col_widths = [max(len(h), max(len(r[i]) for r in table_data)) for i, h in enumerate(headers)]
    lines = []
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    lines.append(header_line)
    lines.append("-" * len(header_line))
    for row in table_data:
        lines.append("  ".join(v.ljust(w) for v, w in zip(row, col_widths)))
    out_txt.write_text("\n".join(lines) + "\n")
    print(f"Saved: {out_txt}")


def compute_stability(result_dir: str) -> int:
    loss_path = RESULTS / result_dir / "loss_curves.npz"
    if not loss_path.exists():
        return -1
    d = np.load(loss_path)
    train_loss = d["train_loss"]
    if len(train_loss) < 14:
        return -1
    window = train_loss[12:60]  # epochs 13-60 (0-indexed 12-59)
    spikes = 0
    for i in range(len(window)):
        # rolling median of up to 5 PRECEDING epochs (not including current)
        start = max(0, i - 5)
        if i == 0:
            continue
        rolling_median = float(np.median(window[start:i]))
        if rolling_median > 0 and window[i] > 1.05 * rolling_median:
            spikes += 1
    return spikes


def main():
    print("Loading run data...")
    runs_data: list[dict | None] = []
    for label, result_dir, iv, if_w, recon in RUNS:
        data = load_run(result_dir)
        runs_data.append(data)
        status = "OK" if data is not None else "MISSING"
        print(f"  {result_dir}: {status}")

    # Compute shared absolute error vmax across all available runs at sample 777
    err_vmax = 0.0
    for data in runs_data:
        if data is not None:
            err = np.abs(data["y_pred"] - data["y_true"])
            err_vmax = max(err_vmax, float(err.max()))
    print(f"Shared absolute error vmax: {err_vmax:.6f}")

    # Per-run figure output filenames for Runs A/B/C (new ones)
    new_run_out_names = [
        None, None, None,  # skip baseline, iv10, iv20 (already have figures)
        FIGURES / f"boundary_correction_gnn_darcy128_seed0_iv5_if05_sample{SAMPLE_IDX}.png",
        FIGURES / f"boundary_correction_gnn_darcy128_seed0_iv5_if1_sample{SAMPLE_IDX}.png",
        FIGURES / f"boundary_correction_gnn_darcy128_seed0_iv5_if1_recon05_sample{SAMPLE_IDX}.png",
    ]

    for (label, result_dir, iv, if_w, recon), data, out_path in zip(RUNS, runs_data, new_run_out_names):
        if out_path is None or data is None:
            continue
        mre = float(data["metrics"].get("mre", float("nan")))
        cfg_label = f"iv={iv:.1f}, if={if_w:.2f}, recon={recon:.1f}"
        make_single_figure(data, cfg_label, mre, err_vmax, out_path)

    # Combined multi-row figure (all 6 runs)
    run_labels = [f"iv={iv:.1f}, if={if_w:.2f}, recon={recon:.1f}" for _, _, iv, if_w, recon in RUNS]
    mre_values = []
    for data in runs_data:
        if data is not None:
            mre_values.append(float(data["metrics"].get("mre", float("nan"))))
        else:
            mre_values.append(float("nan"))

    combined_path = FIGURES / f"boundary_correction_full_sweep_combined_sample{SAMPLE_IDX}.png"
    make_combined_figure(runs_data, run_labels, mre_values, err_vmax, combined_path)

    # Metrics table — collect from metrics.json + runtime.json
    table_rows = []
    for (label, result_dir, iv, if_w, recon), data in zip(RUNS, runs_data):
        if data is None:
            continue
        m = data["metrics"]
        rt_path = RESULTS / result_dir / "runtime.json"
        train_s = float("nan")
        if rt_path.exists():
            rt = json.loads(rt_path.read_text())
            times = rt.get("times", rt)  # some older files are flat, newer are nested
            train_s = float(times.get("nn_train", times.get("end_to_end", float("nan"))))
        spikes = compute_stability(result_dir)
        table_rows.append({
            "iv": iv,
            "if_w": if_w,
            "recon": recon,
            "mre": float(m.get("mre", float("nan"))),
            "ssim": float(m.get("ssim_mean", float("nan"))),
            "mae": float(m.get("mae", float("nan"))),
            "if_jump": float(m.get("interface_jump", float("nan"))),
            "if_flux_jump": float(m.get("interface_flux_jump", float("nan"))),
            "rel_spec_err": float(m.get("relative_spectrum_error", float("nan"))),
            "train_s": train_s,
            "spikes": spikes,
        })

    make_metrics_table(
        table_rows,
        FIGURES / "boundary_correction_full_sweep_table.png",
        RESULTS / "boundary_correction_full_sweep_table.txt",
    )

    print("\n=== Summary of available runs ===")
    print(f"{'Config':<35} {'MRE':>8} {'SSIM':>7} {'if_jump':>10} {'if_flux':>8} {'spec_err':>10} {'spikes':>7}")
    for row in sorted(table_rows, key=lambda r: r["if_w"]):
        label = f"iv={row['iv']:.0f}/if={row['if_w']:.2f}/r={row['recon']:.1f}"
        print(f"{label:<35} {row['mre']:>8.4f} {row['ssim']:>7.4f} {row['if_jump']:>10.2e} "
              f"{row['if_flux_jump']:>8.4f} {row['rel_spec_err']:>10.2e} {row['spikes']:>7}")


if __name__ == "__main__":
    main()
