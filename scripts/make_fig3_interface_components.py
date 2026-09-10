#!/usr/bin/env python
"""Export real-field and trace components for the Fig. 3 interface-loss schematic."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

DEFAULT_RUN = Path(
    "paper_results/headline_poisson_128/poisson/resolution_128/two_scale_interface/seed_0"
)
DEFAULT_OUTPUT = Path("paper_results/manuscript/figures/fig3_interface_loss_components")
DEFAULT_SAMPLE_ID = 8944
INTERFACE_INDEX = 64
PATCH_SIZE = 32
CMAP = "turbo"
LEFT_COLOR = "#D97706"
RIGHT_COLOR = "#2563A8"
TRUTH_COLOR = "#252525"
PREDICTION_COLOR = "#7C3F98"


def _load_sample(run_dir: Path, sample_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = run_dir / "predictions_test.npz"
    with np.load(path, allow_pickle=False) as archive:
        indices = np.asarray(archive["sample_indices"], dtype=np.int64)
        matches = np.flatnonzero(indices == sample_id)
        if len(matches) != 1:
            raise ValueError(f"Sample {sample_id} is not uniquely available in {path}.")
        index = int(matches[0])
        return (
            np.asarray(archive["x_input"][index], dtype=np.float64),
            np.asarray(archive["y_true"][index], dtype=np.float64),
            np.asarray(archive["y_pred"][index], dtype=np.float64),
        )


def _save_field(
    field: np.ndarray,
    output: Path,
    *,
    vmin: float,
    vmax: float,
    patch_grid: bool = False,
) -> None:
    fig, axis = plt.subplots(figsize=(5.4, 5.4), dpi=300)
    axis.imshow(field, origin="lower", cmap=CMAP, vmin=vmin, vmax=vmax, interpolation="bicubic")
    if patch_grid:
        for coordinate in range(PATCH_SIZE, field.shape[0], PATCH_SIZE):
            axis.axhline(coordinate - 0.5, color="#171717", linewidth=1.35, linestyle=(0, (3, 2)))
            axis.axvline(coordinate - 0.5, color="#171717", linewidth=1.35, linestyle=(0, (3, 2)))
    axis.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(output, dpi=300, facecolor="white", pad_inches=0)
    plt.close(fig)


def _save_interface_stencil(output: Path, *, interface_index: int) -> None:
    """Draw a reusable two-patch vertical-interface stencil with index labels."""
    fig, axis = plt.subplots(figsize=(7.1, 3.4), dpi=300)
    axis.set_xlim(-0.2, 10.2)
    axis.set_ylim(-1.55, 5.3)
    axis.axis("off")
    for row in range(4):
        for column in range(10):
            color = "#FFF1CB" if column < 5 else "#DEEAF9"
            axis.add_patch(
                Rectangle((column, row), 1, 1, facecolor=color, edgecolor="#424242", linewidth=0.7)
            )
    axis.axvline(5.0, ymin=0.16, ymax=0.88, color="#252525", linestyle=(0, (4, 3)), linewidth=1.3)
    axis.text(2.5, 4.58, r"patch $j$", ha="center", va="bottom", fontsize=11)
    axis.text(7.5, 4.58, r"patch $j+1$", ha="center", va="bottom", fontsize=11)
    axis.text(4.5, -0.45, r"$b-1$", ha="center", va="top", fontsize=10)
    axis.text(5.0, -0.45, r"$b$", ha="center", va="top", fontsize=10)
    axis.text(5.5, -0.45, r"$b+1$", ha="center", va="top", fontsize=10)
    axis.annotate(
        rf"interface $b={interface_index}$",
        xy=(5.0, 0.0),
        xytext=(2.45, -1.15),
        ha="center",
        va="top",
        fontsize=10,
        arrowprops={"arrowstyle": "-|>", "color": "#252525", "linewidth": 0.9},
    )
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(output, dpi=300, facecolor="none", transparent=True, pad_inches=0)
    plt.close(fig)


def _save_value_trace(
    truth: np.ndarray, prediction: np.ndarray, output: Path, *, interface_index: int
) -> None:
    """Plot exact value traces on each side of a vertical interface."""
    coordinate = np.arange(truth.shape[0])
    true_left, true_right = truth[:, interface_index - 1], truth[:, interface_index]
    predicted_left = prediction[:, interface_index - 1]
    predicted_right = prediction[:, interface_index]
    lower = min(np.min(true_left), np.min(true_right), np.min(predicted_left), np.min(predicted_right))
    upper = max(np.max(true_left), np.max(true_right), np.max(predicted_left), np.max(predicted_right))
    padding = max(0.08 * (upper - lower), 1.0e-12)

    fig, axis = plt.subplots(figsize=(6.2, 3.25), dpi=300)
    axis.plot(coordinate, true_left, color=LEFT_COLOR, linewidth=1.8, label=r"truth: patch $j$")
    axis.plot(coordinate, true_right, color=RIGHT_COLOR, linewidth=1.8, label=r"truth: patch $j+1$")
    axis.plot(coordinate, predicted_left, color=LEFT_COLOR, linewidth=1.25, linestyle=(0, (3, 2)), label=r"prediction")
    axis.plot(coordinate, predicted_right, color=RIGHT_COLOR, linewidth=1.25, linestyle=(0, (3, 2)))
    axis.set_xlim(0, truth.shape[0] - 1)
    axis.set_ylim(lower - padding, upper + padding)
    axis.set_xlabel("Tangential grid index")
    axis.set_ylabel(r"$u$")
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=8)
    axis.legend(loc="upper left", frameon=False, fontsize=7.4, ncol=2)
    fig.tight_layout(pad=0.25)
    fig.savefig(output, dpi=300, facecolor="white", pad_inches=0.03)
    plt.close(fig)


def _save_flux_trace(
    truth: np.ndarray, prediction: np.ndarray, output: Path, *, interface_index: int
) -> None:
    """Plot exact one-sided normal derivatives entering the flux-trace loss."""
    step = 1.0 / (truth.shape[1] - 1)
    coordinate = np.arange(truth.shape[0])
    true_left = (truth[:, interface_index] - truth[:, interface_index - 1]) / step
    true_right = (truth[:, interface_index + 1] - truth[:, interface_index]) / step
    predicted_left = (prediction[:, interface_index] - prediction[:, interface_index - 1]) / step
    predicted_right = (prediction[:, interface_index + 1] - prediction[:, interface_index]) / step
    lower = min(np.min(true_left), np.min(true_right), np.min(predicted_left), np.min(predicted_right))
    upper = max(np.max(true_left), np.max(true_right), np.max(predicted_left), np.max(predicted_right))
    padding = max(0.08 * (upper - lower), 1.0e-12)

    fig, axis = plt.subplots(figsize=(6.2, 3.25), dpi=300)
    axis.plot(coordinate, true_left, color=LEFT_COLOR, linewidth=1.8, label=r"truth: left difference")
    axis.plot(coordinate, true_right, color=RIGHT_COLOR, linewidth=1.8, label=r"truth: right difference")
    axis.plot(coordinate, predicted_left, color=LEFT_COLOR, linewidth=1.25, linestyle=(0, (3, 2)), label=r"prediction")
    axis.plot(coordinate, predicted_right, color=RIGHT_COLOR, linewidth=1.25, linestyle=(0, (3, 2)))
    axis.set_xlim(0, truth.shape[0] - 1)
    axis.set_ylim(lower - padding, upper + padding)
    axis.set_xlabel("Tangential grid index")
    axis.set_ylabel(r"$\partial_n u$")
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=8)
    axis.legend(loc="upper left", frameon=False, fontsize=7.4, ncol=2)
    fig.tight_layout(pad=0.25)
    fig.savefig(output, dpi=300, facecolor="white", pad_inches=0.03)
    plt.close(fig)


def _write_manifest(
    output: Path,
    *,
    run_dir: Path,
    sample_id: int,
    interface_index: int,
    solution_limits: tuple[float, float],
    input_limits: tuple[float, float],
) -> None:
    solution_min, solution_max = solution_limits
    input_min, input_max = input_limits
    output.joinpath("README.md").write_text(
        "# Figure 3 Interface-Loss Components\n\n"
        "All data-driven components use the Poisson-128 headline-study two-scale "
        f"interface-fine-tuned run at `{run_dir}`, test sample `{sample_id}`, and "
        f"vertical patch boundary `b={interface_index}`. Solid curves are truth; "
        "dashed curves are the corresponding prediction.\n\n"
        f"Field panels use `turbo`. Solution limits are [{solution_min:.6g}, "
        f"{solution_max:.6g}], and input limits are [{input_min:.6g}, {input_max:.6g}].\n\n"
        "- `01_input_field_x.png`: actual input field \(x=f\), with the local 4 x 4 grid.\n"
        "- `02_predicted_field_u_hat.png`: two-scale + interface-fine-tuned prediction.\n"
        "- `03_ground_truth_u.png`: aligned true solution.\n"
        "- `04_vertical_interface_stencil.png`: transparent two-patch boundary stencil.\n"
        "- `05_value_trace.png`: left/right values entering \(T_V(u;b)\).\n"
        "- `06_flux_trace.png`: left/right normal differences entering \(T_F(u;b)\).\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-id", type=int, default=DEFAULT_SAMPLE_ID)
    parser.add_argument("--interface-index", type=int, default=INTERFACE_INDEX)
    args = parser.parse_args()

    input_field, truth, prediction = _load_sample(args.run_dir, args.sample_id)
    interface_index = int(args.interface_index)
    if interface_index <= 0 or interface_index >= truth.shape[1] - 1:
        raise ValueError(f"interface index must lie in [1, {truth.shape[1] - 2}].")
    solution_min = float(np.min(np.stack([truth, prediction])))
    solution_max = float(np.max(np.stack([truth, prediction])))
    input_min = float(np.min(input_field))
    input_max = float(np.max(input_field))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _save_field(
        input_field,
        args.output_dir / "01_input_field_x.png",
        vmin=input_min,
        vmax=input_max,
        patch_grid=True,
    )
    _save_field(
        prediction,
        args.output_dir / "02_predicted_field_u_hat.png",
        vmin=solution_min,
        vmax=solution_max,
    )
    _save_field(
        truth,
        args.output_dir / "03_ground_truth_u.png",
        vmin=solution_min,
        vmax=solution_max,
    )
    _save_interface_stencil(
        args.output_dir / "04_vertical_interface_stencil.png",
        interface_index=interface_index,
    )
    _save_value_trace(
        truth,
        prediction,
        args.output_dir / "05_value_trace.png",
        interface_index=interface_index,
    )
    _save_flux_trace(
        truth,
        prediction,
        args.output_dir / "06_flux_trace.png",
        interface_index=interface_index,
    )
    _write_manifest(
        args.output_dir,
        run_dir=args.run_dir,
        sample_id=args.sample_id,
        interface_index=interface_index,
        solution_limits=(solution_min, solution_max),
        input_limits=(input_min, input_max),
    )


if __name__ == "__main__":
    main()
