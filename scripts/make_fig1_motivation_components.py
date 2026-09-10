#!/usr/bin/env python
"""Export reusable Poisson-field components for the Fig. 1 motivation schematic.

The components are intentionally free of titles and axes so they can be placed
directly in an illustration program. They use one aligned headline-study test
sample: the global-PCA and L2L fields are genuine model predictions. The
seam-amplified panel is illustrative: it amplifies the patchwise component of
the actual L2L error to make the failure mode legible at schematic scale.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

DEFAULT_RESULTS = Path("paper_results/headline_poisson_128/poisson/resolution_128")
DEFAULT_OUTPUT = Path("paper_results/manuscript/figures/fig1_components")
DEFAULT_SAMPLE_ID = 8944
GRID_SIZE = 128
PATCH_SIZE = 32
OVERLAP_STRIDE = 16
CMAP = "turbo"


def _load_sample(results_root: Path, sample_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    truth: np.ndarray | None = None
    for method in ("global_pca", "plain_l2l"):
        path = results_root / method / "seed_0" / "predictions_test.npz"
        with np.load(path, allow_pickle=False) as archive:
            indices = np.asarray(archive["sample_indices"], dtype=np.int64)
            matches = np.flatnonzero(indices == sample_id)
            if len(matches) != 1:
                raise ValueError(f"Sample {sample_id} is not uniquely available in {path}.")
            index = int(matches[0])
            candidate_truth = np.asarray(archive["y_true"][index], dtype=np.float64)
            if truth is None:
                truth = candidate_truth
            elif not np.allclose(candidate_truth, truth, rtol=0.0, atol=1.0e-7):
                raise ValueError(f"Ground truth is not aligned for {method}.")
            arrays[method] = np.asarray(archive["y_pred"][index], dtype=np.float64)
    if truth is None:
        raise RuntimeError("No ground-truth field was loaded.")
    return truth, arrays["global_pca"], arrays["plain_l2l"]


def _seam_amplified_l2l(truth: np.ndarray, l2l_prediction: np.ndarray) -> np.ndarray:
    """Amplify genuine patchwise L2L error without inventing a new field."""
    error = l2l_prediction - truth
    blockwise = np.empty_like(error)
    for row in range(0, GRID_SIZE, PATCH_SIZE):
        for column in range(0, GRID_SIZE, PATCH_SIZE):
            patch = error[row : row + PATCH_SIZE, column : column + PATCH_SIZE]
            # The per-patch mean is the visual block-offset component of the error.
            blockwise[row : row + PATCH_SIZE, column : column + PATCH_SIZE] = np.mean(patch)
    return truth + 1.5 * error + 6.0 * blockwise


def _save_field(
    field: np.ndarray,
    output: Path,
    *,
    vmin: float,
    vmax: float,
    dashed_grid: bool = False,
) -> None:
    fig, axis = plt.subplots(figsize=(5.4, 5.4), dpi=300)
    axis.imshow(field, origin="lower", cmap=CMAP, vmin=vmin, vmax=vmax, interpolation="bicubic")
    if dashed_grid:
        for coordinate in range(PATCH_SIZE, GRID_SIZE, PATCH_SIZE):
            axis.axhline(coordinate - 0.5, color="#101010", linewidth=1.5, linestyle=(0, (3, 2)))
            axis.axvline(coordinate - 0.5, color="#101010", linewidth=1.5, linestyle=(0, (3, 2)))
    axis.set_axis_off()
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.2)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(output, dpi=300, facecolor="white", pad_inches=0)
    plt.close(fig)


def _save_overlap_stack(field: np.ndarray, output: Path, *, vmin: float, vmax: float) -> None:
    """Render four true stride-16 patches as a compact overlap-add motif."""
    positions = ((16, 16), (16, 32), (32, 16), (32, 32))
    offsets = ((0.04, 0.03), (0.15, 0.16), (0.28, 0.08), (0.39, 0.21))
    fig, axis = plt.subplots(figsize=(6.2, 4.8), dpi=300)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_axis_off()
    for index, ((row, column), (left, bottom)) in enumerate(zip(positions, offsets)):
        patch = field[row : row + PATCH_SIZE, column : column + PATCH_SIZE]
        extent = (left, left + 0.48, bottom, bottom + 0.48)
        axis.imshow(
            patch,
            extent=extent,
            origin="lower",
            cmap=CMAP,
            vmin=vmin,
            vmax=vmax,
            interpolation="bicubic",
            alpha=0.88 if index else 1.0,
            zorder=index,
        )
        axis.add_patch(
            Rectangle(
                (left, bottom),
                0.48,
                0.48,
                fill=False,
                edgecolor="#151515",
                linewidth=1.35,
                linestyle=(0, (3, 2)),
                zorder=10 + index,
            )
        )
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(output, dpi=300, facecolor="none", transparent=True, pad_inches=0)
    plt.close(fig)


def _save_l2l_patch_stack(field: np.ndarray, output: Path, *, vmin: float, vmax: float) -> None:
    """Render representative nonoverlapping L2L patches as a vertical stack."""
    positions = ((96, 0), (32, 32), (0, 0))
    axes_positions = ((0.20, 0.70, 0.60, 0.21), (0.20, 0.43, 0.60, 0.21), (0.20, 0.08, 0.60, 0.21))
    fig = plt.figure(figsize=(3.6, 8.5), dpi=300)
    for (row, column), axes_position in zip(positions, axes_positions):
        patch = field[row : row + PATCH_SIZE, column : column + PATCH_SIZE]
        axis = fig.add_axes(axes_position)
        axis.imshow(
            patch,
            origin="lower",
            cmap=CMAP,
            vmin=vmin,
            vmax=vmax,
            interpolation="bicubic",
        )
        axis.add_patch(
            Rectangle(
                (-0.5, -0.5),
                PATCH_SIZE,
                PATCH_SIZE,
                fill=False,
                edgecolor="#151515",
                linewidth=1.35,
                linestyle=(0, (3, 2)),
            )
        )
        axis.set_axis_off()
    fig.text(0.50, 0.355, "⋮", ha="center", va="center", fontsize=23, color="#4B4B4B")
    fig.savefig(output, dpi=300, facecolor="none", transparent=True, pad_inches=0)
    plt.close(fig)


def _save_overlap_coverage(field: np.ndarray, output: Path, *, vmin: float, vmax: float) -> None:
    """Show the actual 32-pixel, stride-16 overlapping-patch coverage."""
    fig, axis = plt.subplots(figsize=(5.4, 5.4), dpi=300)
    axis.imshow(field, origin="lower", cmap=CMAP, vmin=vmin, vmax=vmax, interpolation="bicubic")
    for row in range(0, GRID_SIZE - PATCH_SIZE + 1, OVERLAP_STRIDE):
        for column in range(0, GRID_SIZE - PATCH_SIZE + 1, OVERLAP_STRIDE):
            axis.add_patch(
                Rectangle(
                    (column - 0.5, row - 0.5),
                    PATCH_SIZE,
                    PATCH_SIZE,
                    fill=False,
                    edgecolor="white",
                    linewidth=0.7,
                    linestyle=(0, (2, 1.6)),
                    alpha=0.9,
                )
            )
    axis.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(output, dpi=300, facecolor="white", pad_inches=0)
    plt.close(fig)


def _write_manifest(output: Path, sample_id: int, vmin: float, vmax: float) -> None:
    output.joinpath("README.md").write_text(
        "# Figure 1 Motivation Components\n\n"
        "These files are reusable, axes-free raster components for the Fig. 1 "
        "motivation schematic. They were rendered from the aligned Poisson-128 "
        f"headline-study seed-0 test sample `{sample_id}` using the `{CMAP}` color map "
        f"and fixed display limits [{vmin:.6g}, {vmax:.6g}].\n\n"
        "- `01_poisson_solution.png`: true full-domain solution.\n"
        "- `02_global_pca_reconstruction.png`: corresponding Global PCA-Net prediction.\n"
        "- `03_solution_with_l2l_patch_grid.png`: true solution with the 4 x 4, "
        "32-pixel plain-L2L patch grid.\n"
        "- `04_l2l_seam_amplified.png`: illustrative seam-amplified version of the "
        "actual plain-L2L prediction. It amplifies its genuine patchwise error; "
        "it is not a reported quantitative reconstruction.\n"
        "- `05_overlap_patch_stack.png`: transparent-background stack of four real "
        "32-pixel patches extracted at stride 16, suitable for the overlap-add motif.\n"
        "- `06_overlap_coverage.png`: full field with the actual 32-pixel, stride-16 "
        "overlap coverage.\n"
        "- `07_l2l_patch_stack.png`: transparent-background stack of representative "
        "nonoverlapping L2L patches for the local-PCA stage.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-id", type=int, default=DEFAULT_SAMPLE_ID)
    args = parser.parse_args()

    truth, global_prediction, l2l_prediction = _load_sample(args.results_root, args.sample_id)
    seam_amplified = _seam_amplified_l2l(truth, l2l_prediction)
    vmin = float(np.min(truth))
    vmax = float(np.max(truth))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _save_field(truth, args.output_dir / "01_poisson_solution.png", vmin=vmin, vmax=vmax)
    _save_field(
        global_prediction,
        args.output_dir / "02_global_pca_reconstruction.png",
        vmin=vmin,
        vmax=vmax,
    )
    _save_field(
        truth,
        args.output_dir / "03_solution_with_l2l_patch_grid.png",
        vmin=vmin,
        vmax=vmax,
        dashed_grid=True,
    )
    _save_field(
        seam_amplified,
        args.output_dir / "04_l2l_seam_amplified.png",
        vmin=vmin,
        vmax=vmax,
    )
    _save_overlap_stack(
        truth,
        args.output_dir / "05_overlap_patch_stack.png",
        vmin=vmin,
        vmax=vmax,
    )
    _save_overlap_coverage(
        truth,
        args.output_dir / "06_overlap_coverage.png",
        vmin=vmin,
        vmax=vmax,
    )
    _save_l2l_patch_stack(
        truth,
        args.output_dir / "07_l2l_patch_stack.png",
        vmin=vmin,
        vmax=vmax,
    )
    _write_manifest(args.output_dir, args.sample_id, vmin, vmax)


if __name__ == "__main__":
    main()
