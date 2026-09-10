#!/usr/bin/env python
"""Export target-side two-scale decomposition components for the Fig. 2 schematic.

All components come from one true Poisson-128 test solution and the fitted
two-scale PCA encoder used in the headline study. The coarse field is the
coarse-PCA projection of the target, not a neural-network prediction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


DEFAULT_RUN = Path("paper_results/headline_poisson_128/poisson/resolution_128/two_scale/seed_0")
DEFAULT_OUTPUT = Path("paper_results/manuscript/figures/fig2_target_decomposition_components")
DEFAULT_SAMPLE_ID = 8944
CMAP = "turbo"
RESIDUAL_CMAP = "RdBu_r"


def _load_sample(run_dir: Path, sample_id: int) -> tuple[np.ndarray, np.ndarray]:
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
        )


def _decompose_target(
    truth: np.ndarray, encoder: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    from lpcanet.assembly.operators import prolongate_bilinear, restrict_average

    required = ("coarse_factor", "coarse_scaler", "coarse_pca", "patch_size", "stride")
    missing = [name for name in required if not hasattr(encoder, name)]
    if missing:
        raise TypeError(f"PCA artifact is not a fitted two-scale encoder; missing {missing}.")
    factor = int(encoder.coarse_factor)
    patch_size = int(encoder.patch_size)
    stride = int(encoder.stride)
    restricted = restrict_average(torch.as_tensor(truth), factor).cpu().numpy()
    flattened = restricted.reshape(1, -1)
    scaler = encoder.coarse_scaler
    pca = encoder.coarse_pca
    code = pca.transform(scaler.transform(flattened))
    decoded_flat = scaler.inverse_transform(pca.inverse_transform(code))
    decoded_coarse = decoded_flat.reshape(restricted.shape)
    global_field = prolongate_bilinear(torch.as_tensor(decoded_coarse), factor).cpu().numpy()
    return restricted, decoded_coarse, global_field, patch_size, stride


def _save_field(
    field: np.ndarray,
    output: Path,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    interpolation: str = "bicubic",
    dashed_grid: tuple[int, int] | None = None,
) -> None:
    fig, axis = plt.subplots(figsize=(5.4, 5.4), dpi=300)
    axis.imshow(
        field,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation=interpolation,
    )
    if dashed_grid is not None:
        patch_size, stride = dashed_grid
        height, width = field.shape
        for row in range(stride, height, stride):
            axis.axhline(row - 0.5, color="#171717", linewidth=1.25, linestyle=(0, (3, 2)))
        for column in range(stride, width, stride):
            axis.axvline(column - 0.5, color="#171717", linewidth=1.25, linestyle=(0, (3, 2)))
        if patch_size != stride:
            raise ValueError("This component renderer expects nonoverlapping patches.")
    axis.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(output, dpi=300, facecolor="white", pad_inches=0)
    plt.close(fig)


def _save_patch_stack(
    field: np.ndarray,
    output: Path,
    *,
    patch_size: int,
    cmap: str,
    vmin: float,
    vmax: float,
) -> None:
    """Export three square patches for a local-PCA motif."""
    height, width = field.shape
    positions = ((height - patch_size, 0), (patch_size, patch_size), (0, 0))
    axes_positions = (
        (0.20, 0.70, 0.60, 0.21),
        (0.20, 0.43, 0.60, 0.21),
        (0.20, 0.08, 0.60, 0.21),
    )
    fig = plt.figure(figsize=(3.6, 8.5), dpi=300)
    for (row, column), axes_position in zip(positions, axes_positions):
        patch = field[row : row + patch_size, column : column + patch_size]
        axis = fig.add_axes(axes_position)
        axis.imshow(
            patch,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="bicubic",
        )
        axis.add_patch(
            Rectangle(
                (-0.5, -0.5),
                patch_size,
                patch_size,
                fill=False,
                edgecolor="#171717",
                linewidth=1.35,
                linestyle=(0, (3, 2)),
            )
        )
        axis.set_axis_off()
    fig.text(0.50, 0.355, "⋮", ha="center", va="center", fontsize=23, color="#4B4B4B")
    fig.savefig(output, dpi=300, facecolor="none", transparent=True, pad_inches=0)
    plt.close(fig)


def _write_manifest(
    output: Path,
    *,
    run_dir: Path,
    sample_id: int,
    factor: int,
    patch_size: int,
    solution_limits: tuple[float, float],
    input_limits: tuple[float, float],
    residual_limit: float,
) -> None:
    solution_min, solution_max = solution_limits
    input_min, input_max = input_limits
    output.joinpath("README.md").write_text(
        "# Figure 2 Target Decomposition and Input Components\n\n"
        "These axes-free raster components use the true Poisson-128 headline-study "
        f"test sample `{sample_id}` and the fitted two-scale PCA encoder at "
        f"`{run_dir}`. The method uses average restriction by factor `{factor}` and "
        f"nonoverlapping `{patch_size} x {patch_size}` residual patches.\n\n"
        f"Normal solution-field panels use `{CMAP}` with limits "
        f"[{solution_min:.6g}, {solution_max:.6g}]. Residual panels use "
        f"`{RESIDUAL_CMAP}` with symmetric limits +/-{residual_limit:.6g}. The "
        f"input-field panels use `{CMAP}` with limits [{input_min:.6g}, {input_max:.6g}].\n\n"
        "- `01_solution_u.png`: true fine-grid target field \(u\).\n"
        "- `02_restricted_field_uc.png`: average-restricted target \(u_c=R_cu\).\n"
        "- `03_decoded_coarse_field.png`: coarse-grid PCA decode from the target's "
        "coarse score.\n"
        "- `04_global_field_uG.png`: bilinearly prolongated decoded coarse field "
        "\(u_G\).\n"
        "- `05_local_residual_r.png`: exact target-side residual \(r=u-u_G\).\n"
        "- `06_local_residual_with_patch_grid.png`: the same residual with its 4 x 4 "
        "local-PCA partition.\n"
        "- `07_residual_patch_stack.png`: transparent vertical stack of representative "
        "residual patches for the residual-PCA stage.\n"
        "- `08_input_field_x.png`: true Poisson input field \(x=f\), aligned with "
        "the target above and overlaid with the 4 x 4 local-input partition.\n"
        "- `09_input_patch_stack.png`: transparent vertical stack of representative "
        "nonoverlapping input patches for local input PCA.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-id", type=int, default=DEFAULT_SAMPLE_ID)
    args = parser.parse_args()

    input_field, truth = _load_sample(args.run_dir, args.sample_id)
    encoder = joblib.load(args.run_dir / "pca_encoder.joblib")
    restricted, decoded_coarse, global_field, patch_size, stride = _decompose_target(truth, encoder)
    residual = truth - global_field
    factor = int(encoder.coarse_factor)
    solution_min = float(np.min(truth))
    solution_max = float(np.max(truth))
    input_min = float(np.min(input_field))
    input_max = float(np.max(input_field))
    residual_limit = max(float(np.max(np.abs(residual))), 1.0e-12)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _save_field(
        truth,
        args.output_dir / "01_solution_u.png",
        cmap=CMAP,
        vmin=solution_min,
        vmax=solution_max,
    )
    _save_field(
        restricted,
        args.output_dir / "02_restricted_field_uc.png",
        cmap=CMAP,
        vmin=solution_min,
        vmax=solution_max,
        interpolation="nearest",
    )
    _save_field(
        decoded_coarse,
        args.output_dir / "03_decoded_coarse_field.png",
        cmap=CMAP,
        vmin=solution_min,
        vmax=solution_max,
        interpolation="nearest",
    )
    _save_field(
        global_field,
        args.output_dir / "04_global_field_uG.png",
        cmap=CMAP,
        vmin=solution_min,
        vmax=solution_max,
    )
    _save_field(
        residual,
        args.output_dir / "05_local_residual_r.png",
        cmap=RESIDUAL_CMAP,
        vmin=-residual_limit,
        vmax=residual_limit,
    )
    _save_field(
        residual,
        args.output_dir / "06_local_residual_with_patch_grid.png",
        cmap=RESIDUAL_CMAP,
        vmin=-residual_limit,
        vmax=residual_limit,
        dashed_grid=(patch_size, stride),
    )
    _save_patch_stack(
        residual,
        args.output_dir / "07_residual_patch_stack.png",
        patch_size=patch_size,
        cmap=RESIDUAL_CMAP,
        vmin=-residual_limit,
        vmax=residual_limit,
    )
    _save_field(
        input_field,
        args.output_dir / "08_input_field_x.png",
        cmap=CMAP,
        vmin=input_min,
        vmax=input_max,
        dashed_grid=(patch_size, stride),
    )
    _save_patch_stack(
        input_field,
        args.output_dir / "09_input_patch_stack.png",
        patch_size=patch_size,
        cmap=CMAP,
        vmin=input_min,
        vmax=input_max,
    )
    _write_manifest(
        args.output_dir,
        run_dir=args.run_dir,
        sample_id=args.sample_id,
        factor=factor,
        patch_size=patch_size,
        solution_limits=(solution_min, solution_max),
        input_limits=(input_min, input_max),
        residual_limit=residual_limit,
    )


if __name__ == "__main__":
    main()
