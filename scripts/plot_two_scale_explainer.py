#!/usr/bin/env python
"""Visualize how the two-scale output representation suppresses patch seams."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lpcanet.assembly.operators import restrict_average
from lpcanet.metrics.interface import interface_jump
from lpcanet.metrics.torch_ops import interface_boundaries


def plot_two_scale_explainer(
    *,
    results_dir: str | Path,
    output: str | Path,
    dataset: str,
    sample_id: int | None = None,
) -> list[Path]:
    """Generate a decomposition flow and a plain-L2L seam comparison."""
    results_root = Path(results_dir)
    output_path = Path(output)
    plain_path = _prediction_path(results_root, "plain_l2l")
    two_scale_path = _prediction_path(results_root, "two_scale")
    if sample_id is None:
        sample_id = _representative_sample_id(plain_path)

    _, truth, plain_prediction = _load_sample(plain_path, sample_id)
    _, two_truth, two_scale_prediction = _load_sample(two_scale_path, sample_id)
    if not np.allclose(two_truth, truth, rtol=0.0, atol=1.0e-7):
        raise ValueError("Plain L2L and two-scale runs do not share the same ground truth.")

    encoder_path = two_scale_path.parent / "pca_encoder.joblib"
    encoder = joblib.load(encoder_path)
    if not hasattr(encoder, "decompose_outputs"):
        raise TypeError(f"{encoder_path} does not contain a two-scale PCA encoder.")
    latent = encoder.transform_outputs(two_scale_prediction[None, :, :])
    coarse_batch, residual_batch = encoder.decompose_outputs(latent)
    coarse = np.asarray(coarse_batch[0], dtype=np.float64)
    residual = np.asarray(residual_batch[0], dtype=np.float64)
    recomposed = coarse + residual
    if not np.allclose(recomposed, two_scale_prediction, rtol=5.0e-4, atol=5.0e-6):
        maximum_error = float(np.max(np.abs(recomposed - two_scale_prediction)))
        raise ValueError(
            "Two-scale decomposition did not reproduce the saved prediction; "
            f"maximum error={maximum_error:.3e}."
        )
    # Assign the tiny transform round-trip discrepancy to the local correction
    # so the displayed components sum exactly to the saved model prediction.
    residual += two_scale_prediction - recomposed

    patch_size = int(encoder.patch_size)
    stride = int(encoder.stride)
    coarse_factor = int(encoder.coarse_factor)
    include_edges = bool(getattr(encoder, "include_edges", False))
    coarse_grid = (
        restrict_average(torch.as_tensor(coarse), coarse_factor).detach().cpu().numpy()
    )

    paths: list[Path] = []
    paths.extend(
        _plot_decomposition(
            truth=truth,
            coarse_grid=coarse_grid,
            coarse=coarse,
            residual=residual,
            prediction=two_scale_prediction,
            patch_size=patch_size,
            stride=stride,
            include_edges=include_edges,
            dataset=dataset,
            sample_id=sample_id,
            output=output_path.with_name(f"{output_path.stem}_decomposition"),
        )
    )
    paths.extend(
        _plot_seam_comparison(
            truth=truth,
            plain_prediction=plain_prediction,
            two_scale_prediction=two_scale_prediction,
            patch_size=patch_size,
            stride=stride,
            include_edges=include_edges,
            dataset=dataset,
            sample_id=sample_id,
            output=output_path.with_name(f"{output_path.stem}_seam_comparison"),
        )
    )
    return paths


def _plot_decomposition(
    *,
    truth: np.ndarray,
    coarse_grid: np.ndarray,
    coarse: np.ndarray,
    residual: np.ndarray,
    prediction: np.ndarray,
    patch_size: int,
    stride: int,
    include_edges: bool,
    dataset: str,
    sample_id: int,
    output: Path,
) -> list[Path]:
    solution_min = min(float(np.min(truth)), float(np.min(prediction)), float(np.min(coarse)))
    solution_max = max(float(np.max(truth)), float(np.max(prediction)), float(np.max(coarse)))
    residual_limit = max(float(np.max(np.abs(residual))), 1.0e-12)
    error = np.abs(prediction - truth)
    error_limit = max(float(np.max(error)), 1.0e-12)
    mre = _relative_error(prediction, truth)
    residual_ratio = float(np.linalg.norm(residual)) / max(
        float(np.linalg.norm(prediction)), 1.0e-12
    )

    fig, axes = plt.subplots(1, 6, figsize=(17.2, 3.45), constrained_layout=True)
    panels = (
        (truth, "1. Ground truth", "RdBu_r", solution_min, solution_max),
        (
            coarse_grid,
            f"2. Average down\n{coarse_grid.shape[0]} x {coarse_grid.shape[1]} grid",
            "RdBu_r",
            solution_min,
            solution_max,
        ),
        (
            coarse,
            "3. Global PCA + bilinear up\ncontinuous domain-wide carrier",
            "RdBu_r",
            solution_min,
            solution_max,
        ),
        (
            residual,
            f"4. Local residual patches\nonly {residual_ratio:.1%} of field norm",
            "RdBu_r",
            -residual_limit,
            residual_limit,
        ),
        (
            prediction,
            f"5. Add both scales\nMRE {mre:.4f}",
            "RdBu_r",
            solution_min,
            solution_max,
        ),
        (error, "6. Absolute error", "magma", 0.0, error_limit),
    )
    for index, (field, title, cmap, lower, upper) in enumerate(panels):
        image = axes[index].imshow(
            field,
            origin="lower",
            cmap=cmap,
            vmin=lower,
            vmax=upper,
            interpolation="nearest" if index == 1 else "antialiased",
        )
        axes[index].set_title(title)
        axes[index].set_xticks([])
        axes[index].set_yticks([])
        fig.colorbar(image, ax=axes[index], fraction=0.046, pad=0.02)
    _draw_patch_grid(
        axes[3],
        (int(residual.shape[0]), int(residual.shape[1])),
        patch_size,
        stride,
        include_edges=include_edges,
    )
    fig.suptitle(
        f"{dataset.capitalize()} 256 two-scale reconstruction | seed 0 | "
        f"sample {sample_id}\n"
        r"$\hat{u}=$ prolongated global PCA field $+$ patchwise residual correction. "
        "The patches no longer carry the full solution.",
        fontsize=13,
    )
    return _save(fig, output)


def _plot_seam_comparison(
    *,
    truth: np.ndarray,
    plain_prediction: np.ndarray,
    two_scale_prediction: np.ndarray,
    patch_size: int,
    stride: int,
    include_edges: bool,
    dataset: str,
    sample_id: int,
    output: Path,
) -> list[Path]:
    predictions = (plain_prediction, two_scale_prediction)
    labels = ("Plain L2L\npatches predict full field", "Two-scale\npatches predict residual only")
    solution_min = min(float(np.min(truth)), *(float(np.min(field)) for field in predictions))
    solution_max = max(float(np.max(truth)), *(float(np.max(field)) for field in predictions))
    errors = tuple(np.abs(field - truth) for field in predictions)
    error_limit = max(*(float(np.max(error)) for error in errors), 1.0e-12)
    seam_maps = tuple(
        _seam_error_jump_map(
            field - truth,
            patch_size,
            stride,
            include_edges=include_edges,
        )
        for field in predictions
    )
    seam_limit = max(*(float(np.max(field)) for field in seam_maps), 1.0e-12)
    jumps = tuple(
        interface_jump(
            field[None, :, :],
            patch_size,
            stride,
            include_edges=include_edges,
        )
        for field in predictions
    )
    reduction = 100.0 * (1.0 - jumps[1] / max(jumps[0], 1.0e-12))

    fig, axes = plt.subplots(2, 3, figsize=(10.6, 7.0), constrained_layout=True)
    for row, (prediction, error, seam_map, label, jump) in enumerate(
        zip(predictions, errors, seam_maps, labels, jumps)
    ):
        prediction_image = axes[row, 0].imshow(
            prediction,
            origin="lower",
            cmap="RdBu_r",
            vmin=solution_min,
            vmax=solution_max,
        )
        error_image = axes[row, 1].imshow(
            error,
            origin="lower",
            cmap="magma",
            vmin=0.0,
            vmax=error_limit,
        )
        seam_image = axes[row, 2].imshow(
            seam_map,
            origin="lower",
            cmap="magma",
            vmin=0.0,
            vmax=seam_limit,
        )
        axes[row, 0].set_ylabel(
            f"{label}\ninterface jump {jump:.2e}",
            fontsize=10,
            labelpad=8,
        )
        _draw_patch_grid(
            axes[row, 0],
            (int(prediction.shape[0]), int(prediction.shape[1])),
            patch_size,
            stride,
            include_edges=include_edges,
        )
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
        fig.colorbar(prediction_image, ax=axes[row, 0], fraction=0.046, pad=0.02)
        fig.colorbar(error_image, ax=axes[row, 1], fraction=0.046, pad=0.02)
        fig.colorbar(seam_image, ax=axes[row, 2], fraction=0.046, pad=0.02)
    for axis, title in zip(
        axes[0],
        ("Prediction + patch grid", "Absolute error", "Error jump only at patch seams"),
    ):
        axis.set_title(title)
    fig.suptitle(
        f"{dataset.capitalize()} 256 seam comparison | seed 0 | sample {sample_id}\n"
        f"Two-scale interface-jump reduction on this sample: {reduction:.1f}%",
        fontsize=13,
    )
    return _save(fig, output)


def _prediction_path(results_dir: Path, method: str) -> Path:
    return results_dir / "resolution_256" / method / "seed_0" / "predictions_test.npz"


def _representative_sample_id(path: Path) -> int:
    with np.load(path, allow_pickle=False) as data:
        truth = np.asarray(data["y_true"], dtype=np.float32)
        prediction = np.asarray(data["y_pred"], dtype=np.float32)
        sample_indices = np.asarray(data["sample_indices"], dtype=np.int64)
    relative_errors = np.linalg.norm(
        (prediction - truth).reshape(len(truth), -1), axis=1
    ) / np.maximum(np.linalg.norm(truth.reshape(len(truth), -1), axis=1), 1.0e-12)
    median = float(np.median(relative_errors))
    local_index = int(np.argmin(np.abs(relative_errors - median)))
    return int(sample_indices[local_index])


def _load_sample(path: Path, sample_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        sample_indices = np.asarray(data["sample_indices"], dtype=np.int64)
        matches = np.flatnonzero(sample_indices == sample_id)
        if matches.size != 1:
            raise ValueError(f"Expected sample {sample_id} exactly once in {path}.")
        index = int(matches[0])
        return (
            np.asarray(data["x_input"][index], dtype=np.float64),
            np.asarray(data["y_true"][index], dtype=np.float64),
            np.asarray(data["y_pred"][index], dtype=np.float64),
        )


def _seam_error_jump_map(
    error: np.ndarray,
    patch_size: int,
    stride: int,
    *,
    include_edges: bool,
) -> np.ndarray:
    rows, cols = interface_boundaries(
        (int(error.shape[0]), int(error.shape[1])),
        patch_size,
        stride,
        include_edges=include_edges,
    )
    seam_map = np.zeros_like(error, dtype=np.float64)
    for row in rows:
        seam_map[row, :] = np.abs(error[row, :] - error[row - 1, :])
    for col in cols:
        seam_map[:, col] = np.maximum(
            seam_map[:, col],
            np.abs(error[:, col] - error[:, col - 1]),
        )
    return seam_map


def _draw_patch_grid(
    axis: plt.Axes,
    shape: tuple[int, int],
    patch_size: int,
    stride: int,
    *,
    include_edges: bool,
) -> None:
    rows, cols = interface_boundaries(
        shape,
        patch_size,
        stride,
        include_edges=include_edges,
    )
    for row in rows:
        axis.axhline(row - 0.5, color="white", linewidth=0.7, alpha=0.8)
    for col in cols:
        axis.axvline(col - 0.5, color="white", linewidth=0.7, alpha=0.8)


def _relative_error(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(np.linalg.norm((prediction - truth).ravel())) / max(
        float(np.linalg.norm(truth.ravel())), 1.0e-12
    )


def _save(fig: plt.Figure, output: Path) -> list[Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = [output.with_suffix(".png"), output.with_suffix(".pdf")]
    for path in paths:
        fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("poisson", "darcy"), required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output", required=True, help="Output base path without suffix.")
    parser.add_argument("--sample-id", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = plot_two_scale_explainer(
        results_dir=args.results_dir,
        output=args.output,
        dataset=args.dataset,
        sample_id=args.sample_id,
    )
    for path in paths:
        print(f"Wrote figure: {path}")


if __name__ == "__main__":
    main()
