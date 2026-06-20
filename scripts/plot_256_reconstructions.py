#!/usr/bin/env python
"""Plot aligned qualitative reconstructions for a 256-grid method study."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lpcanet.metrics.spectral import energy_spectrum_2d

METHODS = (
    ("global_pca", "Global PCA-Net"),
    ("plain_l2l", "Plain L2L"),
    ("overlap_l2l", "L2L + overlap"),
    ("fno", "FNO"),
    ("two_scale", "Two-scale"),
    ("two_scale_in_loop", "Two-scale + in-loop"),
)
STAGE_SPECS = (
    ("pca_fit", "PCA fit", "#4C78A8"),
    ("latent_transform", "Latent transform", "#72B7B2"),
    ("nn_train", "NN training", "#F58518"),
    ("inference", "Inference", "#E45756"),
)


def _prediction_path(results_dir: Path, method: str) -> Path:
    return results_dir / "resolution_256" / method / "seed_0" / "predictions_test.npz"


def _select_representative_sample(results_dir: Path) -> tuple[int, int]:
    """Select the test sample nearest the median plain-L2L relative error."""
    path = _prediction_path(results_dir, "plain_l2l")
    with np.load(path, allow_pickle=False) as data:
        truth = np.asarray(data["y_true"], dtype=np.float32)
        prediction = np.asarray(data["y_pred"], dtype=np.float32)
        sample_indices = np.asarray(data["sample_indices"], dtype=np.int64)
    error_norm = np.linalg.norm((prediction - truth).reshape(len(truth), -1), axis=1)
    truth_norm = np.linalg.norm(truth.reshape(len(truth), -1), axis=1)
    relative_error = error_norm / np.maximum(truth_norm, 1.0e-12)
    median_error = float(np.median(relative_error))
    local_index = int(np.argmin(np.abs(relative_error - median_error)))
    return local_index, int(sample_indices[local_index])


def _load_sample(
    path: Path,
    *,
    sample_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        sample_indices = np.asarray(data["sample_indices"], dtype=np.int64)
        matches = np.flatnonzero(sample_indices == sample_id)
        if matches.size != 1:
            raise ValueError(f"Expected sample {sample_id} exactly once in {path}.")
        index = int(matches[0])
        input_field = np.asarray(data["x_input"][index], dtype=np.float32)
        truth = np.asarray(data["y_true"][index], dtype=np.float32)
        prediction = np.asarray(data["y_pred"][index], dtype=np.float32)
    return input_field, truth, prediction


def plot_reconstruction_comparison(
    *,
    results_dir: str | Path,
    output: str | Path,
    dataset: str = "poisson",
    sample_id: int | None = None,
) -> list[Path]:
    """Render ground truth, reconstructions, and absolute errors for all methods."""
    results_root = Path(results_dir)
    output_path = Path(output)
    if sample_id is None:
        _, sample_id = _select_representative_sample(results_root)

    dataset_label = dataset.capitalize()
    input_label = "Input forcing" if dataset == "poisson" else "Input coefficient"
    input_color_map = "RdBu_r" if dataset == "poisson" else "viridis"
    input_symmetric = dataset == "poisson"
    input_field: np.ndarray | None = None
    truth: np.ndarray | None = None
    predictions: list[np.ndarray] = []
    for method, _ in METHODS:
        path = _prediction_path(results_root, method)
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed prediction file: {path}")
        method_input, method_truth, prediction = _load_sample(path, sample_id=sample_id)
        if input_field is None:
            input_field = method_input
            truth = method_truth
        else:
            assert truth is not None
            if not np.allclose(method_truth, truth, rtol=0.0, atol=1.0e-7):
                raise ValueError(f"Ground truth mismatch for sample {sample_id} in {path}.")
        predictions.append(prediction)

    assert input_field is not None
    assert truth is not None
    errors = [np.abs(prediction - truth) for prediction in predictions]
    solution_limit = max(
        float(np.max(np.abs(truth))),
        *(float(np.max(np.abs(prediction))) for prediction in predictions),
        1.0e-12,
    )
    pooled_error = np.concatenate([error.ravel() for error in errors])
    positive_error = pooled_error[pooled_error > 0.0]
    error_floor = max(float(np.percentile(positive_error, 1.0)), 1.0e-8)
    error_limit = max(float(np.percentile(pooled_error, 99.5)), error_floor * 10.0)
    input_limit = max(float(np.max(np.abs(input_field))), 1.0e-12)

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(
        2,
        len(METHODS) + 1,
        figsize=(16.2, 5.15),
        constrained_layout=True,
        squeeze=False,
    )

    truth_image = axes[0, 0].imshow(
        truth,
        origin="lower",
        cmap="RdBu_r",
        vmin=-solution_limit,
        vmax=solution_limit,
    )
    axes[0, 0].set_title("Ground truth")
    input_image = axes[1, 0].imshow(
        input_field,
        origin="lower",
        cmap=input_color_map,
        vmin=-input_limit if input_symmetric else float(np.min(input_field)),
        vmax=input_limit,
    )
    axes[1, 0].set_title(input_label)

    error_image = None
    for column, ((_, label), prediction, error) in enumerate(
        zip(METHODS, predictions, errors),
        start=1,
    ):
        numerator = float(np.linalg.norm((prediction - truth).ravel()))
        denominator = max(float(np.linalg.norm(truth.ravel())), 1.0e-12)
        axes[0, column].imshow(
            prediction,
            origin="lower",
            cmap="RdBu_r",
            vmin=-solution_limit,
            vmax=solution_limit,
        )
        axes[0, column].set_title(f"{label}\nMRE {numerator / denominator:.4f}")
        error_image = axes[1, column].imshow(
            error,
            origin="lower",
            cmap="magma",
            norm=LogNorm(vmin=error_floor, vmax=error_limit),
        )
        axes[1, column].set_title("Absolute error")

    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    fig.colorbar(
        truth_image,
        ax=axes[0, :],
        location="right",
        shrink=0.92,
        label="Solution value",
    )
    fig.colorbar(
        input_image,
        ax=axes[1, 0],
        location="bottom",
        shrink=0.82,
        pad=0.03,
        label="Forcing value" if dataset == "poisson" else "Coefficient value",
    )
    assert error_image is not None
    fig.colorbar(
        error_image,
        ax=axes[1, 1:],
        location="bottom",
        shrink=0.92,
        pad=0.03,
        label=(
            "Absolute error (shared log scale; "
            f"{error_floor:.1e} to pooled 99.5th percentile {error_limit:.1e})"
        ),
    )
    fig.suptitle(
        f"{dataset_label} 256 reconstruction comparison | seed 0 | dataset sample {sample_id}"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    paths = [output_path.with_suffix(".png"), output_path.with_suffix(".pdf")]
    for path in paths:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_stage_costs(
    *,
    results_dir: str | Path,
    output: str | Path,
    dataset: str,
) -> list[Path]:
    """Plot cumulative end-to-end runtime split into measured pipeline stages."""
    results_root = Path(results_dir)
    stage_path = results_root / "stage_costs.csv"
    if not stage_path.is_file():
        raise FileNotFoundError(f"Missing stage-cost table: {stage_path}")
    rows = np.genfromtxt(stage_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    rows = np.atleast_1d(rows)
    by_method = {str(row["method"]): row for row in rows}
    missing = [method for method, _ in METHODS if method not in by_method]
    if missing:
        raise ValueError(f"Missing stage-cost rows for: {', '.join(missing)}")

    labels = [label for _, label in METHODS]
    x = np.arange(len(METHODS))
    total = np.asarray(
        [float(by_method[method]["end_to_end"]) for method, _ in METHODS],
        dtype=np.float64,
    )
    components = np.column_stack(
        [
            [float(by_method[method][stage]) for method, _ in METHODS]
            for stage, _, _ in STAGE_SPECS
        ]
    )
    overhead = np.maximum(total - np.sum(components, axis=1), 0.0)

    fig, ax = plt.subplots(figsize=(10.2, 5.2), constrained_layout=True)
    bottom = np.zeros(len(METHODS), dtype=np.float64)
    for column, (_, label, color) in enumerate(STAGE_SPECS):
        values = components[:, column]
        ax.bar(x, values, bottom=bottom, width=0.72, color=color, label=label)
        bottom += values
    ax.bar(
        x,
        overhead,
        bottom=bottom,
        width=0.72,
        color="#B9B9B9",
        label="Other pipeline",
    )
    for index, value in enumerate(total):
        ax.text(index, value + max(total) * 0.015, f"{value:.1f}s", ha="center", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylabel("Cumulative end-to-end wall time (s)")
    ax.set_title(f"{dataset.capitalize()} 256 pipeline cost by stage | seed 0")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    return _save_figure(fig, Path(output))


def plot_qualitative_diagnostics(
    *,
    results_dir: str | Path,
    output: str | Path,
    dataset: str,
    sample_id: int | None = None,
) -> list[Path]:
    """Plot truth, prediction, error, spectrum, and PDF for every method."""
    results_root = Path(results_dir)
    if sample_id is None:
        _, sample_id = _select_representative_sample(results_root)

    truth: np.ndarray | None = None
    predictions: list[np.ndarray] = []
    for method, _label in METHODS:
        path = _prediction_path(results_root, method)
        _input, method_truth, prediction = _load_sample(path, sample_id=sample_id)
        if truth is None:
            truth = method_truth
        else:
            if not np.allclose(method_truth, truth, rtol=0.0, atol=1.0e-7):
                raise ValueError(f"Ground truth mismatch for sample {sample_id} in {path}.")
        predictions.append(prediction)

    assert truth is not None
    errors = [np.abs(prediction - truth) for prediction in predictions]
    solution_min = min(
        float(np.min(truth)),
        *(float(np.min(prediction)) for prediction in predictions),
    )
    solution_max = max(
        float(np.max(truth)),
        *(float(np.max(prediction)) for prediction in predictions),
    )
    error_max = max(*(float(np.max(error)) for error in errors), 1.0e-12)
    pdf_min = solution_min
    pdf_max = solution_max
    if pdf_min == pdf_max:
        pdf_min -= 0.5
        pdf_max += 0.5
    pdf_bins = np.linspace(pdf_min, pdf_max, 40)
    spectrum_k_max = min(truth.shape) // 2

    fig, axes = plt.subplots(
        len(METHODS),
        5,
        figsize=(13.6, 2.55 * len(METHODS)),
        constrained_layout=True,
        squeeze=False,
    )
    columns = ("Ground truth", "Prediction", "Absolute error", "Energy spectrum", "Value PDF")
    for column, title in enumerate(columns):
        axes[0, column].set_title(title)

    for row, ((_method, label), prediction, error) in enumerate(
        zip(METHODS, predictions, errors)
    ):
        truth_image = axes[row, 0].imshow(
            truth,
            origin="lower",
            cmap="RdBu_r",
            vmin=solution_min,
            vmax=solution_max,
        )
        prediction_image = axes[row, 1].imshow(
            prediction,
            origin="lower",
            cmap="RdBu_r",
            vmin=solution_min,
            vmax=solution_max,
        )
        error_image = axes[row, 2].imshow(
            error,
            origin="lower",
            cmap="magma",
            vmin=0.0,
            vmax=error_max,
        )
        denominator = max(float(np.linalg.norm(truth.ravel())), 1.0e-12)
        mre = float(np.linalg.norm((prediction - truth).ravel())) / denominator
        axes[row, 1].text(
            0.03,
            0.97,
            f"MRE {mre:.4f}",
            transform=axes[row, 1].transAxes,
            ha="left",
            va="top",
            color="black",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.0},
        )
        for column in range(3):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
        axes[row, 0].set_ylabel(label)
        fig.colorbar(truth_image, ax=axes[row, 0], fraction=0.046, pad=0.02)
        fig.colorbar(prediction_image, ax=axes[row, 1], fraction=0.046, pad=0.02)
        fig.colorbar(error_image, ax=axes[row, 2], fraction=0.046, pad=0.02)

        _plot_sample_spectrum(
            axes[row, 3],
            truth,
            prediction,
            k_max=spectrum_k_max,
        )
        axes[row, 3].set_ylabel("Energy")
        _plot_sample_pdf(axes[row, 4], truth, prediction, bins=pdf_bins)
        axes[row, 4].set_ylabel("Density")
        if row == len(METHODS) - 1:
            axes[row, 3].set_xlabel("Wavenumber $k$")
            axes[row, 4].set_xlabel("Field value")

    fig.suptitle(
        f"{dataset.capitalize()} 256 qualitative diagnostics | seed 0 | "
        f"dataset sample {sample_id}"
    )
    return _save_figure(fig, Path(output))


def _plot_sample_spectrum(
    axis: plt.Axes,
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    k_max: int,
) -> None:
    truth_energy = energy_spectrum_2d(truth)
    prediction_energy = energy_spectrum_2d(prediction)
    stop = min(k_max, len(truth_energy) - 1, len(prediction_energy) - 1)
    k = np.arange(1, stop + 1)
    axis.plot(k, truth_energy[1 : stop + 1] + 1.0e-30, color="black", linewidth=1.8, label="Truth")
    axis.plot(
        k,
        prediction_energy[1 : stop + 1] + 1.0e-30,
        color="tab:red",
        linewidth=1.8,
        label="Prediction",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(1, max(stop, 2))
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False, fontsize=8)


def _plot_sample_pdf(
    axis: plt.Axes,
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    bins: np.ndarray,
) -> None:
    histogram_bins = bins.tolist()
    axis.hist(
        truth.ravel(),
        bins=histogram_bins,
        density=True,
        histtype="step",
        color="black",
        linewidth=1.8,
        label="Truth",
    )
    axis.hist(
        prediction.ravel(),
        bins=histogram_bins,
        density=True,
        histtype="step",
        color="tab:red",
        linewidth=1.8,
        label="Prediction",
    )
    axis.grid(True, alpha=0.25)


def _save_figure(fig: plt.Figure, output: Path) -> list[Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = [output.with_suffix(".png"), output.with_suffix(".pdf")]
    for path in paths:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return paths


def _companion_output(output: str | Path, suffix: str) -> Path:
    path = Path(output)
    marker = "_reconstruction_comparison"
    stem = path.stem
    if stem.endswith(marker):
        stem = stem[: -len(marker)]
    return path.with_name(f"{stem}_{suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default="results/poisson_256_seed0",
        help="Root of the completed 256-grid study.",
    )
    parser.add_argument("--dataset", choices=("poisson", "darcy"), default="poisson")
    parser.add_argument(
        "--output",
        default="results/figures/poisson_256_reconstruction_comparison",
        help="Output path without extension.",
    )
    parser.add_argument("--sample-id", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = plot_reconstruction_comparison(
        results_dir=args.results_dir,
        output=args.output,
        dataset=args.dataset,
        sample_id=args.sample_id,
    )
    paths.extend(
        plot_stage_costs(
            results_dir=args.results_dir,
            output=_companion_output(args.output, "stage_costs"),
            dataset=args.dataset,
        )
    )
    paths.extend(
        plot_qualitative_diagnostics(
            results_dir=args.results_dir,
            output=_companion_output(args.output, "qualitative_diagnostics"),
            dataset=args.dataset,
            sample_id=args.sample_id,
        )
    )
    for path in paths:
        print(f"Wrote figure: {path}")


if __name__ == "__main__":
    main()
