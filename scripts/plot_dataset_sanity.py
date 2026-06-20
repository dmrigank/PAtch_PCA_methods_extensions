#!/usr/bin/env python
"""Plot representative input and solution fields from a generated dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lpcanet.metrics.residuals import darcy_residual


def _sample_indices(data: np.lib.npyio.NpzFile, count: int) -> tuple[list[int], list[str]]:
    indices: list[int] = []
    labels: list[str] = []
    split_keys = (("train_idx", "train"), ("val_idx", "val"), ("test_idx", "test"))
    for key, label in split_keys:
        split = data[key]
        if split.size:
            indices.append(int(split[0]))
            labels.append(label)
    test_indices = data["test_idx"]
    offset = 1
    while len(indices) < count and offset < test_indices.size:
        indices.append(int(test_indices[offset]))
        labels.append("test")
        offset += 1
    return indices[:count], labels[:count]


def _poisson_residual(forcing: np.ndarray, solution: np.ndarray) -> np.ndarray:
    resolution = solution.shape[-1]
    h = 1.0 / float(resolution - 1)
    laplacian = (
        solution[:, 2:, 1:-1]
        + solution[:, :-2, 1:-1]
        + solution[:, 1:-1, 2:]
        + solution[:, 1:-1, :-2]
        - 4.0 * solution[:, 1:-1, 1:-1]
    ) / h**2
    return laplacian - forcing[:, 1:-1, 1:-1]


def plot_dataset_sanity(
    dataset_path: str | Path,
    *,
    dataset: str,
    output: str | Path,
    count: int = 4,
) -> Path:
    """Render input/solution pairs sampled across train, validation, and test."""
    dataset_path = Path(dataset_path)
    output_path = Path(output)
    input_key = "f" if dataset == "poisson" else "a"
    input_label = "Forcing $f$" if dataset == "poisson" else "Coefficient $a$"

    with np.load(dataset_path, allow_pickle=False) as data:
        indices, split_labels = _sample_indices(data, count)
        inputs = np.asarray(data[input_key][indices], dtype=np.float32)
        solutions = np.asarray(data["u"][indices], dtype=np.float32)
        forcing = (
            np.asarray(data["f"][indices], dtype=np.float32)
            if dataset == "darcy"
            else inputs
        )

    if dataset == "poisson":
        residual = _poisson_residual(inputs, solutions)
    else:
        dx = 1.0 / float(solutions.shape[-1] - 1)
        residual = darcy_residual(solutions, inputs, forcing, dx)
    residual_rms = np.sqrt(np.mean(residual.astype(np.float64) ** 2, axis=(1, 2)))

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(len(indices), 2, figsize=(8.4, 3.1 * len(indices)))
    axes = np.atleast_2d(axes)
    input_limit = float(np.max(np.abs(inputs)))
    solution_limit = float(np.max(np.abs(solutions)))

    for row, (index, split_label) in enumerate(zip(indices, split_labels)):
        input_kwargs = (
            {"cmap": "RdBu_r", "vmin": -input_limit, "vmax": input_limit}
            if dataset == "poisson"
            else {"cmap": "viridis"}
        )
        input_image = axes[row, 0].imshow(inputs[row], origin="lower", **input_kwargs)
        solution_image = axes[row, 1].imshow(
            solutions[row],
            origin="lower",
            cmap="RdBu_r",
            vmin=-solution_limit,
            vmax=solution_limit,
        )
        axes[row, 0].set_title(f"{input_label} | {split_label} #{index}")
        solution_title = f"Solution $u$ | {split_label} #{index}"
        if residual_rms is not None:
            solution_title += f" | residual RMS {residual_rms[row]:.2e}"
        axes[row, 1].set_title(solution_title)
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
        fig.colorbar(input_image, ax=axes[row, 0], fraction=0.046, pad=0.03)
        fig.colorbar(solution_image, ax=axes[row, 1], fraction=0.046, pad=0.03)

    fig.suptitle(
        f"{dataset.capitalize()} dataset sanity check: "
        f"{solutions.shape[-2]} x {solutions.shape[-1]}",
        fontsize=13,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--dataset", choices=("poisson", "darcy"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = plot_dataset_sanity(
        args.dataset_path,
        dataset=args.dataset,
        output=args.output,
        count=args.count,
    )
    print(f"Wrote sanity figure: {output}")


if __name__ == "__main__":
    main()
