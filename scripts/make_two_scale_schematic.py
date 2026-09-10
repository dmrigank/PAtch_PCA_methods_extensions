#!/usr/bin/env python
"""Render a compact manuscript schematic of two-scale patch PCA."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

INK = "#252525"
MUTED = "#666666"
RULE = "#B8B8B8"
INPUT = "#3B6FA1"
COARSE = "#D97706"
RESIDUAL = "#C84A5A"
OUTPUT = "#3E7C59"
LIGHT = "#F6F6F6"


def make_two_scale_schematic(
    *,
    output: str | Path,
    formats: Sequence[str] = ("png", "pdf", "svg"),
) -> list[Path]:
    """Create a two-panel, resolution-agnostic method schematic."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = _synthetic_fields()

    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.size": 8.5,
            "mathtext.fontset": "dejavusans",
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(13.2, 5.9))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    _panel_heading(
        ax,
        0.952,
        "a",
        "Output representation",
        "fitted once from training solutions",
    )
    _draw_representation_panel(ax, fields)

    ax.plot([0.018, 0.982], [0.515, 0.515], color=RULE, linewidth=0.8)
    _panel_heading(
        ax,
        0.488,
        "b",
        "Learned two-scale operator",
        "local input encoding, joint coarse and residual prediction",
    )
    _draw_inference_panel(ax, fields)

    fig.subplots_adjust(left=0.012, right=0.988, top=0.985, bottom=0.045)
    written: list[Path] = []
    for suffix in formats:
        path = output_path.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        written.append(path)
    plt.close(fig)
    return written


def _draw_representation_panel(ax: Axes, fields: dict[str, np.ndarray]) -> None:
    y, image_h = 0.655, 0.19

    _field_node(ax, fields["solution"], (0.025, y, 0.075, image_h), r"$u$", "fine solution")
    _arrow(ax, (0.102, 0.75), (0.125, 0.75))
    _operator(ax, (0.125, 0.69, 0.065, 0.12), r"$R$", "average\ndown", INPUT)

    _arrow(ax, (0.19, 0.75), (0.212, 0.75))
    _field_node(
        ax,
        fields["coarse_grid"],
        (0.212, 0.68, 0.065, 0.14),
        r"$u_c=Ru$",
        "coarse field",
        interpolation="nearest",
    )

    _arrow(ax, (0.279, 0.75), (0.302, 0.75))
    _operator(
        ax,
        (0.302, 0.68, 0.09, 0.14),
        "Global PCA",
        "randomized SVD\n" + r"$k_c$ modes",
        COARSE,
    )

    _arrow(ax, (0.392, 0.75), (0.415, 0.75))
    _mode_icon(ax, (0.416, 0.692), COARSE, count=3)
    ax.text(0.455, 0.665, r"$\Phi_c$", ha="center", va="top", fontsize=10, color=COARSE)

    _arrow(ax, (0.49, 0.75), (0.512, 0.75))
    _operator(
        ax,
        (0.512, 0.68, 0.09, 0.14),
        "Coarse decode",
        r"$\mu_c+\Phi_c y_c$" + "\n" + r"prolongate $P$",
        COARSE,
    )

    _arrow(ax, (0.602, 0.75), (0.624, 0.75))
    _field_node(
        ax,
        fields["coarse_backbone"],
        (0.624, y, 0.075, image_h),
        r"$\widetilde u$",
        "global backbone",
        border=COARSE,
    )

    _arrow(ax, (0.701, 0.75), (0.722, 0.75))
    _operator(
        ax,
        (0.722, 0.69, 0.072, 0.12),
        r"$r=u-\widetilde u$",
        "remove global\ncomponent",
        RESIDUAL,
    )

    _arrow(ax, (0.794, 0.75), (0.815, 0.75))
    _field_node(
        ax,
        fields["residual"],
        (0.815, y, 0.075, image_h),
        r"$r$",
        "fine residual",
        border=RESIDUAL,
        grid=True,
        symmetric=True,
    )

    _arrow(ax, (0.892, 0.75), (0.912, 0.75))
    _operator(ax, (0.912, 0.68, 0.067, 0.14), "Patch PCA", "local residual\nbases", RESIDUAL)

def _draw_inference_panel(ax: Axes, fields: dict[str, np.ndarray]) -> None:
    _field_node(
        ax,
        fields["input"],
        (0.025, 0.19, 0.075, 0.19),
        r"$a$ or $f$",
        "input field",
        border=INPUT,
        grid=True,
    )

    _arrow(ax, (0.102, 0.285), (0.126, 0.285))
    _operator(
        ax,
        (0.126, 0.225, 0.09, 0.12),
        "Local input PCA",
        r"$\{z_p\}$",
        INPUT,
    )

    _arrow(ax, (0.216, 0.285), (0.244, 0.285))
    _operator(
        ax,
        (0.244, 0.215, 0.105, 0.14),
        "Neural latent map",
        r"$\{z_p\}\mapsto(\widehat y_c,\{\widehat y_p^r\})$",
        INK,
    )

    split_x = 0.375
    _arrow(ax, (0.349, 0.285), (split_x, 0.285))
    ax.plot([split_x, split_x], [0.18, 0.39], color=INK, linewidth=1.0)
    ax.add_patch(Circle((split_x, 0.285), 0.0045, facecolor=INK, edgecolor="none"))

    _latent_node(ax, (0.401, 0.337, 0.07, 0.085), r"$\widehat y_c$", COARSE)
    _arrow(ax, (split_x, 0.38), (0.401, 0.38))
    _operator(
        ax,
        (0.498, 0.32, 0.105, 0.12),
        "Global decode",
        r"$P(\mu_c+\Phi_c\widehat y_c)$",
        COARSE,
    )
    _arrow(ax, (0.471, 0.38), (0.498, 0.38))
    _field_node(
        ax,
        fields["coarse_backbone"],
        (0.63, 0.31, 0.065, 0.14),
        r"$\widehat u_c$",
        "",
        border=COARSE,
    )
    _arrow(ax, (0.603, 0.38), (0.63, 0.38))

    _latent_node(ax, (0.401, 0.147, 0.07, 0.085), r"$\{\widehat y_p^r\}$", RESIDUAL)
    _arrow(ax, (split_x, 0.19), (0.401, 0.19))
    _operator(
        ax,
        (0.498, 0.13, 0.105, 0.12),
        "Local decode",
        "residual PCA\n+ mosaic",
        RESIDUAL,
    )
    _arrow(ax, (0.471, 0.19), (0.498, 0.19))
    _field_node(
        ax,
        fields["predicted_residual"],
        (0.63, 0.12, 0.065, 0.14),
        r"$\widehat r$",
        "",
        border=RESIDUAL,
        grid=True,
        symmetric=True,
    )
    _arrow(ax, (0.603, 0.19), (0.63, 0.19))

    plus = (0.735, 0.285)
    ax.add_patch(Circle(plus, 0.018, facecolor="white", edgecolor=INK, linewidth=1.1))
    ax.text(*plus, "+", ha="center", va="center", fontsize=13, fontweight="bold")
    _arrow(ax, (0.697, 0.38), (0.719, 0.302), connectionstyle="arc3,rad=0.10")
    _arrow(ax, (0.697, 0.19), (0.719, 0.268), connectionstyle="arc3,rad=-0.10")

    _arrow(ax, (0.753, 0.285), (0.787, 0.285))
    _field_node(
        ax,
        fields["prediction"],
        (0.787, 0.19, 0.075, 0.19),
        r"$\widehat u=\widehat u_c+\widehat r$",
        "full-field prediction",
        border=OUTPUT,
    )

    ax.annotate(
        "global structure crosses patch boundaries",
        xy=(0.66, 0.455),
        xytext=(0.88, 0.43),
        ha="center",
        va="center",
        fontsize=8.3,
        color=COARSE,
        arrowprops={"arrowstyle": "-", "color": COARSE, "linewidth": 0.8},
    )
def _panel_heading(ax: Axes, y: float, letter: str, title: str, note: str) -> None:
    ax.text(0.02, y, f"({letter})", ha="left", va="top", fontsize=11, fontweight="bold")
    ax.text(0.055, y, title, ha="left", va="top", fontsize=11, fontweight="bold")
    ax.text(0.98, y, note, ha="right", va="top", fontsize=8.5, color=MUTED)


def _operator(
    ax: Axes,
    bounds: tuple[float, float, float, float],
    title: str,
    detail: str,
    color: str,
) -> None:
    x, y, width, height = bounds
    ax.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            facecolor="white",
            edgecolor=color,
            linewidth=1.25,
            zorder=3,
        )
    )
    ax.add_patch(
        Rectangle(
            (x, y + height - 0.009),
            width,
            0.009,
            facecolor=color,
            edgecolor="none",
            zorder=4,
        )
    )
    ax.text(
        x + width / 2,
        y + height * 0.62,
        title,
        ha="center",
        va="center",
        fontsize=8.4,
        fontweight="bold",
        zorder=5,
    )
    ax.text(
        x + width / 2,
        y + height * 0.27,
        detail,
        ha="center",
        va="center",
        fontsize=7.4,
        color=MUTED,
        linespacing=1.15,
        zorder=5,
    )


def _latent_node(
    ax: Axes,
    bounds: tuple[float, float, float, float],
    label: str,
    color: str,
) -> None:
    x, y, width, height = bounds
    ax.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            facecolor=color,
            edgecolor=color,
            linewidth=1.0,
            alpha=0.92,
            zorder=3,
        )
    )
    ax.text(
        x + width / 2,
        y + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=9,
        color="white",
        fontweight="bold",
        zorder=4,
    )


def _field_node(
    ax: Axes,
    field: np.ndarray,
    bounds: tuple[float, float, float, float],
    symbol: str,
    description: str,
    *,
    border: str = INK,
    grid: bool = False,
    symmetric: bool = False,
    interpolation: str = "bilinear",
) -> None:
    x, y, width, height = bounds
    if symmetric:
        limit = max(float(np.max(np.abs(field))), 1.0e-12)
        vmin, vmax = -limit, limit
    else:
        vmin, vmax = float(np.min(field)), float(np.max(field))
    ax.imshow(
        field,
        extent=(x, x + width, y, y + height),
        origin="lower",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        interpolation=interpolation,
        aspect="auto",
        zorder=2,
    )
    ax.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            facecolor="none",
            edgecolor=border,
            linewidth=1.0,
            zorder=4,
        )
    )
    if grid:
        for fraction in (0.25, 0.5, 0.75):
            ax.plot(
                [x + fraction * width] * 2,
                [y, y + height],
                color="white",
                linewidth=0.35,
                alpha=0.8,
                zorder=4,
            )
            ax.plot(
                [x, x + width],
                [y + fraction * height] * 2,
                color="white",
                linewidth=0.35,
                alpha=0.8,
                zorder=4,
            )
    ax.text(
        x + width / 2,
        y - 0.016,
        symbol,
        ha="center",
        va="top",
        fontsize=9.5,
        color=border,
    )
    if description:
        ax.text(
            x + width / 2,
            y - 0.047,
            description,
            ha="center",
            va="top",
            fontsize=7.2,
            color=MUTED,
        )


def _mode_icon(ax: Axes, origin: tuple[float, float], color: str, *, count: int) -> None:
    x, y = origin
    for index in range(count):
        offset = index * 0.010
        ax.add_patch(
            Rectangle(
                (x + offset, y + offset),
                0.052,
                0.105,
                facecolor=LIGHT,
                edgecolor=color,
                linewidth=0.9,
                zorder=3 + index,
            )
        )
        ax.plot(
            [x + offset + 0.026] * 2,
            [y + offset + 0.012, y + offset + 0.093],
            color=color,
            linewidth=0.5,
            alpha=0.55,
            zorder=4 + index,
        )


def _arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    connectionstyle: str = "arc3",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.0,
            color=INK,
            connectionstyle=connectionstyle,
            zorder=6,
        )
    )


def _synthetic_fields() -> dict[str, np.ndarray]:
    size = 64
    coordinate = np.linspace(0.0, 1.0, size)
    xx, yy = np.meshgrid(coordinate, coordinate, indexing="xy")
    envelope = np.sin(np.pi * xx) * np.sin(np.pi * yy)
    coarse = envelope * (
        0.9 * np.sin(1.4 * np.pi * xx)
        - 0.65 * np.cos(1.2 * np.pi * yy)
        + 0.22 * np.sin(np.pi * (xx + yy))
    )
    residual = 0.055 * envelope * (
        np.sin(7.0 * np.pi * xx) * np.cos(5.0 * np.pi * yy)
        + 0.45 * np.cos(10.0 * np.pi * xx + 2.0 * np.pi * yy)
    )
    solution = coarse + residual
    coarse_grid = solution.reshape(16, 4, 16, 4).mean(axis=(1, 3))
    coarse_backbone = _resize_bilinear(coarse_grid, size)
    residual_exact = solution - coarse_backbone
    predicted_residual = 0.92 * residual_exact
    input_field = envelope * (
        np.sin(5.0 * np.pi * xx) + 0.7 * np.cos(6.0 * np.pi * yy)
    )
    return {
        "solution": solution,
        "coarse_grid": coarse_grid,
        "coarse_backbone": coarse_backbone,
        "residual": residual_exact,
        "predicted_residual": predicted_residual,
        "prediction": coarse_backbone + predicted_residual,
        "input": input_field,
    }


def _resize_bilinear(field: np.ndarray, size: int) -> np.ndarray:
    old = np.linspace(0.0, 1.0, field.shape[0])
    new = np.linspace(0.0, 1.0, size)
    rows = np.stack([np.interp(new, old, row) for row in field])
    return np.stack(
        [np.interp(new, old, rows[:, column]) for column in range(size)],
        axis=1,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="paper_results/manuscript/figures/two_scale_method_schematic",
        help="Output base path without a suffix.",
    )
    parser.add_argument(
        "--formats",
        default="png,pdf,svg",
        help="Comma-separated output formats.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    formats = tuple(piece.strip() for piece in args.formats.split(",") if piece.strip())
    for path in make_two_scale_schematic(output=args.output, formats=formats):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
