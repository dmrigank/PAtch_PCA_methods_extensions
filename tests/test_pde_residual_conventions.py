"""Generator-to-residual convention regression tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lpcanet.data.generation import (
    solve_darcy_dirichlet,
    solve_poisson_dirichlet_legacy,
)
from lpcanet.metrics.residuals import (
    POISSON_RESIDUAL_CONVENTION,
    darcy_residual,
    poisson_residual,
)
from lpcanet.metrics.torch_ops import poisson_residual_torch


def test_manufactured_poisson_solution_matches_legacy_convention() -> None:
    rng = np.random.default_rng(17)
    resolution = 12
    dx = 1.0 / float(resolution - 1)
    solution = np.zeros((2, resolution, resolution), dtype=np.float64)
    solution[:, 1:-1, 1:-1] = rng.normal(
        size=(2, resolution - 2, resolution - 2)
    )
    forcing = np.zeros_like(solution)
    forcing[:, 1:-1, 1:-1] = _laplacian(solution, dx)

    residual = poisson_residual(solution, forcing, dx)
    torch_residual = poisson_residual_torch(
        torch.tensor(solution, dtype=torch.float64),
        torch.tensor(forcing, dtype=torch.float64),
        dx,
    )

    np.testing.assert_allclose(residual, 0.0, atol=1.0e-12, rtol=0.0)
    np.testing.assert_allclose(torch_residual.numpy(), residual, atol=1.0e-12, rtol=0.0)

    old_residual = -_laplacian(solution, dx) - forcing[:, 1:-1, 1:-1]
    assert float(np.sqrt(np.mean(old_residual**2))) > 1.0


def test_legacy_poisson_generator_has_near_zero_residual() -> None:
    rng = np.random.default_rng(23)
    forcing = rng.normal(size=(3, 32, 32)).astype(np.float32)
    solution = solve_poisson_dirichlet_legacy(forcing)
    residual = poisson_residual(solution, forcing, 1.0 / 31.0)

    assert float(np.sqrt(np.mean(residual**2))) < 1.0e-4


def test_unsupported_poisson_convention_is_rejected() -> None:
    field = np.zeros((1, 5, 5), dtype=np.float64)
    with pytest.raises(ValueError, match="Unsupported Poisson convention"):
        poisson_residual(
            field,
            field,
            0.25,
            convention="negative_delta_u_equals_f",
        )
    with pytest.raises(ValueError, match="Unsupported Poisson convention"):
        poisson_residual_torch(
            torch.zeros((1, 5, 5), dtype=torch.float64),
            0.0,
            0.25,
            convention="negative_delta_u_equals_f",
        )


def test_darcy_generator_residual_remains_aligned() -> None:
    rng = np.random.default_rng(29)
    resolution = 24
    coefficient = np.where(
        rng.normal(size=(2, resolution, resolution)) >= 0.0,
        12.0,
        4.0,
    )
    solution = np.stack(
        [solve_darcy_dirichlet(sample) for sample in coefficient]
    )
    forcing = np.ones_like(solution)
    residual = darcy_residual(
        solution,
        coefficient,
        forcing,
        1.0 / float(resolution - 1),
    )

    assert float(np.sqrt(np.mean(residual**2))) < 1.0e-4
    assert POISSON_RESIDUAL_CONVENTION == "delta_u_equals_f"


def _laplacian(field: np.ndarray, dx: float) -> np.ndarray:
    return (
        field[:, 2:, 1:-1]
        + field[:, :-2, 1:-1]
        + field[:, 1:-1, 2:]
        + field[:, 1:-1, :-2]
        - 4.0 * field[:, 1:-1, 1:-1]
    ) / (dx**2)
