"""Finite-difference PDE residual diagnostics."""

from __future__ import annotations

import numpy as np

POISSON_RESIDUAL_CONVENTION = "delta_u_equals_f"
PDE_METRICS_VERSION = 2


def poisson_residual(
    u_pred: np.ndarray,
    f: np.ndarray,
    dx: float,
    *,
    convention: str = POISSON_RESIDUAL_CONVENTION,
) -> np.ndarray:
    """Return the interior residual for the legacy equation ``Delta u = f``.

    Boundary values are not evaluated; the returned shape is ``(N, H-2, W-2)``.
    """
    _validate_poisson_convention(convention)
    u_pred = _as_batch_field(u_pred, "u_pred")
    field_shape = (int(u_pred.shape[0]), int(u_pred.shape[1]), int(u_pred.shape[2]))
    f = _broadcast_field(f, field_shape, "f")
    _validate_dx(dx)
    laplacian = (
        u_pred[:, 2:, 1:-1]
        + u_pred[:, :-2, 1:-1]
        + u_pred[:, 1:-1, 2:]
        + u_pred[:, 1:-1, :-2]
        - 4.0 * u_pred[:, 1:-1, 1:-1]
    ) / (dx**2)
    return laplacian - f[:, 1:-1, 1:-1]


def darcy_residual(u_pred: np.ndarray, a: np.ndarray, f: np.ndarray, dx: float) -> np.ndarray:
    """Return interior residual for ``-div(a grad u) = f``.

    Uses centered second-order finite differences with arithmetic face averages
    for the coefficient. Boundary values are not evaluated; the returned shape
    is ``(N, H-2, W-2)``.
    """
    u_pred = _as_batch_field(u_pred, "u_pred")
    field_shape = (int(u_pred.shape[0]), int(u_pred.shape[1]), int(u_pred.shape[2]))
    a = _broadcast_field(a, field_shape, "a")
    f = _broadcast_field(f, field_shape, "f")
    _validate_dx(dx)

    center = u_pred[:, 1:-1, 1:-1]
    a_center = a[:, 1:-1, 1:-1]
    a_e = 0.5 * (a_center + a[:, 1:-1, 2:])
    a_w = 0.5 * (a_center + a[:, 1:-1, :-2])
    a_n = 0.5 * (a_center + a[:, :-2, 1:-1])
    a_s = 0.5 * (a_center + a[:, 2:, 1:-1])

    flux_div = (
        a_e * (u_pred[:, 1:-1, 2:] - center)
        - a_w * (center - u_pred[:, 1:-1, :-2])
        + a_s * (u_pred[:, 2:, 1:-1] - center)
        - a_n * (center - u_pred[:, :-2, 1:-1])
    ) / (dx**2)
    return -flux_div - f[:, 1:-1, 1:-1]


def _as_batch_field(array: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(array, dtype=np.float64)
    if array.ndim == 2:
        array = array[None, :, :]
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape (N, H, W) or (H, W), got {array.shape}.")
    if array.shape[1] < 3 or array.shape[2] < 3:
        raise ValueError(f"{name} grid must be at least 3x3 for residuals, got {array.shape}.")
    return array


def _broadcast_field(array: np.ndarray, shape: tuple[int, int, int], name: str) -> np.ndarray:
    array = np.asarray(array, dtype=np.float64)
    if array.ndim == 0:
        return np.full(shape, float(array), dtype=np.float64)
    if array.ndim == 2:
        array = np.broadcast_to(array[None, :, :], shape)
    if array.ndim == 3 and array.shape[1:] == (1, 1) and array.shape[0] == shape[0]:
        array = np.broadcast_to(array, shape)
    if array.shape != shape:
        raise ValueError(f"{name} must be broadcastable to {shape}, got {array.shape}.")
    return array


def _validate_dx(dx: float) -> None:
    if dx <= 0:
        raise ValueError("dx must be positive.")


def _validate_poisson_convention(convention: str) -> None:
    if convention != POISSON_RESIDUAL_CONVENTION:
        raise ValueError(
            "Unsupported Poisson convention "
            f"{convention!r}; expected {POISSON_RESIDUAL_CONVENTION!r}."
        )
