"""Evaluation, reconstruction, metrics, and diagnostics."""

from lpcanet.metrics.interface import interface_flux_jump, interface_jump, interface_value_jump
from lpcanet.metrics.metrics import aggregate_metrics, mae, mre, mse, ssim_batch
from lpcanet.metrics.residuals import darcy_residual, poisson_residual
from lpcanet.metrics.spectral import (
    batch_energy_spectrum,
    energy_spectrum_2d,
    relative_spectrum_error,
)

__all__ = [
    "aggregate_metrics",
    "batch_energy_spectrum",
    "darcy_residual",
    "energy_spectrum_2d",
    "interface_flux_jump",
    "interface_jump",
    "interface_value_jump",
    "mae",
    "mre",
    "mse",
    "poisson_residual",
    "relative_spectrum_error",
    "ssim_batch",
]
