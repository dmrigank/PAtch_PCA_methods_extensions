"""Evaluation, reconstruction, metrics, and diagnostics."""

from lpcanet.metrics.interface import (
    interface_flux_jump,
    interface_flux_trace_error,
    interface_jump,
    interface_value_jump,
    interface_value_trace_error,
)
from lpcanet.metrics.metrics import aggregate_metrics, mae, mre, mse, ssim_batch
from lpcanet.metrics.residuals import (
    PDE_METRICS_VERSION,
    POISSON_RESIDUAL_CONVENTION,
    darcy_residual,
    poisson_residual,
)
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
    "interface_flux_trace_error",
    "interface_jump",
    "interface_value_jump",
    "interface_value_trace_error",
    "mae",
    "mre",
    "mse",
    "PDE_METRICS_VERSION",
    "POISSON_RESIDUAL_CONVENTION",
    "poisson_residual",
    "relative_spectrum_error",
    "ssim_batch",
]
