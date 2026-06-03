"""Spectral diagnostics for 2D fields."""

from __future__ import annotations

import numpy as np


def energy_spectrum_2d(field: np.ndarray) -> np.ndarray:
    """Compute a radially averaged 2D energy spectrum for one scalar field."""
    field = np.asarray(field, dtype=np.float64)
    if field.ndim != 2:
        raise ValueError(f"field must have shape (H, W), got {field.shape}.")

    fft_field = np.fft.fft2(field)
    energy = np.abs(fft_field) ** 2
    height, width = field.shape
    ky = np.fft.fftfreq(height) * height
    kx = np.fft.fftfreq(width) * width
    grid_kx, grid_ky = np.meshgrid(kx, ky)
    radii = np.rint(np.sqrt(grid_kx**2 + grid_ky**2)).astype(np.int64)
    max_radius = int(radii.max())
    spectrum = np.bincount(radii.ravel(), weights=energy.ravel(), minlength=max_radius + 1)
    counts = np.bincount(radii.ravel(), minlength=max_radius + 1).clip(min=1)
    return (spectrum / counts).astype(np.float64)


def batch_energy_spectrum(fields: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and standard deviation of spectra over a batch."""
    fields = np.asarray(fields)
    if fields.ndim != 3:
        raise ValueError(f"fields must have shape (N, H, W), got {fields.shape}.")
    spectra = np.stack([energy_spectrum_2d(field) for field in fields], axis=0)
    return np.mean(spectra, axis=0), np.std(spectra, axis=0)


def relative_spectrum_error(pred: np.ndarray, true: np.ndarray, eps: float = 1e-12) -> float:
    """Relative L2 error between mean energy spectra."""
    pred_mean, _ = batch_energy_spectrum(pred)
    true_mean, _ = batch_energy_spectrum(true)
    numerator = float(np.linalg.norm(pred_mean - true_mean))
    denominator = max(float(np.linalg.norm(true_mean)), eps)
    return numerator / denominator
