"""Shared differentiable metric kernels.

These functions are the single torch implementation used by physical-field
losses and by NumPy metric wrappers where exact loss/metric parity matters.
Inputs are assembled scalar fields with shape ``(N, H, W)`` or ``(H, W)``.
"""

from __future__ import annotations

import torch

from lpcanet.assembly.patching import get_patch_slices


def as_batch_field(field: torch.Tensor, name: str) -> torch.Tensor:
    """Return ``field`` with an explicit batch dimension."""
    if field.ndim == 2:
        field = field.unsqueeze(0)
    if field.ndim != 3:
        raise ValueError(f"{name} must have shape (N, H, W) or (H, W), got {tuple(field.shape)}.")
    return field


def broadcast_field(
    field: torch.Tensor | float,
    shape: tuple[int, int, int],
    *,
    dtype: torch.dtype,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    """Broadcast a scalar, 2D field, or batch field to ``shape``."""
    tensor = torch.as_tensor(field, dtype=dtype, device=device)
    if tensor.ndim == 0:
        return tensor.reshape(1, 1, 1).expand(shape)
    if tensor.ndim == 2:
        return tensor.unsqueeze(0).expand(shape)
    if tensor.ndim == 3 and tuple(tensor.shape[1:]) == (1, 1) and tensor.shape[0] == shape[0]:
        return tensor.expand(shape)
    if tuple(tensor.shape) == shape:
        return tensor
    raise ValueError(f"{name} must be broadcastable to {shape}, got {tuple(tensor.shape)}.")


def relative_l2_torch(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Global relative L2 error over the full batch."""
    pred, true = matching_batch_fields(pred, true)
    numerator = torch.linalg.vector_norm((pred - true).reshape(pred.shape[0], -1))
    denominator = torch.linalg.vector_norm(true.reshape(true.shape[0], -1)).clamp_min(eps)
    return numerator / denominator


def mse_torch(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Mean squared error over all samples and pixels."""
    pred, true = matching_batch_fields(pred, true)
    return torch.mean((pred - true) ** 2)


def matching_batch_fields(pred: torch.Tensor, true: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate and return matching batch fields."""
    pred = as_batch_field(pred, "pred")
    true = as_batch_field(true, "true")
    if pred.shape != true.shape:
        raise ValueError(f"Prediction and target shapes differ: {tuple(pred.shape)} != {tuple(true.shape)}.")
    return pred, true


def interface_boundaries(
    grid_shape: tuple[int, int],
    patch_size: int,
    stride: int,
    *,
    include_edges: bool = False,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return sorted row/column seam indices induced by the patch grid."""
    height, width = int(grid_shape[0]), int(grid_shape[1])
    slices = get_patch_slices((height, width), patch_size, stride, include_edges=include_edges)
    row_boundaries: set[int] = set()
    col_boundaries: set[int] = set()
    for row_slice, col_slice in slices:
        for boundary in (row_slice.start, row_slice.stop):
            if 0 < boundary < height:
                row_boundaries.add(int(boundary))
        for boundary in (col_slice.start, col_slice.stop):
            if 0 < boundary < width:
                col_boundaries.add(int(boundary))
    return tuple(sorted(row_boundaries)), tuple(sorted(col_boundaries))


def interface_value_jump_torch(
    field: torch.Tensor,
    patch_size: int,
    stride: int,
    *,
    include_edges: bool = False,
) -> torch.Tensor:
    """Mean absolute value jump across stride-induced seams."""
    field = as_batch_field(field, "field")
    _, height, width = field.shape
    rows, cols = interface_boundaries((height, width), patch_size, stride, include_edges=include_edges)
    jumps: list[torch.Tensor] = []
    for row in rows:
        jumps.append(torch.abs(field[:, row, :] - field[:, row - 1, :]).reshape(-1))
    for col in cols:
        jumps.append(torch.abs(field[:, :, col] - field[:, :, col - 1]).reshape(-1))
    if not jumps:
        return field.sum() * 0.0
    return torch.cat(jumps).mean()


def interface_value_traces_torch(
    field: torch.Tensor,
    patch_size: int,
    stride: int,
    *,
    include_edges: bool = False,
) -> torch.Tensor:
    """Return signed adjacent-pixel differences across stride-induced seams."""
    field = as_batch_field(field, "field")
    _, height, width = field.shape
    rows, cols = interface_boundaries((height, width), patch_size, stride, include_edges=include_edges)
    traces: list[torch.Tensor] = []
    for row in rows:
        traces.append((field[:, row, :] - field[:, row - 1, :]).reshape(field.shape[0], -1))
    for col in cols:
        traces.append((field[:, :, col] - field[:, :, col - 1]).reshape(field.shape[0], -1))
    if not traces:
        return field.new_zeros((field.shape[0], 0))
    return torch.cat(traces, dim=1)


def interface_flux_jump_torch(
    field: torch.Tensor,
    patch_size: int,
    stride: int,
    *,
    dx: float,
    include_edges: bool = False,
) -> torch.Tensor:
    """Mean absolute normal first-derivative jump across stride-induced seams."""
    if dx <= 0.0:
        raise ValueError("dx must be positive.")
    field = as_batch_field(field, "field")
    _, height, width = field.shape
    rows, cols = interface_boundaries((height, width), patch_size, stride, include_edges=include_edges)
    jumps: list[torch.Tensor] = []
    for row in rows:
        if 0 < row < height - 1:
            upper_flux = (field[:, row, :] - field[:, row - 1, :]) / dx
            lower_flux = (field[:, row + 1, :] - field[:, row, :]) / dx
            jumps.append(torch.abs(lower_flux - upper_flux).reshape(-1))
    for col in cols:
        if 0 < col < width - 1:
            left_flux = (field[:, :, col] - field[:, :, col - 1]) / dx
            right_flux = (field[:, :, col + 1] - field[:, :, col]) / dx
            jumps.append(torch.abs(right_flux - left_flux).reshape(-1))
    if not jumps:
        return field.sum() * 0.0
    return torch.cat(jumps).mean()


def interface_flux_traces_torch(
    field: torch.Tensor,
    patch_size: int,
    stride: int,
    *,
    dx: float,
    include_edges: bool = False,
) -> torch.Tensor:
    """Return signed normal-derivative differences across stride-induced seams."""
    if dx <= 0.0:
        raise ValueError("dx must be positive.")
    field = as_batch_field(field, "field")
    _, height, width = field.shape
    rows, cols = interface_boundaries((height, width), patch_size, stride, include_edges=include_edges)
    traces: list[torch.Tensor] = []
    for row in rows:
        if 0 < row < height - 1:
            upper_flux = (field[:, row, :] - field[:, row - 1, :]) / dx
            lower_flux = (field[:, row + 1, :] - field[:, row, :]) / dx
            traces.append((lower_flux - upper_flux).reshape(field.shape[0], -1))
    for col in cols:
        if 0 < col < width - 1:
            left_flux = (field[:, :, col] - field[:, :, col - 1]) / dx
            right_flux = (field[:, :, col + 1] - field[:, :, col]) / dx
            traces.append((right_flux - left_flux).reshape(field.shape[0], -1))
    if not traces:
        return field.new_zeros((field.shape[0], 0))
    return torch.cat(traces, dim=1)


def poisson_residual_torch(
    u_pred: torch.Tensor,
    f: torch.Tensor | float,
    dx: float,
) -> torch.Tensor:
    """Interior residual for ``-Delta u = f``."""
    if dx <= 0.0:
        raise ValueError("dx must be positive.")
    u_pred = as_batch_field(u_pred, "u_pred")
    if u_pred.shape[1] < 3 or u_pred.shape[2] < 3:
        raise ValueError(f"u_pred grid must be at least 3x3, got {tuple(u_pred.shape)}.")
    shape = (int(u_pred.shape[0]), int(u_pred.shape[1]), int(u_pred.shape[2]))
    forcing = broadcast_field(f, shape, dtype=u_pred.dtype, device=u_pred.device, name="f")
    laplacian = (
        u_pred[:, 2:, 1:-1]
        + u_pred[:, :-2, 1:-1]
        + u_pred[:, 1:-1, 2:]
        + u_pred[:, 1:-1, :-2]
        - 4.0 * u_pred[:, 1:-1, 1:-1]
    ) / (dx**2)
    return -laplacian - forcing[:, 1:-1, 1:-1]


def darcy_residual_torch(
    u_pred: torch.Tensor,
    a: torch.Tensor | float,
    f: torch.Tensor | float,
    dx: float,
) -> torch.Tensor:
    """Interior residual for ``-div(a grad u) = f``."""
    if dx <= 0.0:
        raise ValueError("dx must be positive.")
    u_pred = as_batch_field(u_pred, "u_pred")
    if u_pred.shape[1] < 3 or u_pred.shape[2] < 3:
        raise ValueError(f"u_pred grid must be at least 3x3, got {tuple(u_pred.shape)}.")
    shape = (int(u_pred.shape[0]), int(u_pred.shape[1]), int(u_pred.shape[2]))
    coeff = broadcast_field(a, shape, dtype=u_pred.dtype, device=u_pred.device, name="a")
    forcing = broadcast_field(f, shape, dtype=u_pred.dtype, device=u_pred.device, name="f")

    center = u_pred[:, 1:-1, 1:-1]
    coeff_center = coeff[:, 1:-1, 1:-1]
    coeff_e = 0.5 * (coeff_center + coeff[:, 1:-1, 2:])
    coeff_w = 0.5 * (coeff_center + coeff[:, 1:-1, :-2])
    coeff_n = 0.5 * (coeff_center + coeff[:, :-2, 1:-1])
    coeff_s = 0.5 * (coeff_center + coeff[:, 2:, 1:-1])

    flux_div = (
        coeff_e * (u_pred[:, 1:-1, 2:] - center)
        - coeff_w * (center - u_pred[:, 1:-1, :-2])
        + coeff_s * (u_pred[:, 2:, 1:-1] - center)
        - coeff_n * (center - u_pred[:, :-2, 1:-1])
    ) / (dx**2)
    return -flux_div - forcing[:, 1:-1, 1:-1]


def residual_norm_torch(residual: torch.Tensor, reduction: str = "rms") -> torch.Tensor:
    """Reduce a residual field to a scalar diagnostic."""
    if reduction == "rms":
        return torch.sqrt(torch.mean(residual**2))
    if reduction == "mse":
        return torch.mean(residual**2)
    if reduction == "mean_abs":
        return torch.mean(torch.abs(residual))
    raise ValueError(f"Unsupported residual reduction {reduction!r}.")


def radial_energy_spectrum_torch(fields: torch.Tensor) -> torch.Tensor:
    """Radially averaged 2D energy spectrum for a batch of fields."""
    fields = as_batch_field(fields, "fields")
    _, height, width = fields.shape
    fft_fields = torch.fft.fft2(fields)
    energy = torch.abs(fft_fields) ** 2
    radii = _radial_bin_indices(height, width, device=fields.device)
    max_radius = int(torch.max(radii).item())
    flat_radii = radii.reshape(1, -1).expand(fields.shape[0], -1)
    flat_energy = energy.reshape(fields.shape[0], -1)
    spectrum = fields.new_zeros((fields.shape[0], max_radius + 1))
    spectrum.scatter_add_(1, flat_radii, flat_energy)
    counts = torch.bincount(radii.reshape(-1), minlength=max_radius + 1)
    return spectrum / counts.clamp_min(1).to(dtype=fields.dtype, device=fields.device)


def relative_spectrum_error_torch(
    pred: torch.Tensor,
    true: torch.Tensor,
    *,
    eps: float = 1e-12,
    high_k_weight_power: float = 0.0,
) -> torch.Tensor:
    """Relative L2 error between mean radial energy spectra."""
    pred, true = matching_batch_fields(pred, true)
    pred_mean = torch.mean(radial_energy_spectrum_torch(pred), dim=0)
    true_mean = torch.mean(radial_energy_spectrum_torch(true), dim=0)
    weights = _spectral_weights(
        pred_mean.shape[0],
        dtype=pred_mean.dtype,
        device=pred_mean.device,
        high_k_weight_power=high_k_weight_power,
    )
    numerator = torch.linalg.vector_norm((pred_mean - true_mean) * weights)
    denominator = torch.linalg.vector_norm(true_mean * weights).clamp_min(eps)
    return numerator / denominator


def _radial_bin_indices(height: int, width: int, *, device: torch.device) -> torch.Tensor:
    ky = torch.fft.fftfreq(height, device=device) * height
    kx = torch.fft.fftfreq(width, device=device) * width
    grid_ky, grid_kx = torch.meshgrid(ky, kx, indexing="ij")
    return torch.round(torch.sqrt(grid_kx**2 + grid_ky**2)).to(dtype=torch.long)


def _spectral_weights(
    size: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    high_k_weight_power: float,
) -> torch.Tensor:
    if high_k_weight_power == 0.0:
        return torch.ones(size, dtype=dtype, device=device)
    radii = torch.arange(size, dtype=dtype, device=device)
    scale = radii / radii[-1].clamp_min(1.0)
    return (1.0 + scale) ** high_k_weight_power
