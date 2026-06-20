"""Standard 2D Fourier Neural Operator baseline."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SpectralConv2d(nn.Module):
    """2D spectral convolution retaining low Fourier modes."""

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive.")
        if modes <= 0:
            raise ValueError("modes must be positive.")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes = int(modes)
        scale = 1.0 / float(in_channels * out_channels)
        self.weights_pos = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat)
        )
        self.weights_neg = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spectral convolution to ``(B, C, H, W)`` tensors."""
        batch_size, _, height, width = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        modes_y = min(self.modes, height)
        modes_x = min(self.modes, width // 2 + 1)
        out_ft[:, :, :modes_y, :modes_x] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :modes_y, :modes_x],
            self.weights_pos[:, :, :modes_y, :modes_x],
        )
        out_ft[:, :, -modes_y:, :modes_x] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, -modes_y:, :modes_x],
            self.weights_neg[:, :, :modes_y, :modes_x],
        )
        return torch.fft.irfft2(out_ft, s=(height, width))


class FNO2d(nn.Module):
    """Compact FNO-2D baseline for scalar elliptic maps."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 32,
        modes: int = 12,
        num_layers: int = 4,
        hidden_size: int = 128,
        padding: int = 8,
    ) -> None:
        super().__init__()
        if width <= 0 or hidden_size <= 0:
            raise ValueError("width and hidden_size must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if padding < 0:
            raise ValueError("padding must be non-negative.")
        self.padding = int(padding)
        self.input_proj = nn.Conv2d(in_channels + 2, width, kernel_size=1)
        self.spectral_layers = nn.ModuleList([SpectralConv2d(width, width, modes) for _ in range(num_layers)])
        self.pointwise_layers = nn.ModuleList([nn.Conv2d(width, width, kernel_size=1) for _ in range(num_layers)])
        self.output_proj = nn.Sequential(
            nn.Conv2d(width, hidden_size, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_size, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, C, H, W)`` input fields to output fields."""
        if x.ndim != 4:
            raise ValueError(f"Expected input with shape (B, C, H, W), got {tuple(x.shape)}.")
        x = torch.cat([x, _coordinate_grid(x)], dim=1)
        x = self.input_proj(x)
        if self.padding:
            x = F.pad(x, (0, self.padding, 0, self.padding))
        for spectral, pointwise in zip(self.spectral_layers, self.pointwise_layers):
            x = F.gelu(spectral(x) + pointwise(x))
        if self.padding:
            x = x[..., : -self.padding, : -self.padding]
        return self.output_proj(x)


def _coordinate_grid(x: torch.Tensor) -> torch.Tensor:
    batch_size, _, height, width = x.shape
    y = torch.linspace(0.0, 1.0, height, dtype=x.dtype, device=x.device)
    z = torch.linspace(0.0, 1.0, width, dtype=x.dtype, device=x.device)
    grid_y, grid_x = torch.meshgrid(y, z, indexing="ij")
    grid = torch.stack((grid_y, grid_x), dim=0).unsqueeze(0)
    return grid.expand(batch_size, -1, -1, -1)
