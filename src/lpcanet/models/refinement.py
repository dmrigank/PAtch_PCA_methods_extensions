"""CNN refinement model for blocky local PCA reconstructions."""

from __future__ import annotations

import torch
from torch import nn

from lpcanet.models.mlp import make_activation


class RefinementNet(nn.Module):
    """Convolutional refinement network preserving ``(B, C, H, W)`` shape."""

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32,
        num_layers: int = 4,
        kernel_size: int = 3,
        activation: str = "relu",
        residual: bool = True,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive.")
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2.")
        if kernel_size not in (3, 5, 7):
            raise ValueError("kernel_size must be one of 3, 5, or 7.")

        padding = kernel_size // 2
        layers: list[nn.Module] = []
        current_channels = in_channels
        for _ in range(num_layers - 1):
            layers.append(
                nn.Conv2d(
                    current_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )
            layers.append(make_activation(activation))
            current_channels = hidden_channels
        layers.append(
            nn.Conv2d(
                current_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=padding,
            )
        )
        self.net = nn.Sequential(*layers)
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Refine a blocky field tensor with shape ``(B, C, H, W)``."""
        if x.ndim != 4:
            raise ValueError(f"Expected input with shape (B, C, H, W), got {tuple(x.shape)}.")
        correction = self.net(x)
        if self.residual:
            return x + correction
        return correction
