"""Thin PCA-Net wrapper around latent neural mappings."""

from __future__ import annotations

import torch
from torch import nn

from lpcanet.models.mlp import MLP


class PCANet(nn.Module):
    """Map latent input codes to latent output codes.

    PCA fitting, transforms, and inverse transforms intentionally live outside
    this Torch model.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int = 128,
        num_layers: int = 4,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.network = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict latent output code from latent input code."""
        return self.network(x)
