"""Configurable multilayer perceptrons for latent PCA-Net mappings."""

from __future__ import annotations

import torch
from torch import nn


def make_activation(name: str) -> nn.Module:
    """Create an activation module by name."""
    normalized = name.lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation {name!r}.")


class MLP(nn.Module):
    """Fully connected network.

    Defaults match the legacy notebook:
    ``input_dim -> 128 -> 128 -> 128 -> output_dim`` with ReLU activations.
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
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive.")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        layers: list[nn.Module] = []
        in_features = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(make_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden_size
        layers.append(nn.Linear(in_features, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map latent input codes to latent output codes."""
        return self.network(x)


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in a model."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
