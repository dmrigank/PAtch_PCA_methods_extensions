"""Patch-wise neural network heads for PCA-Net ablations."""

from __future__ import annotations

import torch
from torch import nn

from lpcanet.models.mlp import MLP


class SingleConcatMLP(nn.Module):
    """Flatten all patch latents and map them with a single MLP."""

    def __init__(
        self,
        num_patches: int,
        input_dim_per_patch: int,
        output_dim_per_patch: int,
        hidden_size: int = 128,
        num_layers: int = 4,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _validate_patch_dims(num_patches, input_dim_per_patch, output_dim_per_patch)
        self.num_patches = num_patches
        self.input_dim_per_patch = input_dim_per_patch
        self.output_dim_per_patch = output_dim_per_patch
        self.model = MLP(
            input_dim=num_patches * input_dim_per_patch,
            output_dim=num_patches * output_dim_per_patch,
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, P, C_in)`` to ``(B, P, C_out)``."""
        _check_patch_input(x, self.num_patches, self.input_dim_per_patch)
        batch_size = x.shape[0]
        output = self.model(x.reshape(batch_size, -1))
        return output.reshape(batch_size, self.num_patches, self.output_dim_per_patch)


class IndependentPatchMLP(nn.Module):
    """Use a separate MLP for each patch position."""

    def __init__(
        self,
        num_patches: int,
        input_dim_per_patch: int,
        output_dim_per_patch: int,
        hidden_size: int = 128,
        num_layers: int = 4,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _validate_patch_dims(num_patches, input_dim_per_patch, output_dim_per_patch)
        self.num_patches = num_patches
        self.input_dim_per_patch = input_dim_per_patch
        self.output_dim_per_patch = output_dim_per_patch
        self.heads = nn.ModuleList(
            [
                MLP(
                    input_dim=input_dim_per_patch,
                    output_dim=output_dim_per_patch,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    activation=activation,
                    dropout=dropout,
                )
                for _ in range(num_patches)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, P, C_in)`` to ``(B, P, C_out)`` with patch-specific heads."""
        _check_patch_input(x, self.num_patches, self.input_dim_per_patch)
        outputs = [head(x[:, patch_index, :]) for patch_index, head in enumerate(self.heads)]
        return torch.stack(outputs, dim=1)


class SharedPatchMLP(nn.Module):
    """Apply the same MLP independently to every patch."""

    def __init__(
        self,
        num_patches: int,
        input_dim_per_patch: int,
        output_dim_per_patch: int,
        hidden_size: int = 128,
        num_layers: int = 4,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _validate_patch_dims(num_patches, input_dim_per_patch, output_dim_per_patch)
        self.num_patches = num_patches
        self.input_dim_per_patch = input_dim_per_patch
        self.output_dim_per_patch = output_dim_per_patch
        self.head = MLP(
            input_dim=input_dim_per_patch,
            output_dim=output_dim_per_patch,
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, P, C_in)`` to ``(B, P, C_out)`` with shared weights."""
        _check_patch_input(x, self.num_patches, self.input_dim_per_patch)
        batch_size = x.shape[0]
        flat = x.reshape(batch_size * self.num_patches, self.input_dim_per_patch)
        output = self.head(flat)
        return output.reshape(batch_size, self.num_patches, self.output_dim_per_patch)


class NeighborAwarePatchMLP(nn.Module):
    """Shared patch MLP that augments each patch with neighbor latents."""

    def __init__(
        self,
        grid_shape: tuple[int, int],
        input_dim_per_patch: int,
        output_dim_per_patch: int,
        neighborhood: int = 4,
        hidden_size: int = 128,
        num_layers: int = 4,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        rows, cols = _validate_patch_grid(grid_shape)
        if neighborhood not in (4, 8):
            raise ValueError("neighborhood must be 4 or 8.")
        _validate_patch_dims(rows * cols, input_dim_per_patch, output_dim_per_patch)
        self.grid_shape = (rows, cols)
        self.num_patches = rows * cols
        self.input_dim_per_patch = input_dim_per_patch
        self.output_dim_per_patch = output_dim_per_patch
        self.neighborhood = neighborhood
        self.offsets = _neighbor_offsets(neighborhood)
        context_width = 1 + len(self.offsets)
        self.head = MLP(
            input_dim=context_width * input_dim_per_patch,
            output_dim=output_dim_per_patch,
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, P, C_in)`` to ``(B, P, C_out)`` with neighbor context."""
        _check_patch_input(x, self.num_patches, self.input_dim_per_patch)
        batch_size = x.shape[0]
        rows, cols = self.grid_shape
        grid = x.reshape(batch_size, rows, cols, self.input_dim_per_patch)
        contexts: list[torch.Tensor] = []
        for row in range(rows):
            for col in range(cols):
                pieces = [grid[:, row, col, :]]
                for row_offset, col_offset in self.offsets:
                    n_row = row + row_offset
                    n_col = col + col_offset
                    if 0 <= n_row < rows and 0 <= n_col < cols:
                        pieces.append(grid[:, n_row, n_col, :])
                    else:
                        pieces.append(torch.zeros_like(grid[:, row, col, :]))
                contexts.append(torch.cat(pieces, dim=-1))
        context = torch.stack(contexts, dim=1)
        flat = context.reshape(batch_size * self.num_patches, -1)
        output = self.head(flat)
        return output.reshape(batch_size, self.num_patches, self.output_dim_per_patch)


class FlatPatchwiseHeadAdapter(nn.Module):
    """Adapt flat concatenated patch latents to fixed-width patchwise heads."""

    def __init__(
        self,
        head: nn.Module,
        input_component_counts: list[int],
        output_component_counts: list[int],
    ) -> None:
        super().__init__()
        if len(input_component_counts) != len(output_component_counts):
            raise ValueError("Input and output component count lists must have the same length.")
        if not input_component_counts:
            raise ValueError("Component count lists cannot be empty.")
        if any(count <= 0 for count in input_component_counts + output_component_counts):
            raise ValueError("All patch component counts must be positive.")
        self.head = head
        self.input_component_counts = [int(count) for count in input_component_counts]
        self.output_component_counts = [int(count) for count in output_component_counts]
        self.num_patches = len(self.input_component_counts)
        self.input_dim = int(sum(self.input_component_counts))
        self.output_dim = int(sum(self.output_component_counts))
        self.input_dim_per_patch = int(max(self.input_component_counts))
        self.output_dim_per_patch = int(max(self.output_component_counts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map flat ``(B, sum(C_in_p))`` latents to flat output latents."""
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"Expected flat input shape (B, {self.input_dim}), got {tuple(x.shape)}.")
        padded = x.new_zeros((x.shape[0], self.num_patches, self.input_dim_per_patch))
        start = 0
        for patch_index, width in enumerate(self.input_component_counts):
            stop = start + width
            padded[:, patch_index, :width] = x[:, start:stop]
            start = stop
        padded_output = self.head(padded)
        pieces = []
        for patch_index, width in enumerate(self.output_component_counts):
            pieces.append(padded_output[:, patch_index, :width])
        return torch.cat(pieces, dim=1)


def _validate_patch_dims(
    num_patches: int,
    input_dim_per_patch: int,
    output_dim_per_patch: int,
) -> None:
    if num_patches <= 0:
        raise ValueError("num_patches must be positive.")
    if input_dim_per_patch <= 0 or output_dim_per_patch <= 0:
        raise ValueError("input_dim_per_patch and output_dim_per_patch must be positive.")


def _validate_patch_grid(grid_shape: tuple[int, int]) -> tuple[int, int]:
    if len(grid_shape) != 2:
        raise ValueError("grid_shape must contain two entries.")
    rows, cols = int(grid_shape[0]), int(grid_shape[1])
    if rows <= 0 or cols <= 0:
        raise ValueError("grid_shape entries must be positive.")
    return rows, cols


def _check_patch_input(x: torch.Tensor, num_patches: int, input_dim_per_patch: int) -> None:
    if x.ndim != 3:
        raise ValueError(f"Expected input with shape (B, P, C), got {tuple(x.shape)}.")
    if x.shape[1] != num_patches or x.shape[2] != input_dim_per_patch:
        raise ValueError(
            f"Expected patch input shape (B, {num_patches}, {input_dim_per_patch}), "
            f"got {tuple(x.shape)}."
        )


def _neighbor_offsets(neighborhood: int) -> list[tuple[int, int]]:
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if neighborhood == 8:
        offsets.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
    return offsets
