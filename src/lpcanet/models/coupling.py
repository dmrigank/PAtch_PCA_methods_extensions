"""Patch-token coupling operators for ragged PCA latent codes."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from lpcanet.models.mlp import MLP, make_activation
from lpcanet.models.patchwise_heads import _neighbor_offsets


class CouplingOperator(nn.Module):
    """Map local input patch codes to coupled ragged output patch codes.

    The module accepts a flat concatenation of heterogeneous patch codes,
    encodes each patch code to a common token dimension, applies either GNN
    message passing or transformer attention over patch tokens, and decodes
    each token back to that patch's output-code length.
    """

    def __init__(
        self,
        *,
        input_component_counts: list[int],
        output_component_counts: list[int],
        grid_shape: tuple[int, int],
        embed_dim: int = 128,
        backend: str = "gnn",
        num_layers: int = 3,
        neighborhood: int = 8,
        attention_heads: int = 4,
        attention_n_max: int = 256,
        dropout: float = 0.0,
        activation: str = "gelu",
        global_output_dim: int = 0,
        local_residual: bool = False,
        local_hidden_size: int = 64,
        local_num_layers: int = 2,
        zero_init_correction: bool = False,
    ) -> None:
        super().__init__()
        rows, cols = _validate_grid(grid_shape)
        if rows * cols != len(input_component_counts):
            raise ValueError(
                f"grid_shape={grid_shape} implies {rows * cols} patches, "
                f"got {len(input_component_counts)} input counts."
            )
        if len(input_component_counts) != len(output_component_counts):
            raise ValueError("Input and output component count lists must have the same length.")
        if any(count <= 0 for count in input_component_counts + output_component_counts):
            raise ValueError("All component counts must be positive.")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if neighborhood not in (4, 8):
            raise ValueError("neighborhood must be 4 or 8.")
        if attention_n_max <= 0:
            raise ValueError("attention_n_max must be positive.")
        if global_output_dim < 0:
            raise ValueError("global_output_dim cannot be negative.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.input_component_counts = [int(count) for count in input_component_counts]
        self.output_component_counts = [int(count) for count in output_component_counts]
        self.grid_shape = (rows, cols)
        self.num_patches = rows * cols
        self.embed_dim = int(embed_dim)
        self.backend = backend.lower()
        self.global_output_dim = int(global_output_dim)
        self.local_residual = bool(local_residual)
        self.input_dim = int(sum(self.input_component_counts))
        self.output_dim = int(self.global_output_dim + sum(self.output_component_counts))

        self.input_encoders = nn.ModuleList(
            [nn.Linear(width, self.embed_dim) for width in self.input_component_counts]
        )
        self.output_decoders = nn.ModuleList(
            [nn.Linear(self.embed_dim, width) for width in self.output_component_counts]
        )
        if zero_init_correction:
            for decoder in self.output_decoders:
                nn.init.zeros_(decoder.weight)
                nn.init.zeros_(decoder.bias)
        self.local_predictors = nn.ModuleList(
            [
                MLP(
                    input_dim=input_width,
                    output_dim=output_width,
                    hidden_size=local_hidden_size,
                    num_layers=local_num_layers,
                    activation=activation,
                    dropout=dropout,
                )
                for input_width, output_width in zip(
                    self.input_component_counts,
                    self.output_component_counts,
                )
            ]
        ) if self.local_residual else nn.ModuleList()
        self.row_embedding = nn.Embedding(rows, self.embed_dim)
        self.col_embedding = nn.Embedding(cols, self.embed_dim)
        positions = torch.tensor(
            [(row, col) for row in range(rows) for col in range(cols)],
            dtype=torch.long,
        )
        self.register_buffer("positions", positions, persistent=False)

        if self.backend == "gnn":
            edge_index = _patch_edge_index((rows, cols), neighborhood)
            self.register_buffer("edge_index", edge_index, persistent=False)
            self.backend_module: nn.Module = GNNCouplingBackend(
                embed_dim=self.embed_dim,
                num_layers=num_layers,
                dropout=dropout,
                activation=activation,
            )
        elif self.backend == "attention":
            assert self.num_patches <= attention_n_max, (
                "Attention guardrail violated: "
                f"token count N={self.num_patches} exceeds N_max={attention_n_max}. "
                "Use backend='gnn' or increase model.coupling.attention_n_max deliberately."
            )
            self.register_buffer("edge_index", torch.empty((2, 0), dtype=torch.long), persistent=False)
            self.backend_module = AttentionCouplingBackend(
                embed_dim=self.embed_dim,
                num_layers=num_layers,
                num_heads=attention_heads,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unsupported coupling backend {backend!r}; expected 'gnn' or 'attention'.")

        self.global_decoder = (
            nn.Linear(self.embed_dim, self.global_output_dim)
            if self.global_output_dim > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map flat ragged input codes to flat ragged output codes."""
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"Expected flat input shape (B, {self.input_dim}), got {tuple(x.shape)}.")

        pieces = []
        input_pieces: list[torch.Tensor] = []
        start = 0
        for encoder, width in zip(self.input_encoders, self.input_component_counts):
            stop = start + width
            input_piece = x[:, start:stop]
            input_pieces.append(input_piece)
            pieces.append(encoder(input_piece))
            start = stop
        tokens = torch.stack(pieces, dim=1)
        tokens = tokens + self._positional_embedding()

        if self.backend == "gnn":
            coupled = self.backend_module(tokens, self.edge_index)
        else:
            coupled = self.backend_module(tokens)

        output_pieces: list[torch.Tensor] = []
        if self.global_decoder is not None:
            output_pieces.append(self.global_decoder(torch.mean(coupled, dim=1)))
        for patch_index, decoder in enumerate(self.output_decoders):
            correction = decoder(coupled[:, patch_index, :])
            if self.local_residual:
                correction = correction + self.local_predictors[patch_index](
                    input_pieces[patch_index]
                )
            output_pieces.append(correction)
        return torch.cat(output_pieces, dim=1)

    def _positional_embedding(self) -> torch.Tensor:
        rows = self.positions[:, 0]
        cols = self.positions[:, 1]
        return (self.row_embedding(rows) + self.col_embedding(cols)).unsqueeze(0)


class GNNInterfaceCorrection(nn.Module):
    """Zero-initialized GNN correction on top of a trained local L2L model.

    The wrapped ``base_model`` predicts the usual patch-latent output. A GNN then
    receives, for each patch, the local input code and the base output code and
    predicts a small residual correction in output-latent space. This makes the
    coupling path a seam/interface corrector rather than a replacement predictor.
    """

    def __init__(
        self,
        *,
        base_model: nn.Module,
        input_component_counts: list[int],
        output_component_counts: list[int],
        grid_shape: tuple[int, int],
        global_output_dim: int = 0,
        embed_dim: int = 128,
        num_layers: int = 3,
        neighborhood: int = 8,
        dropout: float = 0.0,
        activation: str = "gelu",
        freeze_base: bool = True,
        delta_scale: float = 1.0,
        delta_regularization: float = 0.0,
    ) -> None:
        super().__init__()
        rows, cols = _validate_grid(grid_shape)
        if rows * cols != len(input_component_counts):
            raise ValueError(
                f"grid_shape={grid_shape} implies {rows * cols} patches, "
                f"got {len(input_component_counts)} input counts."
            )
        if len(input_component_counts) != len(output_component_counts):
            raise ValueError("Input and output component count lists must have the same length.")
        if any(count <= 0 for count in input_component_counts + output_component_counts):
            raise ValueError("All component counts must be positive.")
        if global_output_dim < 0:
            raise ValueError("global_output_dim cannot be negative.")
        if delta_scale <= 0.0:
            raise ValueError("delta_scale must be positive.")
        if delta_regularization < 0.0:
            raise ValueError("delta_regularization must be non-negative.")

        self.base_model = base_model
        self.input_component_counts = [int(count) for count in input_component_counts]
        self.output_component_counts = [int(count) for count in output_component_counts]
        self.grid_shape = (rows, cols)
        self.num_patches = rows * cols
        self.global_output_dim = int(global_output_dim)
        self.embed_dim = int(embed_dim)
        self.input_dim = int(sum(self.input_component_counts))
        self.output_dim = int(self.global_output_dim + sum(self.output_component_counts))
        self.freeze_base = bool(freeze_base)
        self.delta_scale = float(delta_scale)
        self.delta_regularization = float(delta_regularization)
        self._last_delta: torch.Tensor | None = None

        if self.freeze_base:
            for parameter in self.base_model.parameters():
                parameter.requires_grad_(False)

        self.node_encoders = nn.ModuleList(
            [
                nn.Linear(input_width + output_width, self.embed_dim)
                for input_width, output_width in zip(
                    self.input_component_counts,
                    self.output_component_counts,
                )
            ]
        )
        self.output_decoders = nn.ModuleList(
            [nn.Linear(self.embed_dim, width) for width in self.output_component_counts]
        )
        for decoder in self.output_decoders:
            nn.init.zeros_(decoder.weight)
            nn.init.zeros_(decoder.bias)

        self.global_decoder = (
            nn.Linear(self.embed_dim, self.global_output_dim)
            if self.global_output_dim > 0
            else None
        )
        if self.global_decoder is not None:
            nn.init.zeros_(self.global_decoder.weight)
            nn.init.zeros_(self.global_decoder.bias)

        self.row_embedding = nn.Embedding(rows, self.embed_dim)
        self.col_embedding = nn.Embedding(cols, self.embed_dim)
        positions = torch.tensor(
            [(row, col) for row in range(rows) for col in range(cols)],
            dtype=torch.long,
        )
        self.register_buffer("positions", positions, persistent=False)
        self.register_buffer(
            "edge_index",
            _patch_edge_index((rows, cols), neighborhood),
            persistent=False,
        )
        self.backend = GNNCouplingBackend(
            embed_dim=self.embed_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``base_model(x) + delta`` in flat latent-output coordinates."""
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"Expected flat input shape (B, {self.input_dim}), got {tuple(x.shape)}.")
        with torch.set_grad_enabled(not self.freeze_base):
            base = self.base_model(x)
        if base.ndim != 2 or base.shape[1] != self.output_dim:
            raise ValueError(
                f"Base model must return shape (B, {self.output_dim}), got {tuple(base.shape)}."
            )

        input_pieces = _split_flat(x, self.input_component_counts)
        local_base = base[:, self.global_output_dim:]
        base_pieces = _split_flat(local_base, self.output_component_counts)
        tokens = torch.stack(
            [
                encoder(torch.cat([input_piece, base_piece], dim=1))
                for encoder, input_piece, base_piece in zip(
                    self.node_encoders,
                    input_pieces,
                    base_pieces,
                )
            ],
            dim=1,
        )
        tokens = tokens + self._positional_embedding()
        coupled = self.backend(tokens, self.edge_index)

        delta_pieces: list[torch.Tensor] = []
        if self.global_decoder is not None:
            delta_pieces.append(self.global_decoder(torch.mean(coupled, dim=1)))
        for patch_index, decoder in enumerate(self.output_decoders):
            delta_pieces.append(decoder(coupled[:, patch_index, :]))
        delta = torch.cat(delta_pieces, dim=1) * self.delta_scale
        self._last_delta = delta
        return base + delta

    def regularization_loss(self) -> torch.Tensor:
        """Return latent correction magnitude penalty for the last forward pass."""
        if self._last_delta is None or self.delta_regularization == 0.0:
            parameter = next(self.parameters())
            return parameter.sum() * 0.0
        return self.delta_regularization * torch.mean(self._last_delta**2)

    def load_warm_start_state(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load a plain L2L state dict into the wrapped base model."""
        self.base_model.load_state_dict(state_dict)

    def _positional_embedding(self) -> torch.Tensor:
        rows = self.positions[:, 0]
        cols = self.positions[:, 1]
        return (self.row_embedding(rows) + self.col_embedding(cols)).unsqueeze(0)


class GNNBoundaryCorrection(nn.Module):
    """GNN-driven masked boundary-band correction in physical patch space.

    The wrapped base model produces the standard local PCA latent prediction.
    Those latents are decoded by the provided PCA decoder to patch pixels, then a
    GNN emits a small additive correction image for each patch. The correction is
    multiplied by a fixed boundary-ring mask before assembly, matching the
    boundary-correction coupling idea: the new parameters can affect seams
    directly without competing with the PCA core in the patch interior.
    """

    def __init__(
        self,
        *,
        base_model: nn.Module,
        physical_decoder: Any,
        input_component_counts: list[int],
        output_component_counts: list[int],
        grid_shape: tuple[int, int],
        global_output_dim: int = 0,
        embed_dim: int = 128,
        num_layers: int = 3,
        neighborhood: int = 8,
        dropout: float = 0.0,
        activation: str = "gelu",
        freeze_base: bool = True,
        boundary_width: int = 4,
        smooth_taper: bool = True,
        correction_scale: float = 1.0,
        correction_regularization: float = 0.0,
    ) -> None:
        super().__init__()
        rows, cols = _validate_grid(grid_shape)
        if rows * cols != len(input_component_counts):
            raise ValueError(
                f"grid_shape={grid_shape} implies {rows * cols} patches, "
                f"got {len(input_component_counts)} input counts."
            )
        if len(input_component_counts) != len(output_component_counts):
            raise ValueError("Input and output component count lists must have the same length.")
        if any(count <= 0 for count in input_component_counts + output_component_counts):
            raise ValueError("All component counts must be positive.")
        if global_output_dim < 0:
            raise ValueError("global_output_dim cannot be negative.")
        patch_size = int(getattr(physical_decoder, "patch_size", 0))
        if patch_size <= 0:
            raise ValueError("physical_decoder must expose a positive patch_size.")
        if boundary_width <= 0 or boundary_width * 2 >= patch_size:
            raise ValueError("boundary_width must be positive and smaller than half the patch size.")
        if correction_scale <= 0.0:
            raise ValueError("correction_scale must be positive.")
        if correction_regularization < 0.0:
            raise ValueError("correction_regularization must be non-negative.")

        self.base_model = base_model
        self.physical_decoder = physical_decoder
        self.input_component_counts = [int(count) for count in input_component_counts]
        self.output_component_counts = [int(count) for count in output_component_counts]
        self.grid_shape = (rows, cols)
        self.num_patches = rows * cols
        self.global_output_dim = int(global_output_dim)
        self.embed_dim = int(embed_dim)
        self.input_dim = int(sum(self.input_component_counts))
        self.output_dim = int(self.global_output_dim + sum(self.output_component_counts))
        self.patch_size = patch_size
        self.boundary_width = int(boundary_width)
        self.freeze_base = bool(freeze_base)
        self.correction_scale = float(correction_scale)
        self.correction_regularization = float(correction_regularization)
        self._last_correction: torch.Tensor | None = None

        if self.freeze_base:
            for parameter in self.base_model.parameters():
                parameter.requires_grad_(False)
        for parameter in self.physical_decoder.parameters():
            parameter.requires_grad_(False)

        self.node_encoders = nn.ModuleList(
            [
                nn.Linear(input_width + output_width, self.embed_dim)
                for input_width, output_width in zip(
                    self.input_component_counts,
                    self.output_component_counts,
                )
            ]
        )
        patch_pixels = self.patch_size * self.patch_size
        self.boundary_decoders = nn.ModuleList(
            [nn.Linear(self.embed_dim, patch_pixels) for _ in self.output_component_counts]
        )
        for decoder in self.boundary_decoders:
            nn.init.zeros_(decoder.weight)
            nn.init.zeros_(decoder.bias)

        self.row_embedding = nn.Embedding(rows, self.embed_dim)
        self.col_embedding = nn.Embedding(cols, self.embed_dim)
        positions = torch.tensor(
            [(row, col) for row in range(rows) for col in range(cols)],
            dtype=torch.long,
        )
        self.register_buffer("positions", positions, persistent=False)
        self.register_buffer(
            "edge_index",
            _patch_edge_index((rows, cols), neighborhood),
            persistent=False,
        )
        self.register_buffer(
            "boundary_mask",
            _boundary_ring_mask(self.patch_size, self.boundary_width, smooth_taper),
            persistent=False,
        )
        self.backend = GNNCouplingBackend(
            embed_dim=self.embed_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the base latent prediction for latent-space callers."""
        return self._base_latent(x)

    def predict_physical(self, x: torch.Tensor) -> torch.Tensor:
        """Decode the base PCA core and add masked boundary corrections."""
        base = self._base_latent(x)
        patches, coarse = self.physical_decoder.decode_patches(base)
        correction = self._boundary_correction(x, base, dtype=patches.dtype, device=patches.device)
        corrected_patches = patches + correction
        residual = self.physical_decoder.assemble_patches(corrected_patches)
        if coarse is not None:
            return coarse + residual
        return residual

    def regularization_loss(self) -> torch.Tensor:
        """Return boundary correction magnitude penalty for the last forward pass."""
        if self._last_correction is None or self.correction_regularization == 0.0:
            parameter = next(self.parameters())
            return parameter.sum() * 0.0
        return self.correction_regularization * torch.mean(self._last_correction**2)

    def load_warm_start_state(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load a plain L2L state dict into the wrapped base model."""
        self.base_model.load_state_dict(state_dict)

    def _base_latent(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"Expected flat input shape (B, {self.input_dim}), got {tuple(x.shape)}.")
        if self.freeze_base:
            self.base_model.eval()
        with torch.set_grad_enabled(not self.freeze_base):
            base = self.base_model(x)
        if base.ndim != 2 or base.shape[1] != self.output_dim:
            raise ValueError(
                f"Base model must return shape (B, {self.output_dim}), got {tuple(base.shape)}."
            )
        return base

    def _boundary_correction(
        self,
        x: torch.Tensor,
        base: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        input_pieces = _split_flat(x, self.input_component_counts)
        local_base = base[:, self.global_output_dim:]
        base_pieces = _split_flat(local_base, self.output_component_counts)
        tokens = torch.stack(
            [
                encoder(torch.cat([input_piece, base_piece], dim=1))
                for encoder, input_piece, base_piece in zip(
                    self.node_encoders,
                    input_pieces,
                    base_pieces,
                )
            ],
            dim=1,
        )
        tokens = tokens + self._positional_embedding()
        coupled = self.backend(tokens, self.edge_index)
        corrections = [
            decoder(coupled[:, patch_index, :]).reshape(x.shape[0], self.patch_size, self.patch_size)
            for patch_index, decoder in enumerate(self.boundary_decoders)
        ]
        correction = torch.stack(corrections, dim=1)
        mask = self.boundary_mask.to(dtype=dtype, device=device)
        correction = correction.to(dtype=dtype, device=device) * mask * self.correction_scale
        self._last_correction = correction
        return correction

    def _positional_embedding(self) -> torch.Tensor:
        rows = self.positions[:, 0]
        cols = self.positions[:, 1]
        return (self.row_embedding(rows) + self.col_embedding(cols)).unsqueeze(0)


class GNNCouplingBackend(nn.Module):
    """Residual message-passing backend over a patch adjacency graph."""

    def __init__(
        self,
        *,
        embed_dim: int,
        num_layers: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [GraphMessagePassingLayer(embed_dim, dropout=dropout, activation=activation) for _ in range(num_layers)]
        )

    def forward(self, tokens: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Apply stacked message passing layers."""
        output = tokens
        for layer in self.layers:
            output = layer(output, edge_index)
        return output


class GraphMessagePassingLayer(nn.Module):
    """Mean-neighbor message passing with residual token update."""

    def __init__(self, embed_dim: int, *, dropout: float, activation: str) -> None:
        super().__init__()
        self.self_linear = nn.Linear(embed_dim, embed_dim)
        self.neighbor_linear = nn.Linear(embed_dim, embed_dim)
        self.update = nn.Sequential(
            nn.LayerNorm(embed_dim),
            make_activation(activation),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, tokens: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Update tokens from directed graph edges ``source -> target``."""
        source = edge_index[0]
        target = edge_index[1]
        messages = self.neighbor_linear(tokens[:, source, :])
        aggregated = tokens.new_zeros(tokens.shape)
        aggregated.index_add_(1, target, messages)
        degree = tokens.new_zeros((tokens.shape[1],))
        degree.index_add_(0, target, torch.ones_like(target, dtype=tokens.dtype))
        aggregated = aggregated / degree.clamp_min(1.0).reshape(1, -1, 1)
        proposal = self.self_linear(tokens) + aggregated
        return tokens + self.update(proposal)


class AttentionCouplingBackend(nn.Module):
    """Transformer-encoder backend over patch tokens."""

    def __init__(
        self,
        *,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError("attention_heads must be positive.")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by attention_heads.")
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply transformer coupling."""
        return self.encoder(tokens)


def _validate_grid(grid_shape: tuple[int, int]) -> tuple[int, int]:
    if len(grid_shape) != 2:
        raise ValueError("grid_shape must contain two entries.")
    rows, cols = int(grid_shape[0]), int(grid_shape[1])
    if rows <= 0 or cols <= 0:
        raise ValueError("grid_shape entries must be positive.")
    return rows, cols


def _patch_edge_index(grid_shape: tuple[int, int], neighborhood: int) -> torch.Tensor:
    rows, cols = _validate_grid(grid_shape)
    offsets = _neighbor_offsets(neighborhood)
    edges: list[tuple[int, int]] = []
    for row in range(rows):
        for col in range(cols):
            target = row * cols + col
            for row_offset, col_offset in offsets:
                src_row = row + row_offset
                src_col = col + col_offset
                if 0 <= src_row < rows and 0 <= src_col < cols:
                    source = src_row * cols + src_col
                    edges.append((source, target))
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _boundary_ring_mask(patch_size: int, boundary_width: int, smooth_taper: bool) -> torch.Tensor:
    coords = torch.arange(patch_size, dtype=torch.float32)
    row_distance = torch.minimum(coords, torch.flip(coords, dims=(0,))).reshape(-1, 1)
    col_distance = torch.minimum(coords, torch.flip(coords, dims=(0,))).reshape(1, -1)
    distance = torch.minimum(row_distance, col_distance)
    if smooth_taper:
        mask = ((float(boundary_width) - distance) / float(boundary_width)).clamp(min=0.0, max=1.0)
    else:
        mask = (distance < float(boundary_width)).to(torch.float32)
    return mask.reshape(1, 1, patch_size, patch_size)


def _split_flat(x: torch.Tensor, widths: list[int]) -> list[torch.Tensor]:
    pieces: list[torch.Tensor] = []
    start = 0
    for width in widths:
        stop = start + int(width)
        pieces.append(x[:, start:stop])
        start = stop
    if start != x.shape[1]:
        raise ValueError(f"Widths sum to {start}, but tensor has width {x.shape[1]}.")
    return pieces
