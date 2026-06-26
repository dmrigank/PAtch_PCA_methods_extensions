"""Patch-token coupling operators for ragged PCA latent codes."""

from __future__ import annotations

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
        gat_negative_slope: float = 0.2,
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
        if backend.lower() == "gat" and embed_dim % attention_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by attention_heads ({attention_heads}) for backend='gat'."
            )

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
        elif self.backend == "gat":
            edge_index = _patch_edge_index((rows, cols), neighborhood)
            self.register_buffer("edge_index", edge_index, persistent=False)
            self.backend_module = GATCouplingBackend(
                embed_dim=self.embed_dim,
                num_layers=num_layers,
                num_heads=attention_heads,
                dropout=dropout,
                activation=activation,
                negative_slope=gat_negative_slope,
            )
        else:
            raise ValueError(f"Unsupported coupling backend {backend!r}; expected 'gnn', 'attention', or 'gat'.")

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

        if self.backend in ("gnn", "gat"):
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


class GATCouplingBackend(nn.Module):
    """Graph attention backend restricted to the patch adjacency graph.

    Replaces the uniform mean aggregation in GNNCouplingBackend with learned,
    non-uniform attention weights over each node's spatial neighbors.  The same
    edge_index (4- or 8-neighbor patch grid) is reused, so cost is O(edges·d),
    not O(N²·d) — the graph structure is preserved.
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        activation: str,
        negative_slope: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                GraphAttentionLayer(
                    embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    activation=activation,
                    negative_slope=negative_slope,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, tokens: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Apply stacked GAT layers."""
        output = tokens
        for layer in self.layers:
            output = layer(output, edge_index)
        return output


class GraphAttentionLayer(nn.Module):
    """Single multi-head GAT layer with graph-restricted attention + residual update.

    Formulation (Veličković et al. 2018, adapted for batched token grids):
        e_pq = LeakyReLU(a^T [W h_p ‖ W h_q])   for each edge q→p
        α_pq = softmax over neighbors q of p
        agg_p = Σ_q  α_pq · W_neigh h_q          (per head, then concat/avg)

    The aggregated result is combined with a self-projection and fed through the
    same LayerNorm→activation→dropout→Linear residual block used in
    GraphMessagePassingLayer, so the update style is consistent across backends.

    Multi-head: embed_dim is split into num_heads equal-sized heads; outputs are
    concatenated and projected back to embed_dim via a final linear.
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        dropout: float,
        activation: str,
        negative_slope: float,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Shared linear projection W applied to all tokens before attention.
        self.linear = nn.Linear(embed_dim, embed_dim, bias=False)
        # Per-head attention vector a: shape (num_heads, 2 * head_dim).
        self.attn_vec = nn.Parameter(torch.empty(num_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.attn_vec.unsqueeze(0))
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        # Output projection: concat of heads → embed_dim.
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        # Self-path (mirrors GraphMessagePassingLayer.self_linear).
        self.self_linear = nn.Linear(embed_dim, embed_dim)
        # Residual update block (identical structure to GraphMessagePassingLayer.update).
        self.update = nn.Sequential(
            nn.LayerNorm(embed_dim),
            make_activation(activation),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, tokens: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens:     (B, N, embed_dim)
            edge_index: (2, E)  —  row 0 = source, row 1 = target
        Returns:
            (B, N, embed_dim)
        """
        B, N, D = tokens.shape
        H, d = self.num_heads, self.head_dim
        src, tgt = edge_index[0], edge_index[1]  # each (E,)

        # Project all tokens: (B, N, D) → (B, N, H, d).
        Wh = self.linear(tokens).reshape(B, N, H, d)

        # Gather source and target features for each edge: (B, E, H, d).
        Wh_src = Wh[:, src, :, :]
        Wh_tgt = Wh[:, tgt, :, :]

        # Attention logits: e_pq = LeakyReLU(a^T [Wh_tgt ‖ Wh_src]).
        # concat shape: (B, E, H, 2d); attn_vec broadcast: (1, 1, H, 2d).
        concat = torch.cat([Wh_tgt, Wh_src], dim=-1)
        e = self.leaky_relu((concat * self.attn_vec).sum(-1))  # (B, E, H)

        # Numerically stable softmax per (batch, target-node, head).
        # Max-subtract using scatter_reduce_ over the target dimension.
        tgt_exp = tgt[None, :, None].expand(B, -1, H)  # (B, E, H)
        e_max = tokens.new_full((B, N, H), float("-inf"))
        e_max.scatter_reduce_(1, tgt_exp, e, reduce="amax", include_self=True)
        # Nodes with no incoming edges keep -inf; clamp so exp gives 0.
        e_max = e_max.clamp(min=-1e9)
        e_shifted = e - e_max[:, tgt, :]  # (B, E, H)

        exp_e = self.attn_dropout(torch.exp(e_shifted))  # (B, E, H)
        exp_sum = tokens.new_zeros(B, N, H)
        exp_sum.index_add_(1, tgt, exp_e)
        alpha = exp_e / exp_sum[:, tgt, :].clamp_min(1e-12)  # (B, E, H)

        # Weighted neighbor aggregation: (B, E, H, d) → (B, N, H, d).
        weighted = Wh_src * alpha.unsqueeze(-1)  # (B, E, H, d)
        aggregated = tokens.new_zeros(B, N, H, d)
        aggregated.index_add_(1, tgt, weighted)

        # Concat heads and project back: (B, N, D).
        aggregated = self.out_proj(aggregated.reshape(B, N, D))

        # Residual update (same pattern as GraphMessagePassingLayer).
        proposal = self.self_linear(tokens) + aggregated
        return tokens + self.update(proposal)


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
