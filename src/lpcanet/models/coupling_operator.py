"""GNN-based patch coupling operator replacing the concat MLP in L2L PCA-Net."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.nn import MessagePassing

from lpcanet.assembly.patching import PatchSlice
from lpcanet.models.mlp import make_activation


class PatchGNNLayer(MessagePassing):
    """Mean-aggregation message-passing layer.

    h_p' = W_self · h_p + W_neigh · mean({h_q : q ∈ N(p)})
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__(aggr="mean")
        self.lin_self = nn.Linear(embed_dim, embed_dim)
        self.lin_neigh = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        neigh = self.propagate(edge_index, x=x)
        return self.lin_self(x) + self.lin_neigh(neigh)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j


class CouplingOperator(nn.Module):
    """GNN patch coupling operator: replaces the MLP head in the L2L pipeline.

    Per-patch input latent codes (heterogeneous lengths) are projected to a
    common embedding dimension, coupled via message passing on the 4-neighbor
    patch adjacency graph, then projected back to per-patch output latent codes.

    External interface is flat (B, sum_C_in) → (B, sum_C_out), identical to
    FlatPatchwiseHeadAdapter-wrapped heads, so no adapter wrapper is needed.
    """

    def __init__(
        self,
        input_component_counts: list[int],
        output_component_counts: list[int],
        patch_slices: list[PatchSlice],
        patch_grid_shape: tuple[int, int],
        embed_dim: int = 64,
        num_layers: int = 3,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(input_component_counts) != len(output_component_counts):
            raise ValueError(
                "input_component_counts and output_component_counts must have the same length."
            )
        if len(input_component_counts) != len(patch_slices):
            raise ValueError("Component count lists must match the number of patch slices.")
        if not input_component_counts:
            raise ValueError("Component count lists must be non-empty.")
        if any(c <= 0 for c in input_component_counts + output_component_counts):
            raise ValueError("All component counts must be positive.")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.input_component_counts = [int(c) for c in input_component_counts]
        self.output_component_counts = [int(c) for c in output_component_counts]
        self.num_patches = len(input_component_counts)
        self.embed_dim = embed_dim

        # Per-patch linear encoders: (C_in_p + 2) -> embed_dim
        # +2 for the normalized (row_center, col_center) spatial position.
        self.encoders = nn.ModuleList([
            nn.Linear(c_in + 2, embed_dim) for c_in in self.input_component_counts
        ])

        self.gnn_layers = nn.ModuleList([
            PatchGNNLayer(embed_dim) for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_layers)
        ])
        self.act = make_activation(activation)
        self.dropout = nn.Dropout(dropout)

        # Per-patch linear decoders: embed_dim -> C_out_p
        self.decoders = nn.ModuleList([
            nn.Linear(embed_dim, c_out) for c_out in self.output_component_counts
        ])

        # Precomputed fixed-graph buffers — moved to device automatically with model.
        self.register_buffer("edge_index", _build_edge_index(patch_grid_shape))
        self.register_buffer("node_pos", _build_node_pos(patch_slices))

    def forward(self, x: Tensor) -> Tensor:
        """Map flat (B, sum_C_in) to flat (B, sum_C_out)."""
        B = x.shape[0]
        P = self.num_patches

        # Split flat input into per-patch codes: list of P tensors, each (B, C_in_p).
        patches = x.split(self.input_component_counts, dim=1)

        # Per-patch encode: concat latent code with spatial position, then project to embed_dim.
        node_features = []
        for p in range(P):
            pos = self.node_pos[p].unsqueeze(0).expand(B, -1)  # (B, 2)
            h_p = self.encoders[p](torch.cat([patches[p], pos], dim=1))  # (B, embed_dim)
            node_features.append(h_p)
        H = torch.stack(node_features, dim=1)  # (B, P, embed_dim)

        # Build PyG batch: B independent copies of the same graph, each with different node features.
        data_list = [Data(x=H[i], edge_index=self.edge_index) for i in range(B)]
        batch = Batch.from_data_list(data_list)
        h = batch.x          # (B*P, embed_dim)
        ei = batch.edge_index  # (2, B*E) — node indices offset per graph by Batch

        # GNN layers with residual connections and LayerNorm.
        # Pattern: act → dropout → add residual → LayerNorm.
        for gnn_layer, ln in zip(self.gnn_layers, self.layer_norms):
            h = ln(self.dropout(self.act(gnn_layer(h, ei))) + h)

        # Reshape to (B, P, embed_dim) and decode per patch.
        H_out = h.reshape(B, P, self.embed_dim)
        out_parts = [self.decoders[p](H_out[:, p, :]) for p in range(P)]
        return torch.cat(out_parts, dim=1)  # (B, sum_C_out)


def _build_edge_index(patch_grid_shape: tuple[int, int]) -> Tensor:
    """Directed 4-neighbor edge index for a regular patch grid (row-major node ordering)."""
    R, C = patch_grid_shape
    src: list[int] = []
    dst: list[int] = []
    for r in range(R):
        for c in range(C):
            p = r * C + c
            if c + 1 < C:      # right neighbor — both directions
                src += [p, p + 1]
                dst += [p + 1, p]
            if r + 1 < R:      # down neighbor — both directions
                src += [p, p + C]
                dst += [p + C, p]
    return torch.tensor([src, dst], dtype=torch.long)


def _build_node_pos(patch_slices: list[PatchSlice]) -> Tensor:
    """Normalized [0, 1] patch center coordinates derived from patch slices."""
    field_h = float(max(sl[0].stop for sl in patch_slices))
    field_w = float(max(sl[1].stop for sl in patch_slices))
    positions = [
        [
            (sl[0].start + sl[0].stop) / (2.0 * field_h),
            (sl[1].start + sl[1].stop) / (2.0 * field_w),
        ]
        for sl in patch_slices
    ]
    return torch.tensor(positions, dtype=torch.float32)
