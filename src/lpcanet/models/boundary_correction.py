from __future__ import annotations

import torch
from torch import nn

from lpcanet.models.coupling import GNNCouplingBackend, _patch_edge_index, _validate_grid
from lpcanet.pca.torch_decoder import TorchPCADecoder


def _distance_from_boundary(patch_size: int) -> torch.Tensor:
    """Return an integer distance map (P, P): how far each pixel is from the nearest edge."""
    idx = torch.arange(patch_size)
    dist_from_start = idx
    dist_from_end = patch_size - 1 - idx
    row_dist = torch.minimum(dist_from_start, dist_from_end)
    col_dist = torch.minimum(dist_from_start, dist_from_end)
    return torch.minimum(row_dist.unsqueeze(1), col_dist.unsqueeze(0))  # (P, P)


def _make_tapered_mask(patch_size: int, delta: int) -> torch.Tensor:
    dist = _distance_from_boundary(patch_size).float()
    return torch.clamp(1.0 - dist / float(delta), min=0.0, max=1.0)


class BoundaryCorrectionGNN(nn.Module):
    def __init__(
        self,
        *,
        input_component_counts: list[int],
        output_component_counts: list[int],
        grid_shape: tuple[int, int],
        embed_dim: int = 128,
        num_layers: int = 3,
        neighborhood: int = 8,
        dropout: float = 0.0,
        activation: str = "gelu",
        patch_size: int = 32,
        delta: int = 4,
        pca_decoder: TorchPCADecoder,
    ) -> None:
        """Instantiate per-patch encoders/decoders, GNN, correction heads, tapered mask, and frozen PCA decoder."""
        super().__init__()
        rows, cols = _validate_grid(grid_shape)
        num_patches = rows * cols
        if len(input_component_counts) != num_patches:
            raise ValueError(
                f"grid_shape={grid_shape} implies {num_patches} patches, "
                f"got {len(input_component_counts)} input_component_counts."
            )
        if len(output_component_counts) != num_patches:
            raise ValueError(
                f"grid_shape={grid_shape} implies {num_patches} patches, "
                f"got {len(output_component_counts)} output_component_counts."
            )

        self.input_component_counts = [int(c) for c in input_component_counts]
        self.output_component_counts = [int(c) for c in output_component_counts]
        self.grid_shape = (rows, cols)
        self.num_patches = num_patches
        self.embed_dim = int(embed_dim)
        self.patch_size = int(patch_size)
        self.delta = int(delta)
        self.input_dim = int(sum(self.input_component_counts))

        # Per-patch input encoders: latent_in_p -> embed_dim
        self.input_encoders = nn.ModuleList(
            [nn.Linear(w, self.embed_dim) for w in self.input_component_counts]
        )
        # Per-patch output decoders: embed_dim -> latent_out_p
        self.output_decoders = nn.ModuleList(
            [nn.Linear(self.embed_dim, w) for w in self.output_component_counts]
        )
        # Positional embeddings (row + col)
        self.row_embedding = nn.Embedding(rows, self.embed_dim)
        self.col_embedding = nn.Embedding(cols, self.embed_dim)
        positions = torch.tensor(
            [(r, c) for r in range(rows) for c in range(cols)],
            dtype=torch.long,
        )
        self.register_buffer("positions", positions, persistent=False)

        # GNN backbone
        edge_index = _patch_edge_index((rows, cols), neighborhood)
        self.register_buffer("edge_index", edge_index, persistent=False)
        self.gnn_backend = GNNCouplingBackend(
            embed_dim=self.embed_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
        )

        # Per-patch pixel-space correction heads: W_p: embed_dim -> patch_size^2
        pixel_count = self.patch_size * self.patch_size
        self.correction_heads = nn.ModuleList(
            [nn.Linear(self.embed_dim, pixel_count) for _ in range(num_patches)]
        )
        # Small-scale init: near-zero so training starts from the PCA baseline,
        # but non-zero so Adam's m2 is seeded from step 0 (avoids inflated
        # effective step size that zero-init causes before m2 warms up).
        for head in self.correction_heads:
            nn.init.normal_(head.weight, std=1e-4)
            nn.init.normal_(head.bias, std=1e-4)

        # Smooth tapered boundary mask M_p: (P, P), exactly 0 in interior
        mask = _make_tapered_mask(self.patch_size, self.delta)
        self.register_buffer("boundary_mask", mask, persistent=True)

        # Frozen PCA decoder — buffers only, no nn.Parameters, excluded from optimizer
        self.pca_decoder = pca_decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map flat latent input codes to assembled corrected field.

        Args:
            x: (B, input_dim) flat concatenation of per-patch input latent codes.

        Returns:
            field: (B, H, W) assembled corrected prediction.
        """
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input shape (B, {self.input_dim}), got {tuple(x.shape)}."
            )
        B = x.shape[0]

        # 1. Encode per-patch latent codes to tokens
        pieces: list[torch.Tensor] = []
        start = 0
        for encoder, width in zip(self.input_encoders, self.input_component_counts):
            stop = start + width
            pieces.append(encoder(x[:, start:stop]))
            start = stop
        tokens = torch.stack(pieces, dim=1)           # (B, N, embed_dim)
        tokens = tokens + self._positional_embedding()

        # 2. GNN coupling: produces coupled tokens t'_p
        coupled = self.gnn_backend(tokens, self.edge_index)  # (B, N, embed_dim)

        # 3. Decode coupled tokens to output latent codes
        latent_out: list[torch.Tensor] = []
        for p, decoder in enumerate(self.output_decoders):
            latent_out.append(decoder(coupled[:, p, :]))   # (B, k_p)

        # 4. Core PCA decode: latent -> pixel patches (frozen buffers, no grad)
        core_patches: list[torch.Tensor] = []
        for p, patch_decoder in enumerate(self.pca_decoder.patch_decoders):
            core_patches.append(patch_decoder(latent_out[p]))  # (B, P, P)

        # 5. Pixel-space boundary correction C_p(t'_p) = M_p ⊙ (W_p t'_p)
        mask = self.boundary_mask.to(dtype=x.dtype)     # (P, P)
        corrections: list[torch.Tensor] = []
        for p, head in enumerate(self.correction_heads):
            raw = head(coupled[:, p, :]).reshape(B, self.patch_size, self.patch_size)
            corrections.append(raw * mask)              # interior pixels exactly 0

        # 6. Add correction to core: uhat_p = core_p + C_p(t'_p)
        corrected = [c + e for c, e in zip(core_patches, corrections)]

        # 7. Assemble corrected patches into full field
        patch_tensor = torch.stack(corrected, dim=1)   # (B, N, P, P)
        return self.pca_decoder._assemble_patches(patch_tensor)  # (B, H, W)

    def _positional_embedding(self) -> torch.Tensor:
        rows = self.positions[:, 0]
        cols = self.positions[:, 1]
        return (self.row_embedding(rows) + self.col_embedding(cols)).unsqueeze(0)


class PassthroughDecoder(nn.Module):
    """Identity decoder used when the model already returns an assembled field."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
