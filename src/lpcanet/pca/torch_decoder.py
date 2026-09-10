"""Differentiable torch decoders for fitted PCA output representations."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from lpcanet.assembly.mosaic import make_patch_index_map
from lpcanet.assembly.operators import prolongate_bilinear


class TorchPCADecoder(nn.Module):
    """Torch mirror of fitted local/two-scale output PCA decoders."""

    def __init__(self, encoder: Any) -> None:
        super().__init__()
        if not hasattr(encoder, "output_models"):
            raise ValueError("In-loop physical loss currently requires local-to-local output PCA.")
        self.output_shape = (int(encoder.output_shape_[0]), int(encoder.output_shape_[1]))
        self.patch_size = int(encoder.patch_size)
        self.stride = int(encoder.stride)
        self.include_edges = bool(getattr(encoder, "include_edges", False))
        self.output_guard_band = int(getattr(encoder, "output_guard_band", 0))
        self.output_patch_size = int(getattr(encoder, "output_patch_size", self.patch_size))
        self.patch_decoders = nn.ModuleList(
            [_PatchPCADecoder(model) for model in encoder.output_models]
        )
        self.patch_code_dim = int(sum(decoder.n_components for decoder in self.patch_decoders))
        weights = getattr(encoder, "assembly_weights", None)
        if weights is None:
            weight_array = np.ones((self.patch_size, self.patch_size), dtype=np.float64)
        else:
            weight_array = np.asarray(weights, dtype=np.float64)
        self.register_buffer("assembly_weights", torch.as_tensor(weight_array, dtype=torch.float32))
        index_map = make_patch_index_map(
            self.output_shape,
            self.patch_size,
            self.stride,
            include_edges=self.include_edges,
        )
        flat_indices = index_map.flat_indices
        repeated_weights = torch.as_tensor(weight_array.reshape(-1), dtype=torch.float32).repeat(
            index_map.n_patches
        )
        weight_sums = torch.zeros(self.output_shape[0] * self.output_shape[1], dtype=torch.float32)
        weight_sums.scatter_add_(0, flat_indices, repeated_weights)
        if torch.any(weight_sums <= 0):
            raise ValueError("PCA decoder assembly weights leave uncovered pixels.")
        self.register_buffer("assembly_flat_indices", flat_indices)
        self.register_buffer("assembly_repeated_weights", repeated_weights)
        self.register_buffer("assembly_weight_sums", weight_sums)

        self.coarse_decoder: _CoarsePCADecoder | None = None
        self.coarse_code_dim = 0
        if hasattr(encoder, "coarse_pca"):
            self.coarse_decoder = _CoarsePCADecoder(encoder)
            self.coarse_code_dim = int(self.coarse_decoder.n_components)
        self.output_dim = self.coarse_code_dim + self.patch_code_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode flat PCA output coordinates to assembled scalar fields."""
        patch_tensor, coarse = self.decode_patches(z)
        residual = self.assemble_patches(patch_tensor)
        if coarse is not None:
            return coarse + residual
        return residual

    def decode_patches(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Decode flat PCA output coordinates to patch tensors before assembly."""
        if z.ndim != 2 or z.shape[1] != self.output_dim:
            raise ValueError(f"Expected latent shape (B, {self.output_dim}), got {tuple(z.shape)}.")
        start = 0
        coarse: torch.Tensor | None = None
        if self.coarse_decoder is not None:
            stop = self.coarse_code_dim
            coarse = self.coarse_decoder(z[:, :stop])
            start = stop

        patches: list[torch.Tensor] = []
        for decoder in self.patch_decoders:
            stop = start + decoder.n_components
            patch = decoder(z[:, start:stop])
            if patch.shape[1:] != (self.patch_size, self.patch_size):
                patch = self._crop_patch_core(patch)
            patches.append(patch)
            start = stop
        patch_tensor = torch.stack(patches, dim=1)
        return patch_tensor, coarse

    def _crop_patch_core(self, patch: torch.Tensor) -> torch.Tensor:
        expected = self.patch_size + 2 * self.output_guard_band
        if self.output_guard_band <= 0 or patch.shape[1:] != (expected, expected):
            raise ValueError(
                f"Decoded patch shape {tuple(patch.shape[1:])} does not match "
                f"core {(self.patch_size, self.patch_size)} or guarded {(expected, expected)}."
            )
        start = self.output_guard_band
        stop = start + self.patch_size
        return patch[:, start:stop, start:stop]

    def assemble_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """Assemble decoded patch tensors with the fitted overlap/weight rule."""
        if patches.ndim != 4:
            raise ValueError(f"Expected patch tensor shape (B, N, P, P), got {tuple(patches.shape)}.")
        if patches.shape[1] != len(self.patch_decoders):
            raise ValueError(f"Expected {len(self.patch_decoders)} patches, got {patches.shape[1]}.")
        if patches.shape[2:] != (self.patch_size, self.patch_size):
            raise ValueError(
                f"Expected patch size {(self.patch_size, self.patch_size)}, got {tuple(patches.shape[2:])}."
            )
        batch_size = patches.shape[0]
        flat_indices = self.assembly_flat_indices.to(device=patches.device)
        weights = self.assembly_repeated_weights.to(dtype=patches.dtype, device=patches.device)
        weighted = patches.reshape(batch_size, -1) * weights
        output = patches.new_zeros((batch_size, self.output_shape[0] * self.output_shape[1]))
        output.scatter_add_(1, flat_indices.expand(batch_size, -1), weighted)
        normalization = self.assembly_weight_sums.to(
            dtype=patches.dtype,
            device=patches.device,
        )
        output = output / normalization.unsqueeze(0)
        return output.reshape(batch_size, *self.output_shape)


class _PatchPCADecoder(nn.Module):
    """Decode one patch's PCA coordinates."""

    def __init__(self, model: dict[str, Any]) -> None:
        super().__init__()
        pca = model["pca"]
        scaler = model["scaler"]
        components = np.asarray(pca.components_, dtype=np.float64)
        self.n_components = int(components.shape[0])
        self.patch_size = int(round(np.sqrt(components.shape[1])))
        self.register_buffer("components", torch.as_tensor(components, dtype=torch.float32))
        self.register_buffer("pca_mean", torch.as_tensor(np.asarray(pca.mean_, dtype=np.float64), dtype=torch.float32))
        scaler_mean, scaler_scale = _scaler_params(scaler, components.shape[1])
        self.register_buffer("scaler_mean", torch.as_tensor(scaler_mean, dtype=torch.float32))
        self.register_buffer("scaler_scale", torch.as_tensor(scaler_scale, dtype=torch.float32))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        scaled = z @ self.components.to(dtype=z.dtype, device=z.device)
        scaled = scaled + self.pca_mean.to(dtype=z.dtype, device=z.device)
        flat = scaled * self.scaler_scale.to(dtype=z.dtype, device=z.device)
        flat = flat + self.scaler_mean.to(dtype=z.dtype, device=z.device)
        return flat.reshape(z.shape[0], self.patch_size, self.patch_size)


class _CoarsePCADecoder(nn.Module):
    """Decode two-scale coarse-global coordinates."""

    def __init__(self, encoder: Any) -> None:
        super().__init__()
        components = np.asarray(encoder.coarse_pca.components_, dtype=np.float64)
        self.n_components = int(components.shape[0])
        self.coarse_factor = int(encoder.coarse_factor)
        self.coarse_shape = (
            int(encoder.output_shape_[0] // encoder.coarse_factor),
            int(encoder.output_shape_[1] // encoder.coarse_factor),
        )
        self.register_buffer("components", torch.as_tensor(components, dtype=torch.float32))
        self.register_buffer(
            "pca_mean",
            torch.as_tensor(np.asarray(encoder.coarse_pca.mean_, dtype=np.float64), dtype=torch.float32),
        )
        scaler_mean, scaler_scale = _scaler_params(encoder.coarse_scaler, components.shape[1])
        self.register_buffer("scaler_mean", torch.as_tensor(scaler_mean, dtype=torch.float32))
        self.register_buffer("scaler_scale", torch.as_tensor(scaler_scale, dtype=torch.float32))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        scaled = z @ self.components.to(dtype=z.dtype, device=z.device)
        scaled = scaled + self.pca_mean.to(dtype=z.dtype, device=z.device)
        flat = scaled * self.scaler_scale.to(dtype=z.dtype, device=z.device)
        flat = flat + self.scaler_mean.to(dtype=z.dtype, device=z.device)
        coarse = flat.reshape(z.shape[0], *self.coarse_shape)
        return prolongate_bilinear(coarse, self.coarse_factor)


def _scaler_params(scaler: Any, width: int) -> tuple[np.ndarray, np.ndarray]:
    mean = getattr(scaler, "mean_", None)
    scale = getattr(scaler, "scale_", None)
    if mean is None:
        mean_array = np.zeros(width, dtype=np.float64)
    else:
        mean_array = np.asarray(mean, dtype=np.float64)
    if scale is None:
        scale_array = np.ones(width, dtype=np.float64)
    else:
        scale_array = np.asarray(scale, dtype=np.float64)
    return mean_array, scale_array
