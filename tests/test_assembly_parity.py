from __future__ import annotations

import numpy as np
import torch

from lpcanet.assembly.mosaic import assemble_mosaic, hann_weights, make_patch_index_map
from lpcanet.assembly.operators import prolongate_bilinear, restrict_average
from lpcanet.assembly.patching import get_patch_slices
from lpcanet.assembly.windows import safe_hann2d
from lpcanet.pca.pca import LocalToLocalPCAEncoder, _inverse_patch_models


def _reference_weighted_mosaic(
    patches: np.ndarray,
    grid_shape: tuple[int, int],
    patch_size: int,
    stride: int,
    weights: np.ndarray,
    *,
    include_edges: bool = False,
) -> np.ndarray:
    patches = np.asarray(patches)
    if patches.ndim == 3:
        patches = patches.reshape(patches.shape[0], patches.shape[1], patch_size, patch_size)
    fields = np.zeros((patches.shape[0], *grid_shape), dtype=np.float64)
    weight_sums = np.zeros(grid_shape, dtype=np.float64)
    for patch_index, (row_slice, col_slice) in enumerate(
        get_patch_slices(grid_shape, patch_size, stride, include_edges=include_edges)
    ):
        fields[:, row_slice, col_slice] += patches[:, patch_index].astype(np.float64) * weights
        weight_sums[row_slice, col_slice] += weights
    fields /= weight_sums[None]
    return fields.astype(patches.dtype, copy=False)


def test_differentiable_mosaic_matches_legacy_overlap_add() -> None:
    rng = np.random.default_rng(12)
    grid_shape = (12, 12)
    patch_size = 6
    stride = 3
    n_patches = len(get_patch_slices(grid_shape, patch_size, stride))
    patches = rng.normal(size=(4, n_patches, patch_size, patch_size)).astype(np.float64)
    weights = safe_hann2d(patch_size)

    expected = _reference_weighted_mosaic(patches, grid_shape, patch_size, stride, weights)
    index_map = make_patch_index_map(grid_shape, patch_size, stride)
    actual = assemble_mosaic(
        torch.as_tensor(patches),
        grid_shape,
        patch_size,
        stride,
        torch.as_tensor(weights),
        index_map=index_map,
    )

    np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=1e-12, atol=1e-12)


def test_l2l_inverse_decode_uses_parity_assembly_path() -> None:
    rng = np.random.default_rng(3)
    x_train = rng.normal(size=(24, 8, 8)).astype(np.float32)
    y_train = rng.normal(size=(24, 8, 8)).astype(np.float32)
    weights = safe_hann2d(4)
    encoder = LocalToLocalPCAEncoder(
        patch_size=4,
        stride=2,
        input_variance=0.99,
        output_variance=0.99,
        solver="full",
        standardize=True,
        assembly_weights=weights,
        random_state=0,
    ).fit(x_train, y_train)
    latent_dim = int(encoder.component_counts["output_total"])
    z = rng.normal(size=(5, latent_dim)).astype(np.float32)

    decoded_patches = _inverse_patch_models(z, encoder.output_models, encoder.patch_size)
    expected = _reference_weighted_mosaic(
        decoded_patches,
        encoder.output_shape_,
        encoder.patch_size,
        encoder.stride,
        weights,
    )
    actual = encoder.inverse_transform_outputs(z)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_differentiable_mosaic_gradcheck() -> None:
    torch.manual_seed(0)
    grid_shape = (5, 5)
    patch_size = 3
    stride = 2
    n_patches = len(get_patch_slices(grid_shape, patch_size, stride))
    patches = torch.randn(2, n_patches, patch_size, patch_size, dtype=torch.float64, requires_grad=True)
    weights = hann_weights(patch_size, dtype=torch.float64)
    index_map = make_patch_index_map(grid_shape, patch_size, stride)

    def assemble(input_patches: torch.Tensor) -> torch.Tensor:
        return assemble_mosaic(
            input_patches,
            grid_shape,
            patch_size,
            stride,
            weights,
            index_map=index_map,
        )

    assert torch.autograd.gradcheck(assemble, (patches,), eps=1e-6, atol=1e-4)


def test_restrict_and_prolongate_shapes_and_values() -> None:
    field = torch.arange(16, dtype=torch.float64).reshape(4, 4)
    restricted = restrict_average(field, factor=2)
    expected = torch.tensor([[2.5, 4.5], [10.5, 12.5]], dtype=torch.float64)
    torch.testing.assert_close(restricted, expected)

    prolongated = prolongate_bilinear(restricted, factor=2)
    assert tuple(prolongated.shape) == (4, 4)
