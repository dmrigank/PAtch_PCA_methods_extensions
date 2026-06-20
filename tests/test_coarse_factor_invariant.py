from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lpcanet.assembly.windows import safe_hann2d
from lpcanet.metrics.interface import interface_jump
from lpcanet.pca.pca import LocalToLocalPCAEncoder, TwoScaleLocalToLocalPCAEncoder


def test_coarse_factor_guard_fires_for_illegal_factor() -> None:
    with pytest.raises(ValueError, match="Coarse factor invariant violated"):
        TwoScaleLocalToLocalPCAEncoder(
            patch_size=32,
            stride=16,
            coarse_factor=1,
            grid_size=128,
            n_samples=8000,
        )


def test_two_scale_poisson128_reconstruction_reduces_interface_jump() -> None:
    x_train, y_train = _load_or_make_poisson128_fields(n_samples=280)
    weights = safe_hann2d(32)

    plain = LocalToLocalPCAEncoder(
        patch_size=32,
        stride=16,
        input_variance=1,
        output_variance=8,
        solver="randomized",
        standardize=True,
        oversampling=5,
        n_iter=1,
        random_state=0,
        assembly_weights=weights,
    ).fit(x_train, y_train)
    two_scale = TwoScaleLocalToLocalPCAEncoder(
        patch_size=32,
        stride=16,
        input_variance=1,
        output_variance=7,
        solver="randomized",
        standardize=True,
        oversampling=5,
        n_iter=1,
        random_state=0,
        assembly_weights=weights,
        coarse_factor=8,
        coarse_components=10,
        coarse_variance=0.99,
        grid_size=128,
        n_samples=len(y_train),
    ).fit(x_train, y_train)

    plain_recon = plain.inverse_transform_outputs(plain.transform_outputs(y_train))
    two_scale_latent = two_scale.transform_outputs(y_train)
    two_scale_recon = two_scale.inverse_transform_outputs(two_scale_latent)
    coarse, residual = two_scale.decompose_outputs(two_scale_latent)

    plain_jump = interface_jump(plain_recon, patch_size=32, stride=16)
    two_scale_jump = interface_jump(two_scale_recon, patch_size=32, stride=16)
    np.testing.assert_allclose(coarse + residual, two_scale_recon, rtol=0.0, atol=1.0e-7)
    assert two_scale_jump < plain_jump
    assert int(two_scale.component_counts["output_total"]) <= int(plain.component_counts["output_total"])
    assert two_scale.timings["fit_coarse_svd"] > 0.0


def _load_or_make_poisson128_fields(n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "data" / "processed" / "poisson_128.npz",
        root / "results" / "repro_l2l_overlap_seed0_lowload" / "predictions_test.npz",
        root / "results" / "repro_l2l_overlap_seed0" / "predictions_test.npz",
    ]
    for path in candidates:
        if path.exists():
            data = np.load(path)
            x_key = "f" if "f" in data else "x_input"
            y_key = "u" if "u" in data else "y_true"
            x = np.asarray(data[x_key][:n_samples], dtype=np.float32)
            y = np.asarray(data[y_key][:n_samples], dtype=np.float32)
            if x.shape == (n_samples, 128, 128) and y.shape == (n_samples, 128, 128):
                return x, y
    return _make_smooth_poisson128_fields(n_samples)


def _make_smooth_poisson128_fields(n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(5)
    axis = np.linspace(0.0, 1.0, 128, dtype=np.float64)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    modes = [
        np.sin(np.pi * xx) * np.sin(np.pi * yy),
        np.sin(2.0 * np.pi * xx) * np.sin(np.pi * yy),
        np.sin(np.pi * xx) * np.sin(2.0 * np.pi * yy),
        np.sin(2.0 * np.pi * xx) * np.sin(2.0 * np.pi * yy),
        np.sin(3.0 * np.pi * xx) * np.sin(np.pi * yy),
        np.sin(np.pi * xx) * np.sin(3.0 * np.pi * yy),
    ]
    coeffs = rng.normal(size=(n_samples, len(modes)))
    fields = np.zeros((n_samples, 128, 128), dtype=np.float64)
    forcing = np.zeros_like(fields)
    for mode_index, mode in enumerate(modes):
        amplitude = coeffs[:, mode_index, None, None]
        fields += amplitude * mode
        forcing += (mode_index + 1.0) * amplitude * mode
    fields += 0.01 * rng.normal(size=fields.shape)
    forcing += 0.01 * rng.normal(size=forcing.shape)
    return forcing.astype(np.float32), fields.astype(np.float32)
