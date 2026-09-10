from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from lpcanet.pca.pca import ContextOutputLocalToLocalPCAEncoder, LocalToLocalPCAEncoder
from lpcanet.pca.torch_decoder import TorchPCADecoder
from lpcanet.utils.config import compose_config


def test_context_output_pca_uses_guarded_output_and_disjoint_input() -> None:
    rng = np.random.default_rng(12)
    x = rng.normal(size=(20, 16, 16)).astype(np.float32)
    y = rng.normal(size=(20, 16, 16)).astype(np.float32)

    plain = LocalToLocalPCAEncoder(
        patch_size=8,
        stride=8,
        input_variance=0.99,
        output_variance=0.99,
        solver="full",
        random_state=0,
    ).fit(x, y)
    context = ContextOutputLocalToLocalPCAEncoder(
        patch_size=8,
        stride=8,
        input_variance=0.99,
        output_variance=0.99,
        solver="full",
        random_state=0,
        output_guard_band=2,
    ).fit(x, y)

    assert context.output_patch_size == 12
    assert context.component_counts["input_patches"] == plain.component_counts["input_patches"]
    assert context.transform_inputs(x).shape[1] == plain.transform_inputs(x).shape[1]

    z = context.transform_outputs(y)
    reconstructed = context.inverse_transform_outputs(z)
    assert reconstructed.shape == y.shape

    decoder = TorchPCADecoder(context)
    decoded = decoder(torch.from_numpy(z[:3])).detach().numpy()
    np.testing.assert_allclose(decoded, reconstructed[:3], rtol=1.0e-5, atol=1.0e-6)


def test_poisson_256_context_output_config_composes() -> None:
    config = compose_config(
        root=Path(__file__).resolve().parents[1],
        dataset="poisson",
        model="l2l",
        experiment="poisson_256_context_output_seed0",
    )

    assert config["mechanisms"] == {
        "two_scale": False,
        "coupling": False,
        "in_loop_loss": False,
    }
    assert config["pca"]["context_output"] == {
        "enabled": True,
        "guard_band": 8,
        "padding_mode": "edge",
    }
    assert int(config["patches"]["patch_size"]) == 64
    assert int(config["patches"]["stride"]) == 64
    assert config["patches"]["assembly"] == "average"
