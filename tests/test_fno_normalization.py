from __future__ import annotations

import numpy as np
import pytest
import torch

from lpcanet.models.fno import FNO2d
from lpcanet.train.normalization import FieldNormalizer


def test_field_normalizer_uses_training_scale_and_round_trips_target() -> None:
    x_train = np.arange(32, dtype=np.float32).reshape(2, 4, 4)
    y_train = (1.0e-3 * x_train - 2.0e-2).astype(np.float32)
    normalizer = FieldNormalizer.fit(x_train, y_train)

    assert np.mean(normalizer.transform_input(x_train)) == pytest.approx(0.0, abs=1.0e-6)
    assert np.std(normalizer.transform_target(y_train)) == pytest.approx(1.0, abs=1.0e-6)
    np.testing.assert_allclose(
        normalizer.inverse_transform_target(normalizer.transform_target(y_train)),
        y_train,
        rtol=1.0e-6,
        atol=1.0e-8,
    )


def test_fno_padding_preserves_output_shape_and_input_dependence() -> None:
    torch.manual_seed(0)
    model = FNO2d(width=4, modes=3, num_layers=2, hidden_size=8, padding=2)
    x = torch.randn(2, 1, 12, 12)
    output = model(x)

    assert output.shape == (2, 1, 12, 12)
    assert not torch.allclose(output[0], output[1])
