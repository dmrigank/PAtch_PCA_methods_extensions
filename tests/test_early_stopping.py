from __future__ import annotations

import numpy as np
import torch

from lpcanet.train.loop import train_image_model


def test_image_training_stops_after_configured_validation_patience() -> None:
    x = np.zeros((8, 4, 4), dtype=np.float32)
    y = np.ones((8, 4, 4), dtype=np.float32)
    model = torch.nn.Conv2d(1, 1, kernel_size=1)

    result = train_image_model(
        model,
        x,
        y,
        x[:4],
        y[:4],
        epochs=20,
        batch_size=4,
        lr=0.0,
        early_stopping_patience=2,
    )

    assert result.stopped_early
    assert result.epochs_ran == 3
    assert len(result.train_loss) == 3
    assert len(result.val_loss) == 3
