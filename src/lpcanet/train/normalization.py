"""Training-split normalization for image-to-image operator models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FieldNormalizer:
    """Scalar standardization fitted only on training input and target fields."""

    enabled: bool
    input_mean: float
    input_std: float
    target_mean: float
    target_std: float
    eps: float

    @classmethod
    def fit(
        cls,
        x_train: np.ndarray,
        y_train: np.ndarray,
        *,
        enabled: bool = True,
        eps: float = 1.0e-8,
    ) -> FieldNormalizer:
        if eps <= 0.0:
            raise ValueError("normalization eps must be positive.")
        if not enabled:
            return cls(
                enabled=False,
                input_mean=0.0,
                input_std=1.0,
                target_mean=0.0,
                target_std=1.0,
                eps=eps,
            )
        return cls(
            enabled=True,
            input_mean=float(np.mean(x_train, dtype=np.float64)),
            input_std=max(float(np.std(x_train, dtype=np.float64)), eps),
            target_mean=float(np.mean(y_train, dtype=np.float64)),
            target_std=max(float(np.std(y_train, dtype=np.float64)), eps),
            eps=eps,
        )

    def transform_input(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if not self.enabled:
            return values
        return ((values - self.input_mean) / self.input_std).astype(np.float32, copy=False)

    def transform_target(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if not self.enabled:
            return values
        return ((values - self.target_mean) / self.target_std).astype(np.float32, copy=False)

    def inverse_transform_target(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if not self.enabled:
            return values
        return (values * self.target_std + self.target_mean).astype(np.float32, copy=False)

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "enabled": self.enabled,
            "input_mean": self.input_mean,
            "input_std": self.input_std,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
            "eps": self.eps,
        }
