"""Reusable PCA encoders for Patch PCA-Net experiments."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from lpcanet.assembly.patching import (
    assemble_patches_2d,
    assemble_weighted_patches_2d,
    extract_patches_2d,
    get_patch_slices,
)
from lpcanet.pca.randomized_svd import fit_pca
from lpcanet.utils.paths import normalize_path


class IdentityScaler:
    """Scaler API compatible no-op used when standardization is disabled."""

    def fit(self, data: np.ndarray) -> IdentityScaler:
        del data
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return np.asarray(data)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.fit(data).transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return np.asarray(data)


class BasePCAEncoder:
    """Common configuration and serialization behavior for PCA encoders."""

    def __init__(
        self,
        input_variance: int | float = 0.99,
        output_variance: int | float = 0.99,
        solver: str = "full",
        standardize: bool = True,
        oversampling: int = 20,
        n_iter: int = 4,
        random_state: int | None = None,
        include_edges: bool = False,
    ) -> None:
        self.input_variance = input_variance
        self.output_variance = output_variance
        self.solver = solver
        self.standardize = standardize
        self.oversampling = oversampling
        self.n_iter = n_iter
        self.random_state = random_state
        self.component_counts: dict[str, Any] = {}
        self.timings: dict[str, float] = {}
        self.is_fitted = False

    def save(self, path: str | Path) -> Path:
        """Save this encoder with joblib."""
        out_path = normalize_path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, out_path)
        return out_path

    @classmethod
    def load(cls, path: str | Path) -> BasePCAEncoder:
        """Load an encoder saved with joblib."""
        loaded = joblib.load(normalize_path(path))
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected {cls.__name__}, loaded {type(loaded).__name__}.")
        return loaded

    def _new_scaler(self) -> StandardScaler | IdentityScaler:
        return StandardScaler() if self.standardize else IdentityScaler()

    def _fit_pca(self, data: np.ndarray, n_components: int | float):
        return fit_pca(
            data,
            n_components,
            solver=self.solver,
            oversampling=self.oversampling,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(f"{type(self).__name__} has not been fitted.")


class GlobalPCAEncoder(BasePCAEncoder):
    """PCA encoder using one input PCA and one output PCA on full fields."""

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> GlobalPCAEncoder:
        """Fit global input and output PCA models."""
        x_flat, self.input_shape_ = _flatten_fields(x_train)
        y_flat, self.output_shape_ = _flatten_fields(y_train)

        start = perf_counter()
        self.input_scaler = self._new_scaler()
        x_scaled = self.input_scaler.fit_transform(x_flat)
        self.input_pca = self._fit_pca(x_scaled, self.input_variance)
        self.timings["fit_input_pca"] = perf_counter() - start

        start = perf_counter()
        self.output_scaler = self._new_scaler()
        y_scaled = self.output_scaler.fit_transform(y_flat)
        self.output_pca = self._fit_pca(y_scaled, self.output_variance)
        self.timings["fit_output_pca"] = perf_counter() - start

        self.component_counts = {
            "input": int(self.input_pca.n_components_),
            "output": int(self.output_pca.n_components_),
        }
        self.is_fitted = True
        return self

    def transform_inputs(self, x: np.ndarray) -> np.ndarray:
        """Transform full input fields to global PCA coordinates."""
        self._require_fitted()
        x_flat = _flatten_like(x, self.input_shape_, name="x")
        start = perf_counter()
        result = self.input_pca.transform(self.input_scaler.transform(x_flat))
        self.timings["transform_inputs"] = self.timings.get("transform_inputs", 0.0) + (
            perf_counter() - start
        )
        return result.astype(np.float32, copy=False)

    def transform_outputs(self, y: np.ndarray) -> np.ndarray:
        """Transform full output fields to global PCA coordinates."""
        self._require_fitted()
        y_flat = _flatten_like(y, self.output_shape_, name="y")
        start = perf_counter()
        result = self.output_pca.transform(self.output_scaler.transform(y_flat))
        self.timings["transform_outputs"] = self.timings.get("transform_outputs", 0.0) + (
            perf_counter() - start
        )
        return result.astype(np.float32, copy=False)

    def inverse_transform_outputs(self, z: np.ndarray) -> np.ndarray:
        """Invert global output PCA coordinates back to fields."""
        self._require_fitted()
        start = perf_counter()
        y_scaled = self.output_pca.inverse_transform(np.asarray(z))
        y_flat = self.output_scaler.inverse_transform(y_scaled)
        result = y_flat.reshape((y_flat.shape[0], *self.output_shape_))
        self.timings["inverse_transform_outputs"] = self.timings.get(
            "inverse_transform_outputs", 0.0
        ) + (perf_counter() - start)
        return result.astype(np.float32, copy=False)


class LocalToGlobalPCAEncoder(BasePCAEncoder):
    """Patchwise input PCA with one global output PCA."""

    include_edges: bool = False

    def __init__(
        self,
        patch_size: int,
        stride: int,
        input_variance: int | float = 0.99,
        output_variance: int | float = 0.99,
        solver: str = "full",
        standardize: bool = True,
        oversampling: int = 20,
        n_iter: int = 4,
        random_state: int | None = None,
        include_edges: bool = False,
    ) -> None:
        super().__init__(
            input_variance=input_variance,
            output_variance=output_variance,
            solver=solver,
            standardize=standardize,
            oversampling=oversampling,
            n_iter=n_iter,
            random_state=random_state,
        )
        self.patch_size = patch_size
        self.stride = stride
        self.include_edges = include_edges

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> LocalToGlobalPCAEncoder:
        """Fit patchwise input PCA and global output PCA."""
        self.grid_shape = _field_grid_shape(x_train)
        self.output_shape_ = _field_grid_shape(y_train)
        self.patch_slices = get_patch_slices(
            self.grid_shape,
            self.patch_size,
            self.stride,
            include_edges=self.include_edges,
        )
        x_patches = extract_patches_2d(
            x_train,
            self.patch_size,
            self.stride,
            flatten=True,
            include_edges=self.include_edges,
        )

        start = perf_counter()
        self.input_models = _fit_patch_models(self, x_patches, self.input_variance)
        self.timings["fit_input_pca"] = perf_counter() - start

        y_flat = np.asarray(y_train).reshape(y_train.shape[0], -1)
        start = perf_counter()
        self.output_scaler = self._new_scaler()
        y_scaled = self.output_scaler.fit_transform(y_flat)
        self.output_pca = self._fit_pca(y_scaled, self.output_variance)
        self.timings["fit_output_pca"] = perf_counter() - start

        input_counts = [int(model["pca"].n_components_) for model in self.input_models]
        self.component_counts = {
            "input_patches": input_counts,
            "input_total": int(sum(input_counts)),
            "output": int(self.output_pca.n_components_),
        }
        self.is_fitted = True
        return self

    def transform_inputs(self, x: np.ndarray) -> np.ndarray:
        """Transform fields to concatenated patch input PCA coordinates."""
        self._require_fitted()
        _check_field_shape(x, self.grid_shape, name="x")
        patches = extract_patches_2d(
            x,
            self.patch_size,
            self.stride,
            flatten=True,
            include_edges=self.include_edges,
        )
        start = perf_counter()
        result = _transform_patch_models(patches, self.input_models)
        self.timings["transform_inputs"] = self.timings.get("transform_inputs", 0.0) + (
            perf_counter() - start
        )
        return result.astype(np.float32, copy=False)

    def transform_outputs(self, y: np.ndarray) -> np.ndarray:
        """Transform full output fields to global PCA coordinates."""
        self._require_fitted()
        _check_field_shape(y, self.output_shape_, name="y")
        y_flat = np.asarray(y).reshape(y.shape[0], -1)
        start = perf_counter()
        result = self.output_pca.transform(self.output_scaler.transform(y_flat))
        self.timings["transform_outputs"] = self.timings.get("transform_outputs", 0.0) + (
            perf_counter() - start
        )
        return result.astype(np.float32, copy=False)

    def inverse_transform_outputs(self, z: np.ndarray) -> np.ndarray:
        """Invert global output PCA coordinates back to fields."""
        self._require_fitted()
        start = perf_counter()
        y_scaled = self.output_pca.inverse_transform(np.asarray(z))
        y_flat = self.output_scaler.inverse_transform(y_scaled)
        result = y_flat.reshape((y_flat.shape[0], *self.output_shape_))
        self.timings["inverse_transform_outputs"] = self.timings.get(
            "inverse_transform_outputs", 0.0
        ) + (perf_counter() - start)
        return result.astype(np.float32, copy=False)


class LocalToLocalPCAEncoder(BasePCAEncoder):
    """Patchwise input PCA and patchwise output PCA."""

    include_edges: bool = False

    def __init__(
        self,
        patch_size: int,
        stride: int,
        input_variance: int | float = 0.99,
        output_variance: int | float = 0.99,
        solver: str = "full",
        standardize: bool = True,
        oversampling: int = 20,
        n_iter: int = 4,
        random_state: int | None = None,
        assembly_weights: np.ndarray | None = None,
        include_edges: bool = False,
    ) -> None:
        super().__init__(
            input_variance=input_variance,
            output_variance=output_variance,
            solver=solver,
            standardize=standardize,
            oversampling=oversampling,
            n_iter=n_iter,
            random_state=random_state,
        )
        self.patch_size = patch_size
        self.stride = stride
        self.assembly_weights = assembly_weights
        self.include_edges = include_edges

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> LocalToLocalPCAEncoder:
        """Fit patchwise input and output PCA models."""
        self.grid_shape = _field_grid_shape(x_train)
        self.output_shape_ = _field_grid_shape(y_train)
        if self.grid_shape != self.output_shape_:
            raise ValueError(
                f"Input/output grids must match for L2L PCA, got "
                f"{self.grid_shape} and {self.output_shape_}."
            )
        self.patch_slices = get_patch_slices(
            self.grid_shape,
            self.patch_size,
            self.stride,
            include_edges=self.include_edges,
        )
        x_patches = extract_patches_2d(
            x_train,
            self.patch_size,
            self.stride,
            flatten=True,
            include_edges=self.include_edges,
        )
        y_patches = extract_patches_2d(
            y_train,
            self.patch_size,
            self.stride,
            flatten=True,
            include_edges=self.include_edges,
        )

        start = perf_counter()
        self.input_models = _fit_patch_models(self, x_patches, self.input_variance)
        self.timings["fit_input_pca"] = perf_counter() - start

        start = perf_counter()
        self.output_models = _fit_patch_models(self, y_patches, self.output_variance)
        self.timings["fit_output_pca"] = perf_counter() - start

        input_counts = [int(model["pca"].n_components_) for model in self.input_models]
        output_counts = [int(model["pca"].n_components_) for model in self.output_models]
        self.component_counts = {
            "input_patches": input_counts,
            "input_total": int(sum(input_counts)),
            "output_patches": output_counts,
            "output_total": int(sum(output_counts)),
        }
        self.is_fitted = True
        return self

    def transform_inputs(self, x: np.ndarray) -> np.ndarray:
        """Transform fields to concatenated patch input PCA coordinates."""
        self._require_fitted()
        _check_field_shape(x, self.grid_shape, name="x")
        patches = extract_patches_2d(
            x,
            self.patch_size,
            self.stride,
            flatten=True,
            include_edges=self.include_edges,
        )
        start = perf_counter()
        result = _transform_patch_models(patches, self.input_models)
        self.timings["transform_inputs"] = self.timings.get("transform_inputs", 0.0) + (
            perf_counter() - start
        )
        return result.astype(np.float32, copy=False)

    def transform_outputs(self, y: np.ndarray) -> np.ndarray:
        """Transform fields to concatenated patch output PCA coordinates."""
        self._require_fitted()
        _check_field_shape(y, self.output_shape_, name="y")
        patches = extract_patches_2d(
            y,
            self.patch_size,
            self.stride,
            flatten=True,
            include_edges=self.include_edges,
        )
        start = perf_counter()
        result = _transform_patch_models(patches, self.output_models)
        self.timings["transform_outputs"] = self.timings.get("transform_outputs", 0.0) + (
            perf_counter() - start
        )
        return result.astype(np.float32, copy=False)

    def inverse_transform_outputs(self, z: np.ndarray) -> np.ndarray:
        """Invert concatenated patch output PCA coordinates back to fields."""
        self._require_fitted()
        start = perf_counter()
        patches = _inverse_patch_models(np.asarray(z), self.output_models, self.patch_size)
        if self.assembly_weights is None:
            result = assemble_patches_2d(
                patches,
                self.output_shape_,
                self.patch_size,
                self.stride,
                mode="average",
                include_edges=self.include_edges,
            )
        else:
            result = assemble_weighted_patches_2d(
                patches,
                self.output_shape_,
                self.patch_size,
                self.stride,
                self.assembly_weights,
                include_edges=self.include_edges,
            )
        self.timings["inverse_transform_outputs"] = self.timings.get(
            "inverse_transform_outputs", 0.0
        ) + (perf_counter() - start)
        return result.astype(np.float32, copy=False)


def _fit_patch_models(
    encoder: BasePCAEncoder,
    patches: np.ndarray,
    n_components: int | float,
) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for patch_index in range(patches.shape[1]):
        scaler = encoder._new_scaler()
        patch_scaled = scaler.fit_transform(patches[:, patch_index, :])
        pca = encoder._fit_pca(patch_scaled, n_components)
        models.append({"scaler": scaler, "pca": pca})
    return models


def _transform_patch_models(patches: np.ndarray, models: list[dict[str, Any]]) -> np.ndarray:
    latents: list[np.ndarray] = []
    if patches.shape[1] != len(models):
        raise ValueError(f"Expected {len(models)} patches, got {patches.shape[1]}.")
    for patch_index, model in enumerate(models):
        scaled = model["scaler"].transform(patches[:, patch_index, :])
        latents.append(model["pca"].transform(scaled))
    return np.concatenate(latents, axis=1)


def _inverse_patch_models(
    z: np.ndarray,
    models: list[dict[str, Any]],
    patch_size: int,
) -> np.ndarray:
    patches: list[np.ndarray] = []
    start = 0
    for model in models:
        n_components = int(model["pca"].n_components_)
        end = start + n_components
        if end > z.shape[1]:
            raise ValueError(
                f"Latent output has too few columns: needed at least {end}, got {z.shape[1]}."
            )
        patch_scaled = model["pca"].inverse_transform(z[:, start:end])
        patch = model["scaler"].inverse_transform(patch_scaled)
        patches.append(patch.reshape(z.shape[0], patch_size, patch_size))
        start = end
    if start != z.shape[1]:
        raise ValueError(f"Latent output has extra columns: consumed {start}, got {z.shape[1]}.")
    return np.stack(patches, axis=1)


def _flatten_fields(fields: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    fields = np.asarray(fields)
    if fields.ndim < 2:
        raise ValueError(f"Expected fields with leading sample dimension, got {fields.shape}.")
    return fields.reshape(fields.shape[0], -1), tuple(fields.shape[1:])


def _flatten_like(fields: np.ndarray, expected_shape: tuple[int, ...], name: str) -> np.ndarray:
    fields = np.asarray(fields)
    if tuple(fields.shape[1:]) != expected_shape:
        raise ValueError(
            f"{name} has incompatible sample shape {tuple(fields.shape[1:])}; "
            f"expected {expected_shape}."
        )
    return fields.reshape(fields.shape[0], -1)


def _field_grid_shape(fields: np.ndarray) -> tuple[int, int]:
    fields = np.asarray(fields)
    if fields.ndim != 3:
        raise ValueError(f"Expected fields with shape (N, H, W), got {fields.shape}.")
    return (int(fields.shape[1]), int(fields.shape[2]))


def _check_field_shape(fields: np.ndarray, expected: tuple[int, int], name: str) -> None:
    fields = np.asarray(fields)
    if fields.ndim != 3 or tuple(fields.shape[1:]) != expected:
        raise ValueError(f"{name} must have shape (N, {expected[0]}, {expected[1]}), got {fields.shape}.")
