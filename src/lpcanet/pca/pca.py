"""Reusable PCA encoders for Patch PCA-Net experiments."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from lpcanet.assembly.operators import prolongate_bilinear, restrict_average
from lpcanet.assembly.patching import (
    assemble_patches_2d,
    assemble_weighted_patches_2d,
    extract_patches_2d,
    get_patch_slices,
)
from lpcanet.pca.randomized_svd import fit_pca, truncate_pca_to_variance
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


class _CroppedPCA:
    """PCA wrapper holding guard-band-cropped basis vectors for a single patch.

    After fitting PCA on an enlarged (P+2g)×(P+2g) patch, we discard the guard
    rows/columns from each component vector and retain only the inner P×P core.
    The resulting object is a drop-in replacement for the sklearn PCA used by
    _transform_patch_models and _inverse_patch_models.
    """

    def __init__(self, components: np.ndarray, mean: np.ndarray) -> None:
        self.components_ = components   # (k, P²)
        self.mean_ = mean               # (P²,)
        self.n_components_ = int(components.shape[0])

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X) - self.mean_) @ self.components_.T

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        return np.asarray(Z) @ self.components_ + self.mean_


class _CroppedScaler:
    """Scaler wrapper holding guard-band-cropped mean/scale statistics.

    When standardize=True the per-feature mean and scale are cropped to the
    inner P×P core.  When standardize=False (IdentityScaler), mean=scale=None
    and this wrapper is a transparent no-op.
    """

    def __init__(self, mean: np.ndarray | None, scale: np.ndarray | None) -> None:
        self._mean = mean
        self._scale = scale

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return X if self._mean is None else (X - self._mean) / self._scale

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return X if self._mean is None else X * self._scale + self._mean


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
        guard_band: int = 0,
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
        self.guard_band = int(guard_band)

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

        start = perf_counter()
        self.input_models = _fit_patch_models(self, x_patches, self.input_variance)
        self.timings["fit_input_pca"] = perf_counter() - start

        start = perf_counter()
        if self.guard_band > 0:
            y_patches_ext = _extract_guarded_patches(
                y_train, self.patch_slices, self.patch_size, self.guard_band
            )
            output_models_ext = _fit_patch_models(self, y_patches_ext, self.output_variance)
            self.output_models = [
                _crop_patch_model(m, self.patch_size, self.guard_band)
                for m in output_models_ext
            ]
        else:
            y_patches = extract_patches_2d(
                y_train,
                self.patch_size,
                self.stride,
                flatten=True,
                include_edges=self.include_edges,
            )
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


class TwoScaleLocalToLocalPCAEncoder(LocalToLocalPCAEncoder):
    """Local input PCA with two-scale coarse-global plus local-residual output PCA."""

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
        coarse_factor: int = 4,
        coarse_components: int = 20,
        coarse_variance: float = 0.99,
        grid_size: int | None = None,
        n_samples: int | None = None,
    ) -> None:
        super().__init__(
            patch_size=patch_size,
            stride=stride,
            input_variance=input_variance,
            output_variance=output_variance,
            solver=solver,
            standardize=standardize,
            oversampling=oversampling,
            n_iter=n_iter,
            random_state=random_state,
            assembly_weights=assembly_weights,
            include_edges=include_edges,
        )
        if coarse_factor <= 0:
            raise ValueError("coarse_factor must be positive.")
        if coarse_components <= 0:
            raise ValueError("coarse_components must be positive.")
        if not 0.0 < coarse_variance <= 1.0:
            raise ValueError("coarse_variance must be in (0, 1].")
        self.coarse_factor = int(coarse_factor)
        self.coarse_components = int(coarse_components)
        self.coarse_variance = float(coarse_variance)
        if grid_size is not None and n_samples is not None:
            _validate_coarse_factor(
                (int(grid_size), int(grid_size)),
                int(n_samples),
                self.coarse_factor,
            )

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> TwoScaleLocalToLocalPCAEncoder:
        """Fit local input PCA, coarse output PCA, and residual patch output PCA."""
        self.grid_shape = _field_grid_shape(x_train)
        self.output_shape_ = _field_grid_shape(y_train)
        if self.grid_shape != self.output_shape_:
            raise ValueError(
                f"Input/output grids must match for two-scale L2L PCA, got "
                f"{self.grid_shape} and {self.output_shape_}."
            )
        _validate_coarse_factor(self.output_shape_, int(y_train.shape[0]), self.coarse_factor)
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

        start = perf_counter()
        y_coarse_flat = _flatten_coarse_fields(
            _restrict_fields_numpy(y_train, self.coarse_factor)
        )
        self.coarse_scaler = self._new_scaler()
        y_coarse_scaled = self.coarse_scaler.fit_transform(y_coarse_flat)
        self.coarse_pca = fit_pca(
            y_coarse_scaled,
            min(self.coarse_components, min(y_coarse_scaled.shape) - 1),
            solver="randomized",
            oversampling=self.oversampling,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self.coarse_pca = truncate_pca_to_variance(self.coarse_pca, self.coarse_variance)
        self.timings["fit_coarse_svd"] = perf_counter() - start

        start = perf_counter()
        coarse_latent = self.coarse_pca.transform(y_coarse_scaled)
        coarse_recon = self._coarse_inverse_transform(coarse_latent)
        residual = np.asarray(y_train, dtype=np.float64) - coarse_recon
        residual_patches = extract_patches_2d(
            residual,
            self.patch_size,
            self.stride,
            flatten=True,
            include_edges=self.include_edges,
        )
        self.output_models = _fit_patch_models(self, residual_patches, self.output_variance)
        self.timings["fit_output_residual_pca"] = perf_counter() - start
        self.timings["fit_output_pca"] = (
            self.timings["fit_coarse_svd"] + self.timings["fit_output_residual_pca"]
        )

        input_counts = [int(model["pca"].n_components_) for model in self.input_models]
        output_counts = [int(model["pca"].n_components_) for model in self.output_models]
        coarse_count = int(self.coarse_pca.n_components_)
        self.component_counts = {
            "input_patches": input_counts,
            "input_total": int(sum(input_counts)),
            "coarse_output": coarse_count,
            "output_patches": output_counts,
            "output_residual_total": int(sum(output_counts)),
            "output_total": int(coarse_count + sum(output_counts)),
        }
        self.is_fitted = True
        return self

    def transform_outputs(self, y: np.ndarray) -> np.ndarray:
        """Transform fields to coarse-global plus residual-patch coordinates."""
        self._require_fitted()
        _check_field_shape(y, self.output_shape_, name="y")
        start = perf_counter()
        y_coarse_flat = _flatten_coarse_fields(_restrict_fields_numpy(y, self.coarse_factor))
        coarse_latent = self.coarse_pca.transform(self.coarse_scaler.transform(y_coarse_flat))
        coarse_recon = self._coarse_inverse_transform(coarse_latent)
        residual = np.asarray(y, dtype=np.float64) - coarse_recon
        residual_patches = extract_patches_2d(
            residual,
            self.patch_size,
            self.stride,
            flatten=True,
            include_edges=self.include_edges,
        )
        residual_latent = _transform_patch_models(residual_patches, self.output_models)
        result = np.concatenate([coarse_latent, residual_latent], axis=1)
        self.timings["transform_outputs"] = self.timings.get("transform_outputs", 0.0) + (
            perf_counter() - start
        )
        return result.astype(np.float32, copy=False)

    def inverse_transform_outputs(self, z: np.ndarray) -> np.ndarray:
        """Decode coarse coordinates plus residual patch coordinates to fields."""
        start = perf_counter()
        coarse, residual = self.decompose_outputs(z)
        result = coarse.astype(np.float64, copy=False) + residual.astype(
            np.float64, copy=False
        )
        self.timings["inverse_transform_outputs"] = self.timings.get(
            "inverse_transform_outputs", 0.0
        ) + (perf_counter() - start)
        return result.astype(np.float32, copy=False)

    def decompose_outputs(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decode latent outputs into global-coarse and local-residual fields."""
        self._require_fitted()
        z = np.asarray(z)
        coarse_components = int(self.coarse_pca.n_components_)
        if z.shape[1] < coarse_components:
            raise ValueError(
                f"Latent output has too few columns for coarse code: "
                f"needed {coarse_components}, got {z.shape[1]}."
            )

        coarse = self._coarse_inverse_transform(z[:, :coarse_components])
        residual_z = z[:, coarse_components:]
        residual_patches = _inverse_patch_models(residual_z, self.output_models, self.patch_size)
        if self.assembly_weights is None:
            residual = assemble_patches_2d(
                residual_patches,
                self.output_shape_,
                self.patch_size,
                self.stride,
                mode="average",
                include_edges=self.include_edges,
            )
        else:
            residual = assemble_weighted_patches_2d(
                residual_patches,
                self.output_shape_,
                self.patch_size,
                self.stride,
                self.assembly_weights,
                include_edges=self.include_edges,
            )
        return (
            coarse.astype(np.float32, copy=False),
            residual.astype(np.float32, copy=False),
        )

    def _coarse_inverse_transform(self, coarse_latent: np.ndarray) -> np.ndarray:
        coarse_scaled = self.coarse_pca.inverse_transform(np.asarray(coarse_latent))
        coarse_flat = self.coarse_scaler.inverse_transform(coarse_scaled)
        coarse_shape = (
            coarse_flat.shape[0],
            self.output_shape_[0] // self.coarse_factor,
            self.output_shape_[1] // self.coarse_factor,
        )
        coarse_fields = coarse_flat.reshape(coarse_shape)
        return _prolongate_fields_numpy(coarse_fields, self.coarse_factor)


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


def _validate_coarse_factor(
    grid_shape: tuple[int, int],
    n_samples: int,
    coarse_factor: int,
) -> None:
    height, width = int(grid_shape[0]), int(grid_shape[1])
    if height != width:
        raise ValueError(f"Two-scale output requires square grids, got {grid_shape}.")
    if height % coarse_factor != 0 or width % coarse_factor != 0:
        raise ValueError(
            f"Grid shape {grid_shape} must be divisible by coarse_factor={coarse_factor}."
        )
    coarse_features = (height // coarse_factor) * (width // coarse_factor)
    if coarse_features >= n_samples:
        raise ValueError(
            "Coarse factor invariant violated: "
            f"(D/c)^2={coarse_features} must be < m={n_samples}. "
            "Increase pca.two_scale.coarse_factor for this resolution/sample count."
        )


def _restrict_fields_numpy(fields: np.ndarray, factor: int) -> np.ndarray:
    tensor = torch.as_tensor(np.asarray(fields), dtype=torch.float64)
    return restrict_average(tensor, factor=factor).detach().cpu().numpy()


def _prolongate_fields_numpy(fields: np.ndarray, factor: int) -> np.ndarray:
    tensor = torch.as_tensor(np.asarray(fields), dtype=torch.float64)
    return prolongate_bilinear(tensor, factor=factor).detach().cpu().numpy()


def _flatten_coarse_fields(fields: np.ndarray) -> np.ndarray:
    fields = np.asarray(fields)
    if fields.ndim != 3:
        raise ValueError(f"Expected coarse fields with shape (N, H, W), got {fields.shape}.")
    return fields.reshape(fields.shape[0], -1)


def _extract_guarded_patches(
    fields: np.ndarray,
    patch_slices: list[tuple[slice, slice]],
    patch_size: int,
    guard_band: int,
) -> np.ndarray:
    """Extract (patch_size+2g)×(patch_size+2g) patches using reflect padding at boundaries.

    For boundary patches, the guard band pixels that would fall outside the field are
    filled using reflect (not zero) padding so no artificial sharp edges are introduced.
    For interior patches all guard band pixels come from actual neighboring field data.
    Returns shape (N, num_patches, (patch_size+2g)²).
    """
    g = guard_band
    ext = patch_size + 2 * g
    N = fields.shape[0]
    padded = np.pad(fields, ((0, 0), (g, g), (g, g)), mode="reflect")
    result = np.empty((N, len(patch_slices), ext * ext), dtype=fields.dtype)
    for pidx, (rsl, csl) in enumerate(patch_slices):
        r0, c0 = rsl.start, csl.start
        result[:, pidx] = padded[:, r0 : r0 + ext, c0 : c0 + ext].reshape(N, -1)
    return result


def _crop_patch_model(
    model: dict[str, Any],
    patch_size: int,
    guard_band: int,
) -> dict[str, Any]:
    """Crop a guard-band-fitted patch model to the inner patch_size×patch_size core.

    Slices the PCA component vectors and scaler statistics to the P² pixels
    corresponding to the core patch, discarding the surrounding guard band rows/cols.
    The returned model is a drop-in replacement with the same interface.
    """
    g = guard_band
    ext = patch_size + 2 * g
    idx = np.arange(ext * ext).reshape(ext, ext)[g : g + patch_size, g : g + patch_size].ravel()

    old_pca = model["pca"]
    new_pca = _CroppedPCA(
        components=np.asarray(old_pca.components_[:, idx], dtype=np.float64),
        mean=np.asarray(old_pca.mean_[idx], dtype=np.float64),
    )

    old_scaler = model["scaler"]
    if isinstance(old_scaler, StandardScaler):
        new_scaler = _CroppedScaler(
            mean=np.asarray(old_scaler.mean_[idx], dtype=np.float64),
            scale=np.asarray(old_scaler.scale_[idx], dtype=np.float64),
        )
    else:
        new_scaler = _CroppedScaler(mean=None, scale=None)

    return {"scaler": new_scaler, "pca": new_pca}
