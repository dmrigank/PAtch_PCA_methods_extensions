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
        basis_alignment: bool = False,
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
        self.basis_alignment = bool(basis_alignment)

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
            if self.basis_alignment:
                raise ValueError(
                    "basis_alignment=True is not supported together with guard_band > 0."
                )
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

        if self.basis_alignment:
            # Store reconstructions before rotation for invariance check (10 samples)
            _n_check = min(10, y_patches.shape[0])
            _y_check = y_patches[:_n_check]
            _pre_recon = _patch_encode_decode(_y_check, self.output_models)
            _pre_orth_errors = _basis_orthonormality_errors(self.output_models)
            # Run iterative coupled Procrustes alignment on output bases
            start_align = perf_counter()
            self.alignment_stats = _align_bases_procrustes(
                self.output_models, self.patch_slices, self.patch_size
            )
            self.timings["basis_alignment"] = perf_counter() - start_align
            # Correctness check 1: rotation must not degrade basis orthonormality
            _check_basis_orthonormality(self.output_models, pre_errors=_pre_orth_errors)
            # Correctness check 2: reconstruction with rotated basis must be identical
            _post_recon = _patch_encode_decode(_y_check, self.output_models)
            _check_reconstruction_invariance(_pre_recon, _post_recon)

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


def _align_bases_procrustes(
    models: list[dict[str, Any]],
    patch_slices: list[tuple[slice, slice]],
    patch_size: int,
    *,
    max_iters: int = 50,
    tol: float = 1e-6,
) -> dict[str, Any]:
    """Iterative coupled orthogonal Procrustes alignment of per-patch PCA bases.

    Minimizes sum_{seams} ||E_p R_p - E_q R_q||_F^2 subject to R_p^T R_p = I
    for all patches p, where E_p is the edge-trace matrix (columns of Phi_p at
    seam pixels, transposed to (edge_len, k_p)).  Sweeps patch by patch, solving
    each single-patch Procrustes in closed form via SVD while holding neighbors
    fixed.  Convergence declared when the relative objective change < tol.

    When k_p != k_q at a seam, the smaller-k side is zero-padded to k_p columns
    for the Procrustes solve of patch p's rotation (flagged in the print output).

    Modifies models[p]["pca"].components_ in-place after convergence.
    The scaler and PCA mean are in pixel space and are NOT affected.

    Returns a summary dict with objective_before, objective_after,
    objective_reduction_pct, n_sweeps, and unequal_k.
    """
    P = patch_size
    row_starts = sorted({s[0].start for s in patch_slices})
    col_starts = sorted({s[1].start for s in patch_slices})
    n_rows, n_cols = len(row_starts), len(col_starts)
    if len(models) != n_rows * n_cols:
        raise ValueError(
            f"Expected {n_rows * n_cols} models for a {n_rows}x{n_cols} patch grid, "
            f"got {len(models)}."
        )

    # Edge pixel indices in a row-major P×P patch
    right_edge  = np.arange(P - 1, P * P, P)    # rightmost column
    left_edge   = np.arange(0, P * P, P)          # leftmost column
    bottom_edge = np.arange((P - 1) * P, P * P)  # bottom row
    top_edge    = np.arange(P)                    # top row

    # Seam list: (p_idx, q_idx, edge_pixels_in_p, edge_pixels_in_q)
    seams: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for r in range(n_rows):
        for c in range(n_cols - 1):  # vertical seams (left-right adjacency)
            seams.append((r * n_cols + c, r * n_cols + c + 1, right_edge, left_edge))
    for r in range(n_rows - 1):
        for c in range(n_cols):      # horizontal seams (top-bottom adjacency)
            seams.append((r * n_cols + c, (r + 1) * n_cols + c, bottom_edge, top_edge))

    print(
        f"[basis_alignment] {n_rows}x{n_cols} grid, {len(seams)} seams "
        f"({n_rows * (n_cols - 1)} vertical, {(n_rows - 1) * n_cols} horizontal)"
    )

    k_list = [int(m["pca"].n_components_) for m in models]
    unique_k = sorted(set(k_list))
    unequal_k = len(unique_k) > 1
    if unequal_k:
        print(
            f"[basis_alignment] NOTE: unequal k across patches — unique values: {unique_k}. "
            "For seams with k mismatch, the smaller-k neighbor is zero-padded to k_p "
            "columns when solving for patch p's rotation."
        )

    # Per-patch seam membership: patch_idx → list of (seam_idx, side='p'|'q')
    patch_seam_lookup: dict[int, list[tuple[int, str]]] = {}
    for seam_idx, (p, q, _, _) in enumerate(seams):
        patch_seam_lookup.setdefault(p, []).append((seam_idx, "p"))
        patch_seam_lookup.setdefault(q, []).append((seam_idx, "q"))

    # Initialize rotations to identity
    Rs: list[np.ndarray] = [np.eye(k, dtype=np.float64) for k in k_list]

    def _edge_trace(patch_idx: int, edge_idx: np.ndarray) -> np.ndarray:
        """Return edge-trace matrix from original components_: shape (edge_len, k)."""
        return np.asarray(
            models[patch_idx]["pca"].components_[:, edge_idx], dtype=np.float64
        ).T  # (edge_len, k)

    def _objective(Rs_eval: list[np.ndarray]) -> float:
        total = 0.0
        for p, q, ep_idx, eq_idx in seams:
            k_p, k_q = k_list[p], k_list[q]
            A = _edge_trace(p, ep_idx) @ Rs_eval[p]   # (P, k_p)
            B = _edge_trace(q, eq_idx) @ Rs_eval[q]   # (P, k_q)
            K = max(k_p, k_q)
            if k_p < K:
                A = np.pad(A, [(0, 0), (0, K - k_p)])
            if k_q < K:
                B = np.pad(B, [(0, 0), (0, K - k_q)])
            total += float(np.sum((A - B) ** 2))
        return total

    obj_before = _objective([np.eye(k, dtype=np.float64) for k in k_list])
    print(f"[basis_alignment] Objective BEFORE alignment: {obj_before:.6e}")

    prev_obj = obj_before
    n_sweeps = 0
    rel_change = 0.0
    for sweep in range(max_iters):
        for p in range(len(models)):
            k_p = k_list[p]
            A_parts: list[np.ndarray] = []
            B_parts: list[np.ndarray] = []
            for seam_idx, side in patch_seam_lookup.get(p, []):
                pp, qq, ep_idx, eq_idx = seams[seam_idx]
                if side == "p":
                    # patch p is left/top: its seam pixels are ep_idx
                    E_self = _edge_trace(p, ep_idx)                     # (P, k_p)
                    k_nb   = k_list[qq]
                    B_raw  = _edge_trace(qq, eq_idx) @ Rs[qq]           # (P, k_nb)
                else:
                    # patch p is right/bottom: its seam pixels are eq_idx
                    E_self = _edge_trace(p, eq_idx)                     # (P, k_p)
                    k_nb   = k_list[pp]
                    B_raw  = _edge_trace(pp, ep_idx) @ Rs[pp]           # (P, k_nb)
                # Pad or truncate neighbor contribution to k_p columns
                if k_nb > k_p:
                    B_contrib = B_raw[:, :k_p]
                elif k_nb < k_p:
                    B_contrib = np.pad(B_raw, [(0, 0), (0, k_p - k_nb)])
                else:
                    B_contrib = B_raw
                A_parts.append(E_self)     # (P, k_p)
                B_parts.append(B_contrib)  # (P, k_p)
            if not A_parts:
                continue
            A_stack = np.vstack(A_parts)  # (n_seams_of_p * P, k_p)
            B_stack = np.vstack(B_parts)
            M = A_stack.T @ B_stack       # (k_p, k_p)
            U, _, Vh = np.linalg.svd(M)
            Rs[p] = U @ Vh                # R_p = U V^T (optimal orthogonal Procrustes)

        obj = _objective(Rs)
        rel_change = abs(prev_obj - obj) / max(abs(prev_obj), 1e-30)
        n_sweeps = sweep + 1
        prev_obj = obj
        if rel_change < tol:
            print(
                f"[basis_alignment] Converged after {n_sweeps} sweep(s), "
                f"rel_change={rel_change:.2e}"
            )
            break
    else:
        print(
            f"[basis_alignment] Did not converge in {max_iters} sweeps "
            f"(last rel_change={rel_change:.2e})"
        )

    obj_after = _objective(Rs)
    obj_reduction_pct = (obj_before - obj_after) / max(obj_before, 1e-30) * 100.0
    print(
        f"[basis_alignment] Objective AFTER  alignment: {obj_after:.6e}  "
        f"(reduction: {obj_reduction_pct:.1f}% in {n_sweeps} sweep(s))"
    )
    if obj_reduction_pct < 5.0:
        print(
            "[basis_alignment] WARNING: objective reduction < 5%. "
            "Bases may already be near-aligned, or the misalignment is not "
            "expressible as a pure rotation — null result may be baked in before training."
        )

    # Apply rotations in-place: C_new = R_p^T @ C_old
    # New forward:  z_new = (X - mean) @ C_new.T = z_old @ R_p
    # New inverse:  z_new @ C_new = z_old @ R_p @ R_p^T @ C_old = z_old @ C_old (invariant) ✓
    for p_idx, model in enumerate(models):
        C = np.asarray(model["pca"].components_, dtype=np.float64)
        model["pca"].components_ = Rs[p_idx].T @ C

    return {
        "objective_before": float(obj_before),
        "objective_after": float(obj_after),
        "objective_reduction_pct": float(obj_reduction_pct),
        "n_sweeps": int(n_sweeps),
        "unequal_k": unequal_k,
    }


def _patch_encode_decode(
    y_patches: np.ndarray,
    models: list[dict[str, Any]],
) -> np.ndarray:
    """Encode then decode each patch using the current models; returns (n, n_patches, P²)."""
    n = y_patches.shape[0]
    n_patches = y_patches.shape[1]
    recons: list[np.ndarray] = []
    for pidx in range(n_patches):
        patch = y_patches[:, pidx, :]
        scaled = models[pidx]["scaler"].transform(patch)
        latent = models[pidx]["pca"].transform(scaled)
        back_scaled = models[pidx]["pca"].inverse_transform(latent)
        back = models[pidx]["scaler"].inverse_transform(back_scaled)
        recons.append(back)
    return np.stack(recons, axis=1)  # (n, n_patches, P²)


def _basis_orthonormality_errors(models: list[dict[str, Any]]) -> list[float]:
    """Return ||C C^T - I||_F for each patch's components_."""
    errors = []
    for model in models:
        C = np.asarray(model["pca"].components_, dtype=np.float64)
        errors.append(float(np.linalg.norm(C @ C.T - np.eye(len(C)))))
    return errors


def _check_basis_orthonormality(
    models: list[dict[str, Any]],
    pre_errors: list[float] | None = None,
    degradation_factor: float = 10.0,
) -> None:
    """Verify the rotation did not degrade basis orthonormality.

    The baseline per-patch PCA stores components_ in float32 (when the training
    data is float32), giving pre-rotation ||CC^T - I||_F ~ 1e-6.  An orthogonal
    rotation preserves this error exactly (Frobenius norm is invariant under
    orthogonal transformations), so the post-rotation error should be no worse
    than degradation_factor × pre-rotation error.  A guard-band crop, by
    contrast, can push this error to O(1), which is what we want to rule out.

    If pre_errors is None, checks that post-rotation error < 1e-4 absolutely.
    """
    post_errors = _basis_orthonormality_errors(models)
    max_post = max(post_errors)

    if pre_errors is None:
        tol = 1e-4
        if max_post > tol:
            raise RuntimeError(
                f"[basis_alignment] FAILED orthonormality check: "
                f"max ||CC^T - I||_F = {max_post:.3e} > {tol:.0e}. "
                "This indicates a bug in the rotation application."
            )
        print(
            f"[basis_alignment] Orthonormality check PASSED "
            f"(max ||CC^T - I||_F = {max_post:.2e})."
        )
        return

    max_pre = max(pre_errors)
    threshold = max(max_pre * degradation_factor, 1e-4)
    if max_post > threshold:
        raise RuntimeError(
            f"[basis_alignment] FAILED orthonormality check: "
            f"post-rotation max ||CC^T - I||_F = {max_post:.3e} is more than "
            f"{degradation_factor}x worse than pre-rotation {max_pre:.3e}. "
            "This would indicate a bug in the rotation application."
        )
    print(
        f"[basis_alignment] Orthonormality check PASSED: "
        f"pre-rotation max ||CC^T - I||_F = {max_pre:.2e}, "
        f"post-rotation = {max_post:.2e} "
        f"(float32 source precision; rotation does not degrade orthonormality)."
    )


def _check_reconstruction_invariance(
    pre_recon: np.ndarray,
    post_recon: np.ndarray,
    rtol: float = 1e-4,
) -> None:
    """Verify that rotating the basis does not change the reconstruction.

    Tolerance is set to 1e-4 because PCA components_ may be stored in float32
    (sklearn inherits the dtype from training data), giving rounding errors of
    order 1e-6 per component that can accumulate to ~1e-5 over multiple patches.
    A true reconstruction-changing bug would produce much larger errors (~1e-1).
    """
    max_abs = float(np.max(np.abs(pre_recon - post_recon)))
    scale = float(np.max(np.abs(pre_recon))) + 1e-12
    rel_err = max_abs / scale
    if rel_err > rtol:
        raise RuntimeError(
            f"[basis_alignment] FAILED reconstruction invariance: "
            f"max |Δ| = {max_abs:.3e}, rel = {rel_err:.3e} > {rtol:.0e}. "
            "The rotation changes the reconstruction — bug in rotation application."
        )
    print(
        f"[basis_alignment] Reconstruction invariance check PASSED "
        f"(max relative diff = {rel_err:.2e})."
    )


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
