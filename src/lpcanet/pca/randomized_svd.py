"""Randomized PCA helpers."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.utils.extmath import randomized_svd


def make_pca(
    n_components: int | float,
    *,
    solver: str = "full",
    oversampling: int = 20,
    n_iter: int = 4,
    random_state: int | None = None,
) -> PCA:
    """Create a scikit-learn PCA instance for the requested solver."""
    if solver == "full":
        return PCA(n_components=n_components, svd_solver="full")
    if solver == "randomized":
        if isinstance(n_components, float):
            raise TypeError(
                "Randomized PCA requires an integer n_components during fit; "
                "use fit_pca to support variance thresholds."
            )
        return PCA(
            n_components=n_components,
            svd_solver="randomized",
            iterated_power=n_iter,
            n_oversamples=oversampling,
            random_state=random_state,
        )
    raise ValueError(f"Unsupported PCA solver {solver!r}; expected 'full' or 'randomized'.")


def fit_pca(
    data: np.ndarray,
    n_components: int | float,
    *,
    solver: str = "full",
    oversampling: int = 20,
    n_iter: int = 4,
    random_state: int | None = None,
) -> PCA:
    """Fit PCA with full or deterministic randomized solver.

    For randomized PCA with a variance threshold, use an adaptive low-rank SVD
    search so the fit stops once the requested retained variance is reached.
    This matches the timing benchmark used for the paper's PCA-scaling figure.
    """
    if solver == "full" or isinstance(n_components, int):
        pca = make_pca(
            n_components,
            solver=solver,
            oversampling=oversampling,
            n_iter=n_iter,
            random_state=random_state,
        )
        return pca.fit(data)

    if solver != "randomized":
        raise ValueError(f"Unsupported PCA solver {solver!r}; expected 'full' or 'randomized'.")
    if not 0.0 < n_components <= 1.0:
        raise ValueError("Variance threshold n_components must be in (0, 1].")

    max_components = min(data.shape)
    if max_components <= 1:
        return make_pca(n_components, solver="full").fit(data)
    return _fit_adaptive_randomized_pca(
        data,
        variance_threshold=float(n_components),
        oversampling=oversampling,
        n_iter=n_iter,
        random_state=random_state,
    )


def truncate_pca_to_variance(pca: PCA, variance_threshold: float) -> PCA:
    """Truncate a fitted PCA object to a cumulative explained variance target."""
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n_keep = int(np.searchsorted(cumulative, variance_threshold, side="left") + 1)
    n_keep = min(n_keep, pca.components_.shape[0])
    pca.components_ = pca.components_[:n_keep]
    pca.explained_variance_ = pca.explained_variance_[:n_keep]
    pca.explained_variance_ratio_ = pca.explained_variance_ratio_[:n_keep]
    pca.singular_values_ = pca.singular_values_[:n_keep]
    pca.n_components_ = n_keep
    return pca


def _fit_adaptive_randomized_pca(
    data: np.ndarray,
    *,
    variance_threshold: float,
    oversampling: int,
    n_iter: int,
    random_state: int | None,
    min_k: int = 16,
    growth_factor: float = 1.5,
) -> PCA:
    """Fit a PCA-compatible object by adaptively increasing randomized SVD rank."""
    matrix = np.asarray(data, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"PCA expects a 2D matrix, got shape {matrix.shape}.")
    n_samples, n_features = matrix.shape
    mean = np.mean(matrix, axis=0)
    centered = matrix - mean
    total_variance = float(np.sum(centered * centered))
    max_rank = min(n_samples, n_features)
    if max_rank <= 1 or total_variance <= 0.0:
        return make_pca(variance_threshold, solver="full").fit(matrix)

    # Match sklearn PCA's effective limit for full-rank decompositions.
    max_components = max_rank - 1
    k = min(max_components, max(1, int(min_k)))
    singular_values = np.empty(0, dtype=np.float64)
    components = np.empty((0, n_features), dtype=np.float64)
    while True:
        _, singular_values, components = randomized_svd(
            centered,
            n_components=k,
            n_iter=n_iter,
            n_oversamples=oversampling,
            random_state=random_state,
        )
        cumulative = np.cumsum(singular_values**2) / max(total_variance, 1e-30)
        achieved = float(cumulative[-1]) if cumulative.size else 0.0
        if achieved >= variance_threshold or k >= max_components:
            break
        next_k = int(np.ceil(k * growth_factor))
        k = min(max_components, max(next_k, k + 1))

    if cumulative.size and cumulative[-1] >= variance_threshold:
        n_keep = int(np.searchsorted(cumulative, variance_threshold, side="left") + 1)
    else:
        n_keep = int(len(singular_values))
    n_keep = max(1, min(n_keep, len(singular_values)))

    pca = PCA(
        n_components=variance_threshold,
        svd_solver="randomized",
        iterated_power=n_iter,
        n_oversamples=oversampling,
        random_state=random_state,
    )
    pca.mean_ = mean
    pca.components_ = components[:n_keep]
    pca.singular_values_ = singular_values[:n_keep]
    pca.explained_variance_ = (pca.singular_values_**2) / max(n_samples - 1, 1)
    pca.explained_variance_ratio_ = (pca.singular_values_**2) / max(total_variance, 1e-30)
    pca.n_components_ = n_keep
    pca.n_samples_ = n_samples
    pca.n_features_in_ = n_features
    pca._fit_svd_solver = "randomized"

    total_explained_variance = total_variance / max(n_samples - 1, 1)
    if n_keep < max_rank:
        remaining = max(total_explained_variance - float(np.sum(pca.explained_variance_)), 0.0)
        pca.noise_variance_ = remaining / max(max_rank - n_keep, 1)
    else:
        pca.noise_variance_ = 0.0
    return pca
