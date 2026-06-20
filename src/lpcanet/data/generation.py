"""Deterministic Poisson and Darcy dataset generation."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.fft import dstn, idctn, idstn
from tqdm import tqdm

from lpcanet.data.cache import content_hash
from lpcanet.data.splits import make_splits


def generate_dataset(
    *,
    dataset: str,
    output: str | Path,
    resolution: int,
    n_samples: int = 10_000,
    seed: int = 0,
    alpha: float = 2.0,
    tau: float = 3.0,
    batch_size: int = 32,
    workers: int = 1,
    compress: bool = False,
    overwrite: bool = False,
) -> Path:
    """Generate and cache a steady-state PDE dataset."""
    name = dataset.lower()
    if name not in {"poisson", "darcy"}:
        raise ValueError(f"Unsupported dataset {dataset!r}; expected 'poisson' or 'darcy'.")
    _validate_generation_args(resolution, n_samples, batch_size, workers)

    output_path = Path(output).expanduser()
    metadata = {
        "dataset": name,
        "resolution": int(resolution),
        "n_samples": int(n_samples),
        "seed": int(seed),
        "alpha": float(alpha),
        "tau": float(tau),
        "batch_size": int(batch_size),
        "workers": int(workers),
        "compress": bool(compress),
        "generator_version": 1,
        "poisson_convention": "legacy_delta_u_equals_f",
        "darcy_coefficient": "thresholded_grf_4_12",
        "darcy_forcing": 1.0,
    }
    generation_hash = content_hash(metadata)
    metadata["content_hash"] = generation_hash
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")

    if output_path.exists() and not overwrite:
        existing = _load_metadata(metadata_path)
        if existing.get("content_hash") == generation_hash:
            return output_path
        raise FileExistsError(
            f"Dataset exists with a different generation config: {output_path}. "
            "Pass --overwrite or choose another output path."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{name}_{resolution}_",
        dir=output_path.parent,
    ) as temp_dir:
        temp_root = Path(temp_dir)
        if name == "poisson":
            arrays = _generate_poisson_memmaps(
                temp_root,
                resolution=resolution,
                n_samples=n_samples,
                seed=seed,
                alpha=alpha,
                tau=tau,
                batch_size=batch_size,
            )
        else:
            arrays = _generate_darcy_memmaps(
                temp_root,
                resolution=resolution,
                n_samples=n_samples,
                seed=seed,
                alpha=alpha,
                tau=tau,
                batch_size=batch_size,
                workers=workers,
            )
        splits = make_splits(n_samples, seed=seed)
        payload: dict[str, np.ndarray] = {**arrays, **splits}
        save = np.savez_compressed if compress else np.savez
        temporary_output = output_path.with_suffix(output_path.suffix + ".partial")
        with temporary_output.open("wb") as handle:
            save(handle, **payload)
        temporary_output.replace(output_path)

    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def sample_grf(
    n_samples: int,
    *,
    resolution: int,
    alpha: float,
    tau: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample the legacy DCT Gaussian random field on a square grid."""
    xi = rng.normal(size=(n_samples, resolution, resolution))
    k1, k2 = np.meshgrid(
        np.arange(resolution, dtype=np.float64),
        np.arange(resolution, dtype=np.float64),
        indexing="ij",
    )
    coefficient = tau ** (alpha - 1.0) * (
        np.pi**2 * (k1**2 + k2**2) + tau**2
    ) ** (-alpha / 2.0)
    spectral = resolution * coefficient[None, :, :] * xi
    spectral[:, 0, 0] = 0.0
    return idctn(spectral, axes=(-2, -1), norm="ortho").astype(np.float32)


def solve_poisson_dirichlet_legacy(forcing: np.ndarray) -> np.ndarray:
    """Solve the legacy convention ``Delta u = f`` with zero Dirichlet data."""
    forcing = np.asarray(forcing, dtype=np.float64)
    if forcing.ndim != 3 or forcing.shape[1] != forcing.shape[2]:
        raise ValueError(f"forcing must have shape (N, D, D), got {forcing.shape}.")
    resolution = int(forcing.shape[1])
    interior = resolution - 2
    h = 1.0 / float(resolution - 1)
    rhs_hat = dstn(
        forcing[:, 1:-1, 1:-1],
        type=1,
        axes=(-2, -1),
        norm="ortho",
    )
    modes = np.arange(1, interior + 1, dtype=np.float64)
    eigenvalues = -4.0 / h**2 * np.sin(
        np.pi * modes / (2.0 * (interior + 1))
    ) ** 2
    denominator = eigenvalues[:, None] + eigenvalues[None, :]
    solution_inner = idstn(
        rhs_hat / denominator[None, :, :],
        type=1,
        axes=(-2, -1),
        norm="ortho",
    )
    solution = np.zeros_like(forcing)
    solution[:, 1:-1, 1:-1] = solution_inner
    return solution.astype(np.float32)


def solve_darcy_dirichlet(
    coefficient: np.ndarray,
    *,
    forcing: float = 1.0,
) -> np.ndarray:
    """Solve ``-div(a grad u) = forcing`` using arithmetic face coefficients."""
    coefficient = np.asarray(coefficient, dtype=np.float64)
    if coefficient.ndim != 2 or coefficient.shape[0] != coefficient.shape[1]:
        raise ValueError(f"coefficient must have shape (D, D), got {coefficient.shape}.")
    resolution = int(coefficient.shape[0])
    if resolution < 3:
        raise ValueError("resolution must be at least 3.")
    interior = resolution - 2
    h = 1.0 / float(resolution - 1)
    center = coefficient[1:-1, 1:-1]
    east = 0.5 * (center + coefficient[1:-1, 2:])
    west = 0.5 * (center + coefficient[1:-1, :-2])
    north = 0.5 * (center + coefficient[:-2, 1:-1])
    south = 0.5 * (center + coefficient[2:, 1:-1])

    indices = np.arange(interior * interior, dtype=np.int64).reshape(interior, interior)
    rows = [indices.reshape(-1)]
    cols = [indices.reshape(-1)]
    values = [((east + west + north + south) / h**2).reshape(-1)]

    horizontal_left = indices[:, :-1].reshape(-1)
    horizontal_right = indices[:, 1:].reshape(-1)
    horizontal_values = (-east[:, :-1] / h**2).reshape(-1)
    rows.extend([horizontal_left, horizontal_right])
    cols.extend([horizontal_right, horizontal_left])
    values.extend([horizontal_values, horizontal_values])

    vertical_top = indices[:-1, :].reshape(-1)
    vertical_bottom = indices[1:, :].reshape(-1)
    vertical_values = (-south[:-1, :] / h**2).reshape(-1)
    rows.extend([vertical_top, vertical_bottom])
    cols.extend([vertical_bottom, vertical_top])
    values.extend([vertical_values, vertical_values])

    matrix = scipy.sparse.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(interior * interior, interior * interior),
    ).tocsr()
    rhs = np.full(interior * interior, float(forcing), dtype=np.float64)
    solution = np.zeros((resolution, resolution), dtype=np.float64)
    solution[1:-1, 1:-1] = scipy.sparse.linalg.spsolve(matrix, rhs).reshape(
        interior,
        interior,
    )
    return solution.astype(np.float32)


def _generate_poisson_memmaps(
    temp_root: Path,
    *,
    resolution: int,
    n_samples: int,
    seed: int,
    alpha: float,
    tau: float,
    batch_size: int,
) -> dict[str, np.ndarray]:
    forcing = np.lib.format.open_memmap(
        temp_root / "f.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n_samples, resolution, resolution),
    )
    solution = np.lib.format.open_memmap(
        temp_root / "u.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n_samples, resolution, resolution),
    )
    rng = np.random.default_rng(seed)
    for start in tqdm(range(0, n_samples, batch_size), desc="Poisson data"):
        stop = min(start + batch_size, n_samples)
        forcing_batch = sample_grf(
            stop - start,
            resolution=resolution,
            alpha=alpha,
            tau=tau,
            rng=rng,
        )
        forcing[start:stop] = forcing_batch
        solution[start:stop] = solve_poisson_dirichlet_legacy(forcing_batch)
    forcing.flush()
    solution.flush()
    return {"f": forcing, "u": solution}


def _generate_darcy_memmaps(
    temp_root: Path,
    *,
    resolution: int,
    n_samples: int,
    seed: int,
    alpha: float,
    tau: float,
    batch_size: int,
    workers: int,
) -> dict[str, np.ndarray]:
    coefficient = np.lib.format.open_memmap(
        temp_root / "a.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n_samples, resolution, resolution),
    )
    solution = np.lib.format.open_memmap(
        temp_root / "u.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n_samples, resolution, resolution),
    )
    rng = np.random.default_rng(seed)
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for start in tqdm(range(0, n_samples, batch_size), desc="Darcy data"):
            stop = min(start + batch_size, n_samples)
            normal_fields = sample_grf(
                stop - start,
                resolution=resolution,
                alpha=alpha,
                tau=tau,
                rng=rng,
            )
            coefficient_batch = np.where(normal_fields >= 0.0, 12.0, 4.0).astype(
                np.float32
            )
            coefficient[start:stop] = coefficient_batch
            if executor is None:
                solution_batch = [
                    solve_darcy_dirichlet(sample_coefficient)
                    for sample_coefficient in coefficient_batch
                ]
            else:
                solution_batch = list(
                    executor.map(
                        solve_darcy_dirichlet,
                        coefficient_batch,
                        chunksize=1,
                    )
                )
            solution[start:stop] = np.stack(solution_batch)
    finally:
        if executor is not None:
            executor.shutdown()
    coefficient.flush()
    solution.flush()
    compact_forcing = np.ones((n_samples, 1, 1), dtype=np.float32)
    return {"a": coefficient, "f": compact_forcing, "u": solution}


def _validate_generation_args(
    resolution: int,
    n_samples: int,
    batch_size: int,
    workers: int,
) -> None:
    if resolution < 3:
        raise ValueError("resolution must be at least 3.")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if workers <= 0:
        raise ValueError("workers must be positive.")


def _load_metadata(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}
