from __future__ import annotations

from pathlib import Path

import numpy as np

from lpcanet.data.generation import generate_dataset
from lpcanet.metrics.residuals import darcy_residual


def test_poisson_generation_is_deterministic_and_cached(tmp_path: Path) -> None:
    output = tmp_path / "poisson_8.npz"
    first = generate_dataset(
        dataset="poisson",
        output=output,
        resolution=8,
        n_samples=12,
        seed=3,
        batch_size=5,
    )
    first_mtime = first.stat().st_mtime_ns
    second = generate_dataset(
        dataset="poisson",
        output=output,
        resolution=8,
        n_samples=12,
        seed=3,
        batch_size=5,
    )
    assert second == first
    assert second.stat().st_mtime_ns == first_mtime

    with np.load(output) as data:
        assert data["f"].shape == (12, 8, 8)
        assert data["u"].shape == (12, 8, 8)
        assert sorted(
            len(data[key])
            for key in ("train_idx", "val_idx", "test_idx")
        ) == [1, 2, 9]
        h = 1.0 / 7.0
        u = data["u"].astype(np.float64)
        laplacian = (
            u[:, 2:, 1:-1]
            + u[:, :-2, 1:-1]
            + u[:, 1:-1, 2:]
            + u[:, 1:-1, :-2]
            - 4.0 * u[:, 1:-1, 1:-1]
        ) / h**2
        assert np.sqrt(np.mean((laplacian - data["f"][:, 1:-1, 1:-1]) ** 2)) < 1e-5


def test_darcy_generation_uses_compact_constant_forcing(tmp_path: Path) -> None:
    output = generate_dataset(
        dataset="darcy",
        output=tmp_path / "darcy_8.npz",
        resolution=8,
        n_samples=4,
        seed=2,
        batch_size=2,
    )
    with np.load(output) as data:
        assert data["a"].shape == (4, 8, 8)
        assert data["u"].shape == (4, 8, 8)
        assert data["f"].shape == (4, 1, 1)
        assert set(np.unique(data["a"]).tolist()) == {4.0, 12.0}
        residual = darcy_residual(data["u"], data["a"], data["f"], dx=1.0 / 7.0)
        assert np.sqrt(np.mean(residual**2)) < 1e-4


def test_parallel_darcy_generation_matches_serial(tmp_path: Path) -> None:
    serial = generate_dataset(
        dataset="darcy",
        output=tmp_path / "serial.npz",
        resolution=8,
        n_samples=4,
        seed=7,
        batch_size=4,
        workers=1,
    )
    parallel = generate_dataset(
        dataset="darcy",
        output=tmp_path / "parallel.npz",
        resolution=8,
        n_samples=4,
        seed=7,
        batch_size=4,
        workers=2,
    )

    with np.load(serial) as serial_data, np.load(parallel) as parallel_data:
        np.testing.assert_array_equal(serial_data["a"], parallel_data["a"])
        np.testing.assert_allclose(serial_data["u"], parallel_data["u"], rtol=0.0, atol=0.0)
