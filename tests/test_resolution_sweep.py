from __future__ import annotations

from pathlib import Path

import pytest

from lpcanet.train.factorial import expand_resolution_sweep_runs


def test_resolution_sweep_expands_main_grid_and_coarse_factors() -> None:
    root = Path(__file__).resolve().parents[1]
    runs = expand_resolution_sweep_runs(root=root, dataset="poisson", experiment="resolution_sweep")

    assert len(runs) == 45
    assert {run.resolution for run in runs} == {64, 128, 256}
    assert {run.baseline_model for run in runs} == {"l2l", "l2l_overlap", "full"}
    for resolution in (64, 128, 256):
        for model in ("l2l", "l2l_overlap", "full"):
            model_runs = [
                run for run in runs if run.resolution == resolution and run.baseline_model == model
            ]
            assert sorted(run.seed for run in model_runs) == [0, 1, 2, 3, 4]

    full_runs = [run for run in runs if run.baseline_model == "full"]
    assert full_runs
    for run in full_runs:
        assert run.mechanisms == {
            "two_scale": True,
            "coupling": True,
            "in_loop_loss": True,
        }
        assert int(run.config["pca"]["two_scale"]["coarse_factor"]) == 4
        assert (run.resolution // 4) ** 2 < int(run.config["resolved"]["n_samples"])


def test_resolution_sweep_optional_512_uses_legal_c8() -> None:
    root = Path(__file__).resolve().parents[1]
    runs = expand_resolution_sweep_runs(
        root=root,
        dataset="poisson",
        experiment="resolution_sweep",
        include_optional=True,
    )

    optional_runs = [run for run in runs if run.resolution == 512]
    assert len(optional_runs) == 10
    assert {run.baseline_model for run in optional_runs} == {"l2l", "full"}
    full_512 = [run for run in optional_runs if run.baseline_model == "full"]
    assert full_512
    for run in full_512:
        assert int(run.config["pca"]["two_scale"]["coarse_factor"]) == 8
        assert (512 // 8) ** 2 < int(run.config["resolved"]["n_samples"])


def test_resolution_sweep_rejects_illegal_optional_c4_for_512() -> None:
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="Coarse factor invariant violated"):
        expand_resolution_sweep_runs(
            root=root,
            dataset="poisson",
            experiment="resolution_sweep",
            include_optional=True,
            overrides=["matrix.optional.coarse_factor=4"],
        )
