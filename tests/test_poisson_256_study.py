from __future__ import annotations

from pathlib import Path

import numpy as np

from lpcanet.pca.randomized_svd import fit_pca
from lpcanet.train.factorial import expand_method_study_runs


def test_poisson_256_seed0_study_expands_requested_methods() -> None:
    root = Path(__file__).resolve().parents[1]
    runs = expand_method_study_runs(
        root=root,
        dataset="poisson",
        experiment="poisson_256_seed0",
    )

    assert [run.baseline_model for run in runs] == [
        "global_pca",
        "plain_l2l",
        "overlap_l2l",
        "fno",
        "two_scale",
        "two_scale_in_loop",
    ]
    assert {run.seed for run in runs} == {0}
    assert {run.resolution for run in runs} == {256}
    assert {
        run.config["dataset"]["processed_path"] for run in runs
    } == {"data/processed/poisson_256.npz"}

    pca_runs = [run for run in runs if run.baseline_model != "fno"]
    assert all(run.config["pca"]["solver"] == "randomized" for run in pca_runs)

    two_scale = next(run for run in runs if run.baseline_model == "two_scale")
    assert two_scale.mechanisms == {
        "two_scale": True,
        "coupling": False,
        "in_loop_loss": False,
    }
    assert int(two_scale.config["pca"]["two_scale"]["coarse_factor"]) == 4

    fine_tune = next(run for run in runs if run.baseline_model == "two_scale_in_loop")
    assert fine_tune.mechanisms == {
        "two_scale": True,
        "coupling": False,
        "in_loop_loss": True,
    }
    assert fine_tune.config["warm_start"]["run_dir"] == str(two_scale.run_dir)
    assert int(fine_tune.config["training"]["epochs"]) == 50
    assert float(fine_tune.config["training"]["lr"]) == 1.0e-4

    fno = next(run for run in runs if run.baseline_model == "fno")
    assert int(fno.config["training"]["batch_size"]) == 4
    assert int(fno.config["training"]["eval_batch_size"]) == 8
    assert int(fno.config["training"]["epochs"]) == 100
    assert float(fno.config["training"]["weight_decay"]) == 0.0
    assert fno.config["model"]["normalization"]["enabled"] is True
    assert int(fno.config["model"]["fno"]["padding"]) == 8
    assert fno.config["training"]["early_stopping"] == {
        "enabled": True,
        "patience": 15,
        "min_delta": 0.0,
    }


def test_randomized_pca_preserves_float32_storage() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(64, 128)).astype(np.float32)
    pca = fit_pca(data, 0.9, solver="randomized", random_state=0)

    assert pca.components_.dtype == np.float32
    assert pca.mean_.dtype == np.float32
