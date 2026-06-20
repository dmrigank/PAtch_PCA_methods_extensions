from __future__ import annotations

from pathlib import Path

from lpcanet.train.factorial import expand_method_study_runs


def test_darcy_256_seed0_study_expands_with_darcy_physics() -> None:
    root = Path(__file__).resolve().parents[1]
    runs = expand_method_study_runs(
        root=root,
        dataset="darcy",
        experiment="darcy_256_seed0",
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
    } == {"data/processed/darcy_256.npz"}
    assert all(run.config["dataset"]["input_keys"] == ["a"] for run in runs)
    assert all("darcy_residual" in run.config["evaluation"]["metrics"] for run in runs)
    assert all("poisson_residual" not in run.config["evaluation"]["metrics"] for run in runs)

    fine_tune = next(run for run in runs if run.baseline_model == "two_scale_in_loop")
    two_scale = next(run for run in runs if run.baseline_model == "two_scale")
    assert fine_tune.config["loss"]["pde"] == "darcy"
    assert fine_tune.config["warm_start"]["run_dir"] == str(two_scale.run_dir)

    fno = next(run for run in runs if run.baseline_model == "fno")
    assert fno.config["model"]["normalization"]["enabled"] is True
    assert float(fno.config["training"]["weight_decay"]) == 0.0
