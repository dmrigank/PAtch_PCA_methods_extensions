from __future__ import annotations

import json
from pathlib import Path

import pytest

from lpcanet.train.factorial import (
    FactorialRun,
    build_result_record,
    expand_headline_poisson_runs,
    run_headline_poisson_study,
)

METHODS = [
    "global_pca",
    "plain_l2l",
    "overlap_l2l",
    "l2l_refinement",
    "fno",
    "two_scale",
    "two_scale_interface",
]


def test_headline_poisson_expands_35_paired_runs() -> None:
    root = Path(__file__).resolve().parents[1]
    runs = expand_headline_poisson_runs(root=root)

    assert len(runs) == 35
    assert len({run.run_id for run in runs}) == 35
    assert len({run.config_hash for run in runs}) == 35
    assert {run.seed for run in runs} == {0, 1, 2, 3, 4}
    assert {run.resolution for run in runs} == {128}
    assert {run.baseline_model for run in runs} == set(METHODS)
    for seed in range(5):
        selected = [run for run in runs if run.seed == seed]
        assert [run.baseline_model for run in selected] == METHODS
        assert all(run.config["dataset"]["train_samples"] == 8000 for run in selected)
        assert all(
            run.config["dataset"]["processed_path"]
            == "data/processed/poisson_128.npz"
            for run in selected
        )
        assert all(
            run.config["evaluation"]["retain_predictions"] is (seed == 0)
            for run in selected
        )


def test_headline_poisson_freezes_selected_method_settings() -> None:
    root = Path(__file__).resolve().parents[1]
    selected = {
        run.baseline_model: run
        for run in expand_headline_poisson_runs(root=root)
        if run.seed == 0
    }

    for method, run in selected.items():
        assert run.config["resolved"]["study_type"] == "headline_poisson_128"
        assert run.config["resolved"]["ablation"]["family"] == (
            "headline_poisson_128"
        )
        if method != "fno":
            assert run.config["pca"]["solver"] == "randomized"
            assert run.config["pca"]["randomized"] == {
                "oversampling": 20,
                "n_iter": 4,
            }

    plain = selected["plain_l2l"]
    overlap = selected["overlap_l2l"]
    assert plain.config["patches"] == {
        "patch_size": 32,
        "stride": 32,
        "assembly": "average",
    }
    assert overlap.config["patches"] == {
        "patch_size": 32,
        "stride": 16,
        "assembly": "hann_safe",
    }

    refinement = selected["l2l_refinement"]
    assert refinement.config["base_run_dir"] == str(plain.run_dir)
    assert refinement.config["training"]["epochs"] == 25
    assert refinement.config["resolved"]["postprocessing_stage"] == (
        "refinementnet"
    )

    two_scale = selected["two_scale"]
    assert two_scale.config["pca"]["output_variance"] == pytest.approx(0.995)
    assert two_scale.config["pca"]["two_scale"]["coarse_factor"] == 4
    assert two_scale.config["pca"]["two_scale"]["coarse_components"] == 10
    assert two_scale.config["training"]["latent_loss"] == {
        "mode": "block_balanced",
        "coarse_weight": 0.5,
        "residual_weight": 0.5,
        "variance_floor": 1.0e-8,
    }

    interface = selected["two_scale_interface"]
    assert interface.config["warm_start"]["run_dir"] == str(two_scale.run_dir)
    assert interface.config["loss"]["active_terms"] == [
        "recon",
        "interface_value",
        "interface_flux",
    ]
    assert interface.config["loss"]["weights"]["spectral"] == 0.0
    assert interface.config["loss"]["weights"]["pde_residual"] == 0.0


def test_headline_refinement_cost_includes_plain_base(tmp_path: Path) -> None:
    base_dir = tmp_path / "plain_l2l"
    refinement_dir = tmp_path / "l2l_refinement"
    base_dir.mkdir()
    refinement_dir.mkdir()
    stages = {
        "coarse_svd": 0.0,
        "pca_fit": 4.0,
        "latent_transform": 2.0,
        "nn_train": 20.0,
        "inference": 1.0,
        "end_to_end": 30.0,
    }
    (base_dir / "runtime.json").write_text(
        json.dumps({"stages": stages}), encoding="utf-8"
    )
    child_stages = {
        "coarse_svd": 0.0,
        "pca_fit": 0.0,
        "latent_transform": 0.0,
        "nn_train": 8.0,
        "inference": 0.5,
        "end_to_end": 12.0,
    }
    (refinement_dir / "runtime.json").write_text(
        json.dumps({"stages": child_stages}), encoding="utf-8"
    )
    run = FactorialRun(
        run_id="headline_refinement",
        config={"base_run_dir": str(base_dir)},
        config_hash="hash",
        seed=0,
        mechanisms={"two_scale": False, "coupling": False, "in_loop_loss": False},
        resolution=128,
        run_dir=refinement_dir,
        baseline_model="l2l_refinement",
    )

    record = build_result_record(run, metrics={}, run_dir=refinement_dir)

    assert record["timings"]["pca_fit"] == 4.0
    assert record["timings"]["nn_train"] == 28.0
    assert record["timings"]["end_to_end"] == 42.0
    assert record["timing_accounting"]["includes_base_run"] is True


def test_headline_poisson_dry_run_lists_all_cells(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = Path(__file__).resolve().parents[1]
    runs = run_headline_poisson_study(
        root=root,
        output_dir=tmp_path / "headline",
        dry_run=True,
    )

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(runs) == 35
    assert len(lines) == 35
    assert sum("baseline_model=l2l_refinement" in line for line in lines) == 5
    assert sum("baseline_model=two_scale_interface" in line for line in lines) == 5
    assert not (tmp_path / "headline").exists()


def test_headline_poisson_rejects_schedule_shortcuts() -> None:
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="do not use --limit-samples"):
        run_headline_poisson_study(root=root, limit_samples=100)
    with pytest.raises(ValueError, match="do not use --epochs"):
        run_headline_poisson_study(root=root, epochs=1)
