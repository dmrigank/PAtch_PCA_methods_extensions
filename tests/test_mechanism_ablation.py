from __future__ import annotations

from pathlib import Path

import pytest

from lpcanet.train.factorial import (
    expand_mechanism_ablation_runs,
    run_mechanism_ablation,
)

METHODS = [
    "plain_l2l",
    "plain_l2l_interface",
    "two_scale",
    "two_scale_interface",
    "overlap_l2l",
]


def test_mechanism_ablation_expands_25_paired_runs() -> None:
    root = Path(__file__).resolve().parents[1]
    runs = expand_mechanism_ablation_runs(root=root)

    assert len(runs) == 25
    assert len({run.run_id for run in runs}) == 25
    assert len({run.config_hash for run in runs}) == 25
    assert {run.seed for run in runs} == {0, 1, 2, 3, 4}
    assert {run.resolution for run in runs} == {128}
    for seed in range(5):
        selected = [run for run in runs if run.seed == seed]
        assert [run.baseline_model for run in selected] == METHODS
        assert all(run.config["dataset"]["train_samples"] == 8000 for run in selected)
        assert all(
            run.config["evaluation"]["retain_predictions"] is (seed == 0)
            for run in selected
        )


def test_mechanism_ablation_isolates_representation_and_in_loop_loss() -> None:
    root = Path(__file__).resolve().parents[1]
    selected = {
        run.baseline_model: run
        for run in expand_mechanism_ablation_runs(root=root)
        if run.seed == 0
    }

    plain = selected["plain_l2l"]
    plain_interface = selected["plain_l2l_interface"]
    two_scale = selected["two_scale"]
    two_scale_interface = selected["two_scale_interface"]
    overlap = selected["overlap_l2l"]

    assert plain.mechanisms == {
        "two_scale": False,
        "coupling": False,
        "in_loop_loss": False,
    }
    assert plain_interface.mechanisms == {
        "two_scale": False,
        "coupling": False,
        "in_loop_loss": True,
    }
    assert two_scale.mechanisms == {
        "two_scale": True,
        "coupling": False,
        "in_loop_loss": False,
    }
    assert two_scale_interface.mechanisms == {
        "two_scale": True,
        "coupling": False,
        "in_loop_loss": True,
    }
    assert overlap.mechanisms == {
        "two_scale": False,
        "coupling": False,
        "in_loop_loss": False,
    }

    assert plain_interface.config["warm_start"]["run_dir"] == str(plain.run_dir)
    assert two_scale_interface.config["warm_start"]["run_dir"] == str(
        two_scale.run_dir
    )
    for run in (plain_interface, two_scale_interface):
        assert run.config["training"]["epochs"] == 50
        assert run.config["loss"]["active_terms"] == [
            "recon",
            "interface_value",
            "interface_flux",
        ]
        assert run.config["loss"]["weights"]["spectral"] == 0.0
        assert run.config["loss"]["weights"]["pde_residual"] == 0.0
        assert run.config["loss"]["weight_calibration"]["target_ratios"] == {
            "interface_value": pytest.approx(0.2),
            "interface_flux": pytest.approx(0.2),
        }

    assert plain.config["training"]["latent_loss"]["mode"] == "mse"
    assert plain_interface.config["training"]["latent_loss"]["mode"] == "mse"
    assert two_scale.config["training"]["latent_loss"]["mode"] == "block_balanced"
    assert two_scale_interface.config["training"]["latent_loss"]["mode"] == (
        "block_balanced"
    )
    assert overlap.config["patches"]["stride"] == 16
    assert overlap.config["patches"]["assembly"] == "hann_safe"


def test_mechanism_ablation_uses_frozen_representation_settings() -> None:
    root = Path(__file__).resolve().parents[1]
    runs = expand_mechanism_ablation_runs(root=root)

    for run in runs:
        assert run.config["pca"]["solver"] == "randomized"
        assert run.config["pca"]["randomized"] == {
            "oversampling": 20,
            "n_iter": 4,
        }
        assert run.config["resolved"]["study_type"] == "mechanism_ablation"
        assert run.config["resolved"]["ablation"]["family"] == (
            "mechanism_ablation"
        )
        assert run.mechanisms["coupling"] is False

    two_scale_runs = [run for run in runs if run.mechanisms["two_scale"]]
    for run in two_scale_runs:
        assert run.config["pca"]["output_variance"] == pytest.approx(0.995)
        assert run.config["pca"]["two_scale"]["coarse_factor"] == 4
        assert run.config["pca"]["two_scale"]["coarse_components"] == 10


def test_mechanism_ablation_dry_run_lists_all_cells(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = Path(__file__).resolve().parents[1]
    runs = run_mechanism_ablation(
        root=root,
        output_dir=tmp_path / "mechanism",
        dry_run=True,
    )

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(runs) == 25
    assert len(lines) == 25
    assert sum("baseline_model=plain_l2l_interface" in line for line in lines) == 5
    assert sum("baseline_model=two_scale_interface" in line for line in lines) == 5
    assert not (tmp_path / "mechanism").exists()


def test_mechanism_ablation_rejects_schedule_shortcuts() -> None:
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="do not use --limit-samples"):
        run_mechanism_ablation(root=root, limit_samples=100)
    with pytest.raises(ValueError, match="do not use --epochs"):
        run_mechanism_ablation(root=root, epochs=1)
