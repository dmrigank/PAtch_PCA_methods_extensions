from __future__ import annotations

from pathlib import Path

import pytest

from lpcanet.train.factorial import (
    expand_darcy_generalization_runs,
    run_darcy_generalization_study,
)

ROOT = Path(__file__).resolve().parents[1]
METHOD_ORDER = (
    "global_pca",
    "plain_l2l",
    "overlap_l2l",
    "fno",
    "two_scale",
    "two_scale_interface",
)


def test_darcy_generalization_expands_complete_paired_matrix(tmp_path: Path) -> None:
    runs = expand_darcy_generalization_runs(
        root=ROOT,
        output_dir=tmp_path / "darcy",
    )

    assert len(runs) == 30
    assert {run.seed for run in runs} == set(range(5))
    assert {run.resolution for run in runs} == {256}
    for seed in range(5):
        selected = [run for run in runs if run.seed == seed]
        assert tuple(run.baseline_model for run in selected) == METHOD_ORDER
    assert len({run.config_hash for run in runs}) == len(runs)

    for run in runs:
        config = run.config
        assert config["dataset"]["name"] == "darcy"
        assert config["dataset"]["processed_path"] == "data/processed/darcy_256.npz"
        assert config["dataset"]["train_samples"] == 8000
        assert config["dataset"]["input_keys"] == ["a"]
        assert config["dataset"]["output_key"] == "u"
        assert config["loss"]["pde"] == "darcy"
        assert "darcy_residual" in config["evaluation"]["metrics"]
        assert "poisson_residual" not in config["evaluation"]["metrics"]
        assert config["evaluation"]["retain_predictions"] is (run.seed == 0)
        assert config["resolved"]["study_type"] == "darcy_generalization"
        assert config["resolved"]["dataset"] == "darcy"


def test_darcy_generalization_freezes_final_method_defaults(tmp_path: Path) -> None:
    runs = expand_darcy_generalization_runs(
        root=ROOT,
        output_dir=tmp_path / "darcy",
    )
    selected = {run.baseline_model: run for run in runs if run.seed == 0}

    plain = selected["plain_l2l"].config
    assert plain["patches"] == {
        "patch_size": 64,
        "stride": 64,
        "assembly": "average",
    }
    assert plain["training"]["latent_loss"]["mode"] == "mse"
    assert plain["pca"]["solver"] == "randomized"

    overlap = selected["overlap_l2l"].config
    assert overlap["patches"]["stride"] == 32
    assert overlap["patches"]["assembly"] == "hann_safe"

    two_scale = selected["two_scale"].config
    assert two_scale["mechanisms"] == {
        "two_scale": True,
        "coupling": False,
        "in_loop_loss": False,
    }
    assert two_scale["training"]["latent_loss"]["mode"] == "block_balanced"
    assert two_scale["pca"]["output_variance"] == pytest.approx(0.995)
    assert two_scale["pca"]["two_scale"]["coarse_factor"] == 4
    assert two_scale["pca"]["two_scale"]["coarse_components"] == 10
    assert two_scale["pca"]["two_scale"]["coarse_solver"] == "randomized"

    interface_run = selected["two_scale_interface"]
    interface = interface_run.config
    assert interface["mechanisms"] == {
        "two_scale": True,
        "coupling": False,
        "in_loop_loss": True,
    }
    assert interface["training"]["epochs"] == 50
    assert interface["loss"]["active_terms"] == [
        "recon",
        "interface_value",
        "interface_flux",
    ]
    assert interface["loss"]["weights"]["spectral"] == 0.0
    assert interface["loss"]["weights"]["pde_residual"] == 0.0
    ratios = interface["loss"]["weight_calibration"]["target_ratios"]
    assert ratios == {"interface_value": 0.2, "interface_flux": 0.2}
    assert interface["warm_start"]["run_dir"] == str(selected["two_scale"].run_dir)

    fno = selected["fno"].config
    assert fno["training"]["epochs"] == 100
    assert fno["training"]["batch_size"] == 4
    assert fno["training"]["early_stopping"]["enabled"] is True


def test_darcy_generalization_dry_run_lists_30_without_writes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "darcy"
    runs = run_darcy_generalization_study(
        root=ROOT,
        output_dir=output_dir,
        dry_run=True,
    )
    lines = [line for line in capsys.readouterr().out.splitlines() if line]

    assert len(runs) == 30
    assert len(lines) == 30
    assert all("dataset=darcy" in line for line in lines)
    assert all("resolution=256" in line for line in lines)
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("keyword", "value"),
    (("limit_samples", 16), ("epochs", 1)),
)
def test_darcy_generalization_rejects_shortcuts(
    keyword: str,
    value: int,
) -> None:
    with pytest.raises(ValueError, match="darcy_generalization"):
        run_darcy_generalization_study(
            root=ROOT,
            dry_run=True,
            **{keyword: value},
        )
