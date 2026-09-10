from __future__ import annotations

from pathlib import Path

import pytest

from lpcanet.train.factorial import (
    expand_resolution_sweep_runs,
    run_resolution_sweep,
)

METHODS = (
    "global_pca",
    "plain_l2l",
    "overlap_l2l",
    "two_scale",
    "two_scale_interface",
    "fno",
)
RESOLUTION_SETTINGS = {
    64: {
        "path": "data/processed/poisson_64.npz",
        "patch_size": 16,
        "plain_stride": 16,
        "overlap_stride": 8,
        "coarse_factor": 4,
    },
    128: {
        "path": "data/processed/poisson_128.npz",
        "patch_size": 32,
        "plain_stride": 32,
        "overlap_stride": 16,
        "coarse_factor": 4,
    },
    256: {
        "path": "data/processed/poisson_256.npz",
        "patch_size": 64,
        "plain_stride": 64,
        "overlap_stride": 32,
        "coarse_factor": 4,
    },
}


def test_resolution_sweep_expands_complete_paper_matrix() -> None:
    root = Path(__file__).resolve().parents[1]
    runs = expand_resolution_sweep_runs(root=root)

    assert len(runs) == 54
    assert {run.resolution for run in runs} == {64, 128, 256}
    assert {run.baseline_model for run in runs} == set(METHODS)
    for resolution, settings in RESOLUTION_SETTINGS.items():
        for seed in (0, 1, 2):
            selected = [
                run
                for run in runs
                if run.resolution == resolution and run.seed == seed
            ]
            assert [run.baseline_model for run in selected] == list(METHODS)
            for run in selected:
                config = run.config
                dataset = config["dataset"]
                assert dataset["processed_path"] == settings["path"]
                assert dataset["train_samples"] == 8000
                assert dataset["use_embedded_splits"] is False
                assert dataset["save_split_indices"] is True
                assert config["patches"]["patch_size"] == settings["patch_size"]
                expected_stride = (
                    settings["overlap_stride"]
                    if run.baseline_model == "overlap_l2l"
                    else settings["plain_stride"]
                )
                assert config["patches"]["stride"] == expected_stride
                assert config["resolved"]["ablation"]["common_seam_stride"] == settings[
                    "plain_stride"
                ]
                assert config["evaluation"]["retain_predictions"] is (seed == 0)


def test_resolution_sweep_resolves_method_specific_configs() -> None:
    root = Path(__file__).resolve().parents[1]
    runs = expand_resolution_sweep_runs(root=root)
    selected = {
        run.baseline_model: run
        for run in runs
        if run.resolution == 128 and run.seed == 0
    }

    for method in METHODS:
        run = selected[method]
        assert run.mechanisms["coupling"] is False
        if method != "fno":
            assert run.config["pca"]["solver"] == "randomized"
            assert run.config["pca"]["randomized"] == {
                "oversampling": 20,
                "n_iter": 4,
            }
            if method != "two_scale_interface":
                assert run.config["training"]["epochs"] == 500

    for method in ("global_pca", "plain_l2l", "overlap_l2l", "fno"):
        assert selected[method].mechanisms == {
            "two_scale": False,
            "coupling": False,
            "in_loop_loss": False,
        }

    two_scale = selected["two_scale"]
    assert two_scale.mechanisms == {
        "two_scale": True,
        "coupling": False,
        "in_loop_loss": False,
    }
    assert two_scale.config["pca"]["output_variance"] == pytest.approx(0.995)
    assert two_scale.config["pca"]["two_scale"]["coarse_components"] == 10
    assert two_scale.config["pca"]["two_scale"]["coarse_factor"] == 4
    assert two_scale.config["pca"]["two_scale"]["coarse_solver"] == "randomized"
    assert two_scale.config["training"]["latent_loss"]["mode"] == "block_balanced"

    interface = selected["two_scale_interface"]
    assert interface.mechanisms == {
        "two_scale": True,
        "coupling": False,
        "in_loop_loss": True,
    }
    assert interface.config["warm_start"]["run_dir"] == str(two_scale.run_dir)
    assert interface.config["training"]["epochs"] == 50
    assert interface.config["loss"]["active_terms"] == [
        "recon",
        "interface_value",
        "interface_flux",
    ]
    assert interface.config["loss"]["weight_calibration"]["target_ratios"] == {
        "interface_value": pytest.approx(0.2),
        "interface_flux": pytest.approx(0.2),
    }
    assert interface.config["loss"]["weights"]["spectral"] == 0.0
    assert interface.config["loss"]["weights"]["pde_residual"] == 0.0

    fno = selected["fno"]
    assert fno.config["model"]["type"] == "fno"
    assert fno.config["pca"]["type"] == "none"
    assert fno.config["model"]["normalization"]["enabled"] is True
    assert fno.config["training"]["epochs"] == 100
    assert fno.config["training"]["early_stopping"]["enabled"] is True


def test_resolution_sweep_coarse_factors_are_legal() -> None:
    root = Path(__file__).resolve().parents[1]
    runs = expand_resolution_sweep_runs(root=root)

    for run in runs:
        coarse_factor = int(run.config["resolved"]["coarse_factor"])
        assert (run.resolution // coarse_factor) ** 2 < 8000


def test_resolution_sweep_optional_512_uses_legal_c8() -> None:
    root = Path(__file__).resolve().parents[1]
    runs = expand_resolution_sweep_runs(root=root, include_optional=True)

    optional_runs = [run for run in runs if run.resolution == 512]
    assert len(runs) == 60
    assert len(optional_runs) == 6
    assert {run.baseline_model for run in optional_runs} == {
        "plain_l2l",
        "two_scale",
    }
    for run in optional_runs:
        assert run.config["dataset"]["processed_path"] == (
            "data/processed/poisson_512.npz"
        )
        assert run.config["patches"]["patch_size"] == 128
        assert int(run.config["resolved"]["coarse_factor"]) == 8
        assert (512 // 8) ** 2 < 8000


def test_resolution_sweep_rejects_illegal_optional_c4_for_512() -> None:
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="Coarse factor invariant violated"):
        expand_resolution_sweep_runs(
            root=root,
            include_optional=True,
            overrides=["matrix.optional.coarse_factor=4"],
        )


def test_resolution_sweep_dry_run_lists_all_cells(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = Path(__file__).resolve().parents[1]
    runs = run_resolution_sweep(
        root=root,
        output_dir=tmp_path / "resolution_sweep",
        dry_run=True,
    )

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(runs) == 54
    assert len(lines) == 54
    assert sum("baseline_model=two_scale_interface" in line for line in lines) == 9
    assert sum("baseline_model=fno" in line for line in lines) == 9
    assert not (tmp_path / "resolution_sweep").exists()


def test_resolution_sweep_rejects_schedule_shortcuts() -> None:
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="do not use --limit-samples"):
        run_resolution_sweep(root=root, limit_samples=100)
    with pytest.raises(ValueError, match="do not use --epochs"):
        run_resolution_sweep(root=root, epochs=1)
