from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lpcanet.pca.pca import TwoScaleLocalToLocalPCAEncoder
from lpcanet.train.experiment import _make_encoder, _summarize_pca_encoder
from lpcanet.train.factorial import expand_svd_solver_ablation_runs


def test_svd_solver_ablation_expands_35_paired_runs() -> None:
    root = Path(__file__).resolve().parents[1]
    runs = expand_svd_solver_ablation_runs(root=root)

    assert len(runs) == 35
    assert len({run.run_id for run in runs}) == 35
    assert len({run.run_dir for run in runs}) == 35
    assert len({run.config_hash for run in runs}) == 35
    assert {run.seed for run in runs} == {0, 1, 2, 3, 4}
    assert [run.baseline_model for run in runs[:7]] == [
        "plain_full",
        "plain_randomized",
        "overlap_full",
        "overlap_randomized",
        "two_scale_full",
        "two_scale_hybrid",
        "two_scale_randomized",
    ]

    for run in runs:
        config = run.config
        assert config["dataset"]["train_samples"] == 8000
        assert config["dataset"]["save_split_indices"] is True
        assert config["pca"]["randomized"] == {
            "oversampling": 20,
            "n_iter": 4,
        }
        assert config["training"]["epochs"] == 500
        assert config["training"]["batch_size"] == 32
        assert config["evaluation"]["pca_oracle"] is True
        assert config["evaluation"]["validation_metrics"] is True
        assert config["evaluation"]["retain_predictions"] is False
        assert run.config["resolved"]["ablation"]["family"] == "svd_solver"
        assert run.config["resolved"]["equivalence_gates"] == {
            "learned_mre_relative": 0.05,
            "pca_oracle_mre_relative": 0.02,
            "ssim_absolute": 0.001,
            "secondary_metric_relative": 0.10,
            "minimum_pca_speedup": 1.20,
        }


def test_svd_solver_cells_route_geometry_objective_and_solvers() -> None:
    root = Path(__file__).resolve().parents[1]
    runs = expand_svd_solver_ablation_runs(root=root)
    by_cell = {run.baseline_model: run for run in runs if run.seed == 0}

    for name in ("plain_full", "plain_randomized"):
        run = by_cell[name]
        assert run.config["patches"] == {
            "patch_size": 32,
            "stride": 32,
            "assembly": "average",
        }
        assert run.config["training"]["latent_loss"]["mode"] == "mse"
        assert run.mechanisms["two_scale"] is False

    for name in ("overlap_full", "overlap_randomized"):
        run = by_cell[name]
        assert run.config["patches"] == {
            "patch_size": 32,
            "stride": 16,
            "assembly": "hann_safe",
        }
        assert run.config["training"]["latent_loss"]["mode"] == "mse"
        assert run.mechanisms["two_scale"] is False

    expected = {
        "two_scale_full": ("full", "full", "full"),
        "two_scale_hybrid": ("full", "randomized", "hybrid"),
        "two_scale_randomized": ("randomized", "randomized", "randomized"),
    }
    for name, (local_solver, coarse_solver, solver_class) in expected.items():
        run = by_cell[name]
        assert run.mechanisms == {
            "two_scale": True,
            "coupling": False,
            "in_loop_loss": False,
        }
        assert run.config["pca"]["solver"] == local_solver
        assert run.config["pca"]["two_scale"]["coarse_solver"] == coarse_solver
        assert run.config["pca"]["output_variance"] == 0.995
        assert run.config["training"]["latent_loss"]["mode"] == "block_balanced"
        assert run.config["resolved"]["ablation"]["solver_class"] == solver_class
        encoder = _make_encoder(
            run.config,
            n_train=8000,
            grid_shape=(128, 128),
        )
        assert isinstance(encoder, TwoScaleLocalToLocalPCAEncoder)
        assert encoder.solver == local_solver
        assert encoder.coarse_solver == coarse_solver


@pytest.mark.parametrize("coarse_solver", ["full", "randomized"])
def test_two_scale_coarse_solver_is_independently_routed(
    coarse_solver: str,
) -> None:
    rng = np.random.default_rng(9)
    x = rng.normal(size=(40, 8, 8)).astype(np.float32)
    y = (0.3 * x + rng.normal(scale=0.02, size=x.shape)).astype(np.float32)
    encoder = TwoScaleLocalToLocalPCAEncoder(
        patch_size=4,
        stride=4,
        input_variance=2,
        output_variance=2,
        solver="full",
        coarse_factor=2,
        coarse_components=3,
        coarse_variance=1.0,
        coarse_solver=coarse_solver,
        grid_size=8,
        n_samples=len(y),
        random_state=0,
        n_iter=2,
    ).fit(x, y)

    assert encoder.solver == "full"
    assert encoder.coarse_solver == coarse_solver
    assert encoder.coarse_pca._fit_svd_solver == coarse_solver

    summary = _summarize_pca_encoder(encoder)
    assert summary["configured_solvers"] == {
        "local": "full",
        "coarse": coarse_solver,
    }
    assert summary["groups"]["coarse_output"]["fitted_solvers"] == [
        coarse_solver
    ]
    assert summary["groups"]["coarse_output"]["components_total"] == 3
    assert summary["groups"]["input"]["explained_variance_ratio_min"] > 0.0


def test_invalid_two_scale_coarse_solver_fails_early() -> None:
    with pytest.raises(ValueError, match="coarse_solver"):
        TwoScaleLocalToLocalPCAEncoder(
            patch_size=4,
            stride=4,
            coarse_solver="auto",
        )
