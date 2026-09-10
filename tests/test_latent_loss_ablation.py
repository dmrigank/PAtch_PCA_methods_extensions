from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lpcanet.losses import TwoScaleLatentLoss
from lpcanet.train.factorial import expand_latent_loss_ablation_runs


def test_two_scale_latent_loss_formulas_and_gradients() -> None:
    pred = torch.tensor(
        [[1.0, 3.0, 2.0, 6.0, 10.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    target = torch.zeros_like(pred)

    mse = TwoScaleLatentLoss(mode="mse")(pred, target)
    expected_mse = pred.square().mean()
    assert mse.item() == pytest.approx(float(expected_mse))

    balanced = TwoScaleLatentLoss(
        mode="block_balanced",
        coarse_dim=2,
    )(pred, target)
    expected_balanced = 0.5 * pred[:, :2].square().mean() + 0.5 * pred[:, 2:].square().mean()
    assert balanced.item() == pytest.approx(float(expected_balanced))

    variances = torch.tensor([1.0, 9.0, 4.0, 36.0, 100.0], dtype=torch.float64)
    normalized = TwoScaleLatentLoss(
        mode="score_normalized",
        coarse_dim=2,
        score_variances=variances,
    )(pred, target)
    assert normalized.item() == pytest.approx(1.0)

    (mse + balanced + normalized).backward()
    assert pred.grad is not None
    assert torch.all(torch.isfinite(pred.grad))


def test_latent_loss_ablation_expands_all_paper_cells() -> None:
    root = Path(__file__).resolve().parents[1]
    runs = expand_latent_loss_ablation_runs(
        root=root,
        experiment="two_scale_latent_loss_ablation",
    )

    assert len(runs) == 2 * 2 * 3
    assert {run.config["dataset"]["name"] for run in runs} == {"poisson", "darcy"}
    assert {run.resolution for run in runs} == {128, 256}
    assert {run.seed for run in runs} == {0}
    assert {run.baseline_model for run in runs} == {
        "mse",
        "block_balanced",
        "score_normalized",
    }
    assert all(run.config["pca"]["solver"] == "randomized" for run in runs)
    assert all(
        run.mechanisms
        == {"two_scale": True, "coupling": False, "in_loop_loss": False}
        for run in runs
    )

    for run in runs:
        expected_patch_size = 32 if run.resolution == 128 else 64
        assert int(run.config["patches"]["patch_size"]) == expected_patch_size
        assert int(run.config["patches"]["stride"]) == expected_patch_size
        assert int(run.config["pca"]["two_scale"]["coarse_factor"]) == 4
        assert run.config["training"]["latent_loss"]["mode"] == run.baseline_model
        expected_path = (
            f"data/processed/{run.config['dataset']['name']}_{run.resolution}.npz"
        )
        assert run.config["dataset"]["processed_path"] == expected_path


def test_latent_loss_ablation_seed_override_supports_larger_followup() -> None:
    root = Path(__file__).resolve().parents[1]
    runs = expand_latent_loss_ablation_runs(
        root=root,
        experiment="two_scale_latent_loss_ablation",
        overrides=["matrix.seeds=[0,1]"],
    )

    assert len(runs) == 24
    assert {run.seed for run in runs} == {0, 1}
