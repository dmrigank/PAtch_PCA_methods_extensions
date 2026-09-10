from __future__ import annotations

from pathlib import Path

import numpy as np

from lpcanet.data.io import save_npz
from lpcanet.metrics.evaluate import evaluate_run
from lpcanet.pca.pca import TwoScaleLocalToLocalPCAEncoder
from lpcanet.train.factorial import expand_coarse_representation_ablation_runs
from lpcanet.utils.config import save_config


def test_coarse_representation_ablation_expands_51_unique_runs() -> None:
    root = Path(__file__).resolve().parents[1]
    runs = expand_coarse_representation_ablation_runs(
        root=root,
        experiment="coarse_representation_ablation",
    )

    assert len(runs) == 51
    assert {run.seed for run in runs} == {0, 1, 2}
    assert {run.resolution for run in runs} == {128, 256}
    assert len([run for run in runs if run.resolution == 128]) == 27
    assert len([run for run in runs if run.resolution == 256]) == 24
    assert len({run.run_dir for run in runs}) == 51
    assert len({run.config_hash for run in runs}) == 51

    for run in runs:
        assert run.config["dataset"]["name"] == "poisson"
        assert run.config["pca"]["solver"] == "randomized"
        assert run.config["training"]["latent_loss"]["mode"] == "block_balanced"
        assert run.config["evaluation"]["pca_oracle"] is True
        assert run.mechanisms == {
            "two_scale": True,
            "coupling": False,
            "in_loop_loss": False,
        }
        ablation = run.config["resolved"]["ablation"]
        coarse_features = (
            run.resolution // int(ablation["coarse_factor"])
        ) ** 2
        assert coarse_features < 8000

    cells_128 = {
        run.baseline_model for run in runs if run.resolution == 128 and run.seed == 0
    }
    cells_256 = {
        run.baseline_model for run in runs if run.resolution == 256 and run.seed == 0
    }
    assert cells_128 == {
        "baseline_exact_rank10",
        "coarse_rank_5",
        "coarse_rank_20",
        "coarse_rank_40",
        "coarse_factor_2",
        "coarse_factor_8",
        "residual_variance_0p95",
        "residual_variance_0p995",
        "adaptive_99pct_cap20",
    }
    assert cells_256 == cells_128 - {"coarse_factor_2"}


def test_exact_coarse_rank_is_not_variance_truncated() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(40, 8, 8)).astype(np.float32)
    y = (0.2 * x + rng.normal(scale=0.01, size=x.shape)).astype(np.float32)
    encoder = TwoScaleLocalToLocalPCAEncoder(
        patch_size=4,
        stride=4,
        input_variance=2,
        output_variance=2,
        solver="randomized",
        coarse_factor=2,
        coarse_components=5,
        coarse_variance=1.0,
        grid_size=8,
        n_samples=len(y),
        random_state=0,
    ).fit(x, y)

    assert int(encoder.component_counts["coarse_output"]) == 5


def test_evaluator_records_prefixed_pca_oracle_metrics(tmp_path: Path) -> None:
    true = np.ones((3, 8, 8), dtype=np.float32)
    pred = np.zeros_like(true)
    oracle = 0.9 * true
    save_npz(
        tmp_path / "predictions_test.npz",
        x_input=true,
        y_true=true,
        y_pred=pred,
        y_pca_oracle=oracle,
    )
    save_config(
        {
            "dataset": {"name": "poisson", "grid_size": 8},
            "patches": {"patch_size": 4, "stride": 4},
            "evaluation": {"metrics": ["mse", "mre", "interface_jump"]},
        },
        tmp_path,
    )

    metrics = evaluate_run(tmp_path)

    assert metrics["mre"] == 1.0
    assert metrics["pca_oracle_mre"] < metrics["mre"]
    assert "pca_oracle_interface_jump" in metrics
