from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lpcanet.data.splits import make_splits
from lpcanet.train.experiment import _select_training_subset
from lpcanet.train.factorial import (
    _assert_coarse_factor_legal,
    expand_sample_efficiency_runs,
)


def test_sample_efficiency_expands_48_paired_runs() -> None:
    root = Path(__file__).resolve().parents[1]
    runs = expand_sample_efficiency_runs(
        root=root,
        experiment="sample_efficiency",
    )

    assert len(runs) == 48
    assert len({run.run_id for run in runs}) == 48
    assert len({run.config_hash for run in runs}) == 48
    assert {run.seed for run in runs} == {0, 1, 2}
    assert {
        int(run.config["dataset"]["train_samples"]) for run in runs
    } == {1250, 2000, 4000, 8000}
    assert {run.baseline_model for run in runs} == {
        "plain_l2l",
        "overlap_l2l",
        "two_scale",
        "two_scale_interface",
    }

    for run in runs:
        sample_size = int(run.config["dataset"]["train_samples"])
        assert run.config["dataset"]["save_split_indices"] is True
        assert run.config["pca"]["solver"] == "randomized"
        assert run.config["evaluation"]["pca_oracle"] is True
        assert run.config["evaluation"]["validation_metrics"] is True
        assert run.config["evaluation"]["retain_predictions"] is (
            run.seed == 0 and sample_size in {1250, 8000}
        )
        assert (128 // 4) ** 2 < sample_size

        budget = run.config["resolved"]["optimization_budget"]
        key = (
            "in_loop_planned_steps"
            if run.baseline_model == "two_scale_interface"
            else "latent_planned_steps"
        )
        target_key = (
            "in_loop_target_steps"
            if run.baseline_model == "two_scale_interface"
            else "latent_target_steps"
        )
        assert budget[key] >= budget[target_key]
        assert budget[key] < budget[target_key] + budget["steps_per_epoch"]


def test_sample_efficiency_interface_warm_starts_matching_two_scale_run() -> None:
    root = Path(__file__).resolve().parents[1]
    runs = expand_sample_efficiency_runs(root=root)

    for run in runs:
        sample_size = int(run.config["dataset"]["train_samples"])
        if run.baseline_model == "plain_l2l":
            assert run.config["patches"] == {
                "patch_size": 32,
                "stride": 32,
                "assembly": "average",
            }
        elif run.baseline_model == "overlap_l2l":
            assert run.config["patches"] == {
                "patch_size": 32,
                "stride": 16,
                "assembly": "hann_safe",
            }
        elif run.baseline_model == "two_scale":
            assert run.mechanisms == {
                "two_scale": True,
                "coupling": False,
                "in_loop_loss": False,
            }
            assert run.config["training"]["latent_loss"]["mode"] == "block_balanced"
        elif run.baseline_model == "two_scale_interface":
            expected = (
                Path("paper_results/sample_efficiency")
                / "poisson"
                / "resolution_128"
                / f"train_{sample_size}"
                / "two_scale"
                / f"seed_{run.seed}"
            )
            assert Path(run.config["warm_start"]["run_dir"]) == expected
            assert run.config["loss"]["active_terms"] == [
                "recon",
                "interface_value",
                "interface_flux",
            ]
            assert run.config["loss"]["weight_calibration"]["target_ratios"] == {
                "interface_value": 0.2,
                "interface_flux": 0.2,
            }


def test_training_subsets_are_nested_and_hold_validation_test_fixed() -> None:
    full = make_splits(10_000, seed=2)
    subsets = {
        size: _select_training_subset(full, {"train_samples": size})
        for size in (1250, 2000, 4000, 8000)
    }

    for size, split in subsets.items():
        assert len(split["train_idx"]) == size
        np.testing.assert_array_equal(split["val_idx"], full["val_idx"])
        np.testing.assert_array_equal(split["test_idx"], full["test_idx"])
    for smaller, larger in ((1250, 2000), (2000, 4000), (4000, 8000)):
        np.testing.assert_array_equal(
            subsets[smaller]["train_idx"],
            subsets[larger]["train_idx"][:smaller],
        )

    with pytest.raises(ValueError, match="dataset.train_samples"):
        _select_training_subset(full, {"train_samples": 8001})


def test_original_1000_sample_point_is_illegal_at_c4() -> None:
    with pytest.raises(ValueError, match=r"must be < m=1000"):
        _assert_coarse_factor_legal(128, 4, 1000)
