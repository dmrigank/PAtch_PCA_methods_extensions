from __future__ import annotations

import itertools
import json
from pathlib import Path

from lpcanet.train.factorial import (
    MECHANISM_KEYS,
    FactorialRun,
    build_result_record,
    expand_baseline_runs,
    expand_factorial_runs,
    write_stage_cost_table,
)


def test_factorial_expands_to_8_cells_by_5_seeds() -> None:
    root = Path(__file__).resolve().parents[1]
    runs = expand_factorial_runs(root=root, dataset="poisson", model="l2l", experiment="factorial")

    assert len(runs) == 40
    expected_cells = {
        tuple(zip(MECHANISM_KEYS, flags))
        for flags in itertools.product([False, True], repeat=3)
    }
    observed_cells = {
        tuple((key, run.mechanisms[key]) for key in MECHANISM_KEYS)
        for run in runs
    }
    assert observed_cells == expected_cells

    for cell in expected_cells:
        cell_runs = [
            run
            for run in runs
            if tuple((key, run.mechanisms[key]) for key in MECHANISM_KEYS) == cell
        ]
        assert sorted(run.seed for run in cell_runs) == [0, 1, 2, 3, 4]
        assert {run.resolution for run in cell_runs} == {128}
        assert len({run.config_hash for run in cell_runs}) == 5


def test_baseline_matrix_expands_requested_models() -> None:
    root = Path(__file__).resolve().parents[1]
    runs = expand_baseline_runs(root=root, dataset="poisson", experiment="baselines")

    assert len(runs) == 30
    assert sorted({run.baseline_model for run in runs}) == [
        "fno",
        "global_pca",
        "l2g",
        "l2l",
        "l2l_overlap",
        "l2l_refinement",
    ]
    for model_name in {run.baseline_model for run in runs}:
        model_runs = [run for run in runs if run.baseline_model == model_name]
        assert sorted(run.seed for run in model_runs) == [0, 1, 2, 3, 4]
        assert {run.resolution for run in model_runs} == {128}


def test_warm_start_timings_are_cumulative_and_preserve_invocation(tmp_path: Path) -> None:
    source_dir = tmp_path / "two_scale"
    child_dir = tmp_path / "two_scale_in_loop"
    source_dir.mkdir()
    child_dir.mkdir()
    _write_runtime(
        source_dir / "runtime.json",
        pca_fit=10.0,
        latent_transform=2.0,
        nn_train=20.0,
        inference=1.0,
        end_to_end=40.0,
        coarse_svd=0.5,
    )
    _write_runtime(
        child_dir / "runtime.json",
        pca_fit=0.0,
        latent_transform=3.0,
        nn_train=30.0,
        inference=1.5,
        end_to_end=45.0,
        coarse_svd=0.5,
    )
    run = FactorialRun(
        run_id="two_scale_in_loop",
        config={
            "warm_start": {"enabled": True, "run_dir": str(source_dir)},
        },
        config_hash="hash",
        seed=0,
        mechanisms={"two_scale": True, "coupling": False, "in_loop_loss": True},
        resolution=256,
        run_dir=child_dir,
        baseline_model="two_scale_in_loop",
    )

    record = build_result_record(run, metrics={}, run_dir=child_dir)

    assert record["timings"] == {
        "pca_fit": 10.0,
        "latent_transform": 5.0,
        "nn_train": 50.0,
        "inference": 2.5,
        "end_to_end": 85.0,
        "coarse_svd": 0.5,
    }
    assert record["invocation_timings"]["end_to_end"] == 45.0
    assert record["timing_accounting"]["includes_warm_start"] is True

    table_path = write_stage_cost_table([record], tmp_path)
    header, row = table_path.read_text(encoding="utf-8").splitlines()
    assert "invocation_end_to_end" in header
    assert "85.0" in row
    assert "45.0" in row


def _write_runtime(path: Path, **stages: float) -> None:
    path.write_text(
        json.dumps({"stages": stages, "times": stages}),
        encoding="utf-8",
    )
