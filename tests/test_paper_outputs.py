from __future__ import annotations

import json
import subprocess
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def test_table_and_figure_scripts_run_on_small_fixture(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    results_dir = tmp_path / "results"
    _write_fixture(results_dir)

    tables_dir = tmp_path / "tables"
    table_cmd = [
        sys.executable,
        str(root / "scripts" / "make_tables.py"),
        "--results-dir",
        str(results_dir),
        "--out-dir",
        str(tables_dir),
    ]
    table_result = subprocess.run(table_cmd, cwd=root, capture_output=True, text=True, check=False)
    assert table_result.returncode == 0, table_result.stderr or table_result.stdout
    assert (tables_dir / "headline_factorial_table.csv").exists()
    assert (tables_dir / "resolution_stage_costs_table.csv").exists()

    figures_dir = tmp_path / "figures"
    figure_cmd = [
        sys.executable,
        str(root / "scripts" / "make_figures.py"),
        "--results-dir",
        str(results_dir),
        "--out-dir",
        str(figures_dir),
        "--formats",
        "png",
    ]
    figure_result = subprocess.run(figure_cmd, cwd=root, capture_output=True, text=True, check=False)
    assert figure_result.returncode == 0, figure_result.stderr or figure_result.stdout
    assert (figures_dir / "qualitative_poisson_plain_l2l_vs_full.png").exists()
    assert (figures_dir / "qualitative_poisson_plain_l2l_vs_two_scale.png").exists()
    assert (figures_dir / "qualitative_poisson_plain_l2l_vs_coupling.png").exists()
    assert (figures_dir / "qualitative_poisson_plain_l2l_vs_in_loop_loss.png").exists()
    assert (figures_dir / "qualitative_darcy_plain_l2l_vs_full.png").exists()
    assert (figures_dir / "factorial_interface_spectral_comparison.png").exists()
    assert (figures_dir / "factorial_stage_timing_breakdown.png").exists()


def _write_fixture(results_dir: Path) -> None:
    rng = np.random.default_rng(7)
    factorial_records = []
    keys = ("two_scale", "coupling", "in_loop_loss")
    for seed in [0, 1]:
        for flags_tuple in product([False, True], repeat=3):
            flags = dict(zip(keys, flags_tuple))
            method_name = "full" if all(flags.values()) else "factorial"
            run_dir = results_dir / "factorial" / _cell_name(flags) / f"seed_{seed}"
            _write_run_artifacts(run_dir, dataset="poisson", seed=seed, rng=rng, full=all(flags.values()))
            scale = 0.6 if all(flags.values()) else 1.0 + 0.05 * sum(flags.values())
            factorial_records.append(
                _record(
                    run_dir=run_dir,
                    seed=seed,
                    mechanisms=flags,
                    baseline_model="",
                    run_id=f"{method_name}_{seed}_{_cell_name(flags)}",
                    scale=scale,
                )
            )
    _write_jsonl(results_dir / "factorial" / "results.jsonl", factorial_records)

    baseline_records = []
    for model in ["l2l_overlap", "l2l_refinement"]:
        for seed in [0, 1]:
            run_dir = results_dir / "baselines" / model / f"seed_{seed}"
            _write_run_artifacts(run_dir, dataset="poisson", seed=seed, rng=rng, full=False)
            baseline_records.append(
                _record(
                    run_dir=run_dir,
                    seed=seed,
                    mechanisms={"two_scale": False, "coupling": False, "in_loop_loss": False},
                    baseline_model=model,
                    run_id=f"{model}_{seed}",
                    scale=0.8 if model == "l2l_overlap" else 0.7,
                )
            )
    _write_jsonl(results_dir / "baselines" / "results.jsonl", baseline_records)

    darcy_records = []
    for model, flags, full in [
        ("l2l", {"two_scale": False, "coupling": False, "in_loop_loss": False}, False),
        ("full", {"two_scale": True, "coupling": True, "in_loop_loss": True}, True),
    ]:
        run_dir = results_dir / "darcy" / model / "seed_0"
        _write_run_artifacts(run_dir, dataset="darcy", seed=0, rng=rng, full=full)
        darcy_records.append(
            _record(
                run_dir=run_dir,
                seed=0,
                mechanisms=flags,
                baseline_model=model,
                run_id=f"darcy_{model}_0",
                scale=0.6 if full else 1.0,
            )
        )
    _write_jsonl(results_dir / "darcy" / "results.jsonl", darcy_records)

    stage_dir = results_dir / "resolution_sweep"
    stage_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "method": method,
                "resolution": resolution,
                "seed": seed,
                "pca_fit": 1.0 * resolution / 64,
                "latent_transform": 0.2,
                "nn_train": 2.0,
                "inference": 0.1,
                "end_to_end": 3.3,
            }
            for method in ["l2l", "full"]
            for resolution in [64, 128]
            for seed in [0, 1]
        ]
    ).to_csv(stage_dir / "stage_costs.csv", index=False)


def _write_run_artifacts(
    run_dir: Path,
    *,
    dataset: str,
    seed: int,
    rng: np.random.Generator,
    full: bool,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0.0, 1.0, 8, dtype=np.float64)
    x, y = np.meshgrid(grid, grid, indexing="ij")
    truth = np.stack(
        [
            np.sin(np.pi * x) * np.sin(np.pi * y) + 0.02 * seed,
            np.cos(np.pi * x) * np.sin(np.pi * y) + 0.02 * seed,
        ],
        axis=0,
    )
    noise = 0.02 if full else 0.05
    pred = truth + noise * rng.normal(size=truth.shape)
    source = np.stack([x - y, x + y], axis=0)
    np.savez_compressed(run_dir / "predictions_test.npz", x_input=source, y_true=truth, y_pred=pred)
    config = {
        "dataset": {"name": dataset, "grid_size": 8},
        "experiment": {"seed": seed},
        "mechanisms": {
            "two_scale": full,
            "coupling": full,
            "in_loop_loss": full,
        },
    }
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)


def _record(
    *,
    run_dir: Path,
    seed: int,
    mechanisms: dict[str, bool],
    baseline_model: str,
    run_id: str,
    scale: float,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "config_hash": f"hash-{run_id}",
        "run_dir": str(run_dir),
        "seed": seed,
        "resolution": 8,
        "baseline_model": baseline_model,
        "mechanisms": mechanisms,
        "metrics": {
            "mre": 0.1 * scale + seed * 0.001,
            "mse": 0.01 * scale,
            "mae": 0.02 * scale,
            "ssim_mean": 0.9 + 0.01 / scale,
            "relative_spectrum_error": 0.2 * scale,
            "interface_jump": 0.03 * scale,
            "interface_flux_jump": 0.04 * scale,
            "poisson_residual_rms": 0.05 * scale,
        },
        "timings": {
            "pca_fit": 1.0 * scale,
            "latent_transform": 0.2,
            "nn_train": 2.0 * scale,
            "inference": 0.1,
            "end_to_end": 3.3 * scale,
        },
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _cell_name(flags: dict[str, bool]) -> str:
    return "__".join(f"{key}-{'on' if flags[key] else 'off'}" for key in flags)
