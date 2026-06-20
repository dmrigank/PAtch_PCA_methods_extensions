from __future__ import annotations

from pathlib import Path

from lpcanet.train.factorial import _run_is_complete
from lpcanet.utils.paths import run_dir_has_artifacts


def test_resolved_config_is_not_treated_as_run_output(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    resolved_dir = run_dir / "resolved_configs"
    resolved_dir.mkdir(parents=True)
    (resolved_dir / "config.yaml").write_text("experiment: {}\n", encoding="utf-8")

    assert not run_dir_has_artifacts(run_dir)

    (run_dir / "model.pt").write_bytes(b"checkpoint")
    assert run_dir_has_artifacts(run_dir)


def test_completed_run_requires_metrics_runtime_config_and_predictions(tmp_path: Path) -> None:
    for filename in ("config.yaml", "metrics.json", "runtime.json"):
        (tmp_path / filename).write_text("{}\n", encoding="utf-8")
    assert not _run_is_complete(tmp_path)

    (tmp_path / "predictions_test.npz").write_bytes(b"npz")
    assert _run_is_complete(tmp_path)
