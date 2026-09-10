from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_two_scale_schematic_renders(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "two_scale_method_schematic"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "make_two_scale_schematic.py"),
            "--output",
            str(output),
            "--formats",
            "png,svg",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert output.with_suffix(".png").is_file()
    assert output.with_suffix(".svg").is_file()
    assert output.with_suffix(".png").stat().st_size > 10_000
    assert "Learned two-scale operator" in output.with_suffix(".svg").read_text(
        encoding="utf-8"
    )
