"""Path helpers with Windows/WSL compatibility."""

from __future__ import annotations

import re
from pathlib import Path

_WINDOWS_DRIVE_RE = re.compile(r"^(?P<drive>[A-Za-z]):[\\/](?P<rest>.*)$")


def normalize_path(path: str | Path) -> Path:
    """Return an expanded ``Path`` with basic Windows-to-WSL normalization.

    On WSL/Linux, paths like ``C:/Users/name/file.mat`` are mapped to
    ``/mnt/c/Users/name/file.mat`` when the corresponding mount exists. Native
    POSIX paths are returned unchanged apart from ``~`` expansion.
    """
    raw = str(path).strip()
    match = _WINDOWS_DRIVE_RE.match(raw)
    if match:
        drive = match.group("drive").lower()
        rest = match.group("rest").replace("\\", "/")
        wsl_path = Path("/mnt") / drive / rest
        if Path("/mnt").exists():
            return wsl_path.expanduser()
    return Path(raw).expanduser()


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a normalized path."""
    directory = normalize_path(path)
    directory.mkdir(parents=True, exist_ok=True)
    if not directory.is_dir():
        raise NotADirectoryError(f"Expected directory path, got {directory}")
    return directory


def require_nonexistent_or_empty(path: str | Path, *, overwrite: bool = False) -> Path:
    """Validate an output directory without deleting existing data.

    If ``overwrite`` is false, an existing non-empty directory raises
    ``FileExistsError``. If ``overwrite`` is true, the path is accepted but never
    deleted by this helper.
    """
    directory = normalize_path(path)
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {directory}. "
            "Pass overwrite=True only when the caller will handle existing files safely."
        )
    return directory
