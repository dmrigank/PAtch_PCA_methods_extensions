#!/usr/bin/env python
"""Generate deterministic datasets from composed configs.

This entry point is reserved for the data-generation/cache layer in the build
playbook. Legacy data-prep code remains under `legacy/patch_pca/` until that
step wraps it.
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit("scripts/generate_data.py is scaffolded; data wrapping is not implemented yet.")


if __name__ == "__main__":
    main()
