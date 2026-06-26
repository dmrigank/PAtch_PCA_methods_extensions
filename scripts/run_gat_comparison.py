#!/usr/bin/env python
"""Run 6-way GAT comparison: 3 backends x with/without in-loop loss on Poisson 128."""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lpcanet.train.experiment import train_from_config
from lpcanet.utils.config import compose_config, save_config
from lpcanet.utils.paths import ensure_dir

RUNS = [
    dict(backend="gnn",       in_loop=False),
    dict(backend="gnn",       in_loop=True),
    dict(backend="attention", in_loop=False),
    dict(backend="attention", in_loop=True),
    dict(backend="gat",       in_loop=False),
    dict(backend="gat",       in_loop=True),
]

# Patch grid: 128 grid / 32 patch_size = 4 per side → N=16 tokens.
# Explicit overrides so model="l2l"'s patch_size=64 default can't silently win.
PATCH_OVERRIDES = [
    "patches.patch_size=32",
    "patches.stride=32",
]

# Shared coupling hyperparameters — identical across all six runs.
COUPLING_OVERRIDES = [
    "model.coupling.neighborhood=8",
    "model.coupling.embed_dim=64",
    "model.coupling.num_layers=3",
    "model.coupling.attention_heads=4",
    "model.coupling.attention_n_max=256",
]

for i, run in enumerate(RUNS, 1):
    backend  = run["backend"]
    in_loop  = run["in_loop"]
    tag      = f"{backend}_{'with' if in_loop else 'without'}_inloop"
    out_dir  = str(ROOT / "results" / "gat_comparison_n16" / tag)

    overrides = [
        "mechanisms.coupling=true",
        f"mechanisms.in_loop_loss={'true' if in_loop else 'false'}",
        f"model.coupling.backend={backend}",
        *PATCH_OVERRIDES,
        *COUPLING_OVERRIDES,
    ]

    config = compose_config(
        root=ROOT,
        dataset="poisson",
        model="l2l",
        experiment="single",
        overrides=overrides,
    )
    config.setdefault("experiment", {}).update({
        "name": f"gat_comparison_n16_{tag}",
        "seed": 0,
        "output_dir": out_dir,
    })
    config.setdefault("training", {}).update({
        "epochs": 200,
        "device": "mps",
    })

    patch_size = config["patches"]["patch_size"]
    grid_size  = config["dataset"]["grid_size"]
    n_patches  = (grid_size // patch_size) ** 2

    print(f"\n{'='*64}")
    print(f"  RUN {i}/6 — backend={backend!r}  in_loop_loss={in_loop}")
    print(f"  patch_size={patch_size}  grid_size={grid_size}  N={n_patches} tokens")
    print(f"  output → {out_dir}")
    print(f"{'='*64}\n", flush=True)

    if n_patches != 16:
        raise RuntimeError(f"Expected N=16 tokens, got N={n_patches}. Aborting.")

    run_dir     = ensure_dir(Path(out_dir))
    config_path = save_config(config, ensure_dir(run_dir / "resolved_configs"), filename="config.yaml")

    t0 = perf_counter()
    # Raises immediately on any error — no silent continuation.
    train_from_config(config_path, overwrite=True)
    elapsed = perf_counter() - t0

    print(f"\n  DONE — backend={backend!r}  in_loop_loss={in_loop}  ({elapsed:.1f}s)\n", flush=True)

print("\nAll 6 runs complete.\n")
