#!/usr/bin/env python
"""GNN vs GAT seed sweep: seeds 1-3, in_loop_loss=True only, N=16 (patch_size=32)."""

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

BACKENDS = ["gnn", "gat"]
SEEDS    = [1, 2, 3]

# Patch grid: 128 / 32 = 4 per side → N=16 tokens.
# Explicit so model="l2l"'s patch_size=64 can't silently win.
PATCH_OVERRIDES = [
    "patches.patch_size=32",
    "patches.stride=32",
]

COUPLING_OVERRIDES = [
    "model.coupling.neighborhood=8",
    "model.coupling.embed_dim=64",
    "model.coupling.num_layers=3",
    "model.coupling.attention_heads=4",
    "model.coupling.attention_n_max=256",
]

RUNS = [(b, s) for b in BACKENDS for s in SEEDS]
n_runs = len(RUNS)

for i, (backend, seed) in enumerate(RUNS, 1):
    tag     = f"{backend}_with_inloop_seed{seed}"
    out_dir = str(ROOT / "results" / "gat_comparison_n16" / tag)

    overrides = [
        "mechanisms.coupling=true",
        "mechanisms.in_loop_loss=true",
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
        "name":       f"seed_sweep_{tag}",
        "seed":       seed,
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
    print(f"  RUN {i}/{n_runs} — backend={backend!r}  seed={seed}  in_loop_loss=True")
    print(f"  patch_size={patch_size}  grid_size={grid_size}  N={n_patches} tokens")
    print(f"  output → {out_dir}")
    print(f"{'='*64}\n", flush=True)

    if n_patches != 16:
        raise RuntimeError(f"Expected N=16, got N={n_patches}. Aborting.")

    run_dir     = ensure_dir(Path(out_dir))
    config_path = save_config(config, ensure_dir(run_dir / "resolved_configs"), filename="config.yaml")

    t0 = perf_counter()
    train_from_config(config_path, overwrite=True)
    elapsed = perf_counter() - t0

    print(f"\n  DONE — backend={backend!r}  seed={seed}  ({elapsed:.1f}s)\n", flush=True)

print(f"\nAll {n_runs} seed-sweep runs complete.\n")
