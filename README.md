# Localized PCA-Net with Interface Consistency by Construction

Patch-based PCA neural operators are a fast way to learn solution operators for
elliptic PDEs, but reconstructing a field from independently decoded patches
introduces **patch-interface artifacts** — visible blockiness, large interface
jumps, and spurious high-wavenumber energy. Prior fixes (Hann overlap-add, a CNN
RefinementNet) are *post-processing*.

This repository builds interface consistency **into** the method instead of
correcting it afterward, through three independently toggleable mechanisms:

1. **Representation** — a two-scale *coarse-global + local-residual* basis, so a
   shared smooth backbone carries the inter-patch structure and the local patches
   only represent a small residual.
2. **Architecture** — a *patch-coupling operator* (graph message-passing by
   default, attention for small patch counts) that replaces the concatenation head
   with a learned, multi-hop generalization of neighbor-aware coupling.
3. **Objective** — *differentiable in-loop assembly* with interface (value + flux),
   PDE-residual, and high-wavenumber spectral losses, so smoothness is trained
   rather than post-fixed.

The goal: match or beat the best post-processing baseline on accuracy and spectral
fidelity **with no post-processing stage**, at cost comparable to plain
local-to-local PCA and well below global PCA.

> This is the **methods** paper of a two-paper plan. Transient and harder datasets
> are a separate follow-up and are intentionally out of scope here (see
> `DESIGN.md`).

## Status

Research code. APIs may change. Results are reproduced via configs + fixed seeds.

## Install

```bash
git clone <repo-url>
cd localized-pca-net-interface
pip install -e ".[dev]"
```

Requires Python 3.10+, PyTorch, NumPy, SciPy.

## Quickstart

```bash
# 1. generate a dataset (cached, deterministic)
python scripts/generate_data.py dataset=poisson resolution=128

# Generate the 10,000-sample Poisson dataset used by the 256-grid sweep
conda run -n DiffusionPDE python scripts/generate_data.py \
  dataset=poisson resolution=256 n_samples=10000 seed=0 batch_size=32 \
  output=data/processed/poisson_256.npz

# 2. train the full model
python scripts/run_experiment.py model=full dataset=poisson resolution=128 seed=0

# 3. tables + figures
python scripts/make_tables.py
python scripts/make_figures.py
```

## Datasets (steady-state)

| Dataset | PDE | Notes |
|---|---|---|
| Poisson | $-\nabla^2 u = f$, Dirichlet, GRF forcing | smooth control case; GRF roughness sweep $\alpha = 3.0/2.0/1.5$ |
| Darcy | $-\nabla\!\cdot(a\nabla u) = f$ | binary / sharp-interface coefficients; the stress case |

Resolutions **64 / 128 / 256** (128 primary), $m = 8000$ samples, 99% retained
variance.

The generators preserve the prior paper's GRF and PDE conventions and write
deterministic 80/10/10 splits. A 10,000-sample 256-grid dataset containing two
float32 fields is about 4.9 GiB; generation temporarily requires about 9.8 GiB
while the final NPZ is assembled. Compression is optional but substantially
slower. For Darcy, a short timing pilot can be run before the full generation:

```bash
conda run -n DiffusionPDE python scripts/generate_data.py \
  dataset=darcy resolution=256 n_samples=100 seed=0 batch_size=16 \
  output=data/processed/darcy_256_pilot.npz
```

## Reproducing the paper

```bash
# headline 2x2x2 mechanism ablation (5 seeds, res=128)
python scripts/run_experiment.py -m experiment=factorial

# resolution sweep: plain L2L, L2L+overlap, full model @ 64/128/256
python scripts/run_experiment.py -m experiment=resolution_sweep

# baselines (Global / L2G / L2L / L2L+overlap / RefinementNet / MG-TFNO / FNO)
python scripts/run_experiment.py -m experiment=baselines
```

Metrics: relative error (MRE), SSIM, MSE/MAE, interface jump (value **and** flux),
spectral relative error, and PDE residual. Cost is reported stage-by-stage (PCA
fit / latent transform / training / inference / end-to-end) at every resolution.

## Repository structure

```
src/lpcanet/   data · pca · assembly · models · losses · metrics · train · utils
configs/       base + datasets + models + experiments (composable YAML)
scripts/       generate_data · run_experiment · make_tables · make_figures
tests/         numerical-correctness tests
DESIGN.md      design rationale, math, experiment matrix, invariants
```

## Documentation

- **`DESIGN.md`** — the thesis, the three mechanisms with their math, the
  cost-scaling argument, the experiment matrix, and the correctness invariants and
  scope guardrails.

## Citation

```bibtex
@article{TODO,
  title   = {Localized PCA-Net with Interface Consistency by Construction},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO}
}
```

This work builds on *Localized PCA-Net neural operators for scalable solution
reconstruction of elliptic PDEs* (Dhingra, Maulik, Rasheed, San, 2026).

## License

TODO (e.g. MIT).
