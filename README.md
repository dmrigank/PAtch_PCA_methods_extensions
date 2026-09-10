# Two-Scale Localized PCA-Net for Artifact-Reduced PDE Operator Learning

Patch-based PCA neural operators are a fast way to learn solution operators for
elliptic PDEs, but reconstructing a field from independently decoded patches
introduces **patch-interface artifacts**: visible blockiness, interface jumps,
and spurious high-wavenumber energy. Prior fixes, such as Hann overlap-add and
a CNN RefinementNet, are downstream post-processing.

This repository reduces the dominant artifact at the representation level,
then optionally refines the remaining interface defect in physical-field
space:

1. **Two-scale representation**: a coarse-global + local-residual basis, so a
   shared smooth backbone carries the inter-patch structure and the local patches
   only represent a small residual.
2. **Interface-aware fine-tuning**: a frozen differentiable decoder with
   reconstruction, value-trace, and normal-derivative trace losses. This is an
   optional continuity-oriented refinement, not the default model.

The primary objective is a favorable accuracy--continuity--cost tradeoff within
the PCA-Net family, without overlap or a learned post-processor. Full-field FNO
is retained as a higher-capacity accuracy reference.

The final scientific and implementation scope is defined in `DESIGN.md`.

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
| Poisson | $\nabla^2 u = f$, Dirichlet, GRF forcing | legacy prior-paper convention; final data use $\alpha=2$, $\tau=3$ |
| Darcy | $-\nabla\!\cdot(a\nabla u) = f$ | binary / sharp-interface coefficients; the stress case |

Each processed archive contains 10,000 fields. Standard experiments use an
80/10/10 train/validation/test split, hence $m=8000$ training fields.
Resolutions are **64 / 128 / 256**, with 128 as the primary Poisson setting.

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

## Paper experiments

```bash
# headline Poisson-128 comparison (5 paired seeds)
python scripts/run_experiment.py -m experiment=headline_poisson_128

# representation versus interface-aware fine-tuning (5 paired seeds)
python scripts/run_experiment.py -m experiment=mechanism_ablation

# Poisson resolution study at 64/128/256 (3 paired seeds)
python scripts/run_experiment.py -m experiment=resolution_sweep

# heterogeneous Darcy-256 transfer study (5 paired seeds)
python scripts/run_experiment.py -m experiment=darcy_generalization
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

## Data availability

Processed dataset archives and generated paper results are intentionally not
versioned in Git. The reproducibility release will deposit the exact processed
Poisson and Darcy NPZ archives in a versioned Zenodo dataset record. That record
should include the archives, a SHA-256 checksum manifest, the resolved
generation and experiment configurations, and a short README documenting the
PDE conventions, field names, array dtypes, sample counts, and train/validation/
test split procedure. The version-specific Zenodo DOI should be cited by the
manuscript and inserted here once the dataset record is published.

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
