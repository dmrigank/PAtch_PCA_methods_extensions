# DESIGN

Localized PCA-Net with **interface consistency by construction**.

This document is the source of truth for *why* the code is shaped the way it is.
It is written to be read top-to-bottom before any implementation, and to be
re-read whenever a design decision is in question. User-facing instructions live in `README.md`.

---

## 1. Thesis

The prior work (localized PCA-Net: L2G / L2L, with Hann overlap-add and a CNN
RefinementNet) showed that patch-wise PCA is a fast, accurate dimensionality
reduction for elliptic PDE solution operators, but that the **local-to-local
(L2L)** reconstruction produces patch-interface artifacts — visible blockiness,
large interface jumps, and spurious high-wavenumber energy. Both existing fixes
(Hann overlap-add, RefinementNet) are **post-processing**: they sit downstream of
a representation and a latent map that have no notion of inter-patch continuity.

**Claim of this paper:** the interface artifact can be removed *at the source* —
in the representation, the architecture, and the training objective — so that the
post-processing stage becomes unnecessary, at a cost comparable to plain L2L and
far below global PCA-Net.

Every experiment in this repo must speak to at least one of three things:

1. **Accuracy / spectral parity** — the new method matches or beats `L2L+overlap`
   and `L2L+RefinementNet`.
2. **Stage deletion** — it achieves that *without* a post-processing stage.
3. **Cost** — end-to-end cost stays comparable to plain L2L and well below global.

If a result does not bear on (1), (2), or (3), it is an appendix or it belongs to
paper 2 (transient / harder datasets), which this repo deliberately does **not**
implement.

---

## 2. The three mechanisms

Continuity can enter the pipeline in three places. Each is implemented as an
independently toggleable mechanism so the ablation can attribute the effect.

### 2.1 Representation — two-scale (coarse-global + local-residual)

**Idea.** Give the reconstruction a shared smooth backbone so the local patches
only have to represent a small, near-zero-at-the-boundary residual. Tiling is
suppressed *at the source* because the large-scale field that crosses patch
boundaries is reconstructed from a single global basis, not stitched.

**Construction (output side).**

- Let $u \in \mathbb{R}^{D \times D}$ be a solution field. Choose a coarse factor
  $c$ and restrict to a coarse grid $D_c = D/c$ via an averaging/interpolation
  operator $R$:  $u_c = R\,u \in \mathbb{R}^{D_c \times D_c}$.
- Fit a **global** PCA on the coarse fields: basis $\Phi_c$, mean $\mu_c$,
  retaining $k_c \approx 10\text{–}20$ modes (target 99% of coarse variance).
- Prolongate back with $P$ (bilinear up-sampling): $\tilde u = P\,(\mu_c + \Phi_c
  \Phi_c^\top (u_c - \mu_c))$. Define the **residual** $r = u - \tilde u$.
- Fit **local patch PCA** on the residual $r$, exactly as in the existing L2L
  output path, producing per-patch bases $\{\Phi_p^{\text{res}}, \mu_p^{\text{res}}\}$.

**Decode.** $\hat u = P\,(\mu_c + \Phi_c\,\hat y_c) + \text{mosaic}\big(\{\mu_p^{\text{res}}
+ \Phi_p^{\text{res}}\,\hat y_p^{\text{res}}\}\big)$.

**Cost discipline — this is the crux of the speed argument.** The cost of a thin
SVD of an $m \times n$ matrix is dominated by the *smaller* dimension. Global PCA
in the original paper was expensive because at full resolution
$n = D^2 \gtrsim m$ (e.g. $D=128 \Rightarrow n \approx 16{,}384$ vs $m = 8000$),
putting it in the $\mathcal{O}(m^2 n)$ regime with many modes. The coarse backbone
lives in the **opposite** regime: it is smooth by construction, so it is computed
on a downsampled grid where $n_c = D_c^2 \ll m$ and only $\sim$10–20 modes are
needed. Therefore:

> **Invariant:** choose the coarse factor $c$ so that $n_c = (D/c)^2 < m$.
> Use randomized SVD (already validated in the prior paper's Appendix A) for the
> coarse basis. With $n_c < m$ and few modes, the coarse SVD is *cheaper than the
> local residual stage itself*, so the global-vs-two-scale speedup is essentially
> intact.

Concretely at $m = 8000$: $D{=}128, c{=}4 \Rightarrow n_c{=}1024$ ✓;
$D{=}256, c{=}4 \Rightarrow 4096$ ✓; $D{=}512, c{=}4 \Rightarrow 16{,}384$ ✗ →
use $c{=}8 \Rightarrow 4096$ ✓. The coarse factor is therefore **resolution
dependent** and is a config field, not a constant.

The honest cost note to report: two-scale adds a small fixed cost over plain L2L
(the coarse SVD + restriction/prolongation + a two-stage encode/decode), but the
*whole RefinementNet stage is deleted*. Net pipeline cost should still beat global
comfortably; the cost table must make this legible stage-by-stage.

**Input side** stays local (as in L2L). Only the output representation becomes
two-scale.

### 2.2 Architecture — patch-coupling operator

**Idea.** The prior ablation showed: purely independent / shared patch heads fail,
4-/8-neighbor heads help but underperform the global concat head, and the concat
head wins but its input dimension grows with the patch count. The resolution is a
learned, **multi-hop** generalization of the neighbor-aware head that keeps the
per-token dimension fixed: message passing (GNN) or attention over patch tokens.

**Tokens.** One token per patch. Patch latent codes have *heterogeneous* lengths
(99% variance ⇒ different mode counts per patch), so each patch code is first
mapped to a common embedding dimension $d$ by a per-patch linear encoder (the
inverse linear decoder maps the coupled embedding back to the patch's output code
length). Positional information = the patch's $(row, col)$ grid index, added as a
learned or sinusoidal embedding.

**Two backends, one interface (`CouplingOperator`):**

- **GNN (default).** Message passing on the patch-adjacency graph (4- or
  8-neighbor). Cost $\mathcal{O}(\text{edges})$ ⇒ **linear** in patch count.
  This is the default because (a) it scales to the fine/overlapping configs where
  the token count is large, and (b) it ports directly to unstructured meshes,
  which paper 2 needs. 2–4 message-passing layers.
- **Attention.** A small transformer encoder over the patch tokens. Cost
  $\mathcal{O}(N^2 d)$ in token count $N$. At the main config (128 grid, 32×32
  patches ⇒ $N \approx 16$, or $\sim 64$ with overlap) this is free and is the
  simplest drop-in. **Guardrail:** attention is only selected when $N \le N_{\max}$
  (config; default 256); above that, fall back to GNN. Do not run full attention on
  the fine-patch configs (8×8 @ stride 4 ⇒ $N \sim 10^3$).

The coupling operator *replaces* the concat MLP head; it does not wrap it.

### 2.3 Objective — differentiable in-loop assembly + physics losses

**Idea.** Mosaicking is differentiable, so do it *inside* the training graph and
backprop a loss computed on the assembled physical field. Smoothness becomes a
trained property instead of a post-hoc correction.

**Differentiable assembly.** The PCA decode $\mu + \Phi\,\hat y$ is linear in the
network output $\hat y$, and Hann overlap-add is a weighted scatter. Implement
assembly as:

```
accum_val[ξ] = Σ_q  W_q(ξ) · û_q(ξ)        # scatter-add, weighted
accum_wt [ξ] = Σ_q  W_q(ξ)                  # scatter-add of weights
û(ξ)         = accum_val[ξ] / (accum_wt[ξ] + ε)
```

with precomputed index maps (patch → global pixel) and Hann weights $W_q$. This is
differentiable w.r.t. every decoded patch value, hence w.r.t. $\hat y$. For
two-scale, the prolongated coarse term is added before the normalization is applied
to the residual mosaic (coarse is dense, weight 1 everywhere).

> **Validation invariant:** with random weights and the existing (non-trained)
> decode, the differentiable assembler must reproduce the prior paper's
> `L2L+overlap` forward reconstruction to floating-point tolerance. This is the
> first test to pass before any training. See `tests/test_assembly_parity.py`.

**Losses** (weighted sum; reconstruction-dominant at start, then ablate weights):

| Term | Definition | Targets |
|---|---|---|
| `recon` | MSE / relative L2 of $\hat u$ vs $u$ | pointwise accuracy |
| `interface_value` | mean abs jump across stride-induced seams | blockiness (existing $J$) |
| `interface_flux` | mean abs jump of the **normal first derivative** across seams | conservation / flux continuity |
| `pde_residual` | discrete $-\nabla^2\hat u - f$ (Poisson) or $-\nabla\!\cdot(a\nabla\hat u)-f$ (Darcy) | physical consistency |
| `spectral` | radially-binned energy error $\|E_{\hat u}(k) - E_u(k)\|$, **weighted toward high $k$** | the exact high-wavenumber symptom |

All terms operate on the *assembled* field, all are implemented with
`torch.fft` / finite-difference stencils, all are differentiable. The interface
sets $B_r, B_c$ and finite-difference residual stencils reuse the prior paper's
definitions verbatim so the *loss* and the *metric* are the same operator.

---

## 3. Data

Steady-state only. Two PDE families, reused from the prior paper so comparisons are
apples-to-apples against published numbers.

- **2D Poisson** — $-\nabla^2 u = f$, homogeneous Dirichlet, GRF forcing $f$.
  The smooth-solution control case; two-scale is expected to look best here.
- **Variable-coefficient Darcy** — $-\nabla\!\cdot(a\nabla u) = f$, with the
  binary / sharp-interface coefficient fields from the prior paper. The stress
  case: the smooth-coarse-backbone assumption is genuinely challenged by material
  interfaces, so the coupling operator and interface loss must carry the load.

**GRF roughness sweep** ($\alpha = 3.0 / 2.0 / 1.5$) is reused as a difficulty knob
on Poisson — a roughness axis without introducing new physics (that is paper 2).

**Resolutions.** Sweep **64 / 128 / 256**, $m = 8000$ fixed, **128 primary**
(matches published numbers). One config optionally pushed to **512** as a
scaling-limit demonstration. The cost claim *is* a scaling claim, so the
resolution sweep is not optional polish — it is core evidence.

Data generation lives in `src/lpcanet/data/`; generated arrays are cached to disk
(gitignored) and addressed by a content hash of the generation config so reruns are
deterministic and reproducible.

---

## 4. Baselines

Reused from the prior paper (cite published numbers where the config is identical):

- **Global PCA-Net** — the cost ceiling.
- **L2G PCA-Net** — local input, global output.
- **plain L2L PCA-Net** — the artifact source; accuracy floor for the new methods.
- **L2L + Hann overlap** — the best post-processing; *the number to match while
  deleting the stage*.
- **L2L + RefinementNet** — CNN post-processing.
- **MG-TFNO** — ranks 0.10 / 0.25 / 0.50.

Added because reviewers will expect them:

- **FNO** (standard) — the expected neural-operator yardstick.
- **DeepONet** (optional) — second operator reference.

Randomized SVD (prior Appendix A) is the **default basis construction** for the new
methods so the cost comparison is honest.

---

## 5. Core experiment — the headline ablation

A $2\times2\times2$ factorial over the three mechanisms on top of plain L2L:

| two_scale | coupling | in_loop_loss | reads as |
|:--:|:--:|:--:|---|
| ✗ | ✗ | ✗ | plain L2L (floor) |
| ✓ | ✗ | ✗ | representation only |
| ✗ | ✓ | ✗ | architecture only |
| ✗ | ✗ | ✓ | objective only |
| ✓ | ✓ | ✗ | repr + arch |
| ✓ | ✗ | ✓ | repr + obj |
| ✗ | ✓ | ✓ | arch + obj |
| ✓ | ✓ | ✓ | **full model** |

- Full grid at **128**, 5 seeds, mean ± std, full metric suite + cost.
- The **64 / 256** sweep runs only `{plain L2L, L2L+overlap, full model}` to keep
  the grid tractable.
- Target result: the full model matches/beats `L2L+overlap` and
  `L2L+RefinementNet` on accuracy and spectral fidelity, **with no
  post-processing stage**, at cost comparable to plain L2L. The single-mechanism
  rows give the marginal-contribution / mechanistic story.

---

## 6. Metrics & protocol

**Metrics** (match prior paper for continuity, plus the interface/spectral suite):
`MRE`, `SSIM`, `MSE`, `MAE`; interface jump $J$ (**value and flux**); spectral
relative error $e_E$; PDE residual ($R_P$ Poisson, $R_D$ Darcy).

**Cost** is its own table, broken out by stage — PCA fit / latent transform / NN
train / inference / end-to-end — **at all three resolutions**. The stage breakdown
is what proves "we added a tiny coarse SVD + a GNN but removed RefinementNet, and
net cost still beats global."

**Protocol** (identical to prior paper): fixed train/val/test split, 5 random
seeds, 99% retained variance, Adam, same sample counts; report mean ± std. Seeds
control data split, PCA (randomized SVD), and network init.

---

## 7. Repository layout

```
src/lpcanet/
  data/        # poisson, darcy, grf generation + cached loaders
  pca/         # local / global / two-scale bases; randomized SVD
  assembly/    # differentiable mosaic, hann weights, restrict/prolongate
  models/      # coupling operators (concat-MLP, GNN, attention),
               #   baselines (FNO, MG-TFNO, DeepONet, RefinementNet)
  losses/      # recon, interface (value+flux), pde residual, spectral
  metrics/     # MRE, SSIM, J (value+flux), e_E, pde residual
  train/       # training loop, seeding, checkpointing
  utils/       # config, logging, timing
configs/
  base.yaml
  datasets/    # poisson.yaml, darcy.yaml
  models/      # one yaml per model / mechanism combination
  experiments/ # factorial.yaml, resolution_sweep.yaml, baselines.yaml
scripts/       # generate_data.py, run_experiment.py, make_tables.py, make_figures.py
tests/
results/       # gitignored
```

**Config is the single source of experimental truth.** Every run is fully
specified by a composed config (base + dataset + model + experiment). Mechanism
toggles (`two_scale`, `coupling`, `in_loop_loss`) are first-class config fields so
the factorial is generated, not hand-edited.

---

## 8. Numerical correctness invariants (must always hold)

1. **Assembly parity:** the differentiable assembler reproduces the prior
   `L2L+overlap` forward pass to FP tolerance (test before training anything).
2. **Coarse factor:** $(D/c)^2 < m$ for every resolution; coarse SVD uses
   randomized SVD.
3. **Loss == metric:** the interface and PDE-residual *loss* terms use the same
   stencils/interface sets as the reported *metrics*.
4. **Determinism:** a fixed seed reproduces split, bases, init, and final metrics
   within seed tolerance.
5. **Attention guardrail:** full attention only when token count $\le N_{\max}$;
   else GNN.

---

## 9. Explicit non-goals (scope guardrails — protect paper 2)

These are **out of scope** for this repo and must not be added without an explicit
decision to merge papers:

- **No transient / time-dependent PDEs.** Steady-state only. (Paper 2.)
- **No discretization-invariant / function-space bases.** Fixed-resolution PCA
  only — adding resolution invariance is a separate methods axis that would
  destabilize this paper's scope.
- **No learned-Schwarz / iterative interface solver.** The differentiable assembly
  is partition-of-unity assembly, *not* a convergent interface solve. (Future
  work / possible paper 3.)
- **No irregular geometry** beyond what the GNN trivially supports as a
  forward-looking note; geometry experiments belong to paper 2.
