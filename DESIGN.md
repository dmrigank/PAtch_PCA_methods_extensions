# DESIGN

## Two-Scale Localized PCA-Net for Artifact-Reduced PDE Operator Learning

**Status:** final method and manuscript design. The Poisson residual convention
was corrected in PDE-metrics version 2, affected diagnostics were migrated,
and all six PDE-active A5 cells were recalibrated and retrained.

This document is the scientific and implementation source of truth for the
second localized PCA-Net paper. It describes the two extensions retained for
the final manuscript:

1. **two-scale localized PCA-Net**, which replaces the local-only output basis
   with a coarse-global plus local-residual representation; and
2. **two-scale with interface-aware physical-field fine-tuning**, which
   backpropagates reconstruction and interface losses through the decoded,
   assembled field.

The first method is the primary contribution and recommended default. The
second is an optional accuracy-continuity refinement with substantially higher
offline training cost. Exploratory architecture branches are outside the final
scientific scope and must not appear in the manuscript's method taxonomy,
headline experiments, or claims.

User-facing commands belong in `README.md`. Detailed numerical findings and
paper assets live under `paper_results/<study>/`.

---

## 1. Paper Thesis

### 1.1 Starting point

The prior paper introduced PCA-Net variants for steady elliptic PDE operators:

- **Global PCA-Net**, with global input and output PCA;
- **L2G**, with local input PCA and global output PCA;
- **plain L2L**, with local input and output PCA on nonoverlapping patches;
- **L2L + overlap**, with overlapping patches and Hann-weighted assembly; and
- **L2L + RefinementNet**, with a learned image-space correction after L2L.

Plain L2L is computationally attractive, but independently decoded output
patches create block offsets, interface value and derivative mismatches, and
spurious high-wavenumber content. Overlap and RefinementNet reduce the visible
artifact downstream. They do not change the local-only output subspace that
created the inconsistency.

### 1.2 Central hypothesis

Patch artifacts are best reduced by changing the output representation before
adding a correction stage:

> A small global PCA on a restricted solution field can carry the predictable
> domain-wide structure, while local patch PCAs represent only the remaining
> fine-scale residual.

The prolongated coarse field is shared across every patch boundary. Local
decoders no longer need to reproduce the full solution independently, so their
boundary mismatch has much smaller amplitude. Differentiable field-space
fine-tuning can then target the residual value and normal-derivative errors
that remain.

### 1.3 Final contribution hierarchy

The paper must tell a representation-first story:

1. **Two-scale representation is the dominant mechanism.** It produces the
   large MRE, SSIM, and visible-artifact improvement.
2. **Block-balanced latent supervision is necessary.** A concatenated latent
   MSE underweights the small coarse block relative to the much larger residual
   block.
3. **Physical-field reconstruction and interface losses are complementary.**
   They improve an already strong two-scale model, but are not a substitute for
   the representation.
4. **The selected field-space variant is not a full PDE-residual model.** The
   final A3 objective uses reconstruction, value-trace, and flux-trace terms.
   Explicit spectral and PDE-residual terms were implemented and ablated, but
   failed the reconstruction guard at the tested weights.

### 1.4 Claim boundaries

The completed results support:

- substantially lower MRE and higher SSIM than prior PCA-Net baselines on
  Poisson;
- removal of the dominant visible block-offset artifact without overlapping
  output patches or a RefinementNet;
- approximately half the PCA-fit time of L2L + overlap across 64, 128, and 256;
- near-overlap end-to-end cost, with two-scale becoming faster at 256;
- optional field-space fine-tuning that sharply reduces the remaining value
  and normal-derivative trace errors; and
- qualified transfer to heterogeneous Darcy fields, especially for continuity
  and discrete residual quality.

The results do **not** support:

- claiming strict interface continuity by construction;
- claiming lower flux-trace or spectral error than Hann overlap;
- claiming better absolute accuracy than FNO;
- claiming a large Darcy MRE advantage;
- claiming the explicit spectral or PDE-residual losses are part of the final
  default; or
- aggregating Poisson PDE-residual values without requiring the reconciled
  version-2 convention and provenance; or
- treating the optional fine-tuning cost as free.

The phrase **artifact-reduced** is more accurate than **interface-continuous**.

---

## 2. Problem Definition and Notation

Let an operator map an input field \(x\) to a scalar solution field
\(u \in \mathbb{R}^{D \times D}\). The retained Poisson generator follows the
legacy convention \(x=f\) and

\[
\Delta u = f.
\]

For Darcy, \(x=a\), the forcing is fixed or supplied separately, and

\[
-\nabla \cdot (a \nabla u) = f.
\]

The grid is divided into \(P\) square patches of side \(p\) and stride \(s\).
The primary two-scale runs use a fixed \(4 \times 4\) nonoverlapping topology:

| Resolution \(D\) | Patch size \(p\) | Two-scale stride \(s\) | Overlap stride |
|---:|---:|---:|---:|
| 64 | 16 | 16 | 8 |
| 128 | 32 | 32 | 16 |
| 256 | 64 | 64 | 32 |

For each patch \(j\), input PCA produces a code
\(z^x_j \in \mathbb{R}^{k^x_j}\). Retained dimensions are ragged because each
patch independently retains a requested variance. The input vector is the
ordered concatenation

\[
z_x = [z^x_1,\ldots,z^x_P].
\]

The neural operator is the legacy PCA-Net MLP:

\[
g_\theta: z_x \mapsto \widehat z_u,
\]

implemented as three hidden layers of width 128 with ReLU activations followed
by a linear output layer. The two-scale method changes the output coordinates
and their loss, not the local input representation or the basic latent mapper.

---

## 3. Method I: Two-Scale Output Representation

### 3.1 Design principle

The output is decomposed as

\[
u \approx u_G + r_L,
\]

where \(u_G\) is decoded from one coarse global PCA basis and \(r_L\) is
decoded from local residual patch bases. The global branch carries coherent
low-frequency structure across the domain. The local branch restores detail
that the restricted coarse representation cannot express.

Only the **output side** is two-scale. Input PCA remains local exactly as in
plain L2L.

### 3.2 Restriction

Choose an integer coarse factor \(c\) that divides \(D\). The restriction
operator \(R_c\) averages each nonoverlapping \(c \times c\) block:

\[
u_c = R_c u,
\qquad
u_c \in \mathbb{R}^{D_c \times D_c},
\qquad
D_c = D/c.
\]

The Torch implementation uses average pooling with kernel and stride \(c\).
The NumPy fitting path uses the numerically matching operation.

The coarse factor must satisfy

\[
(D/c)^2 < m,
\]

where \(m\) is the number of training samples. This keeps the coarse feature
dimension below the sample count and preserves the intended low-dimensional
global solve. Construction fails with a clear error if:

- \(D\) is not divisible by \(c\);
- the grid is not square; or
- \((D/c)^2 \geq m\).

With \(m=8000\), the paper uses \(c=4\) at 64, 128, and 256. A future 512 run
would require \(c=8\).

### 3.3 Coarse global PCA

Flatten and standardize the restricted training fields:

\[
\bar u_c^{(i)}
  = S_c\!\left(\operatorname{vec}(R_c u^{(i)})\right).
\]

Fit one randomized PCA basis
\(\Phi_c \in \mathbb{R}^{D_c^2 \times k_c}\). The retained coarse score is

\[
z_c^{(i)}
  = \Phi_c^\top
    \left(\bar u_c^{(i)}-\mu_c\right).
\]

The final paper setting is:

- fixed \(k_c=10\);
- coarse variance threshold 1.0, so all ten requested modes are retained;
- randomized SVD oversampling 20;
- four power iterations; and
- the run seed passed to the PCA solver.

The coarse-rank ablation showed that an adaptive 99% rule capped at 20 also
selected approximately ten modes. Ranks 20 and 40 improved the PCA oracle but
substantially worsened learned reconstruction because the added scores were
harder to infer and enlarged the complete output target. The final rank is
therefore selected for **learned operator performance**, not for the lowest
standalone PCA reconstruction error.

### 3.4 Prolongation

Decode the coarse score on the restricted grid and restore its original units:

\[
u_c^\star(z_c)
  = S_c^{-1}\!\left(\mu_c + \Phi_c z_c\right).
\]

The prolongation operator \(P_c\) applies bilinear interpolation with
`align_corners=False`:

\[
u_G(z_c) = P_c u_c^\star(z_c)
          \in \mathbb{R}^{D \times D}.
\]

The ten coarse modes are not expanded into 256-grid PCA coordinates. Each
coarse basis vector is reshaped to the \(D_c \times D_c\) grid and bilinearly
interpolated to \(D \times D\). A ten-dimensional score vector therefore
remains ten-dimensional; prolongation changes the spatial realization of each
mode, not the number of retained coordinates.

### 3.5 Local residual PCA

For every output training field, form the residual against its own
coarse-PCA reconstruction:

\[
r^{(i)} = u^{(i)} - u_G(z_c^{(i)}).
\]

Extract the same nonoverlapping patches used by plain L2L. Each residual patch
has its own scaler and PCA model:

\[
z^{r,(i)}_j
  = (\Phi^r_j)^\top
    \left(S^r_j(r^{(i)}_j)-\mu^r_j\right),
\qquad j=1,\ldots,P.
\]

The final configuration retains 99.5% variance independently in every
residual patch. This is higher than the legacy 99% local-output setting because
small-amplitude seam corrections live in the residual tail. The ablation found
that moving from 99% to 99.5% reduced MRE by about 10% and flux jump by about
38% at both 128 and 256, with negligible measured end-to-end cost.

The complete output code is

\[
z_u = [z_c,z^r_1,\ldots,z^r_P]
    \in \mathbb{R}^{k_c + \sum_j k^r_j}.
\]

Residual patch lengths \(k^r_j\) are ragged and their ordered offsets are
stored in the fitted encoder.

### 3.6 Block-balanced latent objective

A plain MSE over the concatenated output code is not neutral:

\[
\mathcal{L}_{\mathrm{vec}}
  = \frac{1}{k_c+K_r}
    \left(
      \|\widehat z_c-z_c\|_2^2
      + \|\widehat z_r-z_r\|_2^2
    \right),
\qquad K_r=\sum_j k^r_j.
\]

Because \(K_r \gg k_c\), the residual block receives most of the aggregate
supervision by coordinate count. The selected objective gives the two
mechanistic blocks equal weight:

\[
\mathcal{L}_{\mathrm{block}}
  =
  \frac{1}{2k_c}\|\widehat z_c-z_c\|_2^2
  +
  \frac{1}{2K_r}\|\widehat z_r-z_r\|_2^2.
\]

The means also include the minibatch dimension in implementation. The default
weights are 0.5 coarse and 0.5 residual.

A score-normalized alternative divides each squared coordinate error by its
PCA score variance before applying the block means. It gave the lowest
single-seed Darcy MRE, but increased Darcy normal-derivative jumps by 43-54%
and produced mixed PDE-residual behavior. Block balancing is therefore the
robust cross-dataset default: it corrects block-size imbalance while preserving
the variance hierarchy inside each PCA block.

### 3.7 Decode and assembly

At inference, the network predicts both blocks:

\[
[\widehat z_c,\widehat z^r_1,\ldots,\widehat z^r_P]
  = g_\theta(z_x).
\]

The residual patches decode linearly:

\[
\widehat r_j
  = (S^r_j)^{-1}
    \left(\mu^r_j+\Phi^r_j\widehat z^r_j\right).
\]

The default two-scale topology is nonoverlapping, so residual patches are
assembled by uniform partition-of-unity averaging. The final field is

\[
\widehat u
  = u_G(\widehat z_c)
  + \mathcal{A}(\widehat r_1,\ldots,\widehat r_P).
\]

\(\mathcal{A}\) is the same differentiable scatter-add assembler used
throughout the repository:

\[
\begin{aligned}
a_\xi &= \sum_q W_q(\xi)\widehat r_q(\xi),\\
w_\xi &= \sum_q W_q(\xi),\\
\mathcal{A}(\widehat r)_\xi &= a_\xi/(w_\xi+\epsilon).
\end{aligned}
\]

Patch-to-pixel index maps and repeated weights are precomputed. Uniform
weights implement nonoverlapping/average assembly. The prior paper's safe
separable Hann weights implement the overlap reference. Assembly is
differentiable with respect to every decoded patch value.

### 3.8 Why the artifact is reduced

Plain L2L asks each output patch to predict its full local mean, amplitude, and
shape. Small latent errors create independent patch offsets. Two-scale changes
the division of labor:

1. restriction removes most fine-scale variation;
2. one global basis predicts the domain-wide low-frequency structure;
3. prolongation produces a dense field shared by all patches;
4. local PCA sees only a lower-amplitude residual; and
5. residual patch errors therefore perturb an already coherent global field
   instead of defining the full field independently.

This removes the dominant visible block bias, but does not mathematically force
the local residual values or derivatives to agree at interfaces. That remaining
error motivates Method II.

### 3.9 Computational argument

The coarse feature dimension is \(n_c=(D/c)^2\), with \(n_c<m\) by invariant,
and only ten modes are requested. The coarse randomized SVD is consequently a
small part of PCA fitting. The dominant two-scale PCA work remains:

- local input patch PCA;
- local residual-output patch PCA; and
- data transforms.

Overlap is more expensive because halving the stride changes the fixed
\(4 \times 4\) layout from 16 patches to 49 patches on both input and output
sides. Across the completed resolution sweep, two-scale PCA fitting was 49-53%
faster than overlap:

| Resolution | Overlap PCA fit | Two-scale PCA fit |
|---:|---:|---:|
| 64 | 2.68 s | 1.25 s |
| 128 | 10.25 s | 5.22 s |
| 256 | 39.72 s | 19.79 s |

Neural training, not PCA fitting, dominates end-to-end time. The paper must
therefore report both PCA-only and full pipeline costs.

---

## 4. Method II: Interface-Aware Physical-Field Fine-Tuning

### 4.1 Role

The second method starts from a fully trained Method-I checkpoint. It reuses:

- the fitted local input PCA models;
- the fitted coarse and residual output PCA models;
- the trained latent MLP; and
- the exact train/validation/test split.

PCA bases are frozen and are **not refit**. The MLP parameters are fine-tuned
through a frozen differentiable Torch mirror of the PCA decoder and assembler.
The inference architecture remains identical to Method I.

### 4.2 Differentiable decoder

All fitted PCA means, scales, components, patch offsets, assembly indices, and
weight sums are copied into `TorchPCADecoder` buffers. For a predicted latent
batch, it:

1. splits the coarse and ragged residual code blocks;
2. applies each inverse PCA and inverse scaler;
3. bilinearly prolongates the coarse field;
4. scatter-adds and normalizes residual patches; and
5. adds the coarse and residual fields.

The NumPy and Torch paths must reproduce one another to floating-point
tolerance. The decoder remains fixed; gradients flow through it to
\(g_\theta\).

### 4.3 Available field-space objective

The general implementation supports

\[
\mathcal{L}
  =
  \lambda_R\mathcal{L}_{R}
  + \lambda_V\mathcal{L}_{V}
  + \lambda_F\mathcal{L}_{F}
  + \lambda_E\mathcal{L}_{E}
  + \lambda_P\mathcal{L}_{P}.
\]

All terms operate on the assembled field.

#### Reconstruction

\[
\mathcal{L}_{R}
  = \frac{\|\widehat u-u\|_2}{\|u\|_2+\epsilon}
\]

for each minibatch under the repository's global relative-L2 convention. MSE
is also implemented, but relative L2 is used in the paper fine-tuning runs.

#### Interface value trace

Let \(\mathcal{B}\) be the row and column interfaces induced by patch size and
stride. At a vertical interface \(b\), define the signed adjacent-pixel trace

\[
T_V(u;b) = u_{:,b}-u_{:,b-1},
\]

with the analogous row expression. The truth-referenced loss is

\[
\mathcal{L}_{V}
  = \operatorname{mean}_{b\in\mathcal{B}}
    |T_V(\widehat u;b)-T_V(u;b)|.
\]

This does not force a zero discrete difference across an interface. It asks the
prediction to reproduce the physical variation present in the ground truth.

#### Interface normal-derivative trace

With \(h=1/(D-1)\), compare one-sided normal differences around each seam:

\[
T_F(u;b)
  =
  \frac{u_{:,b+1}-u_{:,b}}{h}
  -
  \frac{u_{:,b}-u_{:,b-1}}{h}.
\]

The selected flux-trace loss is

\[
\mathcal{L}_{F}
  = \operatorname{mean}_{b\in\mathcal{B}}
    |T_F(\widehat u;b)-T_F(u;b)|.
\]

This operator is a normal first-derivative jump proxy. In Darcy experiments it
is not coefficient-weighted \(a\nabla u\); coefficient-weighted physics enters
the Darcy residual metric below.

#### Spectral term

\(\mathcal{L}_E\) is the relative L2 error between minibatch-mean,
radially-binned FFT energy spectra. Optional weights increase with radial
wavenumber. The term is differentiable through `torch.fft`.

#### PDE residual

For the generated Poisson data, the correct interior five-point residual is

\[
\mathcal{R}_P(\widehat u)=\Delta_h\widehat u-f.
\]

PDE-metrics version 2 uses this generator-consistent convention in both the
NumPy and Torch kernels. The correction was motivated by an audit in which 32
exact Poisson-64 solutions produced RMS 0.5830 under the former
\(-\Delta_h\widehat u-f\) expression and \(3.34\times10^{-6}\) under the
correct expression. The paper-result archives have been migrated, and the six
PDE-active A5 cells were recalibrated and retrained in
`paper_results/physics_loss_ablation_v2`. Pre-version-2 residual values remain
historical artifacts and must not be mixed with the corrected results. The
selected A3 method has zero PDE-residual weight, so the kernel correction did
not change its training objective or the paper's MRE, SSIM, interface,
spectral, or timing conclusions.

For Darcy, face coefficients are arithmetic averages and

\[
\mathcal{R}_D(\widehat u)
  =-\nabla_h\cdot(a\nabla_h\widehat u)-f.
\]

RMS, MSE, and mean-absolute reductions are available. The paper reports RMS.

### 4.4 Selected A3 objective

The final second method uses

\[
\boxed{
\mathcal{L}_{A3}
  =
  \mathcal{L}_{R}
  + \lambda_V\mathcal{L}_{V}
  + \lambda_F\mathcal{L}_{F}
}
\]

with \(\lambda_E=\lambda_P=0\). This is best named
**two-scale + interface-aware physical-field fine-tuning**. "Physics in-loop"
may be used as an umbrella description of the infrastructure, but "full
physics loss" is inaccurate for the selected method.

The physics-loss ablation established:

- A1, reconstruction only, gives the lowest MRE;
- A2 adds value-trace matching and sharply reduces value and derivative error;
- A3 adds flux-trace matching and gives the best accuracy-continuity tradeoff;
- A4 reduces spectral error but violates the 5% MRE guard; and
- corrected A5 adds no useful improvement beyond A3: it increases MRE and
  residual RMS at both resolutions under the tested calibration.

### 4.5 Loss calibration

Raw field-space terms differ by orders of magnitude. Active auxiliary weights
are calibrated on at most 128 validation samples from the warm-start model:

\[
\lambda_t
  =
  \rho_t
  \frac{\mathcal{L}_{R,0}}{\mathcal{L}_{t,0}},
\]

clipped to configured numerical bounds. The selected target contribution
ratios are

\[
\rho_V=\rho_F=0.2.
\]

Weights are therefore resolved per run and recorded in its config and metrics.
They must not be hardcoded as universal raw constants in the manuscript.

### 4.6 Warmup and optimization

The selected fine-tuning protocol is:

- warm start from the matching block-balanced two-scale run;
- 50 epochs;
- Adam with learning rate \(10^{-4}\);
- zero weight decay;
- five reconstruction-only epochs;
- a linear ten-epoch ramp for interface terms; and
- full A3 weights for the remaining epochs.

The validation objective always uses the full resolved weights, and the best
validation checkpoint is restored. The current implementation runs all 50
epochs without early stopping.

### 4.7 Cost interpretation

Fine-tuning does not add parameters or an inference postprocessor. It does add
full-field decode, assembly, and trace-stencil backpropagation during training.
Reported cumulative cost must include:

1. the original two-scale PCA fit;
2. latent transforms;
3. latent MLP training;
4. the physical-field fine-tuning invocation; and
5. inference.

In the resolution sweep, A3 added approximately 124 s, 178 s, and 379 s at
64, 128, and 256. It is an optional continuity refinement, not the default
low-cost method.

---

## 5. Shared Numerical Infrastructure

### 5.1 Randomized SVD

Randomized SVD is the default for every final PCA method:

\[
\text{oversampling}=20,
\qquad
\text{power iterations}=4.
\]

The five-seed solver ablation found:

- unchanged retained dimensions;
- PCA-oracle errors equal to practical numerical precision;
- negligible mean learned-quality differences; and
- 2.7-2.8x lower PCA-fit time than full SVD on Poisson 128.

This is a PCA-stage result, not a 2.8x end-to-end speedup. Neural training
dominates total cost.

### 5.2 Assembly

`src/lpcanet/assembly/` is the single assembly implementation:

- average restriction;
- bilinear prolongation;
- prior-paper Hann and safe-Hann windows;
- cached patch-to-pixel index maps; and
- differentiable weighted scatter-add normalization.

The NumPy-facing assembly wrapper calls the same Torch path. Legacy overlap-add
is the numerical reference.

### 5.3 Loss and metric parity

`src/lpcanet/metrics/torch_ops.py` owns the differentiable kernels used by
both losses and metric wrappers:

- relative L2 and MSE;
- interface boundary enumeration;
- value jump and truth-referenced value trace error;
- normal-derivative jump and truth-referenced flux trace error;
- Poisson and Darcy residual stencils; and
- radially averaged spectral error.

No loss may implement a second private stencil that differs from its reported
metric. Loss/metric parity alone is not enough: the shared stencil must also
match the equation convention used to generate the dataset. Generator-linked
regression tests enforce that invariant for Poisson and Darcy. Result records
store the PDE-metrics version and Poisson convention to prevent mixed-version
aggregation.

The post-fix archive audit reports 289 current Poisson records. Its only six
`retrain` findings are the intentionally retained, pre-fix A5 checkpoints in
`paper_results/physics_loss_ablation/`; their recalibrated and retrained
replacements live in `paper_results/physics_loss_ablation_v2/`. The former
directory is historical evidence, not a manuscript data source. The audit is
idempotent and can be repeated without modifying files:

```bash
python scripts/recompute_poisson_residuals.py --results-root paper_results
```

### 5.4 Determinism

A run seed controls:

- split generation or fixed embedded splits;
- training-subset selection;
- randomized SVD;
- NumPy;
- Torch and CUDA initialization; and
- minibatch order.

All final records contain the seed and config hash. Prediction retention is
normally limited to seed 0 for qualitative figures, while aggregate metrics use
all planned seeds.

### 5.5 Stage timing

Every run records:

- `pca_fit`;
- `coarse_svd` where applicable;
- `latent_transform`;
- `nn_train`;
- `inference`; and
- `end_to_end`.

PCA fit is additionally decomposed into input, global/coarse, local output, and
residual-output stages where available. Warm-start studies record both
invocation-only and cumulative accounting. Cumulative values are the manuscript
default.

---

## 6. Final Experimental Protocol

### 6.1 Data

The paper uses the prior work's steady elliptic datasets, generated with GRF
parameters \(\alpha=2\) and \(\tau=3\):

- **Poisson:** GRF forcing \(f\), homogeneous Dirichlet solution \(u\), and
  legacy convention \(\Delta u=f\);
- **Darcy:** thresholded GRF coefficient \(a\in\{4,12\}\), unit forcing, and
  solution of \(-\nabla\cdot(a\nabla u)=1\).

Processed arrays live under `data/processed/` and are not versioned. Final
studies use 8,000 training, 1,000 validation, and 1,000 test samples unless the
sample-efficiency ablation intentionally changes \(m\). Split indices are
paired across methods within each dataset, resolution, and seed.

The main Poisson resolutions are 64, 128, and 256. The main Darcy
generalization result is at 256. Results from independently generated
resolution datasets test distributional resolution robustness, not pointwise
grid convergence on identical realizations.

### 6.2 Final two-scale defaults

| Setting | Final value |
|---|---|
| Input PCA variance | 99% |
| Coarse factor | 4 at 64/128/256 |
| Coarse modes | 10 |
| Coarse retained variance | 100% of requested ten modes |
| Residual PCA variance | 99.5% |
| PCA solver | randomized |
| Oversampling / power iterations | 20 / 4 |
| Patch topology | \(4 \times 4\), nonoverlapping |
| Latent model | 128-wide, four-layer ReLU MLP |
| Latent objective | 0.5 coarse + 0.5 residual block MSE |
| Latent epochs / learning rate | 500 / \(10^{-3}\) |
| Batch size | 32 |

### 6.3 Final interface-aware defaults

| Setting | Final value |
|---|---|
| Warm start | matching trained two-scale run |
| Active terms | relative L2 + value trace + flux trace |
| Interface target | ground-truth trace |
| Target contribution ratios | 0.2 value, 0.2 flux |
| Calibration samples | up to 128 validation samples |
| Fine-tuning epochs / learning rate | 50 / \(10^{-4}\) |
| Reconstruction-only warmup | 5 epochs |
| Interface ramp | 10 epochs |
| Spectral / PDE weight | 0 / 0 |
| Added inference stage | none |

### 6.4 Baselines

The final comparisons use:

- Global PCA-Net;
- plain L2L;
- L2L + Hann overlap;
- L2L + RefinementNet where retained predictions/results exist;
- normalized FNO;
- two-scale; and
- two-scale + interface-aware fine-tuning.

L2G remains an inherited prior-paper method but is not needed in every final
table. FNO is an accuracy ceiling in a larger parameter and runtime regime, not
a method the paper claims to beat.

### 6.5 Metrics

Report:

- MRE, SSIM, MSE, and MAE;
- PCA-oracle MRE and learned/oracle ratio for PCA methods;
- relative radial-spectrum error;
- raw interface value and normal-derivative jumps;
- truth-referenced value and flux trace errors;
- generator-consistent Poisson or Darcy residual RMS; and
- stage-wise and end-to-end time.

Native seam sets differ when overlap uses stride \(p/2\). Direct continuity
claims against overlap must include the common-stride audit. MRE and SSIM do
not have this comparability issue.

### 6.6 Study-to-config map

| Manuscript purpose | Experiment config | Aggregate seeds |
|---|---|---:|
| Poisson 128 headline | `configs/experiments/headline_poisson_128.yaml` | 5 |
| Representation versus objective | `configs/experiments/mechanism_ablation.yaml` | 5 |
| Poisson resolution scaling | `configs/experiments/resolution_sweep.yaml` | 3 |
| Darcy 256 transfer | `configs/experiments/darcy_generalization.yaml` | 5 |
| Coarse representation choices | `configs/experiments/coarse_representation_ablation.yaml` | 3 |
| Latent objective selection | `configs/experiments/two_scale_latent_loss_ablation.yaml` | 1 |
| Field-space loss ladder | `configs/experiments/physics_loss_ablation.yaml` | 3 |
| Sample efficiency | `configs/experiments/sample_efficiency.yaml` | 3 |
| Full versus randomized SVD | `configs/experiments/svd_solver_ablation.yaml` | 5 |

Each harness run writes one stable record containing:

- run ID, run directory, and config hash;
- dataset, resolution, method, and seed;
- final mechanism state and latent-loss mode;
- the complete evaluation metric suite;
- invocation and cumulative timing fields;
- warm-start provenance where applicable; and
- PCA component counts and fit-stage diagnostics where available.

`results.jsonl` is the append-friendly source, while `results.csv` is the
study roll-up consumed by manuscript renderers. Fine-tuned records must retain
their warm-start link so cumulative cost can be reconstructed rather than
guessed from a zero invocation PCA time.

---

## 7. Completed Evidence and Manuscript Interpretation

### 7.1 Poisson 128 headline

Five paired seeds gave:

| Method | MRE (%) | SSIM | Total time (s) |
|---|---:|---:|---:|
| Global PCA-Net | 7.543 | 0.9236 | 192.5 |
| Plain L2L | 4.829 | 0.9427 | 181.2 |
| L2L + overlap | 2.308 | 0.9850 | 208.8 |
| L2L + RefinementNet | 3.210 | 0.9747 | 275.9 |
| FNO | 0.242 | 0.9998 | 725.4 |
| Two-scale | 1.149 | 0.9966 | 218.6 |
| Two-scale + interface | 1.083 | 0.9967 | 416.6 |

Two-scale reduced MRE by 76% relative to plain L2L and 50% relative to overlap,
with only 4.7% more total time than overlap. A3 improved two-scale MRE by 5.8%
and value/flux trace errors by about 65%, but nearly doubled cumulative time.

The two-scale PCA oracle was 0.501% MRE, giving a learned/oracle ratio of 2.29.
The representation has substantial unused capacity; latent prediction is the
remaining bottleneck.

### 7.2 Mechanism ablation

Five paired Poisson-128 seeds showed:

- field-space A3 on plain L2L improved MRE by only 5.3% and flux trace by 18.7%
  while increasing total time by 86.6%;
- two-scale alone improved MRE by 76.2% and flux trace by 92.1% relative to
  plain L2L; and
- A3 on two-scale supplied a further 5.3% MRE and 64.0% flux-trace improvement.

The loss is effective after the output subspace can represent a coherent
field. It cannot repair the local-only representation by itself.

### 7.3 Resolution sweep

Across three paired seeds:

| Method | MRE 64 | MRE 128 | MRE 256 |
|---|---:|---:|---:|
| Plain L2L | 4.739% | 4.794% | 4.835% |
| L2L + overlap | 2.365% | 2.292% | 2.289% |
| Two-scale | 1.060% | 1.155% | 1.221% |
| Two-scale + interface | 1.000% | 1.090% | 1.135% |
| FNO | 0.217% | 0.245% | 0.262% |

Two-scale remained near 1.1-1.2% and reduced overlap MRE by 45-55%. At 256 it
was 5.3% faster end to end than overlap. The learned/oracle ratio grew with
resolution, again identifying latent regression rather than PCA capacity as
the scaling bottleneck.

Overlap retained the best strict flux continuity and radial-spectrum metric.
At 256, A3 reduced two-scale flux-trace error by about 70%, but remained above
overlap.

### 7.4 Coarse-representation ablation

The selected design is supported by three-seed Poisson 128/256 sweeps:

- rank 10 gave the best learned MRE;
- larger ranks improved the PCA oracle but harmed latent trainability;
- \(c=4\) was the best legal setting shared by 128 and 256;
- 99.5% residual retention improved MRE and seam metrics at negligible time
  cost; and
- the adaptive coarse rule independently selected about ten modes.

The ablation establishes a moderate predictable global rank plus a
high-retention residual branch as the best division of labor.

### 7.5 Latent-loss ablation

The seed-0 design-selection study covered Poisson and Darcy at 128 and 256.
Relative to vector MSE:

- block balancing reduced MRE by 42-46% in all four cells;
- score normalization reduced Darcy MRE more strongly;
- score normalization worsened Darcy normal-derivative jumps and gave mixed
  residual behavior.

This study justifies block balancing as the cross-dataset default. Because it
used one seed, exact block-versus-score rankings are not statistical claims.

### 7.6 Physics-loss ablation

Three-seed Poisson 128/256 results selected A3:

- A1 reconstruction-only lowered MRE by about 7.5%;
- A3 lowered flux jump by 61% at 128 and 71% at 256 relative to latent-only A0,
  while retaining a 5-6% MRE improvement;
- A3 lowered generator-consistent Poisson residual RMS by 40% at 128 and 59%
  at 256 without an explicit PDE-residual term;
- A4 spectral loss improved its target but worsened MRE by 13-16% relative to
  A3; and
- corrected A5 worsened MRE by 10% at 128 and 13% at 256 relative to A3, while
  worsening residual RMS by 16% and 1%, respectively.

The corrected residual diagnostic decreases under A3 because matching
interface values and gradients removes large second-difference contributions
near patch boundaries. The A5 result now supports a narrower conclusion: the
tested calibrated domain-wide Laplacian penalty is less effective than direct
interface supervision, not that PDE-residual objectives are universally
unhelpful.

### 7.7 Sample efficiency

At Poisson 128:

- overlap was best at 1,250 samples;
- two-scale became competitive near 2,000 samples;
- two-scale at 4,000 samples beat overlap trained on 8,000; and
- A3 at 2,000 samples beat overlap trained on 8,000.

Local baselines saturated near their PCA-oracle floors. Two-scale continued to
improve because its lower representation floor leaves more learnable headroom.
The sample-efficiency claim applies after a small-data threshold, not
uniformly at every \(m\).

### 7.8 Darcy 256 generalization

Five paired seeds gave:

| Method | MRE (%) | Flux jump | Darcy RMS | Total time (s) |
|---|---:|---:|---:|---:|
| Plain L2L | 3.109 | \(3.20\times10^{-2}\) | 21.08 | 338.8 |
| L2L + overlap | 2.693 | \(2.81\times10^{-4}\) | 3.39 | 717.2 |
| Two-scale | 2.952 | \(3.58\times10^{-3}\) | 4.13 | 388.7 |
| Two-scale + interface | 2.926 | \(6.08\times10^{-4}\) | 3.44 | 781.7 |
| FNO | 0.279 | \(4.27\times10^{-4}\) | 4.00 | 3185.7 |

Two-scale reduced plain-L2L flux jump by 89% and Darcy residual by 80%, but MRE
by only 5%. A3 further reduced flux jump by 83% and residual RMS by 17%, with
almost no MRE change and roughly double cumulative cost.

The two-scale Darcy PCA oracle was 0.278% MRE versus 2.952% learned MRE. The
representation transfers; predicting its scores from a discontinuous
coefficient field is the limiting step.

### 7.9 Solver ablation

Fully randomized PCA is retained because it preserved the practical
representation while reducing PCA-fit time by approximately 64%. Randomizing
only the coarse solve was ineffective because the coarse SVD is already a
small substage. Full randomized SVD should be used for all final quality and
cost tables.

---

## 8. Implementation Map

```text
src/lpcanet/
  data/        deterministic loaders and Poisson/Darcy generation
  pca/         legacy encoders, two-scale encoder, randomized PCA, Torch decoder
  assembly/    patch extraction, restriction, prolongation, windows, mosaic
  models/      latent MLP, FNO, legacy baseline and RefinementNet models
  losses/      block-balanced latent loss and field-space composite terms
  metrics/     reconstruction, interface, spectral, and PDE diagnostics
  train/       experiment execution, warm starts, calibration, seeding, timing
  utils/       config and stage-timing support
configs/
  base.yaml
  datasets/
  models/
  experiments/
scripts/
  run_experiment.py
  make_*_outputs.py
  make_two_scale_schematic.py
tests/
paper_results/  completed paper runs, summaries, tables, and figures
```

Primary implementation ownership:

| Concern | Source |
|---|---|
| Two-scale fit/transform/decode | `src/lpcanet/pca/pca.py` |
| Differentiable PCA decode | `src/lpcanet/pca/torch_decoder.py` |
| Restrict/prolongate | `src/lpcanet/assembly/operators.py` |
| Differentiable mosaic | `src/lpcanet/assembly/mosaic.py` |
| Latent block objective | `src/lpcanet/losses/latent.py` |
| Field-space objective | `src/lpcanet/losses/terms.py` |
| Shared metric/loss stencils | `src/lpcanet/metrics/torch_ops.py` |
| PDE convention and archive migration | `src/lpcanet/metrics/residuals.py`, `src/lpcanet/metrics/migration.py` |
| Warm-start and run orchestration | `src/lpcanet/train/experiment.py` |
| Loss schedule and calibration | `src/lpcanet/train/loop.py` |
| Study expansion and cost accounting | `src/lpcanet/train/factorial.py` |

Final method states are controlled by:

```yaml
mechanisms:
  two_scale: true
  in_loop_loss: false  # Method I
```

and:

```yaml
mechanisms:
  two_scale: true
  in_loop_loss: true   # Method II, warm-started A3
```

The composed run config, not prose or script defaults, is the executable source
for every result.

---

## 9. Numerical and Reporting Invariants

These invariants must remain true for future manuscript revisions.

1. **Coarse legality:** \((D/c)^2<m\), square grids, and exact divisibility.
2. **Local input unchanged:** Method I changes only the output representation.
3. **Residual target consistency:** training residuals use each sample's
   decoded coarse-PCA projection.
4. **Ragged offsets preserved:** no method assumes equal per-patch PCA ranks.
5. **Assembly parity:** weighted differentiable assembly matches the prior
   overlap-add numerical reference.
6. **Decoder parity:** NumPy and Torch output decoders agree to floating-point
   tolerance.
7. **Loss equals metric:** shared interface, spectral, and PDE kernels define
   both optimization terms and evaluation metrics.
8. **Equation convention:** the shared PDE kernel must match the generator;
   Poisson uses \(\Delta u=f\), while Darcy uses
   \(-\nabla\cdot(a\nabla u)=f\).
9. **PDE provenance:** Poisson manuscript aggregation requires
   `pde_metrics_version == 2` and
   `poisson_residual_convention == delta_u_equals_f`.
10. **Block balance:** final two-scale latent training uses equal coarse and
   residual block means.
11. **Warm-start accounting:** Method II reuses PCA and model artifacts, records
   zero invocation PCA fit, and reports cumulative paper cost including Method I.
12. **No external postprocessing:** neither final method uses RefinementNet or a
    Hann post-step after prediction.
13. **Common seam audit:** direct overlap continuity comparisons use a shared
    seam set in addition to native metrics.
14. **Randomized-SVD disclosure:** solver, oversampling, iterations, and seed are
    recorded.
15. **Paired seeds:** methods in an aggregate comparison use the same split seed.
16. **Qualitative selection:** samples follow a documented median-error rule,
    never manual cherry-picking.
17. **Cost separation:** PCA-stage speedups and end-to-end speedups are reported
    separately.

The core tests include:

- `tests/test_assembly_parity.py`;
- `tests/test_coarse_factor_invariant.py`;
- `tests/test_loss_metric_consistency.py`;
- `tests/test_pde_residual_conventions.py`;
- `tests/test_poisson_residual_migration.py`;
- `tests/test_in_loop_training.py`;
- `tests/test_latent_loss_ablation.py`;
- `tests/test_seed_determinism.py`; and
- experiment-expansion/output smoke tests for each paper study.

---

## 10. Manuscript Structure and Asset Map

### 10.1 Recommended paper structure

1. **Introduction:** local PCA efficiency, local-output artifact, and the
   representation-first hypothesis.
2. **Prior localized PCA-Net:** Global, L2G, L2L, overlap, and RefinementNet.
3. **Two-scale method:** restriction, coarse PCA, prolongation, residual PCA,
   block-balanced training, decode, and cost argument.
4. **Interface-aware fine-tuning:** differentiable decoder, trace losses,
   calibration, and warmup.
5. **Experimental protocol:** Poisson/Darcy, baselines, metrics, paired seeds,
   and stage timing.
6. **Main results:** Poisson 128, resolution scaling, and quality-cost tradeoff.
7. **Ablations:** coarse representation, latent objective, field-space losses,
   sample efficiency, and SVD solver.
8. **Darcy generalization:** continuity gain, limited MRE gain, and trainability
   gap.
9. **Limitations and outlook:** strict flux continuity, score prediction,
   heterogeneous inputs, and fine-tuning cost.

### 10.2 Main manuscript assets

- Method schematic:
  `paper_results/manuscript/figures/two_scale_method_schematic.*`
- Headline Poisson figures/tables:
  `paper_results/headline_poisson_128/`
- Resolution accuracy, artifact, and quality-cost plots:
  `paper_results/resolution_sweep/figures/main/`
- Physics-loss ladder and Pareto plot:
  `paper_results/physics_loss_ablation_v2/figures/main/`
- Darcy generalization:
  `paper_results/darcy_generalization/figures/main/`
- Coarse representation ablation:
  `paper_results/coarse_representation_ablation/`
- Latent objective ablation:
  `paper_results/two_scale_latent_loss_ablation/`

Each study's `summary.md` is the detailed numerical source for prose. CSV
tables are the numeric source of truth; generated LaTeX is a formatting
artifact.

### 10.3 Suggested paper title

**Two-Scale Localized PCA-Net: Coarse-Global and Local-Residual
Representations for Artifact-Reduced PDE Operator Learning**

---

## 11. Limitations and Future Work

The final paper should state these limitations directly.

1. **No exact continuity guarantee.** The coarse field is globally coherent,
   but independently decoded residual patches can retain value and derivative
   mismatch.
2. **Overlap remains the strict flux/spectral reference.** Two-scale wins global
   PCA-Net reconstruction accuracy, not every artifact diagnostic.
3. **Latent regression is the main bottleneck.** Two-scale PCA-oracle accuracy is
   much better than learned accuracy, especially for Darcy and fine grids.
4. **Fine-tuning is expensive offline.** It does not affect inference
   architecture, but can roughly double cumulative training time.
5. **Darcy transfer is qualified.** Continuity and residual improve strongly;
   MRE improves only modestly.
6. **Fixed-grid bases.** PCA models are resolution specific and do not define a
   discretization-invariant operator.
7. **Steady, square-grid scope.** The implementation does not establish results
   for transient PDEs, irregular meshes, or complex geometry.
8. **Normal derivative versus Darcy flux.** The interface trace term uses
   \(\partial_n u\), while coefficient-weighted flux is assessed through the
   Darcy residual. A future interface loss could target \(a\partial_n u\)
   explicitly.
9. **Poisson residual provenance.** PDE-metrics version 2 matches the stored
   generator. Pre-version-2 residual values and A5 checkpoints used the
   opposite sign and remain historical only; manuscript aggregation must
   require version-2 provenance.

The strongest next technical direction is not a larger coarse rank. It is a
better predictor for the existing coarse and residual scores, with explicit
cross-block conditioning and no loss of the compact decoder or PCA-stage cost
advantage.
