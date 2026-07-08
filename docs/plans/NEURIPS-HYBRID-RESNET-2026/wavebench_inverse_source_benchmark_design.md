# WaveBench Inverse Source Benchmark Design

## Purpose

This design evaluates WaveBench inverse source reconstruction as an additional
optional evidence lane for the current SRU-Net manuscript package. CDI
`lines128` and PDEBench CNS remain required pillars. WaveBench would add a
controlled 2D wave inverse problem with a known forward model, ready datasets,
and FNO/U-Net baselines if the active roadmap is amended to include the extra
inverse-wave lane.

The benchmark should support two training regimes:

- supervised reconstruction of the hidden initial pressure/source field;
- physics-informed reconstruction using a differentiable wave forward model.

This benchmark is not a geology benchmark and should not be described as one.
It is a controlled wave inverse reconstruction task.

## Roadmap Authority And Status

The active manuscript roadmap requires a two-pillar package:

1. CDI `lines128` reconstruction.
2. PDEBench `2d_cfd_cns` forward prediction.

WaveBench is therefore a candidate addition alongside CDI and CNS.
The only work authorized before a roadmap amendment is a narrow preflight:
inspect data availability, native baseline availability, tensor contracts, and
physics-loop feasibility. Training WaveBench rows or adding them to manuscript
tables requires a checked-in roadmap or evidence-package amendment after
preflight.

This design separates:

- minimum viability gate: enough inspection to decide whether a future
  WaveBench addition is worth planning;
- optional supervised benchmark: shared-encoder architecture comparison;
- optional physics-informed benchmark: only after forward-model reproduction
  passes a quantitative validity gate;
- optional research extensions: ablations, OOD transfer, and extra loss sweeps.

## External Benchmark Facts

WaveBench provides benchmark datasets for linear wave-propagation PDEs. The
README identifies two dataset groups: time-harmonic wave problems and
time-varying wave problems. The time-varying group includes reverse time
continuation and inverse source problems.

For inverse source reconstruction, WaveBench describes the task as predicting
the initial pressure `q(., 0)` from pressure measurements collected at boundary
locations over a time interval `[0, T]`. The README also says the task is
inspired by seismic imaging, where pressure-field measurements are feasible at
the earth's surface.

WaveBench provides PyTorch tooling, Google Colab notebooks, datasets on Zenodo,
and baseline U-Net/FNO checkpoints.

Sources:

- WaveBench README: <https://github.com/wavebench/wavebench>
- WaveBench TMLR paper page: <https://openreview.net/forum?id=6wpInwnzs8>

## Task Definition

Let `Omega` be a 2D spatial domain and let

```math
q_0(x) = q(x,0), \quad x \in \Omega
```

denote the unknown initial pressure/source field. The observed data are
boundary pressure measurements over time:

```math
y(t,b) = u(x_b,t), \quad x_b \in \partial\Omega_{\mathrm{obs}},
\quad t \in [0,T].
```

The inverse problem is:

```math
y \mapsto \hat q_0.
```

The paper-facing interpretation is:

- input: indirect boundary wave observations;
- output: 2D initial pressure/source image;
- reconstruction target: spatial field, not future dynamics and not a 1D
  profile.

## Forward Model

The physics-informed variant uses the scalar linear wave equation. For a fixed
WaveBench inverse-source dataset variant with fixed wavespeed `c(x)`, define:

```math
\frac{1}{c(x)^2}\frac{\partial^2 u(x,t)}{\partial t^2}
- \Delta u(x,t) = 0,
\quad x \in \Omega,\; t \in [0,T].
```

Initial conditions:

```math
u(x,0) = q_0(x),
```

```math
\partial_t u(x,0) = 0,
```

unless the selected WaveBench dataset metadata defines a different initial
velocity/source convention. The implementation must follow the dataset's
actual convention rather than assuming this default.

Boundary measurements are:

```math
\mathcal{M}u = \{u(x_b,t): x_b \in \partial\Omega_{\mathrm{obs}}, t \in [0,T]\}.
```

The full forward map is:

```math
F_c(q_0) = \mathcal{M}u(q_0;c).
```

The physics loss compares observed boundary traces to traces predicted from the
model reconstruction:

```math
\hat y = F_c(\hat q_0).
```

Before training any physics-informed row, the implementation must validate that
the local differentiable forward model reproduces WaveBench observations from
ground-truth `q_0`:

```math
F_c(q_0) \approx y.
```

The reproduction report must record:

- selected WaveBench inverse-source variant and dataset file;
- grid, time step, boundary condition, fixed-wavespeed, receiver, and
  initial-condition metadata;
- normalization applied to `y` and to `F_c(q_0)`;
- sample count, with at least 16 held-out examples for a go/no-go report unless
  preflight data access only permits a smaller smoke;
- waveform MAE, RMSE, relative L2, and normalized residual
  `||F_c(q_0)-y||_2 / ||y||_2`;
- accepted thresholds before training.

Default quantitative validity gate:

```math
\mathrm{median}\left(
\frac{\|F_c(q_0)-y\|_2}{\|y\|_2}
\right) \le 0.02
```

and no more than 5 percent of checked examples above 0.05. If the dataset
metadata or scale makes these thresholds inappropriate, the preflight must
propose alternate thresholds before any physics-informed training begins.

If this check fails, no row may be labeled a WaveBench physics-informed
benchmark row. Later work may still explore approximate-model regularization,
but those rows must be labeled separately and cannot support a
known-forward-model training claim.

## Provenance And Preflight Contract

No training phase is authorized until a preflight summary records:

- exact WaveBench repository revision;
- exact dataset DOI/source URL, file names, local staging path, access/license
  notes, and checksums or size/mtime manifest;
- selected inverse-source variant and split;
- train/validation/test sample counts;
- observed-data, target, and wavespeed tensor shapes;
- normalization and dtype conventions;
- native FNO/U-Net checkpoint availability and checkpoint identifiers;
- whether WaveBench generation or solver code is available for physics-loop
  reproduction;
- final status:
  `ready_for_supervised_plan`, `ready_for_supervised_and_physics_plan`,
  `needs_dataset_or_checkpoint_decision`, or
  `not_suitable_for_current_manuscript`.

## Model Interface

The custom benchmark rows should use a shared boundary-measurement encoder:

```math
E_\phi(y) = h,
```

where:

```math
h \in \mathbb{R}^{128 \times 128 \times C}.
```

The channel count `C` is a hyperparameter:

```math
C \in \{16,32,64,128\}.
```

The first serious runs should use `C=32` and `C=64`.

The reconstruction model then maps:

```math
\hat q_0 = R_\theta(h).
```

This shared input contract standardizes the architecture comparison: every
custom model receives the same 2D latent representation, while native
WaveBench FNO/U-Net rows remain as reference baselines.

Encoder ownership contract:

- In the first supervised benchmark, each shared-encoder row jointly trains its
  own encoder plus reconstruction body using the same encoder architecture,
  channel count, optimizer policy, and data split.
- The reported comparison is therefore `encoder + body`, not body-only.
- Encoder parameters, body parameters, total parameters, runtime, and memory
  must be reported separately when possible.
- The encoder must not be pre-trained on validation/test examples or selected
  after looking at model metrics.
- A frozen shared encoder is an optional later ablation, not the default,
  because it would introduce a separate encoder-training problem before the
  architecture rows are even runnable.

## Boundary-Measurement Encoder

The encoder maps boundary-time measurements to an image-like latent field.

Preferred first version:

1. Treat `y(t,b)` as a 2D boundary-time measurement image.
2. Apply anisotropic convolutional blocks over time and boundary index.
3. Downsample time more aggressively than boundary position.
4. Fuse all observed boundary traces into a compact latent feature map.
5. Project or upsample to `128 x 128 x C`.

The design intentionally keeps this encoder simple at first. Attention-based
boundary fusion and more elaborate measurement geometry encoders are later
ablations, not required for the first benchmark.

## Benchmark Rows

Use two baseline categories.

Native WaveBench reference rows:

| Row | Input | Model | Purpose |
| --- | --- | --- | --- |
| WaveBench U-Net | native WaveBench loader/input | official/reference U-Net | external benchmark reference |
| WaveBench FNO | native WaveBench loader/input | official/reference FNO | external benchmark reference |

Shared-encoder rows use paper-facing labels plus exact repo architecture IDs.
SRU-Net is the manuscript name for the current `hybrid_resnet` family only when
that implemented body is actually used. It is not interchangeable with every
hybrid or spectral row.

| Paper label | Candidate repo architecture ID | Input | Output |
| --- | --- | --- | --- |
| U-Net | verify existing U-Net/CNN image model ID during preflight | `h` | `q_0` |
| SRU-Net | `hybrid_resnet`, if reusable outside CDI without semantic drift | `h` | `q_0` |
| Hybrid-spectral | `spectral_resnet_bottleneck_net` or current spectral-local ID verified during preflight | `h` | `q_0` |
| FNO | `fno_vanilla` or `fno`, selected before launch | `h` | `q_0` |
| FFNO | `ffno` or `ffno_bottleneck_net`, selected before launch | `h` | `q_0` |

Native rows answer "how does this compare to WaveBench's own baselines?"
Shared-encoder rows answer "which reconstruction body works best under the
same measurement encoder?"

## Supervised Training

The supervised objective compares the predicted and true initial pressure:

```math
\mathcal{L}_{\mathrm{sup}}
=
\lambda_1\|\hat q_0-q_0\|_1
+ \lambda_2\|\hat q_0-q_0\|_2^2
+ \lambda_g\|\nabla\hat q_0-\nabla q_0\|_1
+ \lambda_s(1-\mathrm{SSIM}(\hat q_0,q_0)).
```

The first pass should use:

```math
\mathcal{L}_{\mathrm{sup}}=\|\hat q_0-q_0\|_1.
```

Add gradient and SSIM terms only after the L1 baseline is stable. The gradient
term is useful if reconstructions blur compact source structure or edges.

## Physics-Informed Training

The physics-informed objective pushes the reconstructed source through the
wave forward model:

```math
\hat y = F_c(\hat q_0).
```

Boundary waveform loss:

```math
\mathcal{L}_{\mathrm{phys}}
=
\|W(\hat y-y)\|_1,
```

or:

```math
\mathcal{L}_{\mathrm{phys}}
=
\sum_{t,b}\rho(\hat y(t,b)-y(t,b)).
```

Here `W` may include normalization, time-windowing, or frequency weighting, and
`rho` may be L1, L2, or Huber.

Train three regimes:

| Regime | Loss |
| --- | --- |
| Supervised | `L_sup` |
| Physics-only | `L_phys` |
| Hybrid supervised + physics | `lambda_q L_sup + lambda_y L_phys` |

The hybrid regime should be implemented before physics-only if time is limited,
because it is more stable and more directly analogous to adding forward-model
consistency to a learned reconstructor.

## Differentiable Solver Requirements

The physics-informed path requires a differentiable local implementation of the
WaveBench inverse-source forward model or a documented approximation.

Minimum requirements:

- fixed 2D wavespeed `c(x)` for the selected inverse-source variant;
- same spatial grid and time grid as the dataset;
- same boundary/receiver locations used in the dataset;
- same initial-condition convention;
- same boundary conditions, or a validated approximation;
- batched `q_0 -> y` forward pass;
- gradients with respect to `q_0`;
- reproduction check `F_c(q_0) ~= y` on held-out ground-truth examples.

The design should not assume the physics loop is valid simply because the PDE
is known. It is valid only after dataset-generation alignment is checked.

## Metrics

Primary source-reconstruction metrics:

```math
\mathrm{MAE} = \frac{1}{N}\sum_i |\hat q_{0,i}-q_{0,i}|,
```

```math
\mathrm{RMSE} =
\sqrt{\frac{1}{N}\sum_i(\hat q_{0,i}-q_{0,i})^2},
```

```math
\mathrm{RelL2} =
\frac{\|\hat q_0-q_0\|_2}{\|q_0\|_2},
```

```math
\mathrm{SSIM}(\hat q_0,q_0).
```

Local-structure metrics:

```math
\|\nabla\hat q_0-\nabla q_0\|_1
```

and radial high-frequency spectral error on the reconstructed source field.

Physics-consistency metrics:

```math
\|F_c(\hat q_0)-y\|_1,
\quad
\|F_c(\hat q_0)-y\|_2.
```

Report physics-consistency metrics separately from source-image metrics. A row
can reduce waveform residual while hurting source reconstruction, and that
tradeoff should remain visible.

## Visual Evidence

Main reconstruction figure:

| Column | Content |
| --- | --- |
| Ground truth | `q_0` |
| U-Net | reconstruction |
| FNO | reconstruction |
| FFNO | reconstruction |
| SRU-Net | reconstruction |
| Hybrid-spectral | reconstruction |
| Error maps | absolute error for key rows |

Physics-consistency figure:

| Row | Content |
| --- | --- |
| Observed boundary data | `y` |
| Predicted boundary data | `F_c(hat q_0)` |
| Residual | `abs(F_c(hat q_0)-y)` |

Use shared color limits within each row. Do not use separate color scales that
hide failure modes.

## Experiment Sequence

Phase 0: minimum viability preflight.

| Gate | Required decision |
| --- | --- |
| Data | selected variant, source file, checksum/manifest, splits, tensor shapes |
| Native baselines | WaveBench FNO/U-Net checkpoint or reproduction path |
| Repo adapters | candidate architecture IDs and missing adapter/output-head work |
| Physics | `exact_physics_loop_ready`, `approximate_physics_regularization_possible`, `physics_loop_deferred`, or `physics_loop_not_recommended` |
| Roadmap | no manuscript-roadmap change unless preflight recommends one |

Phase 1: optional native WaveBench baseline reproduction.

| Dataset | Rows |
| --- | --- |
| selected inverse-source variant | native WaveBench FNO, native WaveBench U-Net |

Phase 2: optional shared-encoder supervised comparison.

| Dataset | Models | Latent channels | Seeds |
| --- | --- | --- | --- |
| selected inverse-source variant | U-Net, SRU-Net, hybrid-spectral, FNO, FFNO | 32 first; 64 only if needed | 1 smoke, then 3 if promoted |

Phase 3: optional physics-informed extension, only if the reproduction report
passes the validity gate.

| Model | Supervised | Physics-only | Hybrid |
| --- | --- | --- | --- |
| SRU-Net | yes | yes | yes |
| Hybrid-spectral | yes | yes | yes |
| FNO or FFNO | yes | optional | yes |
| U-Net | yes | optional | yes |

Phase 4: robustness or out-of-distribution comparison, if WaveBench variant
metadata supports it.

- Train on one fixed wavespeed variant.
- Evaluate on another fixed wavespeed variant.
- Compare degradation across FNO, U-Net, SRU-Net, and hybrid-spectral rows.

## Optional Research Ablations

These are follow-ups, not gates for the active manuscript path:

1. latent channels: `C=16,32,64,128`;
2. boundary-time encoder depth;
3. direct boundary-time input versus learned `128 x 128 x C` latent;
4. supervised loss: L1 versus L1 + gradient versus L1 + gradient + SSIM;
5. physics loss: raw waveform versus normalized waveform versus
   frequency-weighted waveform;
6. hybrid physics weight `lambda_y`;
7. spectral placement: bottleneck-only versus encoder-plus-bottleneck;
8. train/test wavespeed variant transfer.

The first implementation should not run this sweep. It should establish
preflight readiness, then the smallest supervised row set, then add ablations
only where early results show a real failure mode or tradeoff.

## Expected Paper Claims

Supported supervised claim:

> On WaveBench inverse source reconstruction, spectral-local reconstruction
> models improve 2D source recovery relative to local CNN and pure spectral
> operator bodies under a shared boundary-measurement encoder.

Supported physics-informed claim:

> Adding wave-equation consistency reduces boundary-measurement residuals and
> can regularize source reconstruction when the differentiable forward solver
> matches the benchmark generation model.

Supported architecture claim:

> The global-local inductive bias used for CDI reconstruction transfers to a
> controlled 2D wave inverse problem with established FNO/U-Net baselines.

Unsupported claims:

- This is not a geology material-property inversion benchmark.
- This does not show full waveform inversion performance.
- This does not displace the primary CDI benchmark or the CNS pillar.
- A physics-informed row is not valid unless the forward-model reproduction
  check passes.

## Risks And Mitigations

| Risk | Mitigation |
| --- | --- |
| Native WaveBench rows are not directly comparable to shared-encoder rows. | Report native rows as reference baselines and shared-encoder rows as the fair internal architecture comparison. |
| The dataset-generation solver is not exposed clearly enough for exact physics-loop reproduction. | Add the ground-truth reproduction check before physics training; label any mismatch as approximate-model regularization. |
| Boundary measurements are too sparse for physics-only training. | Implement hybrid supervised + physics rows before physics-only rows. |
| Source fields are smooth enough that FNO saturates the task. | Use high-frequency and gradient metrics, and select the hardest available inverse-source variant. |
| The task reconstructs source, not material properties. | Frame the benchmark as controlled wave inverse reconstruction, not geology. |
| The `128 x 128 x C` latent introduces an artificial architecture advantage. | Keep native WaveBench reference rows and include direct-input/shared-encoder ablations. |

## Decision

Keep the active manuscript evidence package unchanged: CDI `lines128` plus
PDEBench CNS remain the governing pillars. Use WaveBench inverse source
reconstruction only as an additional inverse-wave evidence lane if a preflight
proves dataset, baseline, adapter, and forward-model feasibility and a
checked-in roadmap/evidence-package amendment adds it to the paper scope.
Do not use OpenSWI as the main secondary benchmark for this architecture claim
because its target is 1D.
