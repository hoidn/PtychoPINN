# Inverse-Wave Benchmark Rationale

## Purpose

This note records the inverse-wave benchmark discussion behind the current
candidate lanes. It is meant to prevent future manuscript or workflow tasks
from rediscovering the same options and accidentally changing the evidence
strategy.

The current manuscript evidence package remains:

1. CDI `lines128` reconstruction as the primary ptychographic inverse-imaging
   benchmark.
2. PDEBench `2d_cfd_cns` as the required cross-domain global/local dynamics
   benchmark.

Candidate inverse-wave lanes are additions to CDI and CNS. They may execute
concurrently only when selected by backlog priority and steering, and they do
not replace either required pillar.

## Goal Of The Search

The manuscript already has a primary wave-optics inverse problem in CDI and a
secondary PDEBench CNS dynamics benchmark. The question was whether there is a
second 2D inverse problem with a known forward model that could strengthen the
paper's architecture story without creating the engineering burden of full
OpenFWI physics-loop training.

The desired benchmark properties were:

- 2D spatial target, so SRU-Net and hybrid-spectral image architectures remain
  relevant.
- Known forward model, so supervised and physics-informed variants are both
  conceptually possible.
- Meaningfully different from CDI, so it adds evidence rather than duplicating
  the existing ptychographic benchmark.
- FNO is a credible baseline, not an artificial row.
- Existing ML baselines and benchmark datasets are available.
- The first experiment is feasible on current local compute and deadline
  constraints.

## Existing Pillars

### CDI `lines128`

The CDI benchmark is already the main paper anchor. Its forward model is
ptychographic coherent diffraction:

```math
I_j(q)=\left|\mathcal{F}\{P(x)O(x-r_j)\}\right|^2.
```

It is a strong fit for the paper because it is the actual scientific imaging
problem of interest. The forward model is cheap relative to seismic FWI because
it is FFT-heavy rather than a long sequential time-domain PDE solve.

CDI stays primary.

### PDEBench CNS

PDEBench CNS is a cross-domain dynamics benchmark. It tests whether the
global/local architecture bias transfers to a difficult field-prediction
problem with coherent large-scale flow and local high-gradient structures.

It is not an inverse reconstruction task in the same sense as CDI, but it is
already part of the active roadmap and evidence package. It stays as the
required second pillar unless a separate roadmap decision changes the paper
strategy. This inverse-wave candidate discussion does not make that change.

## Benchmarks Considered

### OpenFWI

OpenFWI is the strongest geophysical inverse-scattering analogue. It maps
seismic shot gathers to subsurface velocity maps:

```math
d = F_{\mathrm{wave}}(v).
```

The acoustic forward model is time-domain wave propagation:

```math
\nabla^2 p -
\frac{1}{v(x,z)^2}\frac{\partial^2 p}{\partial t^2}
= s.
```

Reasons it is attractive:

- 2D spatial target.
- Known forward model.
- Strong inverse-scattering story.
- Published ML baselines such as InversionNet, VelocityGAN, UPFWI, and
  InversionNet3D.
- Hard OpenFWI sets such as `CurveVel-B`, `FlatFault-B`, `CurveFault-B`, and
  `Style-B` are real stress tests.

Reasons it was not chosen as the immediate added benchmark:

- The input geometry is source-time-receiver, not image-to-image. FNO is not a
  natural native baseline without a seismic encoder.
- Physics-informed training requires differentiating through many sequential
  wave-equation time steps and is orders of magnitude more expensive than CDI's
  FFT forward model.
- A serious OpenFWI adaptation needs a seismic encoder, geometry handling,
  baseline reproduction, and careful solver alignment.
- It is better suited to a larger follow-up benchmark project than a quick
  manuscript evidence addition.

Conclusion: OpenFWI is scientifically strong but too expensive and too broad
for the immediate added evidence lane.

### OpenSWI

OpenSWI is a surface-wave dispersion inversion benchmark:

```math
\text{dispersion curves} \rightarrow V_s(z).
```

Reasons it is attractive:

- Geophysics-adjacent.
- Cheaper forward model than full waveform propagation.
- Published ML baselines such as U-Net, FCNN, and Transformer.

Reasons it was not chosen:

- The target is fundamentally 1D.
- It does not test 2D spatial reconstruction, local image features, or
  global/local image decoders.
- It would force SRU-Net/hybrid-spectral into a different 1D architecture
  setting, weakening the connection to the CDI architecture claim.

Conclusion: OpenSWI is useful for profile inversion, but inappropriate as the
main added benchmark for a 2D spectral-convolutional reconstruction paper.

### Fourier Ptychography

Fourier ptychography changes the acquisition geometry from real-space probe
scanning to angle-varying illumination and Fourier-aperture synthesis:

```math
I_j(x)=
\left|
\mathcal{F}^{-1}
\left[
P(k)\,O(k-k_j)
\right]
\right|^2.
```

Reasons it is attractive:

- 2D complex-object reconstruction.
- Known FFT-based forward model.
- Very close to the paper's ptychographic imaging theme.

Reasons it was not chosen:

- It is close enough to CDI that it may not add much independent evidence.
- ML baselines and benchmark protocols are less standardized than WaveBench or
  OpenFWI.
- It risks broadening the optics benchmark without strengthening the
  cross-domain argument.

Conclusion: excellent scientific continuity, but too close to the existing CDI
pillar for the specific goal of adding a distinct inverse-wave stress test.

### Holography / Lensless Holography

Holography uses Fresnel or angular-spectrum propagation:

```math
I_z(x)=|\mathcal{P}_z(O)(x)|^2,
```

with

```math
\mathcal{P}_z(O)=
\mathcal{F}^{-1}\left[
H_z(k_x,k_y)\mathcal{F}(O)
\right].
```

Reasons it is attractive:

- 2D wave inverse imaging.
- Forward model is FFT-cheap.
- More distinct from CDI than Fourier ptychography.

Reasons it was not chosen:

- Public ML benchmark and baseline standardization is fragmented.
- It would likely require building or curating our own benchmark protocol.
- It is still optical wave imaging, so it broadens the imaging story less than
  a non-optical wave benchmark.

Conclusion: promising but under-specified in the first comparison. The later
BRDT design makes the Born/Rytov variant concrete enough to run as an active
candidate preflight on equal footing with WaveBench.

### Diffraction Tomography

Born/Rytov diffraction tomography reconstructs scattering potential or
refractive-index structure from multi-angle scattering data. In linearized form:

```math
u_s \approx G * (q u_i),
```

where `q` is the scattering potential.

Reasons it is attractive:

- 2D or 3D inverse scattering.
- More distinct from CDI than Fourier ptychography.
- Born/Rytov versions can be much cheaper than full multiple-scattering
  solvers.

Reasons it was not chosen in the first discussion:

- Ready ML baselines and benchmark splits are less standardized.
- A paper-grade benchmark would likely require significant design work.
- Full multiple-scattering versions can become expensive and solver-heavy.

Revised conclusion: Born/Rytov diffraction tomography is an active preflight
candidate because the cheap FFT-heavy forward model is close to CDI's
computational shape while adding a distinct inverse-scattering task. Its active
scope is a limited preflight, not a paper table.

### WaveBench Inverse Source Reconstruction

WaveBench inverse source reconstruction predicts an initial pressure/source
field from boundary pressure measurements over time:

```math
y(t,b) \rightarrow q_0(x).
```

The forward model is a scalar linear wave equation with fixed wavespeed:

```math
\frac{1}{c(x)^2}\frac{\partial^2 u(x,t)}{\partial t^2}
- \Delta u(x,t) = 0,
```

with measurements:

```math
y(t,b)=u(x_b,t), \quad x_b\in\partial\Omega_{\mathrm{obs}}.
```

Reasons it is attractive:

- The target is 2D, so image-style reconstruction bodies remain relevant.
- It is a real inverse wave problem, unlike CNS forward prediction.
- The forward model is known and can support supervised, physics-only, and
  hybrid supervised-plus-physics variants if solver alignment is verified.
- WaveBench has dataset infrastructure, Colab notebooks, and native FNO/U-Net
  baselines/checkpoints.
- FNO is a natural baseline because WaveBench is built around wave-PDE operator
  learning.
- It is more controlled and likely lighter than OpenFWI.

Limitations:

- The target is a source/initial pressure field, not a material property.
- It is a controlled benchmark rather than a direct ptychography or geology
  application.
- Physics-informed rows require proving that a local differentiable forward
  model reproduces WaveBench measurements from ground-truth `q_0`.
- Native WaveBench baselines and shared-encoder rows are not automatically the
  same comparison and must be labeled separately.

Conclusion: WaveBench inverse source is also an active preflight candidate. Its
native FNO/U-Net baseline infrastructure is stronger than BRDT's, while its
boundary-time measurement geometry and local forward solver reproduction
requirements create a different integration risk profile.

## Candidate Priority Decision

The active candidate decision is:

1. Born/Rytov diffraction tomography and WaveBench inverse source are both
   active candidate preflights.
2. They are on equal footing as optional additions, not required paper pillars.
3. They share the same candidate priority class so neither is hard-coded ahead
   of the other.

The reason is not that either candidate replaces CDI or CNS. BRDT has a cheaper
and more CDI-like differentiable physics loop. WaveBench has stronger native ML
benchmark infrastructure and FNO/U-Net baselines. The preflights should test
those tradeoffs directly.

Comparison summary:

| Criterion | BRDT | WaveBench inverse source | OpenFWI | OpenSWI |
| --- | --- | --- | --- | --- |
| 2D target | yes | yes | yes | no |
| Known forward model | yes, linearized | yes | yes | yes |
| Cheap differentiable physics loop | yes | maybe, after solver alignment | no | yes, but 1D |
| Close to CDI computational shape | high | medium | low | low |
| Existing ML baselines | weaker | stronger | strong | yes, but 1D |
| Distinct from CDI | yes | yes | yes | yes, but wrong target type |
| Immediate manuscript fit | high as preflight | high as preflight | low | low |

The routing should use backlog priority rather than adding a special gate for
each candidate. `candidate-*` items are allowed by the current gate, and lower
priority numbers stay ahead of candidate preflights.

## Resulting Work Items

Active candidates:

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/born_rytov_dt_candidate_lane_design.md`
- `docs/backlog/active/2026-04-29-brdt-candidate-preflight.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/wavebench_inverse_source_benchmark_design.md`
- `docs/backlog/active/2026-04-29-wavebench-inverse-source-preflight.md`

The BRDT preflight should answer:

- Is the physical target locked as
  `q = k_m^2 ((n / n_m)^2 - 1)`?
- Does the PyTorch Born operator pass independent correctness checks?
- Is normalization safe for the physics loss?
- Can the synthetic dataset and adapters support the four-row decision-support
  roster?
- Should BRDT be promoted by a later evidence-package amendment?

The WaveBench preflight should answer:

- Which WaveBench inverse-source variant should be used?
- What are the exact input/target shapes, sample counts, splits, and data
  provenance?
- Are native WaveBench FNO/U-Net baselines and checkpoints usable?
- Can local repo model bodies consume a shared `128 x 128 x C`
  boundary-measurement latent?
- Can a differentiable forward model reproduce WaveBench boundary traces from
  ground-truth `q_0`?
- Should WaveBench become an additional roadmap evidence lane?

No full BRDT or WaveBench benchmark should run until its preflight is complete
and a later amendment promotes that lane.

## Current Decision

Keep CDI and CNS as the governing manuscript pillars. Run BRDT and WaveBench
inverse source as active concurrent candidate preflights under the broad
`candidate-*` gate and the same low-priority candidate class. Do not use OpenSWI
as a main benchmark because it is 1D. Do not start OpenFWI physics-informed
work now because the solver and input-geometry burden is too large for the
immediate manuscript path.
