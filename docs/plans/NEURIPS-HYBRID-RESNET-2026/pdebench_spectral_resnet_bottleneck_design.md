# PDEBench Spectral ResNet Bottleneck Variant Design

## Design Metadata

- ID: `NEURIPS-HYBRID-RESNET-2026-pdebench-spectral-resnet-bottleneck`
- Title: Spectral ResNet Bottleneck Variant For The PDEBench 128x128 Image Suite
- Status: draft design
- Date: 2026-04-20
- Source brief / issue: define a bottleneck-only variant for the PDEBench image-suite adapter that keeps the current ResNet `3x3` bottleneck blocks, adds a shared factorized spectral residual branch, and exposes it under a distinct model family name instead of `hybrid_resnet`.
- Experiment root: `/home/ollie/Documents/PtychoPINN/`
- Manuscript artifact root: `/home/ollie/Documents/neurips/` (future Phase 5 root; this design must not create it)

## Consumed Inputs And Authority

- `AGENTS.md`
- `docs/index.md`
- `docs/findings.md`
- `docs/INITIATIVE_WORKFLOW_GUIDE.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md`
- `ptycho_torch/generators/hybrid_resnet.py`
- `ptycho_torch/generators/resnet_components.py`
- `scripts/studies/pdebench_image128/models.py`
- Public reference implementation: `alasdairtran/fourierflow`
- Public neural-operator implementation reference: `neuraloperator/neuraloperator`

## Problem Statement

The current PDEBench image-suite adapter exposes a supervised real-channel `hybrid_resnet` path that keeps the existing body:

1. `SpatialLifter`
2. Hybrid spectral/local encoder blocks
3. Downsampling
4. `ResnetBottleneck`
5. `CycleGanUpsampler`
6. Final `Conv2d` projection

That architecture is useful for Darcy and the current PDE image contract, but its bottleneck is purely local convolution. It does not test whether repeated global mixing at the low-resolution latent stage helps the PDEBench tasks, especially Darcy and later CNS.

The requested scope is narrower than a full F-FNO reimplementation. This design covers only a bottleneck replacement:

- keep the current PDEBench supervised input/output contract,
- keep the current encoder/downsample/upsample shell,
- replace only the constant-resolution `ResnetBottleneck`,
- expose the resulting model family under a name other than `hybrid_resnet`.

## Decision Summary

- Introduce a new model family named `spectral_resnet_bottleneck_net`.
- Do not expose this variant as `hybrid_resnet_*`; the architectural identity is materially different enough that reusing the old family name would blur experiment interpretation.
- Replace the current `ResnetBottleneck` with a new `SharedSpectralResnetBottleneck` operating at the inherited bottleneck resolution and channel width.
- Keep the current ResNet `3x3` local-convolution block body as the local branch, but do not reuse `ResnetBlock.forward()` literally inside the new block.
- Add one shared factorized spectral operator reused across bottleneck depth as a residual bypass branch.
- Reuse the current PDEBench image-suite data, normalization, runner, and benchmark protocol unchanged for the first comparison.
- Treat this model as an optional PDEBench image-suite extension or ablation, not as a replacement for the required primary baselines `hybrid_resnet_base`, `fno_base`, and `unet_strong`.
- Treat public GitHub implementations as references for factorized spectral structure and sharing patterns, not as drop-in runtime dependencies.
- Treat the first comparison row as a same-shell, changed-bottleneck row, not a parameter-matched fairness row.
- Keep the variant manual opt-in only until the required primary suite rows are stable for the active PDEBench tasks.

## Naming And Exposure Contract

Public-facing names must make two facts obvious:

1. this is not the existing `hybrid_resnet` family;
2. this is a ResNet-local bottleneck with an added shared spectral branch, not a pure full-resolution F-FNO baseline.

Adopt the following naming:

- Generator / module family: `SpectralResnetBottleneckNet`
- PDE image-suite wrapper: `SpectralResnetBottleneckImageModel`
- Profile namespace: `spectral_resnet_bottleneck_*`
- First profile: `spectral_resnet_bottleneck_base`

Do not register aliases such as `hybrid_resnet_ffno`, `hybrid_resnet_deep`, `hybrid_resnet_bottleneck_ffno`, or `hybrid_resnet_spectral_bottleneck`. Those names imply continuity with the existing Hybrid ResNet profile grid and make benchmark tables harder to read.

## Architecture Contract

### Unchanged Shell

The following structure remains unchanged from the current supervised PDE adapter:

- `SpatialLifter`
- `fno_blocks` Hybrid encoder blocks
- `downsample_steps`
- upsamplers
- final output projection

For the default `128x128` PDE profile with `hidden_channels=32` and `downsample_steps=2`, the spectral ResNet bottleneck therefore operates on the inherited bottleneck tensor after two downsampling steps. For native `128x128` inputs, that means the bottleneck runs at `32x32` resolution with the inherited bottleneck channel width.

### Replaced Component

Replace:

- `ResnetBottleneck(channels, n_blocks=...)`

With:

- `SharedSpectralResnetBottleneck(channels, n_blocks=..., modes=..., share_spectral_weights=...)`

The replacement contract is strict:

- input shape preserved: `(B, C, H, W) -> (B, C, H, W)`
- no channel-count change inside the bottleneck
- no spatial-size change inside the bottleneck
- no change to the task-level `(in_channels, out_channels)` contract

### Required Internal Pieces

Implement three new building blocks:

1. `FactorizedSpectralConv2d`
   - Accepts `(B, C, H, W)`.
   - Performs factorized spectral mixing per spatial axis rather than a joint 2D Fourier kernel.
   - Keeps only the configured low modes per axis.
   - Returns `(B, C, H, W)`.

2. `SpectralResnetBlock`
   - Constant-width residual block.
   - Reuses the current ResNet local `3x3` conv body pattern.
   - Adds a spectral residual bypass computed from a shared factorized spectral operator.
   - Keeps the residual path outside the nonlinear transform.

3. `SharedSpectralResnetBottleneck`
   - Repeats `SpectralResnetBlock` for `n_blocks`.
   - Owns the shared factorized spectral operator and any spectral gates.
   - Preserves the constant-resolution bottleneck contract required by the existing decoder.

### Block Shape

The bottleneck block should keep the current local bottleneck body and add a shared spectral residual branch:

```text
x
-> local ResNet conv body
-> shared factorized spectral op
-> weighted local + weighted spectral residual update
-> residual add with x
```

Required properties:

- residual outside the local and spectral transforms
- the local path remains recognizably the existing CycleGAN-style ResNet body
- the spectral path is additive and shape-preserving
- spectral sharing controlled explicitly, not implicitly by repeated construction

### Local Branch Contract

The local branch must be the raw two-conv body pattern from the current `ResnetBlock`, not the full `ResnetBlock.forward()` residual block.

Explicit rule:

- reuse the current local body structure: reflection pad -> conv -> instance norm -> GELU -> reflection pad -> conv -> instance norm
- do not call a nested `x + layerscale * block(x)` local residual inside `SpectralResnetBlock`
- keep exactly one outer residual add for the combined local-plus-spectral update

Reason:

- the existing `ResnetBlock` already contains its own residual add and shared layerscale
- reusing it literally inside `SpectralResnetBlock` would create a double residual / double scaling path
- the intended experiment is one outer bottleneck residual update with two internal branches, not a residual block nested inside another residual block

So the implementation target is "reuse the current conv/norm/activation body pattern" rather than "instantiate `ResnetBlock` unchanged and call it."

### Weight-Sharing Rule

Depth scaling is the point of this variant, so the stack must support explicit shared spectral weights across layers.

Default rule:

- `share_spectral_weights=True` for the base profile

Scope of sharing:

- share the factorized spectral operator weights across all bottleneck layers
- keep the local ResNet weights per-block
- use either one shared spectral gate for the whole bottleneck or one learned gate per block; start with one shared gate unless a later plan says otherwise

This keeps the design aligned with the motivating depth-scaling idea while preserving the current local-convolution inductive bias.

### Fusion Rule

The default fusion should stay simple and explicit:

```text
x_next = x + local_scale * local_conv_body(x) + spectral_gate * shared_spectral(x)
```

Preferred initial parameterization:

- `local_scale` continues to come from the current shared bottleneck `layerscale`
- `spectral_gate` is a learned scalar initialized small, for example `0.05` or `0.1`

This gives the new branch a conservative starting point and lets the model decide how much global mixing to use.

Implementation note:

- `local_conv_body(x)` here means the raw two-conv local body described above, not the full residual `ResnetBlock.forward()` path

## Configuration Contract

Add a separate config surface for the new model family. Required knobs:

- `spectral_bottleneck_blocks`
- `spectral_bottleneck_modes`
- `spectral_bottleneck_share_weights`
- `spectral_bottleneck_gate_init`
- `spectral_bottleneck_gate_mode`

Inherited knobs remain active:

- `hidden_channels`
- `fno_blocks`
- `hybrid_downsample_steps`
- `in_channels`
- `out_channels`

The first comparison profile should be:

- `spectral_resnet_bottleneck_base`
- hidden width inherited from `hybrid_resnet_base`
- encoder identical to `hybrid_resnet_base`
- downsample schedule identical to `hybrid_resnet_base`
- spectral ResNet bottleneck replacing only the ResNet bottleneck
- default bottleneck config: `modes=12`, `n_blocks=6`, `share_weights=True`, `gate_init=0.1`

This is a bottleneck-depth experiment, not a global architecture sweep. Do not add a wide profile matrix before one base row is buildable and verified.

Comparison policy for the first row:

- compare it as a pure same-shell, changed-bottleneck row against `hybrid_resnet_base`
- do not require parameter matching for the first design tranche
- if the first row looks promising, a later plan may add a parameter-matched comparison as a separate fairness study

## Public Implementation References

Public repositories can reduce design risk, but only as references:

- `alasdairtran/fourierflow`
  - Use as the primary reference for factorized spectral operator layout and shared-weight conventions.
  - Do not mirror its feedforward block, trainer stack, or config surface in the first implementation tranche.

- `neuraloperator/neuraloperator`
  - Use as a secondary reference for spectral-layer implementation details.
  - Do not add a new runtime dependency just to import a factorized bottleneck block if the local implementation can remain smaller and clearer.

Rules for using public code:

- do not vendor large external modules into this repo,
- do not copy repository-specific trainer or config systems,
- do not change the PDEBench runner just to mirror an external API,
- keep any reused ideas behind local modules and local tests,
- record the external repositories in the implementation plan when code-level adaptation starts.

## Integration Surface

Planned file ownership:

- New module: `ptycho_torch/generators/spectral_resnet_bottleneck.py`
  - `FactorizedSpectralConv2d`
  - `SpectralResnetBlock`
  - `SharedSpectralResnetBottleneck`

- Update: `scripts/studies/pdebench_image128/models.py`
  - add `SpectralResnetBottleneckImageModel`
  - add builder branch for `profile.base_model == "spectral_resnet_bottleneck_net"`

- Update: `scripts/studies/pdebench_image128/run_config.py`
  - add `spectral_resnet_bottleneck_base`
  - add validation for required bottleneck config keys

Do not rewrite `ptycho_torch/generators/hybrid_resnet.py` as the first step. The intended integration path is additive and low-risk: keep the current Hybrid ResNet path intact and add a separate model family beside it.

## Benchmark And Interpretation Contract

This variant must inherit the current PDEBench image-suite protocol on its first pass:

- same data splits
- same normalization
- same optimizer and scheduler
- same seed policy
- same task-level metric contract
- same reporting payloads, extended as needed to satisfy the active task contract

Reason: the first question is whether the bottleneck architecture helps under the already-selected local benchmark protocol. Mixing in the full F-FNO training recipe at the same time would confound the result.

Task-contract clarification:

- for Darcy, inherit the same denormalized RMSE / nRMSE / relative-L2 contract already used by the current Darcy adapter
- for CNS, inherit the active CNS contract from `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_design.md`, including `fRMSE_low`, `fRMSE_mid`, and `fRMSE_high`
- if the current runner/reporting stack does not yet emit the required CNS `fRMSE_*` diagnostics, that is a blocker for any meaningful CNS comparison using this variant, not an excuse to drop the diagnostics

Interpretation boundary:

- `spectral_resnet_bottleneck_net` is a new PDE image model with a ResNet local bottleneck plus a shared factorized spectral residual branch.
- It is not the existing `hybrid_resnet` family.
- It is not a paper-faithful full-resolution F-FNO baseline.
- Any comparison against published F-FNO CNS numbers must carry that caveat explicitly.

## Verification Requirements

Before any benchmark or smoke run claim, add at least:

- shape-preservation tests for `FactorizedSpectralConv2d`
- shared-weight tests for `SharedSpectralResnetBottleneck`
- spectral-gate tests for the selected gate mode
- model-builder tests for `spectral_resnet_bottleneck_base`
- forward-pass smoke tests on PDE image-suite tensor shapes
- parameter-count sanity checks so the shared-weight path is actually engaged

If the variant is evaluated on CNS, verification must also include:

- frequency-band `fRMSE_low` / `fRMSE_mid` / `fRMSE_high` metric plumbing
- reporting payload coverage for those `fRMSE_*` outputs
- explicit confirmation that the CNS row is using the same shock-capture reporting contract as the active suite design

If the implementation reaches runtime experiments, the first execution order should be:

1. unit tests for the new bottleneck modules
2. PDE image-suite model-builder tests
3. one readiness-only smoke run on an already-staged task, preferably Darcy
4. optional CNS extension only after the CNS data gate clears

## Non-Goals

This design does not cover:

- replacing the existing Hybrid ResNet encoder with a full FFNO encoder
- renaming or redefining the existing `hybrid_resnet_base` benchmark row
- adopting the full F-FNO training recipe in the first implementation tranche
- adding an FFNO feedforward MLP inside the bottleneck
- claiming equivalence to the original F-FNO paper architecture
- adding a general-purpose FFNO package across unrelated project workflows

## Planning Handoff

This design is not the next required PDEBench tranche by default. It is gated behind the active suite plan's rule that optional model-family extensions wait until the required primary suite rows are stable.

Execution policy:

- keep `spectral_resnet_bottleneck_base` manual `--profiles` opt-in only
- do not add it to a standard benchmark or ablation bundle before the required `hybrid_resnet_base`, `fno_base`, and strong U-Net rows are stable for the active suite tasks
- use Darcy as the first readiness target only after that suite-priority gate is satisfied

Once the suite-priority gate is satisfied, the next implementation plan should stay narrow:

1. add the new bottleneck module and tests,
2. add a separate PDE image-suite model family and profile,
3. verify shape/build/readiness on Darcy first,
4. only then decide whether the variant deserves CNS benchmark budget.

That keeps the change additive, names it clearly, and avoids conflating a useful architectural experiment with either the existing `hybrid_resnet` family or a full faithful F-FNO baseline.
