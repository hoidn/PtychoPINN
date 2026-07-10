# CI Model Compatibility Ablation Design

**Date:** 2026-07-09  
**Status:** Approved design

## Purpose

Determine whether the Torch `hybrid_resnet` architecture is compatible with
the manuscript-defined contrast-invariant (CI) training and inference contract.
The study must establish physics compatibility and recognizable reconstruction
quality. It is not a model-ranking study, and CI Poisson NLL does not need to
outperform the established MAE control in image quality.

The implementation must also create a reusable ablation driver. Future studies
must be able to vary architectures such as CNN, Hybrid ResNet, and FNO together
with model, data, training, inference, execution, and data-generation settings
without adding architecture-specific branches to the driver.

## Authorities And Claim Boundary

- `docs/superpowers/specs/2026-07-09-absolute-scaling-contract-design.md`
  owns the CI measurement-domain, probe-gauge, loss, and VarPro contract.
- `docs/specs/spec-ptycho-core.md` and
  `docs/specs/spec-ptycho-interfaces.md` own the implemented physics and batch
  interfaces.
- `ptycho_torch.reassembly.reconstruct_image_barycentric` is the canonical CI
  inference and stitching path.
- `docs/findings.md` entries `CI-ABSOLUTE-SCALE-CONTRACT-001`,
  `REASSEMBLY-METRIC-FRAME-001`, `TORCH-REASSEMBLY-SIGN-001`, and
  `TORCH-N128-FLAT-AMP-001` constrain interpretation and regression coverage.

Passing this study supports the bounded claim that Hybrid ResNet can train and
infer under the current CI contract on the selected synthetic ptychography
condition. It does not establish universal architecture compatibility,
superiority over MAE, experimental-data generalization, or compatibility with
every model family.

## Study Budget

The initial study uses four core arms, three seeds, and ten epochs per run. A
post-training dose inference sweep reuses each target checkpoint. The intended
single-RTX-3090 runtime is approximately 45--90 minutes, subject to a measured
one-arm runtime preflight. The driver must support overriding seeds and epochs
from the command line without changing the manifest.

## Reusable Driver Architecture

Create a thin CLI at `scripts/studies/torch_ablation_driver.py` backed by
focused modules under `scripts/studies/ablation/`. The CLI owns orchestration
only. Manifest loading, config resolution, run expansion, execution, metrics,
artifact writing, and report rendering remain independently testable units.

Canonical study definitions are TOML files under `scripts/studies/specs/`.
The driver accepts:

```text
--spec PATH
--dry-run
--only SELECTOR
--seeds CSV
--epochs N
--output-root PATH
--resume
--fail-fast
```

`--dry-run` prints the fully expanded run matrix, resolved config changes,
validation result, artifact directory, and estimated run count without loading
training data or allocating a GPU. `--only` selects arm ids or dimension
predicates. `--resume` skips completed runs only after validating their stored
manifest hash and resolved configuration.

The driver must not contain an architecture enum or architecture-specific
execution branch. It invokes the repository's canonical Torch model factory,
loader, trainer, checkpoint loader, and inference functions using resolved
configuration objects.

## Manifest And Matrix Contract

The manifest supplies base overrides plus a Cartesian product of named
dimensions. Each dimension value contributes schema-validated dotted
overrides. `exclude` entries remove invalid or unwanted combinations;
`include` entries add sparse arms. Comparisons and verdict gates are also
manifest-defined so future studies can reuse the driver without changing code.

Conceptually:

```toml
[study]
id = "hybrid-resnet-ci-compatibility"
seeds = [3, 11, 29]

[base.overrides]
"training.epochs" = 10
"inference.patch_weighting" = "probe"

[[matrix.dimensions]]
name = "architecture"

[[matrix.dimensions.values]]
id = "hybrid_resnet"
[matrix.dimensions.values.overrides]
"model.generator_type" = "hybrid_resnet"

[[matrix.dimensions.values]]
id = "cnn"
[matrix.dimensions.values.overrides]
"model.generator_type" = "cnn"

[[matrix.dimensions]]
name = "physics_profile"

[[matrix.dimensions.values]]
id = "ci_nll"
[matrix.dimensions.values.overrides]
"data.scale_contract_version" = "ci_intensity_v2"
"data.measurement_domain" = "count_intensity"
"model.physics_forward_mode" = "rectangular_scaled"
"training.torch_loss_mode" = "poisson"

[[matrix.exclude]]
architecture = "cnn"
physics_profile = "legacy_nll"
```

The precise field name used to select a model comes from the authoritative
Torch config schema; the example above is illustrative. The checked-in
manifest must use the actual field resolved by the model factory.

The initial matrix contains:

| Architecture | Profile | Role |
|---|---|---|
| Hybrid ResNet | CI rectangular + Poisson NLL | compatibility target |
| CNN | CI rectangular + Poisson NLL | architecture control |
| Hybrid ResNet | explicit legacy + Poisson NLL | scaling-contract control |
| Hybrid ResNet | established explicit legacy + MAE | recognizable-quality control |

CI scaling is active only for the CI Poisson arm. Selecting CI with MAE,
supervised loss, or another primary loss is invalid and must fail before model
or data construction. Missing scale-contract metadata receives the repository's
agreed CI defaults; historical behavior requires explicit legacy selection.

## Configuration Control

The resolver supports every relevant authoritative namespace:

- `data.*`: measurement domain, scaling contract, normalization, grouping,
  `probe_scale`, `probe_normalize`, image geometry, and photon settings;
- `model.*`: architecture, output mode, FNO/Hybrid/CNN parameters, probe masks,
  physics mode, trainable scales, and training reassembly;
- `training.*`: loss, optimizer, learning rate, epochs, batching, scheduler,
  clipping, precision, and regularization;
- `inference.*`: VarPro, patch weighting, trim, padding, windowing, and batching;
- `execution.*`: accelerator, devices, workers, deterministic mode, precision,
  logging, and checkpoint behavior;
- `datagen.*`: synthetic object/probe sources and generation parameters.

Resolution uses an explicit namespace registry backed by dataclass
introspection. It maps each dotted path to one authoritative config object,
coerces TOML values to the declared type, runs repository validators, and
rejects unknown, duplicated, derived, or contradictory fields with a suggested
valid path. Every run stores the fully resolved configuration.

`probe_scale` and `probe_normalize` are controllable training/gauge choices.
The physical probe, its count-dose calibration, measurement domain, and
provenance are physical dataset inputs. They may be changed only through a
named dataset/probe transform that emits updated provenance, not an unchecked
runtime override. Probe masks and inference patch weighting remain separate
knobs.

The implementation starts with a feasibility fixture proving that one
manifest can resolve and execute representative overrides from `data`,
`model`, `training`, `inference`, and `execution`. Failure of this fixture is a
design blocker; it must not be bypassed with model-specific parsing.

## Dataset Contract

The compatibility dataset must provide:

- raw measured count intensity;
- a count-calibrated physical probe;
- normalized training-probe fields produced by the CI loader;
- scan coordinates with explicit `(x, y)` provenance;
- complex object truth for quality evaluation;
- measured counts per image and checksums for every split.

Synthetic Poisson inputs must satisfy the existing minimum of one million
measured photons per image. The balanced N=64 study therefore requires a
calibrated high-dose dataset rather than the existing approximately 108-counts
per-pixel `lines_N64` fixture. Dataset preparation is a separate, deterministic
prerequisite; the ablation driver consumes immutable paths and never silently
regenerates data.

The dose inference sweep holds object, scan geometry, and trained checkpoint
fixed while using test sets whose counts and physical probes are calibrated
together. Holding the probe fixed while changing counts is allowed only as a
named negative control.

## Training And Inference Flow

For each expanded arm and seed:

1. Resolve and validate all configuration before loading data.
2. Record the source manifest, manifest hash, Git commit, environment, dataset
   checksums, probe provenance, seed, and resolved configuration.
3. Construct canonical loaders and model objects.
4. Train for the requested budget while recording losses, gradient norms,
   output statistics, and checkpoint identity.
5. Reload the selected checkpoint through the production checkpoint loader.
6. Run canonical inference on the base test set and dose sweep.
7. Calculate metrics, render visuals, and write a per-run verdict.
8. Aggregate seeds, arms, and comparisons into the study report.

An arm failure remains visible in the aggregate report and does not stop other
arms unless `--fail-fast` is selected.

## Reassembly Contract

Canonical inference uses
`ptycho_torch.reassembly.reconstruct_image_barycentric` for every arm so model
comparisons do not change placement code. CI arms require:

- `inference.patch_weighting = "probe"`;
- `inference.varpro_scaling = true`;
- full-detector VarPro basis accumulation before stitching crop;
- an incoherent-mode probe weight equal to the sum of `|P_p|^2` over modes;
- `coords_global` interpreted in the established `(x, y)` convention;
- the padded scan-center-of-mass canvas and returned `canvas_anchor` metadata.

Truth alignment must use `canvas_anchor.scan_com`. Object-centered truth
cropping and the simplified uniform-stitching inference helper are forbidden
for primary CI metrics. Uniform weighting and disabled VarPro may be selected
only as explicit ablation values.

Training-time patch reassembly is a separate model setting. CI production arms
select `model.training_patch_weighting = "probe"`. The report records whether
that choice was effective. If `C=1`, `object_big=false`, or another guard makes
the path inert, the study must not claim to have tested training-time
probe-weighted reassembly.

The object-frame direct-placement metric remains a secondary diagnostic that
isolates patch quality from stitching. It does not replace the canonical
probe-weighted stitched reconstruction in the primary verdict.

## Metric Contract

Metrics are divided by the transformation they permit. Each output key records
its basis and alignment policy.

### Absolute-Scale And Physics Metrics

These metrics do not normalize reconstructed amplitude:

- amplitude MAE and NRMSE against truth;
- complex NRMSE after unit-magnitude global-phase alignment only;
- reconstructed/truth amplitude mean and selected quantile ratios;
- fitted `s1`, `s2`, VarPro residual, and solve condition diagnostics;
- physical count-space relative intensity error;
- physical count-space Poisson NLL;
- dose-to-dose variation of reconstructed amplitude and fitted scales.

The VarPro result must not increase its own physical least-squares intensity
objective relative to the unscaled texture. No fitted real amplitude gauge may
be applied before absolute-scale metrics.

### Recognizability Metrics

These may remove scale or unavoidable phase ambiguity, but must say so:

- amplitude Pearson correlation;
- amplitude and phase SSIM and MS-SSIM;
- amplitude and phase FRC curves, FRC50, and FRC1/7;
- phase error after the configured global phase or plane ambiguity correction;
- object-frame patch Pearson correlation as a secondary diagnostic.

`ptycho.evaluation.eval_reconstruction` mean-normalizes predicted amplitude.
Its outputs are recognizability metrics only and must never be presented as
absolute-scale evidence. Evaluation may crop to a common anchor-correct
footprint but may not resize either reconstruction or truth.

### Stability Metrics

- finite losses, gradients, outputs, and metrics;
- gradient norm history and clipping frequency;
- output variance and decoder rail occupancy;
- checkpoint reload parity;
- seed-level pass rate and metric dispersion;
- patch inclusion and covered-pixel fractions.

## Initial Compatibility Verdict

The checked-in Hybrid ResNet study manifest defines the final thresholds. Its
initial contract is:

- at least two of three Hybrid ResNet CI seeds finish with finite training and
  inference outputs;
- median anchor-correct stitched amplitude Pearson correlation is at least
  `0.50`;
- median amplitude SSIM is at least `0.25`;
- median CI amplitude Pearson is at least 70% of the matched Hybrid ResNet MAE
  control, without requiring CI to exceed MAE;
- checkpoint reload reproduces inference within registered numerical
  tolerance;
- physical-probe VarPro is finite, does not worsen its fitted intensity
  objective, and yields finite absolute-scale metrics;
- the calibrated dose sweep has reconstructed object-scale coefficient of
  variation no greater than 15% over its declared valid dose range;
- visual review confirms recognizable line/object structure and no flat,
  checkerboard, mirrored, or decoder-rail failure mode.

The report must separate numeric gate results from the visual sanity result.
A visual failure cannot be overridden merely because a scalar metric passes,
and a visually recognizable result cannot erase a failed physics gate.

## Artifacts

Each run directory contains:

- source and resolved TOML/JSON configuration;
- invocation, environment, Git, dataset, and probe provenance;
- training history and checkpoint reference;
- raw and stitched reconstruction arrays;
- VarPro basis/scale diagnostics sufficient to audit the solve;
- tidy metrics JSON and one-row CSV;
- per-run verdict JSON;
- logs and failure details.

The study root contains:

- `report.md` with the bounded compatibility conclusion;
- aggregate tidy CSV and JSON;
- arm-by-seed status and verdict tables;
- reconstruction/truth/error grids with shared row color limits;
- training and gradient curves;
- seed-distribution plots;
- dose-response and VarPro-scale plots;
- an absolute-scale/stability dashboard;
- machine-readable manifest expansion and exclusion records.

Plots must label normalized versus absolute quantities. The report links every
figure row to its resolved run id.

## Error Handling

Preflight rejects unknown config paths, type mismatches, invalid enum values,
CI with a non-Poisson primary loss, ambiguous or contradictory scale metadata,
missing physical-probe fields, uncalibrated dose transforms, duplicate run ids,
manifest cycles, output collisions, and resume hash mismatches.

Runtime failures are captured with the failing stage and traceback. Partially
written runs use an incomplete status and are never treated as resumable
successes. Aggregate rendering tolerates failed arms while displaying them
prominently.

## Test Strategy

Unit tests cover TOML parsing, Cartesian expansion, include/exclude behavior,
dotted-path ownership, type coercion, typo suggestions, deterministic run ids,
CLI overrides, resume hashes, and manifest-defined comparisons and gates.

Physics/config tests cover default-CI resolution, explicit legacy selection,
CI-plus-MAE rejection, physical probe protection, calibrated probe transforms,
and checkpoint/config round trips.

Reassembly/metric tests cover multimode `|P|^2` weighting, `(x, y)` placement,
canvas padding, anchor-correct truth alignment, patch inclusion, no-resize
evaluation, separation of normalized and absolute metrics, global-phase-only
absolute comparison, and a synthetic perfect-reconstruction oracle.

Integration tests run a tiny two-arm, one-seed, one-epoch manifest through
training, checkpoint reload, canonical reconstruction, metrics, resume, and
report generation. The full four-arm GPU study is execution evidence, not a
routine CI test.

## Rejected Alternatives

- Extending `varpro_probe_ablation_runner.py` would preserve architecture and
  arm assumptions that make future studies cumbersome.
- A Python-only study definition is flexible but weakens auditability and
  encourages executable configuration.
- A flat list of hand-written arms avoids matrix logic but duplicates settings
  and makes multi-model studies error-prone.
- Free-form mutation of nested objects is concise but cannot reliably validate
  ownership, types, derived values, or physical invariants.
- Using `eval_reconstruction` alone would hide absolute-scale errors through
  amplitude mean normalization.
- Using object-frame direct placement as the only quality path would avoid the
  known anchor problem by bypassing the production stitching behavior that the
  study is required to validate.

## Completion Criteria

The implementation is complete when the generic-driver test suite passes, the
dry-run expansion is independently auditable, the four-arm/three-seed study and
dose inference sweep finish or record explicit failures, all required artifacts
are generated, and the report gives a bounded Hybrid ResNet CI compatibility
verdict supported separately by physics, quality, stability, and visual
evidence.
