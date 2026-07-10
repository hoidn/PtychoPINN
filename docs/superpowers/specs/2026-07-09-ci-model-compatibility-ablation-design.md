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
with model, data, training, inference, and execution settings
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
--dataset ID
--dataset-spec PATH
--dry-run
--only SELECTOR
--seeds CSV
--epochs N
--output-root PATH
--resume
--rerun
--fail-fast
--visual-review PATH
```

`--dry-run` prints the fully expanded run matrix, resolved config changes,
validation result, artifact directory, and estimated run count without loading
training data or allocating a GPU. `--only` selects arm ids or dimension
predicates. `--resume` skips completed runs only after validating their stored
run fingerprint and required artifacts. `--visual-review` imports a completed
machine-readable review record before aggregate verdict calculation.
`--rerun` archives a mismatched or completed attempt and starts a new attempt;
it never overwrites evidence in place.

`--dataset ID` selects a declared bundle. If `dataset` is a matrix dimension,
it filters that dimension; otherwise it replaces the base `dataset.id`.
`--dataset-spec PATH` loads one additional versioned dataset descriptor without
editing the study manifest, enabling a local experimental NPZ to use the same
ablation matrix. Descriptor ids must be unique and their paths/checksums become
part of every run fingerprint.

The driver must not contain an architecture enum or architecture-specific
execution branch. It invokes the repository's canonical Torch model factory,
loader, trainer, checkpoint loader, and inference functions using resolved
configuration objects.

## Manifest And Matrix Contract

Every manifest declares `schema.version = 1`; unknown versions fail. The
manifest supplies immutable dataset identities, base overrides, and a Cartesian
product of named dimensions. Each value contributes schema-validated dotted
overrides. `exclude` entries remove combinations and `include` entries add
complete sparse combinations. Comparisons and gates use a small typed operator
set so future studies can reuse the driver without changing code.

`dataset.id` is the only non-config override namespace. It selects one
immutable dataset bundle declared in the manifest; it cannot modify arrays or
metadata.

Conceptually:

```toml
[study]
id = "hybrid-resnet-ci-compatibility"
seeds = [3, 11, 29]

[schema]
version = 1

[base.overrides]
"training.epochs" = 10
"inference.patch_weighting" = "probe"

[[matrix.dimensions]]
name = "architecture"

[[matrix.dimensions.values]]
id = "hybrid_resnet"
[matrix.dimensions.values.overrides]
"model.architecture" = "hybrid_resnet"

[[matrix.dimensions.values]]
id = "cnn"
[matrix.dimensions.values.overrides]
"model.architecture" = "cnn"

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

Expansion is deterministic:

1. Parse dimensions and values in declaration order; reject duplicate ids.
2. Produce the Cartesian product in declaration order.
3. Remove assignments matching every key in an `exclude` table.
4. Add each `include`, which must name one value from every dimension and must
   not duplicate an existing assignment.
5. Apply base overrides, then dimension-value overrides in declaration order,
   then optional include overrides. Base values may be specialized by a
   dimension. Two dimensions or an include may not assign different values to
   the same field; that is an error rather than implicit precedence.
6. Apply CLI dataset selection, `--epochs`, `--seeds`, and `--output-root`
   last. No other config field has a CLI precedence path.

`--only VALUE` matches either one exact run/arm id or a comma-separated
conjunction such as `architecture=hybrid_resnet,physics_profile=ci_nll`.
Unknown dimensions or values fail. The logical arm id is the study id, resolved
dataset id, and dimension ids in declaration order. If dataset is itself a
dimension it appears exactly once. The run id appends `seed-<integer>`.
Canonical JSON uses sorted keys, UTF-8, no insignificant whitespace, and
JSON-native scalar values.

Comparisons and gates target dimension selectors resolved after dataset
selection, rather than embedding dataset-specific logical arm ids. An optional
`dataset` key narrows a target; omitting it means the currently selected
dataset. Gates support only `ge`, `le`,
`finite`, `paired_ratio_ge`, `status_count_ge`, and `manual_review`, with
`median`, `mean`, or `cv` aggregation where applicable. Missing operands yield
`INCONCLUSIVE`, never an implicit pass.

Dataset paths are repository-root-relative and carry expected hashes:

```toml
[datasets.ci_tmc432]
kind = "synthetic"
format = "npz_mmap"
scale_contract_version = "ci_intensity_v2"
measurement_domain = "count_intensity"
truth = "object_truth"
measurement_key = "diff3d"
probe_key = "probeGuess"
x_key = "xcoords"
y_key = "ycoords"
truth_key = "objectGuess"
truth_location = "embedded_test"
coords_convention = "xy_pixels"
detector_shape = [64, 64]
grouping_max_C = 4
probe_modes = 1
train = ".artifacts/ci_compatibility/datasets/ci_tmc432_train.npz"
test = ".artifacts/ci_compatibility/datasets/ci_tmc432_test.npz"
provenance = ".artifacts/ci_compatibility/datasets/provenance.json"
train_sha256 = "..."
test_sha256 = "..."
provenance_sha256 = "..."

[datasets.ci_tmc432.probe]
source = "synthetic_calibrated"
calibration = "count_amplitude"
gauge = "physical_count_amplitude"
mask_policy = "model_config"
sha256 = "..."

[datasets.ci_tmc432.dose.train]
counts_mean = 432.0
photons_per_image_min = 1000000.0
photons_per_image_mean = 1769472.0
max_observed_count = 1900
dtype_max = 65535
saturation_fraction = 0.0

[datasets.ci_tmc432.dose.test]
counts_mean = 432.0
photons_per_image_min = 1000000.0
photons_per_image_mean = 1769472.0
max_observed_count = 1900
dtype_max = 65535
saturation_fraction = 0.0
```

A standalone experimental descriptor uses the same fields:

```toml
[schema]
version = 1

[dataset]
id = "fly001_experimental"
kind = "experimental"
format = "npz_mmap"
scale_contract_version = "ci_intensity_v2"
measurement_domain = "count_intensity"
truth = "reference_reconstruction"
measurement_key = "diff3d"
probe_key = "probeGuess"
x_key = "xcoords"
y_key = "ycoords"
truth_key = "object"
truth_location = "external_npz"
coords_convention = "xy_pixels"
detector_shape = [64, 64]
grouping_max_C = 4
probe_modes = 1
train = "/data/fly001_train.npz"
test = "/data/fly001_test.npz"
reference = "/data/fly001_reference.npz"
provenance = "/data/fly001_provenance.json"
train_sha256 = "..."
test_sha256 = "..."
reference_sha256 = "..."
provenance_sha256 = "..."

[dataset.probe]
source = "iterative_reconstruction"
calibration = "count_amplitude"
gauge = "physical_count_amplitude"
mask_policy = "model_config"
sha256 = "..."

[dataset.dose.train]
counts_mean = 2500.0
photons_per_image_min = 1000000.0
photons_per_image_mean = 10000000.0
max_observed_count = 18000
dtype_max = 65535
saturation_fraction = 0.0

[dataset.dose.test]
counts_mean = 2600.0
photons_per_image_min = 1000000.0
photons_per_image_mean = 10600000.0
max_observed_count = 19000
dtype_max = 65535
saturation_fraction = 0.0
```

Absolute paths are accepted only in standalone descriptors; checked-in study
manifests use repository-root-relative paths. The descriptor is data and
provenance only: it cannot inject model/training overrides.

Gate syntax is typed and closed:

```toml
[[gates]]
id = "ci_seed_success"
target = { architecture = "hybrid_resnet", physics_profile = "ci_nll" }
operator = "status_count_ge"
status = "success"
threshold = 2
requested = 3

[[gates]]
id = "ci_amp_pearson"
target = { architecture = "hybrid_resnet", physics_profile = "ci_nll" }
metric = "truth_quality.amp_pearson"
operator = "ge"
aggregation = "median"
threshold = 0.50
min_successful = 2
requires = ["has_object_truth"]
when_dataset_kind = "synthetic"

[[comparisons]]
id = "ci_vs_mae_recognizability"
left = { architecture = "hybrid_resnet", physics_profile = "ci_nll" }
right = { architecture = "hybrid_resnet", physics_profile = "legacy_mae" }
metric = "truth_quality.amp_pearson"
operator = "paired_ratio_ge"
aggregation = "median"
threshold = 0.70
min_pairs = 2
when_dataset_kind = "synthetic"
```

Each selector must resolve to exactly one logical arm per selected dataset;
zero or multiple matches fail unless the gate explicitly aggregates a dataset
dimension. Metric paths must exist in the declared metric registry. Operators
reject irrelevant keys, and all gate/comparison ids are unique. Failed runs remain in
the requested denominator but do not contribute numeric values; the verdict
rules below decide when that becomes `FAIL` versus `INCONCLUSIVE`.

`status_count_ge` names an arm, required terminal status (`success`), requested
seed denominator, and threshold. It counts completed attempts with that status;
completed failed attempts count against the threshold. Missing or incomplete
requested attempts make the gate `INCONCLUSIVE` rather than reducing the
denominator.

Gates and comparisons may declare `requires` capabilities and
`when_dataset_kind = "synthetic" | "experimental"`. Nonmatching conditional
gates are recorded `not_applicable`. The closed field
`on_missing_capability = "error" | "not_applicable"` defaults to `error` and
decides how a matching gate handles an absent required capability. This permits one reusable study manifest to
carry synthetic truth-quality gates and experimental measurement-consistency
or reference-agreement gates without confusing their meanings.

The initial matrix contains:

| Architecture | Profile | Role |
|---|---|---|
| Hybrid ResNet | CI rectangular + Poisson NLL | compatibility target |
| CNN | CI rectangular + Poisson NLL | architecture control |
| Hybrid ResNet | explicit legacy amplitude + Poisson NLL | legacy pipeline control |
| Hybrid ResNet | explicit legacy amplitude + MAE | recognizable-quality control |

CI scaling is active only when all of these are true: the resolved pair is
`ci_intensity_v2/count_intensity`, `model.mode="Unsupervised"`,
`model.physics_forward_mode="rectangular_scaled"`, and
`training.torch_loss_mode="poisson"`. Selecting rectangular CI with MAE,
supervised mode, or another primary loss fails before model or data
construction. Amplitude mode does not activate CI even when absent profile
fields receive CI defaults. Historical behavior requires the explicit
`legacy_v1/normalized_amplitude` pair.

### Initial Resolved Arms

The checked-in manifest pins these shared values: `N=64`,
`grid_size=(2,2)`, `C=C_model=C_forward=4`, `model.mode="Unsupervised"`,
`model.object_big=true`, `model.probe_big=false`,
`model.training_patch_weighting="probe"`, `model.probe_mask=false`,
`data.probe_normalize=true`, `data.probe_scale=4.0`, batch size 16, ten
epochs, real/imag output, and `inference.middle_trim=32`. It also pins Adam,
learning rate `2e-4`, norm clipping at `1.0`, `ReduceLROnPlateau` with factor
`0.5`, patience `2`, minimum LR `1e-4`, deterministic execution, one CUDA
device, and no mixed precision. Every other allowlisted field is explicit in
the checked TOML; the claim-grade manifest does not inherit mutable dataclass
defaults.

Architecture values resolve output fields explicitly: Hybrid ResNet uses
`generator_output_mode="real_imag"`; CNN uses
`cnn_output_mode="real_imag"` and `use_shared_decoder=false`. Profile values
are:

| Profile | Dataset | Scale pair | Forward | Model/training loss | Trainable `s1/s2` | VarPro |
|---|---|---|---|---|---:|---:|
| `ci_nll` | `ci_tmc432` | CI/count | rectangular | Poisson/`poisson`, `nll=true` | true | true |
| `legacy_nll` | `legacy_amp` | legacy/amplitude | amplitude | Poisson/`poisson`, `nll=true` | false | false |
| `legacy_mae` | `legacy_amp` | legacy/amplitude | amplitude | MAE/`mae`, `nll=false` | false | false |

All arms use canonical probe-weighted inference stitching. Only the two CI
arms receive the calibrated dose sweep. The matrix excludes both CNN legacy
profiles, leaving exactly four arms times three seeds.

## Configuration Control

The resolver supports proven effective fields from these namespaces:

- `data.*`: measurement domain, scaling contract, normalization, grouping,
  `probe_scale`, `probe_normalize`, image geometry, and photon settings;
- `model.*`: architecture, output mode, proven FNO/Hybrid/CNN parameters,
  probe masks, physics mode, CI `s1/s2`, and training reassembly;
- `training.*`: loss, optimizer, learning rate, epochs, batching, scheduler,
  clipping, and regularization;
- `inference.*`: VarPro, patch weighting, trim, padding, windowing, and batching;
- `execution.*`: accelerator, devices, strategy, workers, deterministic mode,
  precision, logging, progress, and checkpoint behavior.

The namespace registry is normative:

| Prefix | Owner | Consumer |
|---|---|---|
| `dataset` | manifest immutable dataset bundle | runtime adapter |
| `data` | `ptycho_torch.config_params.DataConfig` | loader/model/reassembly |
| `model` | `ptycho_torch.config_params.ModelConfig` | model/checkpoint/reassembly |
| `training` | `ptycho_torch.config_params.TrainingConfig` | Lightning/model |
| `inference` | `ptycho_torch.config_params.InferenceConfig` | reassembly/evaluation |
| `execution` | `ptycho.config.config.PyTorchExecutionConfig` | Lightning runtime only |

The registry combines dataclass introspection with an explicit allowlist.
Fields marked unimplemented, known inert in `docs/findings.md`, or owned as a
derived alias are rejected even if present on a dataclass. In particular,
`intensity_scale_trainable`, execution-side model fields, execution-side
`middle_trim`/`pad_eval`, and execution-side optimization duplicates are not
sweepable. Execution aliases such as learning rate and gradient clipping are
derived from `training.*` by the runtime adapter and asserted equal; they are
not independent controls.

No TF config class owns a study override. Resolution creates the four Torch
config objects and the runtime-only execution object directly, coerces values
to declared types, runs repository validators, and rejects unknown,
duplicated, derived, inert, or contradictory fields with a suggested valid
path. Every run stores all resolved objects.

`train_lightning_only.main` still requires a fifth tuple member,
`DatagenConfig`. The adapter supplies one fixed default compatibility object;
it has no manifest namespace, performs no generation, is included in the
fingerprint, and must round-trip unchanged when persisted.

`probe_scale` and `probe_normalize` are controllable training/gauge choices.
The physical probe, its count-dose calibration, measurement domain, and
provenance are immutable dataset inputs and are not runtime overrides. Probe
masks and inference patch weighting remain separate knobs. Dataset generation
and physical-probe transforms are out of scope for this driver and require a
separate materialization command and provenance contract.

The implementation starts with a feasibility fixture proving that one
manifest can resolve and execute representative effective overrides from
`data`, `model`, `training`, `inference`, and `execution`. Failure of this fixture is a
design blocker; it must not be bypassed with model-specific parsing.

### Runtime Prerequisite Extensions

Before claim-grade execution, `PyTorchExecutionConfig` gains an effective
`devices` field (`int | "auto"`) and precision field (`32-true`, `16-mixed`,
or `bf16-mixed`), and
`train_lightning_only.main` gains an explicit `seed` argument. That entry point
must construct Lightning `Trainer` values from every allowlisted execution
field instead of hardcoding accelerator, device count, strategy,
deterministic mode, progress, checkpointing, workers, or precision. Seed may
not be sourced only from an environment variable.

`reconstruct_image_barycentric` receives the same resolved precision mapping;
`32-true` sets `use_mixed_precision=false`, while mixed modes use the declared
dtype. Each run writes `effective_runtime.json` from the constructed Trainer,
dataloaders, and inference call. Preflight and tests compare these effective
values to resolved config values. A mismatch is a hard failure, not merely
provenance.

The adapter derives the compatibility fields
`TrainingConfig.n_devices=execution.devices`,
`TrainingConfig.strategy=execution.strategy`, and
`TrainingConfig.device=execution.accelerator`, then asserts equality before
model construction and after checkpoint reload. Those `training.*` aliases are
not separately allowlisted, preserving one-owner semantics.

## Dataset Contract

Every manifest dataset bundle declares:

- `kind = "synthetic" | "experimental"` and a stable dataset id;
- storage format and immutable train/test paths with SHA-256 values;
- measurement domain and scale-contract pair, with no value-range inference;
- coordinate convention, detector shape, scan grouping support, and probe-mode
  count;
- physical-probe source, calibration method, gauge, mask policy, and checksum;
- measured dose statistics and saturation checks when count data are present;
- `truth = "object_truth" | "reference_reconstruction" | "none"`, plus the
  source/checksum when present.

Dataset descriptor v1 is closed. Besides `id` (required only for standalone
descriptors), its allowed scalar/list fields are exactly:

| Field | Type / enum |
|---|---|
| `kind` | `synthetic | experimental` |
| `format` | `npz_mmap` |
| `scale_contract_version` | `ci_intensity_v2 | legacy_v1` |
| `measurement_domain` | `count_intensity | normalized_amplitude` |
| `train`, `test`, `provenance` | nonempty path |
| `train_sha256`, `test_sha256`, `provenance_sha256` | lowercase SHA-256 |
| `measurement_key`, `probe_key`, `x_key`, `y_key` | nonempty NPZ key |
| `coords_convention` | `xy_pixels` |
| `detector_shape` | two positive equal integers |
| `grouping_max_C`, `probe_modes` | positive integer |
| `truth` | `object_truth | reference_reconstruction | none` |
| `truth_location` | `embedded_test | external_npz | none` |
| `truth_key` | NPZ key, required unless truth is `none` |
| `reference`, `reference_sha256` | required only for `external_npz` |

The required `probe` table contains only `source: str`,
`calibration: count_amplitude | legacy_normalized`,
`gauge: physical_count_amplitude | legacy_normalized`,
`mask_policy: model_config | pre_masked`, and array-content `sha256`. For each
count-domain split, required `dose.train` and `dose.test` tables contain only
finite nonnegative `counts_mean`, `photons_per_image_min`,
`photons_per_image_mean`, integer `max_observed_count`, integer `dtype_max`, and
`saturation_fraction` in `[0,1]`. Dose tables are forbidden for
normalized-amplitude bundles. Unknown fields or tables fail.

Preflight verifies file hashes; required keys; array shapes, lengths, dtypes,
and finite values; nonnegative integer count measurements; detector shape;
probe rank/mode count and canonical array-content hash; coordinate convention;
truth role/location; and exact agreement between measured dose statistics,
descriptor values, and provenance. CI requires count calibration plus physical
count-amplitude gauge. Legacy requires the legacy pair and normalized probe
gauge. Capability derivation occurs only after these checks.

Preflight derives capabilities from validated content rather than trusting
free-form flags: `has_object_truth`, `has_reference`, `has_physical_probe`,
`supports_count_metrics`, `supports_dose_sweep`, and `supports_grouping_C`.
Matrix expansion may use `dataset.id` as a dimension, but each arm must pass
profile/domain, probe, grouping, and required-metric compatibility checks before
execution.

Synthetic compatibility bundles must provide:

- raw measured count intensity;
- a count-calibrated physical probe;
- normalized training-probe fields produced by the CI loader;
- scan coordinates with explicit `(x, y)` provenance;
- complex object truth for quality evaluation;
- measured counts per image and checksums for every split.

Experimental bundles may omit object truth. They still require raw calibrated
measurements, scan coordinates, and the physical known probe for CI. A
conventional reconstruction is `reference_reconstruction`, never relabeled as
ground truth. Its image metrics are emitted under `reference_agreement.*` and
cannot satisfy synthetic absolute-truth gates. Experimental bundles without a
reference use count-space physics, stability, coverage, and manual visual
review only.

Gate applicability is capability-based. A gate marked required for a selected
dataset that lacks its capability is a preflight error; an explicitly optional
metric is written as `not_applicable` with the missing capability. Reports
separate `truth_quality`, `reference_agreement`, and `measurement_consistency`
sections so cross-dataset tables cannot silently mix them.

Typical selection is therefore:

```bash
python scripts/studies/torch_ablation_driver.py \
  --spec scripts/studies/specs/hybrid_resnet_ci_compatibility.toml \
  --dataset synthetic_lines_ci --only physics_profile=ci_nll

python scripts/studies/torch_ablation_driver.py \
  --spec scripts/studies/specs/hybrid_resnet_ci_compatibility.toml \
  --dataset-spec /data/fly001_dataset.toml \
  --dataset fly001_experimental --only physics_profile=ci_nll
```

Selecting a count-domain experimental bundle with a legacy normalized-
amplitude arm is an explicit incompatibility, not an automatic conversion.
Such a study must select only compatible profiles or register a separately
materialized legacy bundle.

Synthetic Poisson inputs must satisfy the existing minimum of one million
measured photons per image. The balanced N=64 study therefore requires a
calibrated high-dose dataset rather than the existing approximately 108-counts
per-pixel `lines_N64` fixture. Dataset preparation is a separate, deterministic
prerequisite. The manifest names train, validation/test, dose-test,
provenance, and expected SHA-256 values; the driver verifies them and never
regenerates or mutates data.

The dose inference sweep holds object, scan geometry, and trained checkpoint
fixed while using test sets whose counts and physical probes are calibrated
together. Holding the probe fixed while changing counts is allowed only as a
named negative control.

The initial invocation of the reusable compatibility manifest selects
synthetic truth-bearing data because it must detect recognizable versus
collapsed solutions and absolute-scale error. This bounds that report to the
selected synthetic condition. A later experimental invocation uses the same
conditional manifest with an experimental dataset id and produces a separate
dataset-scoped report; reports are never silently aggregated across dataset
kinds. A dedicated manifest is optional convenience, not a different contract.

### Materialization Prerequisite

Dataset creation is a separate command,
`scripts/studies/materialize_ci_compatibility_datasets.py`, not a driver mode.
It reuses the repository's calibrated synthetic truth/count builder and emits:

- `ci_tmc432`: train and test count-intensity NPZs at target mean 432;
- `ci_tmc864` and `ci_tmc1728`: dose-only test twins;
- `legacy_amp`: normalized-amplitude train/test twins from the identical
  latent object, scan positions, and uncalibrated probe geometry;
- one provenance JSON containing generator commit, seeds, source-object and
  coordinate digests, scale profiles, measured photons/image, count means,
  saturation checks, physical-probe norms, and every output SHA-256.

The materializer is deterministic and refuses to overwrite an existing output
whose checksum differs. A parity check proves that the CI and legacy bundles
share object truth and coordinates and that each CI dose changes counts and
physical-probe amplitude together according to the scaling contract. The
driver accepts only the resulting immutable bundle ids and checksums.

## Training And Inference Flow

For each expanded arm and seed, the architecture-neutral runtime adapter:

1. Resolve and validate all configuration before loading data.
2. Record the source manifest, manifest hash, Git commit, environment, dataset
   checksums, probe provenance, seed, and resolved configuration.
3. Instantiate four sweep-owned Torch configs, one runtime execution config,
   and the fixed `DatagenConfig` compatibility object. Stage only the immutable
   training NPZ in an isolated directory and invoke
   `ptycho_torch.train_lightning_only.main(existing_config=(data, model,
   training, inference, datagen), execution_config=execution, seed=seed)`.
   This deliberately bypasses the overlapping flat-override factory.
4. Train for the requested budget while recording losses, gradient norms,
   output statistics, and checkpoint identity.
5. Select with `ptycho_torch.lightning_utils.find_best_checkpoint` and reload
   with `load_checkpoint_with_configs`; assert every persisted data, model,
   training, inference, and compatibility-datagen field matches the resolved
   configuration.
6. Build each held-out mmap with `ptycho_torch.dataloader.PtychoDataset` from
   the reloaded configs, then call
   `ptycho_torch.reassembly.reconstruct_image_barycentric` with the resolved
   inference config. Use the same path for declared dose sets.
7. Calculate metrics, render visuals, and write a per-run verdict.
8. Aggregate seeds, arms, and comparisons into the study report.

The adapter passes the resolved `InferenceConfig` through model construction
and inference; no default replacement is permitted. An arm failure remains
visible in the aggregate report and does not stop other
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

Truth alignment is a coordinate transform, not a center crop. For canvas pixel
`(row, col)`, reference coordinates are
`x = col - floor(W_canvas/2) + scan_com_x` and
`y = row - floor(H_canvas/2) + scan_com_y`. Real and imaginary truth are
bilinearly sampled at `(y, x)`; out-of-bounds samples are invalid. The common
mask is `(canvas_weights > 0)` intersected with valid truth coordinates. A
deterministic largest all-valid axis-aligned rectangle (ties: greatest area,
then lexicographically smallest `(top,left,bottom,right)`) supplies the SSIM
image footprint. FRC uses the largest centered square inside that rectangle;
for odd excess, the extra row/column is removed from the bottom/right. The FRC
helper must receive square arrays and may not perform another crop. Pointwise
metrics use every common-mask pixel. Neither path resizes the reconstruction,
fits a translation, nor center-crops truth.

The study metric adapter calls low-level metric functions on these prepared
arrays; it does not call `eval_reconstruction`, because that function applies
an additional global-offset trim and amplitude normalization. Its formulas may
be reused only after this adapter has made both transformations explicit.
Object-centered truth cropping and the simplified uniform-stitching inference
helper are forbidden for primary CI metrics. Uniform weighting and disabled
VarPro may be selected only as explicit ablation values.

Training-time patch reassembly is a separate model setting. CI production arms
select `model.training_patch_weighting = "probe"`. The report records whether
that choice was effective. If `C=1`, `object_big=false`, or another guard makes
the path inert, the study must not claim to have tested training-time
probe-weighted reassembly.

The object-frame direct-placement metric remains a secondary diagnostic that
isolates patch quality from stitching. It does not replace the canonical
probe-weighted stitched reconstruction in the primary verdict.

### Structured Reassembly Diagnostics

The positional diagnostics list is insufficient and must be replaced or
supplemented by a versioned `ReassemblyDiagnostics` object. It returns timings,
`s1`, `s2`, accumulated `ATA`, `ATb`, `sum_I2`, pixel count, `cond(ATA)`,
unit-scale and fitted objectives, scale-profile id, effective probe-mask
digest, `canvas_anchor`, `canvas_weights`, and patch accepted/total counts. It
must not expose the final batch's basis as though it represented the dataset.

For coefficient vector `z=[s1^2,s2^2,s1*s2]`, the accumulated least-squares
objective is `z^T ATA z - 2 z^T ATb + sum_I2`. Unit-scale uses
`z=[1,1,1]`; the fitted objective uses solved `s1,s2`. Both are divided by
pixel count for reporting. The fitted objective must be no greater than
`unit_objective + 1e-12 + 1e-10*abs(unit_objective)`.

A deterministic second physics pass over the held-out loader applies the
solved scales and physical masked probe without storing detector stacks. It
accumulates:

- `relative_l2_intensity_error = sqrt(sum((I_pred-I_meas)^2) / sum(I_meas^2))`;
- `mean_raw_poisson_nll = mean(I_pred - I_meas*log(max(I_pred,1e-8)))`.

Both reductions cover every declared sample, channel, and detector pixel and
report sample and pixel counts. This API extension is shared instrumentation
in `ptycho_torch.reassembly`, not study-only reconstruction code.

Aggregate VarPro sufficient statistics and the physical count-space second
pass are mandatory only for CI/count arms. Legacy normalized-amplitude arms
still return stitching anchor, weights, timing, and patch coverage, but their
VarPro and count metric records are explicit
`{"status":"not_applicable","reason":"legacy_normalized_amplitude"}`.
These entries satisfy legacy artifact completeness and are never treated as
zero, missing, or passing CI physics evidence.

## Metric Contract

Metrics are divided by the transformation they permit. Each output key records
its basis and alignment policy.

The metric-path registry is closed:

| Namespace | Allowed metric paths |
|---|---|
| `truth_quality` | `absolute_amp_mae`, `absolute_amp_nrmse`, `absolute_complex_nrmse`, `amp_mean_ratio`, `amp_quantile_ratio_p05`, `amp_quantile_ratio_p50`, `amp_quantile_ratio_p95`, `amp_pearson`, `amp_ssim`, `amp_ms_ssim`, `phase_ssim`, `phase_ms_ssim`, `amp_frc50`, `amp_frc1over7`, `phase_frc50`, `phase_frc1over7`, `amp_frc_curve`, `phase_frc_curve`, `phase_wrapped_mae`, `patch_amp_pearson` |
| `reference_agreement` | the same image/patch paths except names beginning `absolute_`; these describe agreement only |
| `measurement_consistency` | `relative_l2_intensity_error`, `mean_raw_poisson_nll`, `varpro.s1`, `varpro.s2`, `varpro.condition`, `varpro.unit_objective`, `varpro.fitted_objective`, `dose.object_scale`, `dose.object_scale_cv` |
| `stability` | `finite`, `gradient_norm_mean`, `gradient_norm_median`, `gradient_norm_p99`, `gradient_norm_max`, `clip_fraction`, `amp_variance`, `phase_variance`, `cross_patch_cv`, `spatial_gradient_energy`, `reload_max_abs_error`, `patches_accepted`, `patches_total`, `coverage_fraction` |
| `runtime` | `train_seconds`, `inference_seconds`, `assembly_seconds`, `peak_memory_bytes` |

Unknown metric paths fail at manifest load and artifact validation. A metric
computed against `object_truth` is written only under `truth_quality`; the
same formula against `reference_reconstruction` is written only under
`reference_agreement`. Count/VarPro/dose metrics are always under
`measurement_consistency`, never under either image-comparison namespace.

### Absolute-Scale And Physics Metrics

These metrics do not normalize reconstructed amplitude:

- amplitude MAE and NRMSE against truth;
- complex NRMSE after unit-magnitude global-phase alignment only;
- reconstructed/truth amplitude mean and selected quantile ratios;
- fitted `s1`, `s2`, VarPro residual, and solve condition diagnostics;
- physical count-space relative intensity error;
- physical count-space Poisson NLL;
- dose-to-dose variation of reconstructed amplitude and fitted scales.

Truth-bearing datasets write these as `truth_quality.absolute_*` and related
ratio paths. Experimental reference comparisons may record raw differences in
the artifact payload, but they are not registered as absolute correctness
metrics or accepted by absolute-truth gates.

The VarPro result must not increase its own physical least-squares intensity
objective relative to the unscaled texture. No fitted real amplitude gauge may
be applied before absolute-scale metrics.

### Recognizability Metrics

These remove only declared nuisance factors:

- amplitude Pearson uses raw `|O_recon|` and `|O_truth|` on every common-mask
  pixel; Pearson supplies its own centering and is scale invariant;
- amplitude SSIM/MS-SSIM multiply predicted amplitude by
  `mean(target)/mean(prediction)` on the valid rectangle. The SSIM data range
  is `max(target)-min(target)`, amplitude MS-SSIM uses `sigma=1.0`, and a
  nonpositive/near-zero prediction mean or target range is a metric failure;
- define `c = sum(conj(O_recon)*O_truth) / abs(sum(conj(O_recon)*O_truth))`
  on the common mask, with zero correlation treated as failure. Phase metrics
  use `angle(c*O_recon)` and `angle(O_truth)` with wrapped phase residuals in
  `[-pi,pi]`; no fitted phase plane or amplitude gauge is allowed;
- phase SSIM/MS-SSIM map those globally aligned phase arrays from
  `[-pi,pi]` to `[0,1]` with data range `1.0` and no Gaussian smoothing;
- amplitude and phase FRC use the explicit square footprint, `frc_sigma=0`,
  and report full curves, FRC50, and FRC1/7;
- object-frame patch Pearson remains a secondary diagnostic.

The resulting paths use `truth_quality.*` for synthetic object truth and
`reference_agreement.*` for an experimental conventional reference.

`ptycho.evaluation.eval_reconstruction` mean-normalizes predicted amplitude.
Historical outputs from it are recognizability metrics only and must never be
presented as absolute-scale evidence. This study computes equivalent metrics
through the explicit adapter above and may not resize either image.

### Stability Metrics

- finite losses, gradients, outputs, and metrics;
- gradient norm history and clipping frequency;
- amplitude/phase variance, cross-patch coefficient of variation, and spatial
  gradient energy as architecture-neutral collapse diagnostics;
- checkpoint reload parity;
- seed-level pass rate and metric dispersion;
- patch inclusion and covered-pixel fractions.

Architecture-specific rail checks are optional manifest diagnostics and cannot
be a generic compatibility requirement.

## Initial Compatibility Verdict

Verdicts are typed `PASS`, `FAIL`, or `INCONCLUSIVE`. A run succeeds only when
training, checkpoint reload, inference, metrics, and required artifacts finish.
The successful-seed gate uses all three requested seeds as its denominator and
requires at least two successes. If all three attempts are terminal and fewer
than two succeed, the gate is `FAIL`; if any requested attempt is missing or
incomplete, it is `INCONCLUSIVE`. Other medians use successful seeds only after
the status gate passes.

The Hybrid ResNet CI aggregate gates are:

- at least two of three requested seeds succeed;
- median anchor-correct `truth_quality.amp_pearson` is at least `0.50`;
- median anchor-correct `truth_quality.amp_ssim` is at least `0.25`;
- for each seed where both CI and Hybrid ResNet MAE succeed, calculate
  `pearson_ci / max(pearson_mae, 1e-12)`; require at least two matched pairs
  and a median paired ratio of at least `0.70`;
- checkpoint reload reproduces the pre-reload fixed-batch texture and stitched
  canvas with `rtol=1e-5, atol=1e-6`;
- physical-probe VarPro is finite, passes the objective non-increase rule, and
  produces finite `truth_quality.absolute_*` and
  `measurement_consistency.*` metrics;
- dose sets have calibrated target means `[432, 864, 1728]` counts/pixel and
  each clears the photon floor. For each seed/dose define object scale as
  `median(|O_recon|) / median(|O_truth|)` on the common pointwise mask. Require
  at least two complete CI seed sweeps and median seed-level
  `std(scale, ddof=1)/mean(scale) <= 0.15`;
- a manual review approves the shared-limit reconstruction/error grid and
  records whether line/object structure is recognizable and whether flat,
  checkerboard, mirrored, or saturation/collapse artifacts are present.

Any evaluated mandatory numeric or visual gate that fails yields `FAIL`.
Missing attempts, matched controls, dose rows, diagnostics, or manual review
yields `INCONCLUSIVE`; completed failed attempts are adjudicated by the status
gate above. `visual_review.json` contains schema version, reviewer, UTC
timestamp, reviewed figure SHA-256, `approve|reject`, `recognizable: bool`,
`flat: bool`, `checkerboard: bool`, `mirrored: bool`, `saturation: bool`,
`collapse: bool`, and notes. Approval requires `recognizable=true` and every
failure-mode field false. The driver writes a pending template but never
self-approves it.

The report separates numeric gates from manual review. A visual failure cannot
be overridden by a scalar pass, and a visual pass cannot erase a physics
failure.

## Artifacts

Each run directory contains:

- source and resolved TOML/JSON configuration;
- invocation, environment, Git, dataset, and probe provenance;
- training history and checkpoint reference;
- raw and stitched reconstruction arrays;
- structured VarPro sufficient statistics, objectives, scale, and mask
  diagnostics sufficient to audit the solve;
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

## Run Fingerprint And Resume

Claim-grade execution runs from a clean checkout. The training fingerprint is
SHA-256 of canonical JSON containing schema version, manifest hash, logical
arm/run id, all resolved configs, seed, Git commit, clean-tree status,
environment lock/version digest, and every dataset/provenance/probe SHA-256.
The inference fingerprint additionally includes the selected checkpoint
SHA-256. Dirty checkouts are allowed only for smoke tests; their tracked patch
and relevant untracked-source hashes are recorded and their outputs are marked
non-claim-grade.

A run becomes complete by atomically renaming a temporary completion record
after all required artifacts have been hashed and listed. `--resume` reuses a
run only when both fingerprints, the completion record, and every required
artifact hash match. Incomplete runs restart in a new attempt directory;
corrupt or mismatched completed runs fail with an explicit `--rerun` remedy
rather than being overwritten or silently reused.

## Error Handling

Preflight rejects unknown config paths, type mismatches, invalid enum values,
CI with a non-Poisson primary loss, ambiguous or contradictory scale metadata,
missing physical-probe fields, dataset checksum/calibration failures, duplicate
run ids, conflicting overrides, output collisions, and fingerprint mismatches.

Runtime failures are captured with the failing stage and traceback. Partially
written runs use an incomplete status and are never treated as resumable
successes. Aggregate rendering tolerates failed arms while displaying them
prominently.

## Test Strategy

Unit tests cover TOML parsing, Cartesian expansion, include/exclude behavior,
dotted-path ownership, allowlist rejection, type coercion, typo suggestions,
deterministic run ids, CLI overrides, fingerprints, atomic completion/resume,
status-count/manual-review schemas, and manifest-defined comparisons and gates.

Physics/config tests cover default-CI resolution, explicit legacy selection,
CI-plus-MAE and CI-plus-supervised rejection, amplitude-mode CI-default
non-activation, physical probe protection, immutable calibrated datasets, and
checkpoint/config round trips.

Runtime tests assert explicit seed use and effective Trainer/dataloader/
reassembly values for device count, strategy, deterministic mode, precision,
workers, progress, checkpointing, and inference mixed precision. They also pin
the fixed `DatagenConfig` tuple member and full inference-config round trip.

Materialization tests cover deterministic checksums, no-overwrite behavior,
CI/legacy latent-object and coordinate parity, physical-probe/count co-scaling,
photon-floor enforcement, saturation checks, and provenance completeness.

Dataset-selection tests cover synthetic truth, experimental reference, and
experimental no-reference bundles; domain/profile incompatibility; unavailable
grouping or dose capabilities; and separation of truth-quality,
reference-agreement, and measurement-consistency metric namespaces.

Reassembly/metric tests cover multimode `|P|^2` weighting, `(x, y)` placement,
canvas padding, exact anchor-to-truth sampling, common-mask and largest-valid-
rectangle selection, patch inclusion, no-resize evaluation, separation of
normalized and absolute metrics, global-phase-only absolute comparison,
structured VarPro objectives/count reductions, and a perfect-reconstruction
oracle.

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
