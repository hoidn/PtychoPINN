# CI Model Compatibility Ablation Design

**Date:** 2026-07-09  
**Contract revision:** 2026-07-13

**Quality corrective revision 2026-07-11:** The sealed 30-run result is retained
as diagnostic evidence but its compatibility PASS is withdrawn. The accepted
quality contract was too weak, the study did not first reproduce the known-good
Hybrid ResNet integration baseline, and the matrix changed several material
variables at once. In particular, the lines Hybrid-CI median amplitude SSIM was
`0.6928` versus approximately `0.876` in the integration fixture, while both
CNN lines NLL arms produced constant-interior reconstructions. Tasks 19--24 and
30 in the implementation plan own the corrective bridge, isolation ladder,
optimization diagnosis, CNN contract recovery, rerun, and publication. No current artifact root is
claim-grade evidence for CI model compatibility until those tasks complete.

**Corrected-physics prerequisite evidence:** Canonical rung1a under unit
`dictionary_parity` passed at amp/phase SSIM
`0.8913340876617375`/`0.9632217816205675`; absolute deltas from rung0 were
`0.0054688232603687`/`0.0013551856818027`, inside locked `0.02`/`0.01` gates.
The fresh root is
`.artifacts/bridge_ladder/task28_gain16_seed3_unit_scaling_rerun`; rung
evidence/report SHA-256 values are
`a8741f93511cc90eb26777b1ff2cab0235ce812cff3661bad168632fce9ce711` and
`2f95c0e4651a2a48449797f4a34aed557ce70962c8e82dced0d40b4aa95a0cf6`.
This rung1a PASS is bridge-regression evidence, not evidence that the corrected
compatibility matrix has run or passed. Live task state and dependency routing
belong only to the implementation plan.

**CNN contract recovery revision 2026-07-13:** The Task 22 seed-3 convergence
pilot found every Hybrid ResNet profile recognizable at epoch 80 and every CNN
profile collapsed and saturated from epoch 5 onward. Follow-up tensor and
history audits identified two invalid inputs to the planned CNN claim: the
canonical-lines materializer treated the arbitrary magnitude of
`mk_lines_img` as object transmission, and the CNN configuration emitted one
real component channel against `C` imaginary component channels and relied on
implicit broadcasting. The fixed epoch-80 budget remains useful diagnostic
evidence, but the failed CNN arms do not authorize multi-seed execution. The
corrected claim path requires the architecture-independent physical lines
mapping, the generic per-patch component contract for `object_big` models, and
correct active-support saturation diagnostics before the protocol is locked
again.

**Performance-reference clarification 2026-07-11:** Reference alignment is a
reconstruction-performance requirement, not an internal-equivalence
requirement. Hybrid ResNet and CNN must reproduce their known high-resolution
grid-lines amplitude/phase SSIM within locked tolerances through the study
driver. Patch, canvas, mask, crop, probe, and checkpoint hashes remain required
provenance and debugging evidence, but exact equality between historical and
generic internals is not itself a compatibility gate. Hidden resizing, leaked
truth, or undeclared gauge fitting remain prohibited because they can inflate
SSIM.

**Revision 2026-07-10:** Claim-grade execution covers both canonical synthetic
Dead Leaves and canonical synthetic lines. Each family runs five explicit arms,
including the added CNN legacy-NLL control, for 30 runs total at three seeds.
The preliminary 12-run Dead Leaves result is retained only as superseded
development evidence. Required report figures must contain eligible plotted
evidence or an explicit not-applicable panel; empty axes cannot be sealed.

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

Passing the corrected study supports the bounded claim that Hybrid ResNet can train and
infer under the current CI contract on the selected Dead Leaves and lines
synthetic ptychography conditions. It does not establish universal architecture
compatibility,
superiority over MAE, experimental-data generalization, or compatibility with
every model family.

## Study Budget

The historical diagnostic study used five explicit arms for each of two object
families, three seeds, and twenty epochs per run: 30 runs total. The corrected
budget is not locked until the reference-performance qualification and convergence
pilot establish a defensible epoch/step budget. The final budget must be fixed
before the multi-seed rerun and may not vary by observed seed quality.
(Revision 2026-07-10: the previously planned post-training dose inference
sweep is removed — dose response is not a study parameter.) The intended
single-GPU runtime is projected from the measured preliminary study before
launch. The driver must support overriding seeds and epochs
from the command line without changing the manifest.

The convergence pilot is a one-seed lines-only matrix of exactly six arms:
Hybrid ResNet and CNN crossed with legacy MAE, legacy NLL, and CI NLL. Each arm
trains once, deterministically, for 80 epochs and emits exact post-epoch
milestones at 5, 20, 40, and 80. All four checkpoints are evaluated from that
single trajectory. Four independent trajectories, best-within-budget
substitution, and CI with MAE are prohibited. The pilot is diagnostic sidecar
evidence and does not itself change the claim-grade selected checkpoint or
verdict.

## Corrective Isolation Protocol

The corrected study proceeds through ordered gates; a failed gate blocks later
claim execution.

1. **Reference performance qualification.** Run the known-good N=128/C=1
   grid-lines Hybrid ResNet and CNN MAE configurations through the study driver.
   Require locked amplitude and phase SSIM floors derived from their historical
   results; retain MAE as a supporting guard.
2. **Diagnostic alignment.** Record loader, probe, mask, boundary, crop,
   checkpoint, patch, canvas, and valid-mask provenance for both the historical
   and study paths. Differences are diagnostic unless they invalidate a fair
   metric comparison, introduce hidden resize/gauge fitting, or violate the
   declared experimental condition.
3. **One-variable ladder.** Change only one of these groups per rung:
   loader/schema; reassembly/alignment; probe source and border transform;
   N=128 to N=64; `object_big=false`/C=1 to `object_big=true`/`C>1` plus training weighting; normalized-amplitude to
   count-intensity/Poisson; rectangular scaling and VarPro. Each rung retains a
   control from the preceding rung and records the first material degradation.
   The grouping rung must account for every source scan and fail on unintended
   omissions; inference must reuse training normalization statistics.
4. **Optimization diagnosis.** Train one deterministic 80-epoch trajectory for
   every required one-seed lines arm and evaluate exact post-epoch checkpoints
   5, 20, 40, and 80. Record the compact trajectory table and reconstruction
   grid defined below for each arm.
   Select one common final budget and convergence rule before viewing multi-seed
   results. Do not treat a still-decreasing loss at the budget boundary as
   convergence and do not replace a milestone with a validation-best checkpoint.
5. **CNN contract recovery.** Before a claim-grade matrix, replace arbitrary
   lines-image magnitude with a declared physical object mapping, enforce equal
   per-patch component shapes when `object_big=true`, and rerun the six-arm
   seed-3 qualification. A failed mandatory CNN arm blocks the matrix; it is
   not a diagnostic comparison that can be ignored.
6. **Corrected matrix.** Include Hybrid and CNN legacy-MAE, legacy-NLL, and
   CI-NLL controls wherever the physics contract permits. CI remains active
   only for rectangular Poisson NLL.

The Hybrid performance reference uses the integration test's Run1084 probe, including smoothing
sigma `0.5` and `pad_extrapolate` to N=128, Hybrid convolutional hidden scale
`2.0`, gridsize/C=1, central-mask training weighting, and historical stitching.
The current FLY/N=64/C=4/probe-weighted setup is a later ladder endpoint, not a
performance baseline. The CNN reference must likewise freeze the dataset,
probe, model, training, and evaluator provenance of the known high-resolution
grid-lines run before its SSIM floor is locked.

## CNN Component And Synthetic Object Contract

The CNN component-channel rule is:

```text
component_channels = C_model if object_big else 1
```

Both branches use that count for amplitude/phase and real/imaginary output, and
their shapes must match before complex combination. The observed `C=4` failure
is one instance of the generic `C>1` bug. Implicit broadcast is invalid; an
explicit override may exist only to inspect historical checkpoints.

The normal Torch `object_big=true` CNN path also requires
`model.probe_big=true`: each component head must learn the complementary outer
support of the full patch. `probe_big=false` is permitted only for an explicitly
named historical-checkpoint or zero-border diagnostic. Shared grouped-study
manifests keep the value true even when a non-CNN generator does not consume
that decoder branch.

`ptycho.diffsim.mk_lines_img` supplies morphology, not calibrated complex object
transmission. Normalize it to `t in [0,1]` and construct

```text
A = 0.3 + 0.7*t
phi = 0.5*(2*t - 1)
O = A*exp(i*phi)
```

This bounds amplitude to `[0.3,1.0]` and phase to `[-0.5,0.5]` without clipping
to a particular network. CI and legacy twins share truth, coordinates, probe
geometry, and splits. Existing Dead Leaves arrays do not change.

CI keeps `amplitude_physics_gain=1`; trainable `s1/s2` own training scale and
physical-probe VarPro owns inference scale. The legacy gain is a single
dataset-scale normalization derived from the exact amplitude forward, not an
architecture-, loss-, or reconstruction-quality hyperparameter.

For a sealed legacy training split with observed amplitude `Y`, truth patches
`O`, detector width `N`, effective loader probe `P_eff`, and Batch/Parseval
input/output scale `r`, define

```text
P_eff = normalize_probe_like_tf(P_stored, probe_scale) / probe_scale
A0 = fftshift(abs(sum_p FFT(O * P_eff[p]))) / N
r = sqrt(N^2 / mean_samples(sum_hw(Y^2)))
G_phys = r * sqrt(sum(Y^2) / sum(A0^2))
```

This is also the closed-form Poisson maximum-likelihood scalar because the
amplitude loss constructs `lambda=(q*G_phys*A0/r)^2` and
`k=(q*Y)^2`; the common loss-side physics scale `q` and final scalar loss
normalizer cancel. Compute `G_phys` once from the exact sealed training input
consumed by the qualification run,
record the input identities and factors, and use that same value for legacy MAE
and legacy NLL across CNN and Hybrid ResNet. Held-out test data does not enter
the derivation.

For the sealed Task 30 v3 lines training archive (SHA-256
`97e3933abf1ff27e443d1d0541e776ebb5e52c0d6edb2f2e3f2e3a744bdbf38f`),
the factors are
`N=64`, probe-normalization multiplier `27.67109393796515`, effective-probe
multiplier versus the stored probe `6.9177734844912875`,
`r=86.1417236328125`, and loss-side `q=0.023132076486945152`. The resulting
value is `G_phys=12.452229360013307`. A narrow numerical check evaluates the
same sealed truth forward once: amplitude least squares gives
`12.450350059079451` and the exact MAE weighted-median scalar gives
`12.450986331825641`, both within relative `2e-4` of the Poisson expression.

The historical flat-probe multiplier `B=16` is only the exact
broadcast-equivalent conditioner for a 16-sample batch. Initialization RMS
matching and gain sweeps measure initialization or training behavior, not
physical normalization; they are inadmissible for selecting this dataset-scale
constant and remain diagnostic history only.

Before corrected multi-seed execution, qualify all six seed-3 lines roles with
the physical mapping, symmetric component heads, and full-support policy. This
pre-qualification may combine fresh corrected Hybrid completions with earlier
support-on CNN metric records when their effective configs already contain all
three corrections. Those records retain distinct source-manifest identities
and must be described as prior compatibility evidence; they are not imported,
resealed, or presented as one aggregate report. The later claim-grade
multi-seed execution remains one coherent locked manifest. Both contract
changes are independently required, so promotion does not require a factorial
attribution study. If a CNN arm still fails, isolate that arm and change one
optimization variable at a time; do not broaden the matrix first.

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
--integration-bridge-evidence PATH
```

`--dry-run` prints the fully expanded run matrix, resolved config changes,
validation result, artifact directory, and estimated run count without loading
training data or allocating a GPU. `--only` selects arm ids or dimension
predicates. `--resume` skips completed runs only after validating their stored
run fingerprint and required artifacts. `--visual-review` imports a completed
machine-readable review record before aggregate verdict calculation.
`--rerun` archives a mismatched or completed attempt and starts a new attempt;
it never overwrites evidence in place.

Claim-locked execution with an integration-bridge requirement must receive one
sealed `--integration-bridge-evidence` artifact. The runtime parses and
adjudicates those bytes against the manifest requirement before dataset loading
or training, passes the typed evidence into claim eligibility, and records the
artifact path and SHA-256 in invocation evidence. A locked claim run without a
passing artifact fails preflight; post-hoc report repair is not an alternative.

`--dataset ID` selects a declared bundle. If `dataset` is a matrix dimension,
it filters that dimension; otherwise it replaces the base `dataset.id`.
`--dataset-spec PATH` loads one additional versioned dataset descriptor without
editing the study manifest, enabling a local experimental NPZ to use the same
ablation matrix. Descriptor ids must be unique and their paths/checksums become
part of every run fingerprint.

The historical checked manifest declared a closed claim-grade budget: seeds
`[3,17,29]`, twenty epochs, all ten logical arms, both checked synthetic families,
and no `--only` filter. The invocation record and report completion contain
`claim_grade_eligible: bool` plus a closed list of disqualifying reasons.
`--epochs`, `--seeds`, `--only`, dataset replacement, a dirty checkout, or an
incomplete arm/family selection makes the invocation non-claim-grade even when
it completes cleanly. Such development/smoke reports may calculate diagnostic
verdicts but must display `NON_CLAIM_GRADE` prominently and cannot publish the
bounded compatibility `PASS`. `--output-root`, fingerprint-identical
`--resume`, and importing the matching visual review do not alter eligibility.
The reason enum is exactly `epochs_override`, `seeds_override`,
`matrix_filter`, `dataset_override`, `external_dataset_spec`, `dirty_checkout`,
`manifest_budget_mismatch`, and `fixture_dataset`. Reasons are de-duplicated and emitted in this
order. A terminal failed run affects the typed verdict, not protocol
eligibility; it is not itself a disqualification reason.

The driver must not contain an architecture enum or architecture-specific
execution branch. It invokes the repository's canonical Torch model factory,
loader, trainer, checkpoint loader, and inference functions using resolved
configuration objects.

The checked Task 22 diagnostic manifest may declare:

```toml
[diagnostics.milestones]
epochs = [5, 20, 40, 80]
timing = "post_epoch"
```

Unknown timing values fail closed. The resolved milestone contract participates
in dry-run output and the training fingerprint. A generic CLI milestone
override is optional and not required by this design. Manifests without this
table preserve ordinary single-best-checkpoint execution exactly.

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
seeds = [3, 17, 29]

[schema]
version = 1

[base.overrides]
"training.epochs" = 20
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
physics_profile = "legacy_mae"
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
`median`, `mean`, `cv`, or `all_successful` aggregation where applicable.
`all_successful` is valid only for scalar `finite`, `ge`, or `le` gates and
evaluates every terminal-success attempt selected by the gate; it does not
reduce values. Missing operands ordinarily yield `INCONCLUSIVE`, never an
implicit pass. The deliberate exception is `all_successful`: a missing or
not-applicable operand on a terminal-success attempt is `FAIL`, while a missing
or incomplete requested attempt remains `INCONCLUSIVE`.

Dataset paths are repository-root-relative and carry expected hashes:

```toml
[datasets.deadleaves_ci_3p5m]
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
train = ".artifacts/ci_compatibility/datasets_v2/deadleaves_ci_3p5m_train.npz"
test = ".artifacts/ci_compatibility/datasets_v2/deadleaves_ci_3p5m_test.npz"
provenance = ".artifacts/ci_compatibility/datasets_v2/provenance.json"
train_sha256 = "..."
test_sha256 = "..."
provenance_sha256 = "..."

[datasets.deadleaves_ci_3p5m.probe]
source = "synthetic_calibrated"
calibration = "count_amplitude"
gauge = "physical_count_amplitude"
mask_policy = "model_config"
train_sha256 = "..."
test_sha256 = "..."

[datasets.deadleaves_ci_3p5m.dose.train]
counts_mean = 864.0
photons_per_image_min = 1000000.0
photons_per_image_mean = 3538944.0
max_observed_count = 7600
dtype_max = 65535
saturation_fraction = 0.0

[datasets.deadleaves_ci_3p5m.dose.test]
counts_mean = 864.0
photons_per_image_min = 1000000.0
photons_per_image_mean = 3538944.0
max_observed_count = 7600
dtype_max = 65535
saturation_fraction = 0.0
```

CI train and test splits are calibrated independently and therefore use
`train_sha256` plus `test_sha256`. Legacy twins use the `sha256` shorthand when
their canonical normalized train and test probe arrays are identical.

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

The standalone example also uses the shorthand form and therefore asserts
identical canonical train/test probes.

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

The superseded matrix contained these arms for each object family:

| Architecture | Profile | Role |
|---|---|---|
| Hybrid ResNet | CI rectangular + Poisson NLL | compatibility target |
| CNN | CI rectangular + Poisson NLL | architecture control |
| Hybrid ResNet | explicit legacy amplitude + Poisson NLL | legacy pipeline control |
| Hybrid ResNet | explicit legacy amplitude + MAE | recognizable-quality control |
| CNN | explicit legacy amplitude + Poisson NLL | architecture-matched scale-contract control |

CI scaling is active only when all of these are true: the resolved pair is
`ci_intensity_v2/count_intensity`, `model.mode="Unsupervised"`,
`model.physics_forward_mode="rectangular_scaled"`, and
`training.torch_loss_mode="poisson"`. Selecting rectangular CI with MAE,
supervised mode, or another primary loss fails before model or data
construction. Amplitude mode does not activate CI even when absent profile
fields receive CI defaults. Historical behavior requires the explicit
`legacy_v1/normalized_amplitude` pair.

### Historical Resolved Arms (Superseded)

The checked-in manifest pins these shared values: `N=64`,
`grid_size=(2,2)`, `C=C_model=C_forward=4`, `model.mode="Unsupervised"`,
`model.object_big=true`, `model.probe_big=false`,
`model.training_patch_weighting="probe"`, `model.probe_mask=false`,
`data.probe_normalize=true`, `data.probe_scale=4.0`, batch size 16, twenty
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
| `ci_nll` | `<family>_ci_3p5m` | CI/count | rectangular | Poisson/`poisson`, `nll=true` | true | true |
| `legacy_nll` | `<family>_legacy_amp` | legacy/amplitude | amplitude | Poisson/`poisson`, `nll=true` | false | false |
| `legacy_mae` | `<family>_legacy_amp` | legacy/amplitude | amplitude | MAE/`mae`, `nll=false` | false | false |

All arms used canonical probe-weighted inference stitching. The historical
matrix excluded CNN legacy MAE. Its valid set was five arms for each of `deadleaves` and
`lines`, times three seeds, for exactly 30 runs. This is an explicit valid-arm
set rather than an unconstrained Cartesian product.

## Normalization Ownership

Normalization is owned by the producer that establishes the measurement
domain. `ptycho.diffsim.illuminate_and_diffract` already conditions synthetic
grid-lines measurements: it multiplies the diffracted field by
`intensity_scale` before Poisson sampling and divides the measured amplitude by
the same scale. Those arrays are simulator-owned normalized amplitudes, not raw
experimental detector inputs. Re-expressing byte-identical arrays in the generic
mmap schema must not apply a second RMS or physics normalization.

The canonical generic bridge therefore declares
`mmap_scale_convention="dictionary_parity"`, which resolves to the already
supported `DataConfig.normalize="None"` mode and unit legacy
`rms_scaling_constant`/`physics_scaling_constant`. The dictionary baseline
carries the same ladder convention as an inert declaration, so its behavior is
unchanged. The ladder runtime resolves this convention once and supplies the
same concrete mode to both the prebuilt dataset payload and the internal
training overrides; persisted configuration may not disagree with the dataset
that training consumes.

`DataConfig` keeps its public `normalize="Batch"` default. Batch normalization
remains valid when a dataset or compatibility study explicitly selects the
loader convention for raw/legacy inputs. The CI/count contract is unchanged:
the count-domain rung explicitly restores the loader convention (`Batch` in the
persisted config), while CI statistics and count-intensity physics remain owned
by the CI path rather than this legacy-amplitude mapping.

Task 28 diagnostics established this ownership boundary. The generic twin and
dictionary measurements were byte-identical; the dictionary flow carried unit
legacy constants, while the original generic rung1a used Batch-derived RMS
approximately `1.33047` and physics scaling approximately `1.9797e-4`. That
rung failed at amp/phase SSIM `0.856505683935826`/`0.9498293416806348`; the
unit-scaling rung1c recovered to `0.8913340876617375`/`0.9632217816205675`, and
sampler-only controls passed. The diagnostic conclusion is normalization
ownership, not sampler behavior and not a reason to change global defaults or
CI/count semantics. The fresh canonical rung1a reproduced that recovery and
passed. Rungs 1c-1f are archived from the current TOML; the remaining
rung1b-through-rung8 scaffold plus historical injection/parser support are
conservatively retained until Task 29 producer retirement. Task 28 adjudicates
only rung0 versus canonical rung1a; retention is not remaining Task 28 work.

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

The required `probe` table is closed. It always contains only `source: str`,
`calibration: count_amplitude | legacy_normalized`,
`gauge: physical_count_amplitude | legacy_normalized`, and
`mask_policy: model_config | pre_masked`, plus exactly one array-identity form:

- `sha256` when canonical train and test physical-probe arrays are identical;
- `train_sha256` and `test_sha256` when calibration makes the split probes
  differ, including calibrated dose twins.

Mixed forms, partial split forms, and equal `train_sha256`/`test_sha256` values
are invalid. Equal split probes must use the `sha256` shorthand. For each
count-domain split, required `dose.train` and `dose.test` tables contain only
finite nonnegative `counts_mean`, `photons_per_image_min`,
`photons_per_image_mean`, integer `max_observed_count`, integer `dtype_max`, and
`saturation_fraction` in `[0,1]`. Dose tables are forbidden for
normalized-amplitude bundles. Unknown fields or tables fail.

This one-of keeps identical-probe base and standalone descriptors concise while
still allowing a dataset whose train and test probes differ to bind both hashes
without weakening array identity.

Preflight verifies file hashes; required keys; array shapes, lengths, dtypes,
and finite values; nonnegative integer count measurements; detector shape;
probe rank/mode count and the selected canonical split array-content hash;
coordinate convention; truth role/location; and exact agreement between
measured dose statistics,
descriptor values, and provenance. CI requires count calibration plus physical
count-amplitude gauge. Legacy requires the legacy pair and normalized probe
gauge. Capability derivation occurs only after these checks.

Compatibility-materializer NPZs additionally preserve the uncalibrated probe geometry
as canonical `(P,H,W)` array `probeGeometry`. Provenance records its exact
dtype/shape/content digest. Every calibrated split probe must be a positive-real
scalar multiple of that raw whole array within relative tolerance, with one
shared scalar across all modes so relative mode powers are preserved.
The provenance-v2 source-object digest also binds the latent full-object `objectGuess`
array in every train and test NPZ across all CI and legacy twins. Generic
experimental truth and reference roles remain separate and are not inferred
from runtime initialization arrays.

Descriptor-v1 grouping capability uses the initial study's canonical runtime
baseline: `DataConfig` defaults `neighbor_function='Nearest'`, `K=6`, `C=4`,
and `grid_size=(2,2)`, with `n_subsample=1`. Preflight deterministically checks
candidate viability from the Nearest/K=6 neighbor rows on both splits and
requires every row to contain at least `C` distinct in-bounds candidates. It
does not seed or sample exact groups; runtime sampling remains separately
seeded and configured. Alternate resolved grouping policies belong to Task 5
compatibility checks rather than descriptor-v1 capability inference.

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
  --dataset lines_ci_3p5m --only physics_profile=ci_nll

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
measured photons per image, and the governing dose quantity is mean measured
photons per image (per-pixel counts are an implementation detail of the
calibration). The N=64 study calibrates to a target mean of approximately 3.5
million photons per image (864 counts per pixel at N=64, i.e., 3,538,944) so
the weakest frame clears the floor with margin, and uses
5000 train and 1250 test scan positions (revision 2026-07-10: raised from
512/128 for visually adequate reconstruction quality; the previous
432-counts-per-pixel base violated the per-image floor on its dimmest frames).
Dataset preparation is a separate, deterministic prerequisite. The manifest
names train, validation/test, provenance, and expected SHA-256 values; the
driver verifies them and never regenerates or mutates data.

The claim-grade invocation selects two synthetic truth-bearing object families
because it must detect recognizable versus collapsed solutions, morphology
sensitivity, and absolute-scale error. Gates and report rows remain family-
scoped; values are never pooled across morphologies. A later experimental
invocation uses the same
conditional manifest with an experimental dataset id and produces a separate
dataset-scoped report; reports are never silently aggregated across dataset
kinds. A dedicated manifest is optional convenience, not a different contract.

### Materialization Prerequisite

Dataset creation is a separate command,
`scripts/studies/materialize_ci_compatibility_datasets.py`, not a driver mode.
It reuses the repository's calibrated synthetic truth/count builder and emits:

- `deadleaves_ci_3p5m` and `lines_ci_3p5m`: train and test count-intensity
  NPZs calibrated independently to a target mean of 3.5 million measured
  photons per image;
- `deadleaves_legacy_amp` and `lines_legacy_amp`: normalized-amplitude
  train/test twins from the corresponding identical latent object, scan
  positions, and uncalibrated probe geometry;
- one provenance JSON containing generator commit, seeds, source-object and
  coordinate digests, scale profiles, measured photons/image, count means,
  saturation checks, split physical-probe hashes and norms, the exact canonical
  `probeGeometry` digest, and every output SHA-256.

(Revision 2026-07-10: the `ci_tmc864`/`ci_tmc1728` dose-only test twins are
removed together with the dose sweep.)

Dead Leaves uses `ptycho_torch.datagen.objects.create_dead_leaves`. Lines uses
the canonical lower-level `ptycho.diffsim.mk_lines_img` texture as morphology.
The materializer normalizes it to `t in [0,1]` and constructs
`A=0.3+0.7*t`, `phi=0.5*(2*t-1)`, and `O=A*exp(i*phi)`. This keeps physical
transmission bounded while retaining nonzero phase contrast. Generation uses
an explicit local RNG contract that does not mutate ambient NumPy state.
Both families use 5000 train and 1250 test scan positions. Coordinates and raw
probe geometry are identical across families; only the latent
morphology changes.

The compatibility materializer does not call
`scripts/studies/grid_lines_torch_runner.py`. That runner is a training and
reporting consumer of cached grid-lines-workflow NPZs, whose
`diffraction/Y_I/Y_phi/coords_nominal` schema, legacy amplitude normalization,
scan geometry, and reassembly path differ from this study's canonical mmap and
CI contracts. Reusing the runner would confound morphology with data schema,
normalization, geometry, and execution. Instead, the materializer reuses the
same public line-morphology and phase helpers as the grid-lines workflow, then
passes the resulting complex object through the exact same coordinates, probe,
diffraction forward model, and CI/legacy twin conversion as Dead Leaves. The
materializer stops after immutable dataset/provenance publication. The generic
study runtime subsequently owns loading, training, and reassembly for both
families.

The seeded morphology oracle uses `object_resolution=320`, calls
`mk_lines_img(N=640, nlines=400)` with a local `numpy.random.RandomState` seeded
from `object_seed`, and crops `[160:-160,160:-160,0]` exactly as
`sim_object_image(size=320)`. Production then applies the declared rectangular
mapping above. The helper call must leave ambient NumPy state byte-identical;
the raw morphology hash and final complex-object hash are recorded before count
calibration.

The materializer is deterministic and refuses to overwrite an existing output
whose checksum differs. For each family, a parity check proves that the CI and
legacy bundles share object truth and coordinates and that the CI counts and
physical-probe amplitude are calibrated together according to the scaling
contract. Cross-family checks prove exact coordinate and raw-probe identity
while requiring distinct source-object digests.

The shared provenance file is closed at every level. `schema_version` is the
literal string `ci_compatibility_provenance_v2`; unknown or missing fields at
any level fail. Required shapes are:

- top level: `schema_version: str`, `materializer_id: str`,
  `materializer_version: positive int`, `generator_commit: 40-lowercase-hex`,
  `materialization_profile: claim_grade | fixture`,
  `expected_dataset_ids: list[str]`, `seeds: object`,
  `source_objects: object`, `coordinate_sets: object`,
  `probe_geometries: object`, and `datasets: object`;
- `expected_dataset_ids`: exactly the four sorted revised dataset ids;
- `seeds`: exactly `object: int`, `train_coordinates: int`,
  `test_coordinates: int`, and `measurements: object`; `measurements` maps each
  dataset id to exactly `train: int` and `test: int`;
- `source_objects`: exactly `deadleaves` and `lines`; each record contains
  `generator: create_dead_leaves | grid_lines_rectangular_v1`, `parameters: object`,
  `dtype: complex64`, `shape: [positive int, positive int]`, and
  `sha256: lowercase SHA-256`. Dead-Leaves parameters are exactly
  `max_iters: positive int`, `r_min_frac: float in (0,1)`,
  `r_max_frac: float in (r_min_frac,1)`, `r_sigma: positive float`,
  `phase_max: positive float`, and `seed: int`; claim-grade values are
  `700`, `0.02`, `0.18`, `3.0`, `0.5`, and the manifest object seed. Lines parameters are exactly
  `canvas_size`, `object_resolution`, `crop_start`, `crop_stop`, `nlines`,
  `mapping=rectangular_v1`, `amplitude_min=0.3`, `amplitude_max=1.0`,
  `phase_min=-0.5`, `phase_max=0.5`, and seed;
- `coordinate_sets`: exactly `shared_scan`, whose `train` and `test` records
  each contain `count: positive int`, `dtype: float32`, `shape: [count]`,
  `x_sha256`, and `y_sha256`;
- `probe_geometries`: exactly `raw_probe`, containing
  `array_key: probeGeometry`, `dtype: complex64`,
  `shape: [modes,height,width]`, and `sha256`;
- `datasets`: exactly the four dataset ids. Each record contains
  `family: deadleaves | lines`,
  `scale_contract_version: ci_intensity_v2 | legacy_v1`,
  `measurement_domain: count_intensity | normalized_amplitude`, and `splits`.
  `splits` contains exactly `train` and `test`; each split contains
  `path: str` as the normalized path relative to the shared provenance file's
  parent directory,
  `file_sha256`, `truth_sha256`, `xcoords_sha256`, `ycoords_sha256`,
  `raw_probe_sha256`, `stored_probe_sha256`, and `probe_scale: positive float`.
  Each split also records `stored_probe_l2_norm: positive float64`. CI splits
  additionally require a closed `dose` object with the six fields
  from the descriptor dose contract; legacy splits forbid `dose`.

Every digest is lowercase 64-hex SHA-256 over the canonical contiguous array
bytes after conversion to the declared dtype; shape is bound separately.

Claim-grade preflight verifies each staged file hash, recomputes all declared
truth, coordinate, and probe array hashes from the NPZs, checks them against
the provenance record, then enforces within-family CI/legacy truth-coordinate
identity and cross-family coordinate/raw-probe identity. A relationship failure
rejects the study before run expansion. The driver accepts only the resulting
immutable bundle ids and checksums.

`materialization_profile=claim_grade` additionally requires detector size 64,
object resolution 320, lines canvas size 640 with crop `[160:480]`, exactly
5000 train and 1250 test positions, and all claim-grade dose/probe policies.
Any other positive-size deterministic materialization is `fixture`; fixture
bundles remain valid for unit/integration tests but cannot satisfy claim-grade
eligibility. Moving the complete bundle directory is allowed because split
paths are bundle-relative; moving a file within the bundle without updating
and regenerating provenance is rejected.

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

### Task 22 Milestone Diagnostic Contract

Task 22 adds only an optional exact-epoch checkpoint capture path around the
existing architecture-neutral boundary
`runtime_execution.execute_canonical_run -> runtime_records`. It solves the
specific problem that one validation-best checkpoint cannot show whether an arm
was still improving, had converged, or had collapsed earlier. The capture path
emits post-epoch checkpoints 5, 20, 40, and 80 from one 80-epoch trajectory and
then sends each checkpoint through the existing canonical evaluator. It does
not duplicate inference or change ordinary `find_best_checkpoint` selection,
main run metrics, or verdict semantics.

External milestone epochs are one-based; checkpoint payload epochs are
zero-based, so milestones 5, 20, 40, and 80 bind payload epochs 4, 19, 39, and
79. A missing or off-by-one required checkpoint fails that arm's Task 22
trajectory. No per-milestone reload-parity duplication or nested sidecar seal is
required; the ordinary run-level checkpoint/reload contract remains in force.

The retained requirements and artifact scopes are intentionally narrow:

| Requirement | Problem solved | Required artifact scope |
|---|---|---|
| Checked six-arm seed-3 lines manifest, gain 1, 80 epochs, milestones 5/20/40/80 | Prevents arm, budget, gain, normalization, or dataset drift | One checked Task 22 TOML using current pinned `lines_ci_3p5m` and `lines_legacy_amp` datasets; legacy is explicit `Batch`, CI is count/rectangular/Poisson NLL, and CI+MAE remains invalid |
| Optional exact-epoch capture from one trajectory | Makes convergence shape observable without four independent trainings | Four milestone checkpoint files in each of the six Task 22 run directories; ordinary studies emit none |
| Canonical evaluator reuse | Prevents a Task 22-only inference path from changing reconstruction semantics | Existing evaluator outputs consumed by Task 22 collation; no new reassembly contract |
| Compact trajectory table | Puts the minimum convergence, quality, collapse, saturation, and CI-physics signals in one auditable view | One JSON and one CSV per arm with epoch, validation loss, LR, amplitude/phase SSIM, stitched amplitude standard deviation, centered phase variance, CNN rail occupancy when available, and CI Poisson NLL, relative count error, and fitted scales when applicable |
| Compact reconstruction grid and review | Catches recognizable collapse or saturation that scalar trends can obscure | One four-column milestone reconstruction grid per arm and one concise `recognizable/collapsed/saturated` review record per arm |
| Task 22 summary and common budget rule | Prevents per-arm or best-within-budget cherry-picking in the diagnostic pilot | One Task 22 summary linking the six tables/grids, recording run outcomes, and selecting one common budget/rule |

Task 22 reused the then-pinned N=64/C=4 lines dataset bytes. Every baseline
arm explicitly uses `model.amplitude_physics_gain=1.0`; Task 27 gain 16 is
scoped to N=128/C=1/Run1084 legacy amplitude and Task 28 `dictionary_parity` is
scoped to its N=128 bridge twin. Neither transfers to this N=64 baseline.

Raw-patch metrics, Pearson/FRC, gradient norms, detailed residual plots, typed
N/A objects, per-milestone reload parity, generic CLI milestone overrides,
nested sidecar schemas/seals, per-visual hashes, gain sweeps, and architecture
changes were optional for the Task 22 diagnostic. Their absence does not
invalidate its convergence result, but its failed CNN arms require the separate
CNN recovery prerequisite before Task 23. Existing CNN rail diagnostics are reused; changes to
`ptycho_torch/reassembly.py` require implementation evidence that those fields
are unavailable or insufficient.

The canonical stitched path remains unchanged: no truth-dependent alignment,
hidden resize, crop, gauge, placement, or weighting change is authorized by
Task 22.

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
| `stability` | `finite`, training/validation loss summaries and validation slope, learning-rate trajectory, checkpoint identity/epoch, `gradient_norm_mean`, `gradient_norm_median`, `gradient_norm_p99`, `gradient_norm_max`, `clip_fraction`, `amp_variance`, `phase_variance`, `cross_patch_cv`, `spatial_gradient_energy`, `reload_max_abs_error`, `reload_allclose`, `patches_accepted`, `patches_total`, `coverage_fraction`, source/filtered scan counts and utilization, and separate bounded-head rail saturation fractions |
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
- physical count-space Poisson NLL.

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
- patch inclusion and covered-pixel fractions;
- unique participating source-scan utilization, whose numerator includes every
  center/neighbor ID in the loader stream and whose denominator is every source
  scan in the held-out archive (`1250` for each claim dataset), before
  coordinate-bound filtering. Four-quadrant neighbors may lie outside the
  bounds-eligible center population. Filtered utilization is therefore defined
  separately as unique used center IDs divided by bounds-eligible center IDs;
  it cannot satisfy the source-completeness gate;
- mmap schema v3 requires `center_scan_id` and
  `center_scan_id_available` for every sample. Genuine legacy five-value loader
  results with two-element grouping records use sentinel `-1` plus availability
  `false`; they never fabricate centers. Participating/source utilization
  remains valid, while center count and filtered-center utilization are omitted
  as unavailable. Persisted v1/v2 stale maps must fail early with an explicit
  `remake_map=True` rebuild instruction;
- validation-loss slope plus the recorded learning-rate trajectory and selected
  checkpoint identity/epoch. Gradient logging is enabled in the checked
  protocol so convergence evidence includes gradient norms rather than inferring
  them from loss alone.

Architecture-specific rail checks apply only to architectures with a declared
bounded output head. They are mandatory for bounded CNN CI arms and
not-applicable to Hybrid ResNet's unbounded rectangular projection; stitched
canvas values may not be labeled decoder-head saturation evidence. Occupancy is
computed over the decoder's active support, never over zero padding around that
support. The CNN real
head has asymmetric rails at `-0.8` and `+1.2`; the imaginary head has rails at
`-1.2` and `+1.2`. Lower and upper occupancy are measured independently within
`0.001` of the corresponding head span. Each rail has a predeclared maximum
occupancy of `0.05`. This is intentionally strict: it rejects the historical
near-total upper-rail collapse while allowing a small population of legitimately
clipped pixels. A combined saturation fraction or a `0.95` threshold is invalid.

Claim qualification retains the existing gradient and output-variance
diagnostics alongside active-support rail occupancy. These are diagnostics for
explaining a failed quality run, not a second state machine: recognizable
reconstruction quality, rail occupancy, and the existing physics checks decide
whether the corrected run passes.

## Initial Compatibility Verdict

Verdicts are typed `PASS`, `FAIL`, or `INCONCLUSIVE`. A run succeeds only when
training, checkpoint reload, inference, metrics, and required artifacts finish.
The corrected successful-seed gate requires all three requested seeds. A
terminal failed attempt makes the gate `FAIL`; a missing or incomplete attempt
makes it `INCONCLUSIVE`. Other medians are evaluated only after this status gate
passes.

The corrected Hybrid ResNet CI gates are evaluated independently for Dead
Leaves and lines. No family verdict is evaluated until sibling performance
references reproduce the N=128, gridsize-1 grid-lines MAE quality for Hybrid
ResNet and CNN through the study driver. The Hybrid reference uses the Run1084
probe and five-epoch integration command; its fixture implies amplitude SSIM
`>=0.8409` and phase SSIM `>=0.9404`. The existing CNN reference is approximately
amplitude SSIM `0.886` and phase SSIM `0.928`; its final floors are locked from
the frozen reference artifact before execution. SSIM is the primary gate and
MAE is a supporting guard. Internal historical/generic equality is not required.
The reference uses the legacy dictionary loader: its transformed probe passes
through unchanged even though `DataConfig` declares `probe_normalize=true` and
`probe_scale=4`. The mmap loader would invoke `normalize_probe_like_tf` under
those same values and is a later ladder rung. Reference evidence records the
Run1084 raw archive/array hashes, pad-then-smooth transform and
transformed hash, dictionary/mmap normalization policies, disabled mask plus
`probe_mask_tensor=None` and identity resolved mask, inactive sigma/diameter,
model `edge_pad`, and grid-lines boundary/outer-offset parameters. A difference
does not automatically fail performance qualification; it must be surfaced and
classified as harmless, performance-relevant, or comparison-invalidating.

The prerequisite is execution-backed, not declaration-backed. Its sealed result
must include amplitude/phase MAE and SSIM, checkpoint identity, diagnostic
patch/canvas/mask hashes, declared crop and gauge handling, and a no-hidden-resize
assertion. It must also record the command operands: hidden scale `2`,
central patch weighting, legacy amplitude forward mode, MAE loss, seed `3`, and
five epochs. Copying the requirement mapping into an evidence object is invalid;
the execution-result schema is distinct and requires measured metrics plus
declared provenance.
The evidence schema version is exact, every declared artifact hash must be non-sentinel,
and eligibility accepts evidence parsed from sealed JSON bytes so the seal hash
is recomputed rather than declared by the producer.

The overall bounded compatibility verdict can pass only when both
family-scoped gate sets pass:

- the promoted CNN configuration uses equal `(B,C,H,W)` component shapes when
  `object_big=true`, and no historical component-broadcast override is active;
- all three requested seeds succeed under a predeclared convergence budget;
- every seed passes no-collapse checks for valid-mask standard deviation and
  dynamic range; bounded CNN heads additionally pass raw decoder saturation
  gates;
- scan utilization and canvas coverage satisfy predeclared completeness floors;
- lines median anchor-correct amplitude Pearson is at least `0.90` and
  amplitude SSIM is at least `0.75`;
- each architecture/family CI arm retains at least 85% of its matched
  legacy-MAE control's median amplitude SSIM, with an absolute SSIM floor of
  `0.50`; the corrected matrix includes CNN legacy MAE;
- amplitude and phase quality are both reported. Phase thresholds and the
  claim-grade epoch budget are locked from the bridge and one-seed pilot before
  the multi-seed rerun;
- quality is reported before and after VarPro. A lower fitted count objective
  does not waive the post-VarPro recognizability gates;
- absolute amplitude NRMSE, mean ratio, and declared quantile ratios have
  numeric bounds. Merely being finite is insufficient;
- held-out physical-count relative L2 error is bounded relative to the
  truth-forward Poisson-noise oracle for the same split, using a multiplier
  locked before the multi-seed rerun;
- checkpoint reload reproduces the pre-reload fixed-batch texture and stitched
  canvas with `rtol=1e-5, atol=1e-6`; runtime emits
  `stability.reload_allclose=1.0` only when both comparisons pass and
  `stability.reload_max_abs_error` as the maximum across both arrays;
- physical-probe VarPro is finite, passes the objective non-increase rule, and
  produces bounded `truth_quality.absolute_*` and
  `measurement_consistency.*` metrics;
- a manual review approves the shared-limit reconstruction/error grid and
  records whether line/object structure is recognizable and whether flat,
  checkerboard, mirrored, or saturation/collapse artifacts are present.

Task 22 locked epoch 80 as the common diagnostic budget, but its failed CNN arms
invalidate the subsequently frozen claim protocol. Before Task 30 executes, set
`budget_threshold_contract_locked=false`; restore it only after the corrected
dataset hashes, component contract, derived legacy normalization,
phase-quality thresholds, and Poisson-oracle multiplier are re-pinned together.
An unlocked invocation is
ineligible for a compatibility claim even if every numeric diagnostic passes.

The truth-forward Poisson oracle uses the exact ordered source-scan ID stream
consumed by the count-metric loader, including grouping multiplicity. The sample
IDs and digest are persisted in run evidence. Oracle FFT results are cached by
immutable dataset identity plus sample-policy digest so seeds and architectures
sharing a dataset/stream do not repeat the same truth propagation.

Structural report grids normalize reconstruction amplitude using only the
common-valid mask. Reconstruction, truth/reference, and error are masked and
cropped identically before display; full-canvas means are not admissible.

Reports contain separate structural/gauge-normalized and absolute-scale views.
Absolute shared limits remain required for scale claims, but they may not be the
only visual used to judge morphology.

For each family, bounded absolute-scale and physical-count gates plus finite
VarPro and exact reload
gates select the Hybrid ResNet CI arm and use `all_successful`: every terminal
successful seed must provide a finite applicable operand and satisfy the gate.
All three successful seeds are required by the status gate. A terminal
successful seed with a missing, not-applicable, nonfinite, or failing operand
makes that family gate `FAIL`; a requested seed that is missing or incomplete
makes the family result `INCONCLUSIVE`; completed failed attempts remain
governed by the status-count rule. The reload gate requires
`stability.reload_allclose == 1.0` for every successful seed. Median Pearson,
SSIM, and paired-ratio gates retain the aggregation rules stated above.

The report also computes family-scoped diagnostic comparisons for Hybrid CI
versus Hybrid legacy NLL, CNN CI versus CNN legacy NLL, Hybrid CI versus Hybrid
legacy MAE, CNN CI versus CNN legacy MAE, and Hybrid CI versus CNN CI. These
comparisons have no model- or
loss-superiority gate. No metric is aggregated across object families.

Any evaluated mandatory numeric or visual gate that fails yields `FAIL`.
Missing attempts, matched controls, diagnostics, or manual review
yields `INCONCLUSIVE`; completed failed attempts are adjudicated by the status
gate above. `visual_review.json` contains schema version, reviewer, UTC
timestamp, reviewed figure SHA-256, and one closed review record for each of
`deadleaves` and `lines`. Each family record contains `approve|reject`,
`recognizable: bool`, `flat: bool`, `checkerboard: bool`, `mirrored: bool`,
`saturation: bool`, `collapse: bool`, and notes. Family approval requires
`recognizable=true` and every failure-mode field false; overall approval
requires both families. The driver writes a pending template but never
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

Task 22 milestone runs additionally contain four exact-epoch checkpoints. Its
collation writes one trajectory JSON/CSV pair and one compact reconstruction
grid per arm, plus one study summary. These artifacts are diagnostic and do not
replace the ordinary run artifact or verdict contract.

The study root contains:

- `report.md` with the bounded compatibility conclusion;
- aggregate tidy CSV and JSON;
- arm-by-seed status and verdict tables;
- reconstruction/truth/error grids with shared row color limits;
- training and gradient curves;
- seed-distribution plots;
- VarPro plots sourced directly from typed `s1` and `s2` metric records, plus
  manuscript-derived `c_A=sqrt(s1^2+s2^2)` and `c_phi=atan2(s2,s1)`;
- an absolute-scale/stability dashboard;
- machine-readable manifest expansion and exclusion records.

The absolute-scale dashboard plots amplitude mean ratio against the target line
at one and absolute amplitude NRMSE. Its physics/stability panel uses the
VarPro fitted-to-unit objective ratio and checkpoint reload maximum error;
exact zeros are annotated. Gradient history is shown only in the dedicated
training-curves figure and its absence when `training.log_grad_norm=false`
cannot empty another required figure.

Plots must label normalized versus absolute quantities and group rows by object
family. The report links every figure row to its resolved run id. Every required
figure must either contain all expected eligible run ids or render a visible
`Not applicable` panel with a machine-readable reason. Eligible typed metrics
with zero plotted marks are a report-generation error; empty axes cannot be
published or included in a completion record.

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

Study-level publication follows the same transaction boundary. Frozen source
manifest/config copies, `invocation.json`, `expansion.json`, aggregate tables,
figures, semantic sidecars, report, verdicts, and their hashes are all staged
before `report_completion.json` is replaced last. A publication failure
restores the previous complete bundle byte-for-byte. `--rerun` archives the
prior run evidence and invalidates any previous visual review, reviewed-figure
hash, copied figures, and semantic sidecars; new numeric evidence can never be
paired with an old visual approval. Plain `--resume` may preserve a review only
when the complete reviewed figure hash and all source run fingerprints remain
identical.

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
canonical seeded Dead Leaves and lines generation without ambient RNG mutation,
within-family CI/legacy latent-object and coordinate parity, cross-family
coordinate/raw-probe identity, physical-probe/count co-scaling, photon-floor
enforcement, saturation checks, and provenance completeness.

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

Reporting tests require visible typed VarPro marks, manuscript-derived scales,
family grouping, dashboard evidence when gradient logging is disabled,
not-applicable annotations, empty-required-figure rejection, rerun visual-review
invalidation, and atomic inclusion of invocation/expansion provenance.

Integration tests run a tiny two-arm, one-seed, one-epoch manifest through
training, checkpoint reload, canonical reconstruction, metrics, resume, and
report generation. The full 30-run GPU study is execution evidence, not a
routine CI test.

Task 22 tests cover the checked six-arm manifest, exact post-epoch capture from
one trajectory, canonical evaluator reuse, ordinary best-checkpoint invariance,
the compact trajectory columns, and four-column reconstruction grids.

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
dry-run expansion is independently auditable, the corrected six-arm/two-family/
three-seed study finishes or records explicit failures, all required artifacts
are generated, and the report gives a bounded Hybrid ResNet CI compatibility
verdict for both Dead Leaves and lines supported separately by physics,
quality, stability, and visual evidence.
