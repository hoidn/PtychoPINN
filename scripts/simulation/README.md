# PtychoPINN Synthetic and Simulation Workflows

For new synthetic PyTorch work, use `ptycho_synthetic`. It is the supported
generic runner for simulation, grouping, training, strict model reload,
reconstruction, and evaluation. The lower-level simulation tools remain useful
when only a prepared dataset is needed.

## Supported Generic PyTorch Runner

### Default full workflow

The shortest complete invocation resolves the coherent
`synthetic-lines` profile and runs all four stages:

```bash
ptycho_synthetic --output-root outputs/synthetic-lines
```

The equivalent source-tree entry point is:

```bash
python -m scripts.simulation.synthetic_pipeline \
  --output-root outputs/synthetic-lines
```

The default is a real 50-epoch run, not a smoke test.

### Profile selection and precedence

A profile is a resolver-registered named starting bundle. It is expanded before
ordinary configuration objects are constructed and validated. "Preset" is an
informal description that may also refer to an unregistered combination of
configuration controls; there is no separate generic preset registry or
resolver.
The runner registers two profiles:

| Profile | Recipe | Purpose |
|---|---|---|
| `synthetic-lines` | `synthetic-lines-v2` | Default legacy normalized-amplitude CNN lines workflow |
| `cnn-lines-ci` | `cnn-lines-ci-v2` | Count-intensity Poisson CNN lines workflow with dose-closure startup |

With no selection, the runner uses `synthetic-lines`. A YAML, TOML, or JSON
workflow may select either profile with the root `profile` field; explicit
`--profile` wins. Value precedence is:

```text
selected profile < --config file values < explicit CLI values
```

The profiles are overrideable starting bundles with no profile-specific lock
set; final values still pass the coherent scaling, measurement, forward-model,
and loss validators. The config filename and path never select a profile.

The workflow file passed to `ptycho_synthetic` is named `--config` and may be
YAML, TOML, or JSON. This is different from the simulation-only
`simulate_and_save.py --simulation-config` interface described later.

### GS2/custom-probe example

This ordinary five-epoch example selects grid size 2 (`C=4`), the established
legacy custom-probe transform, all 4,096 train frames, and independent train
and validation group counts. It is not one of the sealed quality gates; those
are CNN GS1/C1, GS2/C4, and C4-CI (count-intensity), documented in
`docs/TESTING_GUIDE.md`.

```bash
ptycho_synthetic \
  --output-root outputs/synthetic-lines-gs2 \
  --gridsize 2 \
  --epochs 5 \
  --probe-source custom \
  --probe-path datasets/custom_probe.npz \
  --probe-transform 'pad_extrapolate:128|smooth:0.5' \
  --train-patterns 4096 \
  --test-patterns 1024 \
  --train-raw-selection 4096 \
  --training-groups 4096 \
  --validation-groups 1024 \
  --neighbor-count 4 \
  --neighbor-pool-size 4 \
  --groups-per-center 1
```

Simulation-time and model-time probe masks both default off, `pad_object`
remains enabled, and the GS2 geometry derives grouped-patch/relative-overlap
object assembly with probe weighting. The runner fails before training if the
selected architecture, `N`, grid size, channel count, probe shape, or grouping
requirements disagree.

### Config file and partial-stage replay

The same recipe can be authored as a structured workflow document:

```yaml
# configs/synthetic-gs2.yaml
profile: synthetic-lines

simulation:
  N: 128
  gridsize: 2
  train_patterns: 4096
  test_patterns: 1024
  probe:
    source: custom
    source_path: datasets/custom_probe.npz
    transform_pipeline: "pad_extrapolate:128|smooth:0.5"

training:
  train_raw_selection: 4096
  training_groups: 4096
  validation_groups: 1024
  neighbor_count: 4
  neighbor_pool_size: 4
  epochs: 5

inference:
  groups_per_center: 1

workflow:
  output_root: outputs/synthetic-lines-gs2
  accelerator: auto
  devices: 1
```

Run the whole document with:

```bash
ptycho_synthetic --config configs/synthetic-gs2.yaml
```

Or split an exact replay across commands while retaining the same config and
output root:

```bash
ptycho_synthetic --config configs/synthetic-gs2.yaml \
  --stages simulate,train

ptycho_synthetic --config configs/synthetic-gs2.yaml \
  --stages reconstruct,evaluate
```

`--stages` accepts a unique, ordered subsequence of
`simulate,train,reconstruct,evaluate`. A later stage requires every predecessor
to be complete under the selected output root, even when that predecessor is
not selected in the current command.

### Profile semantics and defaults

This section is the operational profile summary. The
[configuration guide](../../docs/CONFIGURATION.md#profiles-and-presets) owns the
profile-versus-preset semantics. The default named profile is
`synthetic-lines`, recipe
`synthetic-lines-v2`. A second profile, `cnn-lines-ci`
(recipe `cnn-lines-ci-v2`), selects the count-intensity Poisson
contract and `model.rect_s1s2_init=dose_closure`. The amplitude profile keeps
`rect_s1s2_init=ones`. Every resolved field is written to
`resolved_workflow.json`; this
table highlights the user-facing defaults:

| Area | Default |
|---|---|
| Simulation | `N=128`, `gridsize=1`, seed 3, nongrid scan, buffer 64, `position_layout=uniform_random` |
| Object | Shared 392×392 registered object producer; default `lines` / `lines-object-v1`, `set_phi=true` |
| Probe | Ideal source, scale 0.7, `smooth:0.5|pad_preserve:128`; simulation mask off |
| Raw frames | 4,096 train, 1,024 test; normalized-amplitude `legacy_v1` |
| Sampling | `dictionary_parity` adapter; 4,096 selected train frames; 1,024 train groups; 1,024 validation groups; neighbor/pool size 4; oversampling off |
| Model | Unsupervised `cnn`, real/imaginary output, model mask off, geometry-derived layout, derived amplitude physics gain |
| Training | 50 epochs, batch 16, Adam `2e-4`, plateau scheduler, MAE with prediction-L2 matching |
| Inference | Batch 16, probe-weighted barycentric assembly, VarPro on, `groups_per_center=1` |
| Execution | One auto-selected device, deterministic FP32, zero workers, CSV logger, one best checkpoint |

### Scan position layout

`--scan-position-layout` selects how scan positions are placed inside the
buffered object extent:

| Value | Behavior |
|---|---|
| `uniform_random` (default) | Positions drawn uniformly at random from the split's coordinate seed stream. |
| `raster` | Span-filling square grid: `pitch = (extent - 2*buffer) / (side - 1)`, row-major, no jitter, no randomness consumed. |
| `fixed_pitch_raster` | Exact-slice row-major square lattice: legacy translation origin `N // 2`, split pitch `outer_offset_train/2` or `outer_offset_test/2`, no jitter or position randomness. |

Both raster modes require a perfect-square pattern count per object (`side =
sqrt(M)`), at least 4 positions, and full patch fit in the source canvas. The
sealed 4,489 / 729 per-object counts qualify (67² and 27²); a non-square count
is rejected up front with the nearest squares named. Both splits must share the
layout. For even `N`, the geometric pixel center corresponding to the first
`fixed_pitch_raster` translation is `N / 2 - 0.5`; persisted flat coordinates
remain in the translation frame.

The layout is default-elided from recipe and workflow digests, so adding it
did not move any pre-existing identity; a `raster` workflow gets its own
digest. The dataset manifest records the layout and, for `raster`, the
realized per-axis pitch under `scan_geometry`.

```bash
ptycho_synthetic \
  --profile cnn-lines-ci \
  --scan-position-layout raster \
  --train-patterns 4489 --test-patterns 729 \
  --output-root outputs/raster-ci
```

This is the nongrid simulation pipeline emitting a grid layout; it is
distinct from the TensorFlow grid simulation pipeline described in
`docs/DATA_GENERATION_GUIDE.md`.

### Count-intensity contract flags

`--scale-contract-version`, `--measurement-domain`, `--physics-forward-mode`,
`--cnn-output-mode`, and `--torch-loss-mode` select the measurement/objective
contract. The units triple is inseparable: `ci_intensity_v2` +
`count_intensity` + `rectangular_scaled` must be selected together with
Poisson loss, and any partial combination is rejected naming the offending
field. `--profile cnn-lines-ci` selects the whole coherent set.

### CI rectangular gauge initialization

The CI profile also initializes rectangular `s1`/`s2` from the fixed
representative sample of exactly 256 logical detector slots before fitting:

```bash
ptycho_synthetic \
  --profile cnn-lines-ci \
  --rect-s1s2-init dose_closure \
  --output-root outputs/ci-dose-closure
```

`--rect-s1s2-init ones` restores exact unit initialization. Bare Torch
`ModelConfig` defaults to `ones`; this synthetic CI profile and the
training-only `ci` profile opt into
`dose_closure`. The solve runs the actual resolved forward with a unit complex
object. It is named *dose closure* because it computes
`c* = sum(measured counts) / sum(predicted unit-object intensity)` and sets
`s1=s2=sqrt(c*)`, closing the sampled predicted/observed count totals at
startup. It improves conditioning but does not calibrate the stored probe or
identify physical object units. The
[core contract](../../docs/specs/spec-ptycho-core.md#ci-rectangular-gauge-initialization-normative)
owns the fixed seed/draw policy, logical population, no-fallback behavior, and
record compatibility.

#### Result persistence and historical evidence

Each invocation records its command in `OUTPUT/invocation.sh`. The training
stage writes the startup gauge to `OUTPUT/training/training_summary.json`, and
the evaluation stage writes `OUTPUT/reconstruction/metrics.json`. Fresh
summaries use
`rect-s1s2-initialization-v2`; dose closure records the positive finite gauge,
method `dose_closure_seeded_uniform_unit_object`, and exactly 256 sampled
slots. `ones` records the unit no-solve result. Strict v1 reading is retained
only for prefix-era artifacts.

Post-training learned `s1`/`s2` and reconstruction-time VarPro are separate
values. On the grid-lines path, the CI adapter derives
`ci_count_amplitude_scale` independently; the legacy-only `count_scale_mode`
flag is ignored when CI is active.

The original five-epoch Phase 1 metrics and command used the v1 prefix policy.
They remain available as historical evidence in the
[preserved plan](../../docs/plans/2026-08-04-ci-gauge-invariant-scaling.md) and
[durable finding](../../docs/findings.md#ci-gauge-initialization-001---dose-closure-selects-a-startup-gauge-not-physical-calibration),
not as a current v2 quality gate or runnable reproduction recipe.

Probe transform defaults are source-aware:

- An ideal or already-`N` custom probe uses
  `smooth:0.5|pad_preserve:N`.
- A smaller custom probe uses `pad_extrapolate:N|smooth:0.5`.
- A larger custom probe fails unless an explicit supported downsampling
  transform is supplied.

### Flat storage and grouped model samples

Simulation persists canonical flat acquisitions. Each train/test NPZ contains
`diff3d` with shape `(M, N, N)`, one `xcoords`/`ycoords` value per frame, the
transformed training `probeGuess`, the exact acquisition illumination
`probe_simulated`, per-frame truth `Y`, and `scan_index` plus `object_index`
vectors. `scan_index` retains its acquisition/probe meaning and may repeat;
`object_index` identifies the independent object canvas for grouping. A
single-object split also stores `objectGuess`; a multi-object split binds its
rows to the source object bank instead of collapsing that bank into one array.
The manifest's array digests and row order pin acquisition identity. Flat
storage does **not** imply `gridsize=1`.

The shared loader is the only owner that groups these rows for the model. For
grid size `g`, each model sample has `C = g ** 2` distinct raw-row/channel
indices. Pre-grouped arrays must not be passed back through this boundary.

The sampling flags name separate lifecycle decisions:

| Flag | Meaning |
|---|---|
| `--train-patterns` / `--test-patterns` | Raw frames physically generated in each split |
| `--train-raw-selection` | Train frames selected before grouping |
| `--training-groups` | Exact grouped samples built for training |
| `--validation-groups` | Exact grouped samples independently built from the complete test acquisition |
| `--neighbor-count` | Neighbor candidates queried for ordinary grouping |
| `--neighbor-pool-size` | Candidate pool for explicit oversampling |
| `--groups-per-center` | Reconstruction-only repeated groups per eligible center |

Structured configuration also accepts `simulation.train_objects` and
`simulation.test_objects`. Pattern counts are split totals and must divide
evenly by the corresponding object count. `shared_object=true` requires the
default 1/1 bank. Reconstruction/evaluation currently require one test object.

The legacy profile's `training.data_adapter=dictionary_parity` supplies raw
stored amplitudes and `probeGuess` with unit RMS/physics factors. The CI profile
uses `data_adapter=loader`. A positive
`simulation.probe.simulation_normalization_scale` creates a distinct
`probe_simulated` with the versioned legacy rule, while
`training.torch_training_seed` pins Torch initialization independently of the
simulation seed.

`simulation.object.patch_amplitude_normalization=mean_patch_max` computes one
scale independently per split as `mean_i(max_xy(abs(Y_i)))`, pooling all frames
and objects. It divides `Y` and diffraction amplitude before count conversion
and persists the positive float64 `object_amplitude_scale`. This option requires
`fixed_pitch_raster` and strict tiled reconstruction.

The persisted training `DataConfig.n_raw_frames_selected` records
`train_raw_selection` (4,096 in the profile). Reconstruction starts from the
strictly loaded persisted `DataConfig` and threads `groups_per_center` to the
dataset constructor as an explicit runtime argument (default 1, no dataclass
field round-trip); it never rewrites the saved train selection.

### Stage identity and reuse

Completed stages are reusable by default. Reuse is fail-closed:

- every stage compares `schema_version`, `profile`, and `recipe_version`;
- simulation compares the resolved `simulation` namespace;
- training compares `simulation`, `model`, and `training`;
- reconstruction/evaluation also compare `inference`;
- evaluation additionally checks
  `metric_contract_version=synthetic-quality-metrics-v1`;
- the `workflow` namespace, including execution controls, is excluded from
  stage identity;
- each required stage-manifest entry and artifact path must be complete;
- NPZ content is verified separately through recorded array and file digests;
- a required identity mismatch or a partial artifact requires a new output
  root rather than an in-place overwrite.

The complete `ResolvedSyntheticWorkflow` is persisted as
`resolved_workflow.json`, including `schema_version`, `profile`,
`recipe_version`, the derived `data` namespace, and every resolved simulation,
model, training, inference, and workflow value. Reuse compares the
stage-specific portions listed above, not the spelling of the invocation or
config path.

Stage selection, output-root spelling, and the reuse switch itself are not
scientific identity. Downstream-only settings therefore do not redefine an
already complete simulation, but an exact replay must retain the complete
identity required by every selected stage.

The current manifest schema is `synthetic-stage-manifest-v2`; its completed
training entry requires both `training/wts.h5.zip` and
`training/training_summary.json`. Reuse strictly parses the initialization
record and requires its mode to match the resolved workflow. Historical
`synthetic-stage-manifest-v1` roots do not contain this contract: use a new
output root or retrain rather than trying to reuse them.

### Reconstruction and stitching

The workflow always disables the older generic `do_stitching` route. That path
reduces multiple predicted channels at group centers and is not a valid
multi-channel reconstruction. `inference.reconstruction_method` selects one of two
public mmap-backed adapters:

- `barycentric` is the general coordinate-aware path. It retains all `C`
  channel indices and supports probe or uniform weights.
- `tiled` requires GS1, one test object, `fixed_pitch_raster`, one group per
  center, uniform weights, and complete source-row coverage. Its tile size and
  pitch are `outer_offset_test/2`; `outer_offset_test` must be divisible by four
  and no larger than `2*N`.

CI requires VarPro for both methods. Tiled output stores
`measurement_gauge_canvas` for fitted count diagnostics and publishes
`complex_canvas` in the raw-source object gauge after applying any split object
scale exactly once. `metric_crop_border` affects only the aligned metric mask.

### Output tree

One complete run writes:

```text
OUTPUT/
  invocation.json
  invocation.sh
  resolved_workflow.json
  stage_manifest.json
  stage_logs/
    simulate_request.json
    simulate.log
  datasets/
    source.npz
    train.npz
    test.npz
    manifest.json
  training/
    wts.h5.zip
    training_summary.json
    effective_runtime.json
    checkpoint_selection.json
    checkpoints/
      <monitored-best>.ckpt
      last.ckpt
    lightning_logs/
  reconstruction/
    reconstruction.npz   # complex_canvas + measurement_gauge_canvas
    metrics.json
    diagnostics.json
    comparison.png
```

Invocation and resolved-input records are written before expensive work.
Stage and dataset manifests use relative artifact paths and are updated
atomically. The runner owns managed descendants beneath `--output-root`; it
does not recursively delete an arbitrary caller directory.

## Retained Simulation-Only Tools

### Two-stage architecture

The low-level simulation workflow remains modular:

1. Prepare an NPZ containing the ground-truth `objectGuess` and `probeGuess`.
   The source may be experimental data, a previous reconstruction, or a
   programmatically generated object/probe.
2. Run `simulate_and_save.py` with a canonical `SimulationConfig` to generate
   diffraction and coordinate arrays.

When `simulation.probe.source_path` is set, that archive supplies the probe.
The Stage-1 `probeGuess` remains required by the input schema but is not a
competing owner.

| Entry point | Status and purpose |
|---|---|
| `ptycho_synthetic` / `synthetic_pipeline.py` | **Supported default:** complete generic PyTorch synthetic workflow |
| `simulate_and_save.py` | **Supported low-level tool:** simulate from a prepared object/probe NPZ |

The simulation-only CLI keeps its own canonical simulation-file flag:

```bash
python scripts/simulation/simulate_and_save.py \
  --simulation-config configs/lines64.toml \
  --input-file prepared_lines64.npz \
  --output-file outputs/lines64.npz
```

`simulate_and_save.py` consumes the `objectGuess` already present in
`--input-file`; it does not synthesize a requested object family. This
restriction applies to that low-level tool, not to the registered object
producers in `ptycho_synthetic`. Prepare unsupported kinds such as
`natural_patch` with their dedicated producer first.
`grf` is not a supported `SyntheticObjectConfig.kind` and must not be recorded
under another label.

### SimulationConfig example and probe transforms

This complete low-level configuration uses the available boundary-matched
probe transform. It is a distinct dataset recipe, not the GS2 example shown
earlier:

```toml
[simulation]
N = 128
seed = 3

[simulation.probe]
source = "custom"
source_path = "datasets/custom_probe.npz"
transform_pipeline = "smooth:0.5|pad_extrapolate_boundary_matched:128"

[simulation.object]
kind = "lines"
image_size = [392, 392]
objects_per_probe = 1
diffractions_per_object = 2000
set_phi = true

[simulation.scan]
kind = "nongrid"
grid_size = [1, 1]
offset = 4
outer_offset_train = 8
outer_offset_test = 20
train_groups = 1
test_groups = 1
buffer = 64

[simulation.detector]
photons_per_pattern = 1e9
```

Probe transform meanings:

- `pad_extrapolate:N` is the legacy global quadratic phase, including the
  center, followed by any later operations.
- `smooth:0.5|pad_extrapolate_boundary_matched:N` smooths at source resolution,
  preserves that complex center exactly, and applies the C0
  boundary-conditioned outer phase only outside it. The boundary-matched
  operation is terminal.
- `pad_preserve:N` center-pads the complex probe.
- `interp:N` interpolates real and imaginary parts.

Changing a probe pipeline or simulation-time mask changes dataset identity.
The simulation-time mask is separate from the model-time support prior.
`simulate_and_save.py` records `simulation_config_sha256` and
`dataset_recipe_sha256` and rejects mismatched reuse at an existing explicit
output path.

## Output Data Contract

Generated NPZ files conform to the project's standalone data contract. See the
[data contracts](../../specs/data_contracts.md) for required keys and shapes,
and the [Data Generation Guide](../../docs/DATA_GENERATION_GUIDE.md) for grid
versus nongrid programmatic APIs.

### Object producer selection

Synthetic object generation is selected by the pair
`simulation.object.kind` and `simulation.object_recipe`. The runner dispatches
that pair through one producer registry. Generated recipes receive explicit
seed-derived NumPy streams; a source-backed recipe receives a validated object
bank. An unsupported or mismatched pair fails before simulation.

Registered recipes are:

| Object kind | Recipe |
|---|---|
| `lines` | `lines-object-v1` |
| `dead_leaves` | `dead-leaves-object-v2` |
| `lines` or `dead_leaves` | `frozen-object-bank-v1` |

`--object-kind` derives the current default recipe for that kind. A structured
workflow may state both fields explicitly:

```yaml
profile: synthetic-lines

simulation:
  object_recipe: dead-leaves-object-v2
  object:
    kind: dead_leaves

workflow:
  output_root: outputs/synthetic-dead-leaves
```

Dead Leaves v2 derives independent named geometry and material RNG streams and
uses a fixed phase law (`max=1.1`, `mean=0.95`), so one object's phase does not
depend on its split or bank companions. The backend-qualified v1 recipe remains
available for seeded compatibility work, but it cannot recover a historical
caller that left Python geometry randomness unseeded. Dataset manifest v3 records RNG and phase identities,
the realized hash, and fixed-raster morphology diagnostics. The diagnostics do
not replace the reconstruction quality preflight.

For an unreconstructible historical canvas, use `frozen-object-bank-v1` and
set `simulation.object.source_path` to an NPZ containing exactly finite
complex64 `trainObjectGuess` and `testObjectGuess` banks. Manifest v4 binds the
input byte snapshot, both ordered banks, and every selected canvas; cache reuse
rereads the external file. It records coordinate/noise acquisition streams but
forbids fictional object-seed lineage. Diffraction, coordinates, detector
noise, truth-forward closure, and output NPZs are still produced by this runner.
