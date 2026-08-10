# PtychoPINN Synthetic and Simulation Workflows

For new synthetic PyTorch work, use `ptycho_synthetic`. It is the supported
generic runner for simulation, grouping, training, strict model reload,
reconstruction, and evaluation. For anything that runs the workflow more than
once — ablations, architecture comparisons, sweeps — drive this runner with
the `ptycho_study` layer instead of hand-rolled loops (see
[Multi-Arm Studies](#multi-arm-studies) below). The lower-level simulation
tools remain useful when only a prepared dataset is needed.

## Supported Generic PyTorch Runner

### Default full workflow

The shortest complete invocation resolves the coherent
`synthetic-lines` profile and runs all four stages:

```bash
ptycho_synthetic --output-root outputs/synthetic-cnn
```

The equivalent source-tree entry point is:

```bash
python -m scripts.simulation.synthetic_pipeline \
  --output-root outputs/synthetic-cnn
```

The default is a real 50-epoch run, not a smoke test.

### Profile selection and precedence

A profile is a resolver-registered named starting bundle, expanded before
ordinary configuration objects are constructed and validated. The runner
registers two profiles:

| Profile | Recipe | Purpose |
|---|---|---|
| `synthetic-lines` | `synthetic-lines-v1` | Default legacy normalized-amplitude lines workflow |
| `cnn-lines-ci` | `cnn-lines-ci-v1` | Count-intensity lines workflow with the CNN/Poisson contract |

With no selection, the runner uses `synthetic-lines`. A YAML, TOML, or JSON
workflow may select a profile with the root `profile` field; an explicit
`--profile` overrides that field. Value precedence is:

```text
selected profile < --config file values < explicit CLI values
```

This precedence applies to overrideable fields. A selected profile's locked
fields may only be restated with equal values; any contradiction fails closed.

The config filename and path never select a profile. The workflow file passed
to `ptycho_synthetic` is named `--config`; this differs from the simulation-only
`simulate_and_save.py --simulation-config` interface described later.

A complete count-intensity CNN run is:

```bash
ptycho_synthetic --profile cnn-lines-ci \
  --output-root outputs/synthetic-cnn-ci
```

### GS2/custom-probe example

This five-epoch example selects grid size 2 (`C=4`), the established legacy
custom-probe transform, all 4,096 train frames, and independent train and
validation group counts:

```bash
ptycho_synthetic \
  --output-root outputs/synthetic-cnn-gs2 \
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
  output_root: outputs/synthetic-cnn-gs2
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

### Profile contracts and defaults

This is an operational summary of the profile contract and its public flags.
The [configuration guide](../../docs/CONFIGURATION.md#profiles-are-starting-bundles)
is the conceptual authority for profile/preset semantics, locks, and defaults.

`cnn-lines-ci` locks the fields that define its count-intensity CNN contract:

| Namespace | Locked values |
|---|---|
| Simulation | `scale_contract_version=ci_intensity_v2`, `measurement_domain=count_intensity` |
| Model | `architecture=cnn`, `physics_forward_mode=rectangular_scaled`, `cnn_output_mode=real_imag`, `loss_function=Poisson` |
| Training | `torch_loss_mode=poisson`, `nll=true` |

The public CI contract and initialization flags are:

| Flag | Value allowed with `cnn-lines-ci` |
|---|---|
| `--scale-contract-version` | `ci_intensity_v2` |
| `--measurement-domain` | `count_intensity` |
| `--physics-forward-mode` | `rectangular_scaled` |
| `--cnn-output-mode` | `real_imag` |
| `--torch-loss-mode` | `poisson`; also resolves `model.loss_function=Poisson` and `training.nll=true` |
| `--rect-s1s2-init` | `dose_closure` (default) or `ones` |

For `cnn-lines-ci`, the first five flags may only restate the locked values
shown above; a contradiction fails closed. `--rect-s1s2-init` selects either
supported initialization without changing the locked contract. Equal values in
a workflow file are accepted for the same reason.

The profile's gradient-clipping defaults
(`gradient_clip_val=1.0`, `gradient_clip_algorithm=norm`) are ordinary
overrideable defaults. `dose_closure` solves a shared startup gauge from the
training data; `ones` starts `s1=s2=1`. The
[configuration guide](../../docs/CONFIGURATION.md#dose-closure-initialization)
owns the naming, selection identity, and retirement rules.

For this profile, NPZ diffraction (`diff3d`) is Poisson-realized detector
counts, not square-root intensity. `probeGuess` is the CI-scaled physical
forward probe in the same count convention.

`synthetic-lines` has no named profile locks, although every final resolved
configuration still passes the normal cross-field validators. Its main
user-facing defaults are:

| Area | Default |
|---|---|
| Simulation | `N=128`, `gridsize=1`, seed 3, nongrid scan, buffer 64 |
| Object | Shared 392×392 `lines-object-v1`, `set_phi=true` |
| Probe | Ideal source, scale 0.7, `smooth:0.5|pad_preserve:128`; simulation mask off |
| Raw frames | 4,096 train, 1,024 test; normalized-amplitude `legacy_v1` |
| Sampling | 4,096 selected train frames; 1,024 train groups; 1,024 validation groups; neighbor/pool size 4; oversampling off |
| Model | Unsupervised `cnn`, amplitude/phase output, amplitude forward, model mask off, geometry-derived layout, derived amplitude physics gain |
| Training | 50 epochs, batch 16, Adam `2e-4`, plateau scheduler, MAE with prediction-L2 matching |
| Inference | Batch 16, probe-weighted barycentric assembly, VarPro on, `groups_per_center=1` |
| Execution | One auto-selected device, deterministic FP32, zero workers, CSV logger, one best checkpoint |

Probe transform defaults are source-aware:

- An ideal or already-`N` custom probe uses
  `smooth:0.5|pad_preserve:N`.
- A smaller custom probe uses `pad_extrapolate:N|smooth:0.5`.
- A larger custom probe fails unless an explicit supported downsampling
  transform is supplied.

### Flat storage and grouped model samples

Simulation persists canonical flat acquisitions. Each train/test NPZ contains
`diff3d` with shape `(M, N, N)`, one `xcoords`/`ycoords` value per frame, the
contract-appropriate `probeGuess`, shared `objectGuess`, and one
acquisition/probe `scan_index` value per row. For legacy normalized-amplitude
data, `probeGuess` is the transformed simulation probe; for CI count data, it
is the scaled physical forward probe described above. `scan_index` may repeat;
the manifest's array digests and row order pin acquisition identity. Flat
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

The persisted training `DataConfig.n_subsample` records
`train_raw_selection` (4,096 in the profile). Reconstruction creates an
evaluation-only copy whose `n_subsample=groups_per_center` (default 1). That
runtime copy is neither serialized back into the model bundle nor interpreted
as raw train selection.

### Stage identity and reuse

Completed stages are reusable by default. Reuse is fail-closed:

- every stage compares `schema_version`, `profile`, and `recipe_version`;
- simulation compares the resolved `simulation` namespace;
- training compares `simulation`, `model`, and `training`;
- reconstruction/evaluation also compare `inference`;
- the `workflow` namespace, including execution controls, is excluded from
  stage identity;
- each required stage-manifest entry and artifact path must be complete;
- NPZ content is verified separately through recorded array and file digests;
- a required identity mismatch or a partial artifact requires a new output
  root rather than an in-place overwrite.

The resolved `ResolvedSyntheticWorkflow` is persisted as
`resolved_workflow.json`, including `profile`, `recipe_version`, and every
resolved simulation, model, training, inference, and workflow value. Reuse
compares the stage-specific portions listed above, not the spelling of the
invocation or config path.

Stage selection, output-root spelling, and the reuse switch itself are not
scientific identity. Downstream-only settings therefore do not redefine an
already complete simulation, but an exact replay must retain the complete
identity required by every selected stage.

Fresh training writes the strict `rect-s1s2-initialization-v2` record to
`OUTPUT/training/training_summary.json`; dose closure records method
`dose_closure_seeded_uniform_unit_object`. The current
`synthetic-stage-manifest-v2` training entry requires that file alongside the
model bundle. Reuse reparses the record and requires its mode to match the
resolved workflow. Strict historical prefix-era v1 initialization records
remain readable and reusable without being rewritten. Historical
`synthetic-stage-manifest-v1` roots lack the initialization-record contract;
use a new output root or retrain them.

### Reconstruction and stitching

The workflow always disables the older generic `do_stitching` route. That path
reduces multiple predicted channels at group centers and is not a valid GS2
quality reconstruction. Instead, the public mmap-backed barycentric workflow
retains all `C` channel indices, places every accepted patch in global
coordinates, applies probe weighting, and records reassembly diagnostics.

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
    checkpoints/
      <monitored-best>.ckpt
      last.ckpt
    lightning_logs/
  reconstruction/
    reconstruction.npz
    metrics.json
    diagnostics.json
    comparison.png
```

Invocation and resolved-input records are written before expensive work.
Stage and dataset manifests use relative artifact paths and are updated
atomically. The runner owns managed descendants beneath `--output-root`; it
does not recursively delete an arbitrary caller directory.

For a fitted count-intensity reconstruction,
`reconstruction/diagnostics.json` records
`metric_validity.count_diagnostics.status="complete"` with
`relative_l2_intensity_error`, `mean_raw_poisson_nll`, `n_samples`, and
`n_pixels`. The two floating-point metrics must be finite and both counts must
be positive. A legacy normalized-amplitude reconstruction instead records
`status="not_applicable"` with reason
`legacy_normalized_amplitude`; it does not fabricate count-domain metrics.

## Multi-Arm Studies

Best practice for any multi-run experiment is one config tree plus one
command, not a shell loop or a bespoke driver script. A study factors the
experiment into a base `config.yaml` (everything shared) and its ablation
axes; `ptycho_study` composes base + axis values + CLI overrides per arm and
invokes the generic runner once per arm.

A minimal complete study — a two-arm architecture comparison — is one file:

```yaml
# studies/arch_compare/conf/config.yaml
profile: synthetic-lines

model:
  architecture: cnn        # swept axis; the base value must exist to override

simulation:
  train_patterns: 512
  test_patterns: 256

training:
  epochs: 5
  train_raw_selection: 512
  training_groups: 512
  validation_groups: 256

study:                     # study-layer node; removed before arm.yaml is written
  name: arch_compare
  output_root: .artifacts/studies/arch_compare
  runner_script: scripts/simulation/synthetic_pipeline.py
  sentinel: reconstruction/metrics.json

hydra:
  run:
    dir: ${study.output_root}/adhoc/${now:%Y%m%d_%H%M%S}
  sweep:
    dir: ${study.output_root}
    subdir: arch_${model.architecture}
  job:
    chdir: false
```

Everything outside the `study`/`hydra` nodes is ordinary runner schema — the
same document you would pass to `ptycho_synthetic --config`. Run the matrix:

```bash
ptycho_study --config-dir studies/arch_compare/conf --config-name config -m \
    model.architecture=cnn,fno
```

This writes `.artifacts/studies/arch_compare/arch_cnn/` and `arch_fno/`.
Every swept key must appear in `hydra.sweep.subdir`, or arms collide on one
directory. Axes that change several keys at once live in per-axis delta
files instead of a dotted override; the study guide linked below covers
that layout.

Each arm's output root receives `arm.yaml` — a complete, self-contained
workflow document in this runner's `--config` schema, directly replayable
with `ptycho_synthetic --config <arm>/arm.yaml` — plus `runner.log` and
`study_provenance.json`. All validation still happens in the runner, on the
composed document; the study layer holds no schema of its own.

Rerunning the same sweep command resumes it: arms whose sentinel
(`reconstruction/metrics.json` by default) exists are skipped, and the
fail-closed stage identity described above still governs any per-arm reuse.
Cross-arm collation and the comparison table come from
`scripts/studies/collate_study_metrics.py` and
`scripts/studies/render_study_comparison.py`.

The [study workflow guide](../../docs/workflows/hydra_studies.md) owns the
config-tree structure, delta-file conventions, failure semantics, shared
datasets, and how to define a new study. Until the package is reinstalled,
invoke the layer as `python -m ptycho.workflows.study_runner`.

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
| `ptycho_study` / `ptycho.workflows.study_runner` | **Supported study layer:** composes per-arm configs and runs the generic runner once per arm |
| `simulate_and_save.py` | **Supported low-level tool:** simulate from a prepared object/probe NPZ |
| `run_with_synthetic_lines.py` | **Deprecated compatibility/history only:** delegates to `ptycho_synthetic --stages simulate` and contains no simulation logic |

The simulation-only CLI keeps its own canonical simulation-file flag:

```bash
python scripts/simulation/simulate_and_save.py \
  --simulation-config configs/lines64.toml \
  --input-file prepared_lines64.npz \
  --output-file outputs/lines64.npz
```

`simulate_and_save.py` consumes the `objectGuess` already present in
`--input-file`; it does not synthesize a requested object family. Prepare
`dead_leaves` or `natural_patch` inputs with their dedicated producer first.
`grf` is not a supported `SyntheticObjectConfig.kind` and must not be recorded
under another label.

### Deprecated lines-wrapper migration

`scripts/simulation/run_with_synthetic_lines.py` is retained only so
historical simulation-only commands can delegate to the generic runner; do
not use it for new recipes. It rejects its old `--simulation-config` option
(migrate the document to `ptycho_synthetic --config`) and the historical
`--n-images` (split it into explicit `--train-patterns`/`--test-patterns`,
then choose `--training-groups`/`--validation-groups` independently). The
adapter translates only unambiguous names:

| Historical wrapper name | Generic name |
|---|---|
| `--output-dir` | `--output-root` (adapter-owned) |
| `--probe-size` | `--N` |
| `--n-photons` | `--photons-per-pattern` |
| `--buffer` | `--scan-buffer` |

### SimulationConfig example and probe transforms

This complete low-level configuration uses the available boundary-matched
probe transform. It is a distinct dataset recipe, not the locked GS2 recipe
shown earlier:

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
[data contracts](../../specs/data_contracts.md) for required keys and shapes.
The retained simulation-only tools above cover the grid and nongrid
programmatic APIs available on this branch.
