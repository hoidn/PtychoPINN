# PtychoPINN Configuration Guide

This guide has two layers:

- **Users and study authors:** start with [Which configuration should I use?](#which-configuration-should-i-use).
- **Developers:** see [Developer architecture](#developer-architecture) for the
  canonical/Torch split, `ModelSpec`, artifact versions, and the legacy bridge.

Dataclass defaults describe valid raw construction. They are not necessarily the
best scientific starting point. Use [Model Baselines](model_baselines.md) for
the current recommended combinations.

## Which Configuration Should I Use?

Configure the stage where a choice first changes behavior:

| You want to change… | Configure… | What it owns |
|---|---|---|
| A synthetic dataset | `SimulationConfig` | Probe construction, synthetic object, scan, detector/noise, `N`, and generation seed |
| The model or differentiable physics | `ModelConfig` | Architecture, output representation, object grouping/assembly, and model-time probe behavior |
| Optimization | `TrainingConfig` | Loss, optimizer family, schedule, epochs, batch size, sampling, and training paths |
| Reconstruction/evaluation | `InferenceConfig` | Checkpoint, test data, grouping, and inference-only reconstruction behavior |
| Torch execution mechanics | `ExecutionRequest` / explicit CLI runtime values | Requested device, DDP strategy, workers, precision, logging, and Lightning `Trainer` mechanics; capability resolution returns `PyTorchExecutionConfig` |
| The measured diffraction, positions, or actual probe | The dataset/acquisition record | Physical inputs such as `diff3d`, coordinates, `probeGuess`, and an optional exact simulation probe such as `probe_simulated`; these are data, not model settings |

In normal CLI and study workflows, supply config-file values and explicit
overrides. The entry point constructs and validates the dataclasses for you.
Do not manually construct both the public and Torch representations just to
keep duplicate fields synchronized.

The practical ownership rules are:

1. If changing a value changes generated arrays or their identity, it belongs
   to `SimulationConfig`.
2. If it changes the graph or forward model, it belongs to `ModelConfig`.
3. If it changes parameter updates, it belongs to `TrainingConfig`.
4. If it changes only reconstruction after training, it belongs to
   `InferenceConfig`.
5. If it changes Torch devices, processes, loaders, or Trainer mechanics, it
   belongs to the unresolved execution request. Model and optimizer choices
   stay with their canonical Model/Training owners; `PyTorchExecutionConfig`
   is the resolved runtime output.
6. If it is measured or saved in an NPZ, it is data.

Fields such as `N` and grid size appear at multiple boundaries because they are
validated join keys. They are not independent choices: disagreement is an
error.

## Generic PyTorch Synthetic Workflow

`ptycho_synthetic` is the supported user-facing entry point for a complete
synthetic PyTorch run. It resolves one document with five namespaces:

| Namespace | Owns |
|---|---|
| `simulation` | Values baked into train/test acquisitions, including geometry, object, probe, detector, and seed |
| `model` | Torch architecture and differentiable physics identity |
| `training` | Raw train selection, independent train/validation grouping, optimizer, loss, and schedule |
| `inference` | Reconstruction-only policy such as VarPro, patch weighting, and groups per center |
| `workflow` | Stage selection, output root, artifact reuse, and execution mechanics |

### Profiles and presets

A profile is a resolver-registered named bundle of starting values. It is
expanded before ordinary configuration-object construction and validation.
"Preset" is informal and may also describe an unregistered combination of
configuration controls, such as the TF-parity controls. There is no separate generic
preset registry or resolver, and profiles do not bypass downstream validation.

The synthetic runner registers two profiles:

| Profile | Recipe | Measurement path |
|---|---|---|
| `synthetic-lines` | `synthetic-lines-v3` | Default legacy normalized-amplitude CNN workflow |
| `cnn-lines-ci` | `cnn-lines-ci-v3` | Count-intensity Poisson CNN workflow with dose-closure startup |

With no selection, `ptycho_synthetic` uses `synthetic-lines`. A YAML, TOML,
or JSON workflow may set root `profile`; explicit `--profile` wins. Value
precedence is selected profile, then file values, then explicit CLI values. The
profiles are overrideable starting bundles rather than profile-specific lock
sets, but the final scaling, measurement, forward-model, and loss validators
still require a coherent contract. A config filename or path never selects a
profile. CLI arguments have no implicit argparse values, so an omitted flag
does not overwrite the file or profile.

The resulting `ResolvedSyntheticWorkflow`, including `profile`,
`recipe_version`, and all resolved namespaces, is written to
`resolved_workflow.json`. `DataConfig` is derived and persisted for model
reload; it is not a sixth user-authored namespace. See the
[simulation workflow guide](../scripts/simulation/README.md#stage-identity-and-reuse)
for exact stage-reuse and NPZ-digest behavior.

### Default profile

The default profile is `synthetic-lines` (recipe
`synthetic-lines-v3`):

| Area | Important resolved defaults |
|---|---|
| Data | `N=128`, `gridsize=1`, nongrid scan, shared 392×392 lines object, seed 3 |
| Probe | Ideal probe at scale 0.7, `smooth:0.5|pad_preserve:128`, simulation and model masks off |
| Raw acquisitions | 4,096 train patterns and 1,024 test patterns, normalized-amplitude legacy contract |
| Grouping | `dictionary_parity` data adapter, 4,096 selected train frames, 1,024 train groups, 1,024 validation groups, four-neighbor pool |
| Model | Unsupervised `hybrid_resnet`, geometry-derived object/probe layout, derived amplitude physics gain |
| Optimization | 50 epochs, batch 16, Adam at `2e-4`, `ReduceLROnPlateau`, MAE with prediction-L2 matching |
| Reconstruction | Probe-weighted mmap barycentric assembly, VarPro on, `groups_per_center=1` |
| Execution | One auto-selected device, deterministic FP32, zero workers, CSV logging, best checkpoint |

The shortest full invocation is therefore:

```bash
ptycho_synthetic --output-root outputs/synthetic-hybrid-resnet
```

### Synthetic object producer selection

The synthetic workflow selects object generation with the inseparable pair
`simulation.object.kind` and `simulation.object_recipe`. Both values are
resolved before simulation and dispatched through one registered producer
boundary; a mismatched pair fails. `lines` / `lines-object-v1` is the default,
and `dead_leaves` / `dead-leaves-object-v2` is the current deterministic
Dead Leaves selection. V2 derives independent named geometry and material
streams from each object seed and applies a fixed phase law with reference
maximum 1.1 and mean 0.95. This prevents phase from changing with object-bank
membership. The backend-qualified `dead-leaves-object-v1` recipe remains
available for seeded compatibility studies. Its phase law calls TensorFlow `dummy_phi`, so its exact
bytes are qualified to the TensorFlow runtime and execution device; the
manifest binds the realized array hash, but the same seed is not a
backend-independent byte contract. It also cannot reconstruct older datasets
whose caller left the Python geometry stream unseeded. Use
`dead-leaves-object-v2` for portable generation.

Manifest v3 records each object's RNG lineage, phase-law identity, realized
hash, and source commit. For fixed-pitch Dead Leaves datasets it also records
model-blind morphology descriptors on the tiled evaluation support. Those
descriptors diagnose a finite split; they are not a learnability gate, so a
quality preflight is still required when only a few independent canvases are
used.

`frozen-object-bank-v1` is registered for both object kinds when an old canvas
cannot be reconstructed from a complete RNG contract. Set
`simulation.object.source_path` to an NPZ containing exactly
`trainObjectGuess` and `testObjectGuess`, both finite complex64 arrays with
shape `(object_count, H, W)`. The base seed still owns coordinates, detector
noise, grouping, and Torch initialization; it does not claim to generate the
supplied canvases. Manifest v4 records the source-file, bank, and per-canvas
hashes and revalidates the external file before cached data can be reused.
Generated recipes reject `object.source_path`.

The CLI's `--object-kind` derives the registered recipe. Structured files may
pin both explicitly:

```yaml
profile: synthetic-lines

simulation:
  object_recipe: dead-leaves-object-v2
  object:
    kind: dead_leaves

workflow:
  output_root: outputs/synthetic-dead-leaves
```

A source-backed historical replay changes only the object selection:

```yaml
simulation:
  object_recipe: frozen-object-bank-v1
  object:
    kind: dead_leaves
    source_path: inputs/deadleaves_object_bank.npz
```

### Object banks, probe gauge, and training seed

`train_patterns` and `test_patterns` are total rows per split. The optional
`train_objects` and `test_objects` fields divide those rows into independent,
deterministically seeded object banks, so each total must be divisible by its
object count. `shared_object: true` is valid only for the default one-train,
one-test-object case. Reconstruction and evaluation currently require
`test_objects: 1`; simulation and training may use more.

The synthetic legacy profile selects `training.data_adapter:
dictionary_parity`: the grouped model input uses raw stored amplitudes, the
stored `probeGuess`, no loader probe normalization, and unit RMS/physics
factors. The CI profile selects `loader`. Set
`training.torch_training_seed` when the Torch seed is part of a reproduction;
when omitted, the workflow derives its normal independent seed.

`training.batch_order_recipe` defaults to `torch-generator-v1`. The explicit
`torch-implicit-july2026-v1` recipe reproduces the July 2026 single-device
shuffle schedule without relying on ambient global RNG state. It is rejected
for distributed Lightning strategies because sampler replacement would change
that historical order, and it requires a validation loader evaluated every
epoch because the sealed schedule includes that loader's RNG draw. Before
training, the recipe checks immutable CPU permutation fingerprints from its
reference Torch runtime and fails closed if the current runtime disagrees.

`simulation.frame_order_recipe` controls the semantic row order written to
flat train/test NPZs. `object-major-v1` is the normal default. The explicit
`coordinate-major-interleaved-v1` recipe requires a raster layout, traverses
it with the second coordinate varying fastest, and interleaves object canvases
at each coordinate. Pair it with `torch-implicit-july2026-v1` only when reproducing
the July multi-object sample sequence; freezing the shuffle alone is
insufficient if the underlying row layout differs.

`simulation.probe.simulation_normalization_scale` optionally separates the
probe stored for training from the illumination used to generate diffraction.
A positive value applies the versioned legacy normalization and persists that
illumination as `probe_simulated`; `null` uses the transformed `probeGuess`
unchanged. For example:

```yaml
profile: synthetic-lines

simulation:
  train_patterns: 8978
  test_patterns: 729
  train_objects: 2
  test_objects: 1
  shared_object: false
  probe:
    simulation_normalization_scale: 4.0

training:
  train_raw_selection: 8978
  training_groups: 8978
  validation_groups: 729
  torch_training_seed: 3
  data_adapter: dictionary_parity
```

For historical regular-grid acquisition, set
`simulation.scan.position_layout: fixed_pitch_raster`. Each object's rows form
a row-major square lattice with translation-coordinate origin `N // 2` and
split pitch `outer_offset_train / 2` or `outer_offset_test / 2`. For even `N`,
the corresponding geometric pixel center is one half-pixel smaller.

`simulation.object.patch_amplitude_normalization: mean_patch_max` computes one
scale per split across every frame and object,
`mean_i(max_xy(abs(Y_i)))`, then divides stored `Y` and diffraction amplitude
by it before count conversion. The scalar is stored as
`object_amplitude_scale`; this mode requires strict tiled reconstruction so the
raw-source object gauge can be restored exactly once.

### Structured GS2 example

This config selects an ordinary five-epoch GS2/custom-probe experiment. It is
not one of the sealed quality gates; those are CNN GS1/C1, GS2/C4,
and C4-CI (count-intensity), documented in `docs/TESTING_GUIDE.md`. The 4,096
training groups and 1,024 validation groups
are independent counts; validation is built from the complete test acquisition
rather than copied from the train count.

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
  epochs: 5

inference:
  groups_per_center: 1

workflow:
  output_root: outputs/synthetic-hybrid-resnet-gs2
  accelerator: auto
  devices: 1
```

Run it directly or override a file value explicitly:

```bash
ptycho_synthetic --config configs/synthetic-gs2.yaml
ptycho_synthetic --config configs/synthetic-gs2.yaml --epochs 10
```

For exact partial-stage replay, keep the same config and output root:

```bash
ptycho_synthetic --config configs/synthetic-gs2.yaml \
  --stages simulate,train
ptycho_synthetic --config configs/synthetic-gs2.yaml \
  --stages reconstruct,evaluate
```

Stages must be unique and follow `simulate,train,reconstruct,evaluate` order.
The second command verifies the completed prerequisites and their manifests
before strict reload. Completed selected stages are reused by default only when
their stage-specific resolved identity and required artifacts match. A required
identity mismatch, a changed consumed NPZ digest, or a partial artifact fails
closed; use a new output root for a changed experiment.

### CI gauge initialization in YAML and TOML

The synthetic `cnn-lines-ci` profile selects count-intensity Poisson
training and defaults `model.rect_s1s2_init` to `dose_closure`. Writing the
field explicitly makes that choice visible in review:

```yaml
profile: cnn-lines-ci
model:
  rect_s1s2_init: dose_closure
workflow:
  output_root: outputs/ci-dose-closure
```

The same configuration in TOML is:

```toml
profile = "cnn-lines-ci"

[model]
rect_s1s2_init = "dose_closure"

[workflow]
output_root = "outputs/ci-dose-closure"
```

Use `ones` to preserve unit initialization explicitly. Bare Torch
`ModelConfig` and non-CI `ptycho_train` runs default to `ones`; both the
training-only `ci` profile and this `ptycho_synthetic` CI profile default to
`dose_closure`. The field is part of `ModelSpec` and workflow identity but not
simulation identity. Read the result at
`<output_root>/training/training_summary.json`; see
[Data Normalization](DATA_NORMALIZATION_GUIDE.md#ci-gauge-initialization-is-not-calibration)
for its interpretation and the [core contract](specs/spec-ptycho-core.md#ci-rectangular-gauge-initialization-normative)
for the fixed representative-sampling and record rules.

The name *dose closure* refers to closing the fixed sample's aggregate count
budget: `c* = sum(measured counts) / sum(predicted unit-object intensity)`, then
`s1=s2=sqrt(c*)`. This startup conditioning does not calibrate the stored probe
or identify absolute object units.

### Torch Training-Only `ci` Profile

The native Torch training CLI accepts `--profile ci`. Programmatic callers use
`resolve_training_payload(..., profile="ci")` on the modern path or
`create_training_payload(..., profile="ci")` at the compatibility boundary.
This profile configures training against an existing count-intensity NPZ; it is
separate from the synthetic runner's `cnn-lines-ci` profile, which
also owns simulation and defaults to dose-closure startup. With `profile=None`,
the training factory performs ordinary resolution without applying a named
bundle.

The training-only profile locks these coherent contract fields:

| Field | Required value | Meaning |
|---|---|---|
| `scale_contract_version` | `ci_intensity_v2` | Selects the versioned scaling and units contract persisted with the resolved model and artifact. It is an identity tag, not a numerical multiplier. |
| `measurement_domain` | `count_intensity` | Declares that NPZ diffraction contains detector counts/intensity rather than normalized amplitude. |
| `physics_forward_mode` | `rectangular_scaled` | Selects the real/imaginary intensity forward with per-dataset `s1`/`s2` gauge factors. |
| `torch_loss_mode` | `poisson` | Selects the primary Torch/Lightning Poisson objective that compares predicted intensity with measured counts. |
| `loss_function` | `Poisson` | Keeps the shared/legacy model loss identity aligned; it does not override `torch_loss_mode` in Lightning. |

It also supplies these overrideable non-contract defaults:

| Field | Default | Meaning |
|---|---|---|
| `amplitude_physics_gain` | `1.0` | Adds no legacy amplitude-forward gain. Normal validation requires exactly `1.0` while the rectangular forward is active. |
| `rect_s1s2_trainable` | `True` | Lets the optimizer update the per-dataset real/imaginary gauge factors after initialization. |
| `rect_s1s2_init` | `dose_closure` | Solves the fixed representative startup gauge before fitting. Explicit `ones` starts `s1=s2=1` without reading training data. |
| `cnn_output_mode` | `real_imag` | Makes CNN heads represent real and imaginary object components. Non-CNN generators use their `generator_output_mode` contract. |

`cnn_output_mode` and `physics_forward_mode` are coupled but not aliases: the
first chooses the CNN's object representation, while the second chooses the
downstream diffraction/scaling calculation and its prediction domain. The CI
profile selects both because `rectangular_scaled` requires an effective
real/imaginary generator output; real/imaginary output can also be used with the
amplitude forward. See the
[PyTorch output/forward compatibility matrix](workflows/pytorch.md#35-cnn-output-and-physics-forward-knobs).

An explicit contradiction of a locked field fails closed. User overrides win
for non-contract defaults, after which the normal scaling and model validators
enforce coherence. The profile name is an authoring convenience, not an
inference selector: the persisted resolved model, data, training, inference,
and artifact identity is authoritative when the model is loaded.

The native `python -m ptycho_torch.train` CLI owns
`--rect-s1s2-init {ones,dose_closure}` directly. Its argparse default is
`None`, so omission preserves the profile or bare default; an explicit flag is
forwarded as the caller override. The native CLI does not load a workflow
`--config` document. `ptycho_synthetic --config` is a separate structured
workflow boundary.

### Sampling and reconstruction meanings

The generic runner deliberately gives overloaded compatibility fields distinct
public names:

| Public control | Lifecycle meaning |
|---|---|
| `train_patterns` / `test_patterns` | Flat raw frames physically generated in each NPZ |
| `train_objects` / `test_objects` | Independent object canvases per split; each pattern total is divided evenly across them |
| `train_raw_selection` | Train frames selected before grouping; persisted as Torch `DataConfig.n_raw_frames_selected` |
| `training_groups` | Exact train groups and unique designated centers; cannot exceed the selected candidate-row count |
| `validation_groups` | Exact grouped samples independently built from the complete test NPZ |
| `neighbor_count` | K nearest non-center candidates considered per group; must be at least `gridsize² - 1` |
| `groups_per_center` | Reconstruction-only repeats per bounded inference center; passed directly to the mmap dataset and never persisted |

Synthetic inference has four coupled controls:

| Control | Meaning |
|---|---|
| `reconstruction_method` | `barycentric` for general coordinate-aware assembly; `tiled` for strict fixed-pitch, nonoverlapping GS1 assembly |
| `patch_weighting` | Barycentric supports probe or uniform weights; tiled requires uniform weights |
| `varpro_scaling` | Fits the acquisition/count gauge. CI requires it for either reconstruction method |
| `metric_crop_border` | Symmetric border removed from the aligned metric mask only; it does not resize the saved canvas |

Each stored NPZ remains a flat acquisition with one diffraction pattern and one
coordinate per row. The loader alone constructs `C = gridsize ** 2` channel
groups for model input. In the default profile, persisted
`DataConfig.n_raw_frames_selected=4096` records raw train selection;
`groups_per_center` is a runtime inference argument (default 1) and is never
persisted in the torch-artifact-v4 wire. These two values must not be
interpreted as the same sample count.

The workflow also keeps the generic `do_stitching` path disabled. That older
path reduces multi-channel predictions at group centers, so it cannot preserve
or evaluate all four GS2 channels. Reconstruction dispatches to the public
mmap-backed barycentric or tiled adapter. Barycentric retains every grouped
channel with explicit source indices; tiled requires GS1, one test object,
`fixed_pitch_raster`, one group per center, uniform weights, and complete source
coverage.

## The Probe Lifecycle

Several fields contain the word “probe,” but they act at different stages:

```text
SimulationConfig.probe
  source + transform_pipeline + optional simulation mask
                         │
                         ▼
             generated dataset acquisition record
             ├─ probeGuess: transformed stored guess
             └─ probe_simulated: exact illumination used for diffraction
                                      │
                                      ▼
                       loader / selected scaling contract
             ├─ synthetic legacy dictionary parity: probeGuess
             └─ CI/grid-lines compatibility: contract-selected probe
                                      └─► probe_physical + probe_training
                         │
                         ▼
              differentiable forward model
                         ▲
                         │
       optional ModelConfig.probe_mask support prior
```

| Name | Meaning |
|---|---|
| `SimulationConfig.probe.transform_pipeline` | Constructs the transformed probe stored as `probeGuess`. Extension from 64×64 to 128×128, for example, happens here. |
| `SimulationConfig.probe.simulation_normalization_scale` | Optional positive legacy normalization divisor. When set, the resulting distinct illumination is stored as `probe_simulated`; `null` leaves it equal to the transformed probe. |
| `SimulationConfig.probe.mask_diameter` | Applies a simulation-time mask before diffraction is generated. Its result is baked into `probeGuess` and the dataset identity. |
| Dataset `probeGuess` | The stored complex acquisition guess. For a synthetic dataset, it contains the configured simulation transforms and mask, but a simulator may subsequently normalize it before producing diffraction. |
| Dataset `probe_simulated` | Exact illumination used to generate synthetic diffraction. Current synthetic manifest-v3 datasets record it on source, train, and test artifacts; compatibility readers may still encounter older files where it is absent. |
| CI `probe_physical` / `probe_training` | Named acquisition and normalized training views derived after selecting the dataset probe for the count-intensity/CI contract. `probe_physical` may retain an arbitrary global scalar relative to the object; the name does not prove physical calibration. Legacy normalized-amplitude paths use their existing generic probe carrier instead. These are data representations, not independently chosen configs. |
| `ModelConfig.probe_mask`, `probe_mask_diameter`, `probe_mask_sigma` | Apply an additional model-time support prior inside the differentiable forward model. They do not alter the saved dataset. |
| `ModelConfig.probe_big` | Historical name for the CNN decoder’s learned complementary outer spatial support. It does **not** resize, pad, extrapolate, or construct the physical probe. |

For an exact matched synthetic replay, use the probe field selected by the
dataset contract. `probeGuess` embodies the configured transform and mask, but
grid-lines CI prefers `probe_simulated` because it also captures the simulator's
post-`set_probe` normalization. `ModelConfig.probe_mask=False` then avoids
applying a second model-time mask. Enable a model-time mask only when the
experiment intentionally adds that support prior.

Simulation probe settings and model probe settings are not automatically
inherited from one another, and the current factory does not infer a model mask
from simulation lineage. Canonically generated datasets record the simulation
recipe and probe hashes so the relationship can be audited.

See [Data Generation](DATA_GENERATION_GUIDE.md) for probe construction and
[Data Normalization](DATA_NORMALIZATION_GUIDE.md) for the legacy and CI probe
representations.

## Object Layout and Training Assembly

New code should use the three explicit public fields below:

| Field | Choices | Meaning |
|---|---|---|
| `object_layout` | `single_patch`, `grouped_patches` | Whether model components represent independent patches or a grouped set of neighboring patches |
| `training_canvas` | `independent`, `relative_overlap` | Whether training evaluates patches independently or places them on one relative-overlap canvas |
| `training_patch_weighting` | `central_mask`, `uniform`, `probe` | How overlapping grouped patches are combined for the training forward model |

Only these layout/canvas pairs are valid:

```yaml
# Independent single-patch reconstruction
object_layout: single_patch
training_canvas: independent
training_patch_weighting: central_mask
```

```yaml
# Position-aware grouped reconstruction
object_layout: grouped_patches
training_canvas: relative_overlap
training_patch_weighting: probe
```

`object_layout` and `training_canvas` must be supplied together. PyTorch
supports all three weighting modes; TensorFlow currently supports
`central_mask` only.

`object_big` is a deprecated compatibility alias:

- `object_big: false` maps to `single_patch` + `independent`.
- `object_big: true` maps to `grouped_patches` + `relative_overlap`.
- Supplying contradictory old and new fields is an error.
- When all object-policy fields are omitted, the resolved default is
  `grouped_patches` + `relative_overlap` + `central_mask`.

The raw `None` defaults are intentional: they preserve whether a caller omitted
a field, which lets the resolver distinguish a defaulted canonical policy from
an explicitly supplied legacy alias. After resolution, all four fields,
including the derived `object_big` readback, are materialized.

`probe_big` and `pad_object` are independent choices. They are not implied by
the object layout.

## Developer Architecture

### One Meaning, Several Representations

The configuration system has multiple representations because it serves a
public API, two backends, checkpoint reconstruction, and legacy modules. These
representations are not co-equal sources of truth:

| Representation | Role | Should users edit it directly? |
|---|---|---|
| `ptycho.config.config` dataclasses | Public/shared configuration contract and legacy projection | Yes, when using the Python API |
| Factory-resolved `ptycho_torch.config_params` dataclasses | Torch data, topology, physics, training, and inference carriers after defaults, aliases, and object policy are materialized | Usually no; use the closed factory or a study wrapper |
| `TrainingPayload` / `InferencePayload` | Phase-local bundle returned by the factory | No; consume it |
| `ModelSpec("torch-model-spec-v2")` | Derived, sealed Torch graph/state identity used for construction and reload | No |
| `ExecutionRequest` | Explicit unresolved Torch runtime/Trainer request with presence provenance | Yes, normally through the CLI or request builder |
| `PyTorchExecutionConfig` | Capability-resolved runtime output; never an unresolved request or model/training owner | No |
| `ptycho.params.cfg` | Flat compatibility projection for legacy consumers | Never as a new configuration source |

The `tf_training_config` member of `TrainingPayload` is historically named. In
a native Torch run it is the canonical compatibility projection used to update
`params.cfg`; it is not a second training plan and does not construct the Torch
model.

The supported modern Torch training flow is:

```text
User / study / CLI values + optional ExecutionRequest
              │
              ▼
    resolve_training_payload()
              │
              ├─ DataConfig
              ├─ Torch ModelConfig
              ├─ Torch TrainingConfig
              ├─ Torch InferenceConfig
              ├─ canonical TrainingConfig compatibility projection
              └─ applied-overrides audit
              │
              ├─ shared model fields + Torch extensions + data joins
              │                         │
              │                         ▼
              │              ModelSpec("torch-model-spec-v2")
              │                         │
              │                         ▼
              │                 application factory
              │                         │
              │                         ▼
              │                PtychoPINN_Lightning
              │
              └─ capability resolution of ExecutionRequest
                                      └─► PyTorchExecutionConfig
                                           └─► Trainer / DataLoader setup
```

`resolve_training_payload()` may observe runtime capabilities and emit notices,
but it does not read or write `params.cfg`, mutate global configuration, or
create filesystem state. Compatibility/native entry points use
`create_training_payload()`, which performs the same resolution and then
commits the canonical projection through the scoped legacy bridge. Modern
workflow code keeps the resolved payload explicit; a surviving legacy leaf
owns any narrower compatibility scope it needs.

Studies that already hold the five resolved Torch records use
`create_training_payload_from_resolved_configs()` to adapt those exact objects;
the adapter does not run default resolution again. The shared Lightning
service consumes that payload and any prebuilt mmap DataModule directly.

The canonical and Torch model records overlap only where the backends share a
public concept. Torch-only topology and physics fields remain in the Torch
carrier. `derive_model_spec()` checks the shared fields rather than silently
choosing one representation.

### Model and Artifact Identity

`ModelSpec` is derived after configuration resolution. It freezes every Torch
structural field needed to reconstruct the model and makes checkpoint identity
independent of later mutable defaults.

Current Torch artifacts use:

- `torch-model-spec-v3` for sealed model identity;
- `torch-artifact-v4` for the enclosing data/model/training/inference identity.

The runtime load paths accept `torch-artifact-v3` and `torch-artifact-v4`.
Pre-v3 (v1/v2) artifacts are recovered through
`python -m ptycho_torch.migrate_bundle`, which deterministically upgrades them
to the current era. TensorFlow artifact formats are unchanged by this Torch
schema migration.

### Validation Boundaries

Structural validation is family-specific:

- complete simulation recipes and complete public
  Model/Training/Inference snapshots use cached Pydantic `TypeAdapter`
  boundaries over the existing stdlib dataclasses;
- alias precedence, object policy, cross-record semantics, runnable/resource
  checks, and legacy projection remain explicit Python;
- Torch Data/Model/Training/Inference keeps its explicit transactional
  resolver and manual validation because its measured 157-field adapter
  replacement would add more policy and infrastructure than it deletes; and
- execution requests, partial patches, `params.cfg`, ModelSpec, artifacts,
  checkpoints, and MLflow dictionaries do not use Pydantic.

Pydantic is therefore neither the YAML/TOML parser nor a serializer. Parsed
mappings and explicit CLI patches are merged first; only a complete snapshot
enters an adopted adapter.

Factories and bridges fail closed on ambiguous composition:

- `SimulationConfig.N` must agree with `ModelConfig.N`.
- `SimulationConfig.scan.grid_size` must agree with `ModelConfig.gridsize`.
- Torch `DataConfig.C`, `C_model`, and `C_forward` must agree.
- Object layout/canvas pairs must be complete and supported.
- Deprecated aliases may agree with canonical fields but may not contradict
  them.
- Unknown simulation keys and unknown flat Torch training overrides are errors.
- `rect_s1s2_init="data"` is retired completely. Direct config construction,
  structured mappings, ModelSpec/artifact and checkpoint decoding all reject
  it; maintained MLflow whole-model loaders revalidate immediately after
  unpickling because that path bypasses construction. There is no alias or
  automatic migration.
- `PyTorchExecutionConfig` excludes model topology and optimizer semantics.
  Historical execution aliases for those fields are retired; generators read
  Torch `ModelConfig`, while optimization reads Torch `TrainingConfig`.

Public code materializes the object policy with
`resolve_model_object_policy()`. Torch code uses
`resolve_torch_model_object_policy()` at its boundary. Downstream model code
must consume the resolved fields instead of reinterpreting `object_big`.

### Accepted Domains That Differ by Boundary

The public, Torch, execution, and protected legacy boundaries are related but
are not one interchangeable schema:

| Concept | Accepted contract |
|---|---|
| Public `ModelConfig.N` | Exactly `64`, `128`, or `256`. This is the supported authoring domain. A protected legacy model may tolerate additional powers of two; that tolerance is not a public configuration promise. |
| Training `batch_size` | An exact built-in positive integer. There is no power-of-two requirement. Backend memory limits remain runtime constraints. |
| Public `TrainingConfig.scheduler` | `Default`, `Exponential`, `WarmupCosine`, or `ReduceLROnPlateau`. |
| Torch resolved `TrainingConfig.scheduler` | The public four plus the Torch-only `MultiStage` and `Adaptive` schedules. A bridge must not silently reinterpret a Torch-only value as a public value. |
| Unresolved execution accelerator | `auto`, `cpu`, `gpu`, `cuda`, or `mps`. Capability resolution removes `auto`; `tpu` is rejected. |
| File and CLI mapping keys | Unknown root and nested keys fail closed. Direct stdlib-dataclass construction is not a mapping-resolution boundary and is validated only when passed to an explicit validator or resolver. |

### Legacy Compatibility

Some TensorFlow-era modules still read the process-local
`ptycho.params.cfg`. Supported entry points therefore perform a one-way bridge:

```text
resolved dataclass ──► update_legacy_dict(params.cfg, config) ──► legacy consumer
```

New code must not read `params.cfg` as a source for structured configuration.
Generation bridges `SimulationConfig` immediately around legacy simulation.
Training and inference project resolved runtime values only at named
legacy/archive/TensorFlow leaves. Supported modern Torch cores consume their
resolved payloads directly and do not read the global dictionary.

For the normative field mappings and CONFIG-001 lifecycle rules, see
[Configuration Bridge Specification](specs/spec-ptycho-config-bridge.md).

## Usage

Configuration precedence is entry-point specific:

- `ptycho_synthetic` uses explicit CLI values over its complete workflow
  `--config`, then the named profile.
- Retained low-level generation CLIs apply explicit compatibility overrides
  over `--simulation-config` values. Those simulation-only files may be TOML,
  YAML, or JSON; omitted file fields use dataclass defaults, while omitting the
  file invokes the entry point's historical compatibility defaults.
- Training and inference resolution uses dataclass defaults, then config-file
  values, then explicitly supplied CLI values. Omitted CLI options do not
  overwrite file values.
- Unknown simulation keys and conflicting legacy aliases are errors; not every dataclass field has a CLI flag.

## Parameter Reference

### Generated data (SimulationConfig)

`SimulationConfig` is a frozen nested recipe with `probe`, `object`, `scan`, and `detector` sections. Load TOML, YAML, or JSON with `load_simulation_config()`; unknown keys are errors. Low-level generation CLIs use explicit CLI value over `--simulation-config` value over the historical no-file default. The generic runner embeds these fields under `simulation` in its broader `--config` document and resolves them over the profile instead.

Supported probe pipeline operations are ordered and composable:

| Operation | Meaning |
|---|---|
| `smooth:0.5` | Smooth complex amplitude and unwrapped phase at the current resolution. |
| `pad_preserve:128` | Center-pad the prepared complex probe without changing its values. |
| `pad:128` | Accepted legacy spelling for `pad_preserve:128`; use `pad_preserve` in new authored recipes. |
| `interp:128` | Cubic real/imaginary interpolation. |
| `pad_extrapolate:128` | Legacy behavior: fit and evaluate one quadratic phase over the entire target probe, including the center. |
| `pad_extrapolate_boundary_matched:128` | Center-copy the prepared source exactly and solve a C0 harmonic Dirichlet correction only outside it, relaxing to the fitted quadratic at the target perimeter. This operation must be last. |

The available outer-only form is `smooth:0.5|pad_extrapolate_boundary_matched:128`: smoothing happens before extension, and no post-extension operation may alter the copied center. It remains useful when exact source-center preservation is the intended probe contract, but it is not the locked GS2/custom-probe recipe above. Changing a pipeline changes the simulation and dataset recipe digests; it cannot reuse a dataset generated by another pipeline.

Grid-lines generation writes beneath `<output_dir>/datasets/N<N>/gs<gridsize>/simulation-<simulation_config_sha256>/`. Explicit-output simulation records both `simulation_config_sha256` and `dataset_recipe_sha256` and rejects mismatched reuse; see the [Data Generation Guide](DATA_GENERATION_GUIDE.md).

```toml
[simulation]
N = 128
seed = 3

[simulation.probe]
source = "custom"
source_path = "datasets/Run1084_recon3_postPC_shrunk_3.npz"
transform_pipeline = "smooth:0.5|pad_extrapolate_boundary_matched:128"

[simulation.object]
kind = "lines"
image_size = [392, 392]
objects_per_probe = 4
diffractions_per_object = 7000
set_phi = true

[simulation.scan]
kind = "grid"
grid_size = [1, 1]
offset = 4
outer_offset_train = 8
outer_offset_test = 20
train_groups = 2
test_groups = 1
buffer = 0

[simulation.detector]
photons_per_pattern = 1e9
```

### Model Architecture (ModelConfig)

These parameters define the structure and physics of the neural network.

To add a new selectable architecture—not merely tune one of the registered
values—follow the [Custom PyTorch CDI Architecture
Guide](workflows/custom_torch_architecture.md). It covers the additional Torch
config, construction, `ModelSpec`, training, and inference boundaries.

**Illustrative mixed subset.** Shared fields are defined by `ModelConfig` in
`ptycho/config/config.py`. Torch-only extensions—including
`rect_s1s2_init` and the generator-specific rows marked below—are defined by
`ptycho_torch.config_params.ModelConfig`. Consult both dataclasses for the full
field lists and `docs/specs/spec-ptycho-config-bridge.md` §3 for ownership.
The shared `architecture` field's authoritative 14-value `Literal` lives on
the public dataclass.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `N` | `Literal[64, 128, 256]` | `64` | The dimension of the input diffraction patterns (e.g., 64×64 pixels). This is a critical parameter that defines the network's input shape. |
| `gridsize` | `int` | `1` | For PINN models, the number of neighboring patches to process together (e.g., 2 for a 2×2 grid). For supervised models, this defines the input channel depth. |
| `n_filters_scale` | `int` | `2` | A multiplier for the number of filters in the U-Net's convolutional layers. |
| `model_type` | `Literal['pinn', 'supervised']` | `'pinn'` | The type of model to use. 'pinn' is the main physics-informed model. |
| `architecture` | `ModelConfig.architecture` literal | `'cnn'` | The generator architecture for PINN models. The authoritative literal set lives in `ModelConfig` in `ptycho/config/config.py` and is mirrored in `docs/specs/spec-ptycho-config-bridge.md` §3. Common PyTorch options include `ffno`, `fno`, `hybrid`, `stable_hybrid`, `fno_vanilla`, `neuralop_uno`, `hybrid_resnet`, and the spectral/hybrid bottleneck variants. See `docs/architecture_torch.md` §4.1. |
| `fno_modes` | `int` | `12` | Number of spectral modes retained in FNO/Hybrid spectral convolutions (PyTorch only). |
| `fno_width` | `int` | `32` | Hidden channel width for FNO/Hybrid blocks (PyTorch only). |
| `fno_blocks` | `int` | `4` | Number of spectral blocks in the FNO/Hybrid encoder (PyTorch only). |
| `fno_cnn_blocks` | `int` | `2` | Number of local CNN refiner blocks for PyTorch FNO-family generators. For `architecture='fno'`, this is the Cascaded FNO refiner count. For `architecture='ffno'`, positive values create a local-refiner proxy after the factorized Fourier stack; paper-facing pure FFNO rows must set `fno_cnn_blocks=0`. |
| `fno_input_transform` | `Literal['none','sqrt','log1p','instancenorm']` | `'none'` | Optional input dynamic-range transform for FNO/Hybrid lifter (PyTorch only). |
| `resnet_width` | `Optional[int]` | `None` | Fixed bottleneck width for `hybrid_resnet`. Must be divisible by 4 when set (PyTorch only). |
| `amp_activation` | `str` | `'sigmoid'` | The activation function for the amplitude output layer. Choices: 'sigmoid', 'swish', 'softplus', 'relu'. |
| `object_layout` | `Optional[Literal['single_patch','grouped_patches']]` | `None` | Public component-layout policy. Must be supplied with `training_canvas`; omitted fields resolve through the compatibility policy. |
| `training_canvas` | `Optional[Literal['independent','relative_overlap']]` | `None` | Public training-canvas policy paired with `object_layout`. |
| `training_patch_weighting` | `Optional[Literal['central_mask','uniform','probe']]` | `None` | Training-forward overlap weighting. The resolved default is `central_mask`; TensorFlow supports only that value. |
| `object_big` | `Optional[bool]` | `None` | **Deprecated alias.** `False` maps to `single_patch`/`independent`; `True` maps to `grouped_patches`/`relative_overlap`. Contradictory dual input is rejected. |
| `probe_big` | `bool` | `True` | Historical name for the CNN decoder's learned complementary outer spatial support. It does not resize or extend the physical probe. See `docs/model_baselines.md`. |
| `probe_mask` | `bool` | `False` | If true, applies an additional model-time circular support mask inside the forward model. A simulation-time mask is already baked into dataset `probeGuess`. |
| `rect_s1s2_init` | `Literal['ones','dose_closure']` | `'ones'` | **PyTorch only.** Rectangular-scale initialization defined in `ptycho_torch/config_params.py`. `dose_closure` fails closed outside the coherent CI contract and uses the fixed representative 256-slot solve defined by the [core contract](specs/spec-ptycho-core.md#ci-rectangular-gauge-initialization-normative). The training-only and `ptycho_synthetic` CI profiles override this raw default to `dose_closure`; other `ptycho_train` and synthetic profiles retain `ones` unless explicitly overridden. |
| `pad_object` | `bool` | `True` | Controls padding behavior in the model. |
| `probe_scale` | `float` | `4.0` | A normalization factor for the probe's amplitude. |
| `gaussian_smoothing_sigma` | `float` | `0.0` | TensorFlow `ProbeIllumination` applies this Gaussian smoothing to the illuminated exit wave after multiplying object and probe; `0.0` disables it. Canonical Torch carries and seals the field for shared identity but does not currently consume it in model construction or the forward path. |

### PyTorch Execution Requests and Resolved Runtime

**Illustrative subset — full field list: `PyTorchExecutionConfig` in `ptycho/config/config.py`.**

Callers provide unresolved runtime values through `ExecutionRequest`, normally
via the request builder; CLI-exposed fields are collected into the same request.
Capability resolution produces `PyTorchExecutionConfig`, which owns only
effective device, distributed strategy, DataLoader, logging, checkpoint, and
Trainer runtime mechanics. A bare resolved carrier is not accepted as a new
request.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `accelerator` | `str` | request: `'auto'` | The request accepts `auto`, `cpu`, `gpu`, `cuda`, or `mps`; the resolved carrier contains the selected concrete runtime. |
| `devices` | `Union[int, Literal['auto']]` | `1` | Number of devices supplied to Lightning. |
| `strategy` | `str` | `'auto'` | Lightning execution strategy, including `ddp` for multi-device execution. |
| `precision` | `Literal['32-true','16-mixed','bf16-mixed']` | `'32-true'` | Torch numerical precision policy. |
| `num_workers` | `int` | `0` | DataLoader worker-process count. |
| `logger_backend` | `Optional[str]` | `'csv'` | Logging backend: CSV, TensorBoard, MLflow, or disabled. |

#### Distributed Data Parallel (DDP)

DDP is execution configuration, not model topology or optimization. The
current native and unified CLIs do not expose `devices` or `strategy`;
programmatic callers request them at the high-level Torch workflow boundary:

```python
from ptycho_torch.execution_request import ExecutionRequest

ddp_request = ExecutionRequest(
    values={"accelerator": "cuda", "devices": 2, "strategy": "ddp"},
    explicit_fields=frozenset({"accelerator", "devices", "strategy"}),
)
```

Pass this request as `execution_config`; do not construct
`PyTorchExecutionConfig` as input. Before capability resolution, `devices`
accepts a positive integer or `"auto"`; the resolved carrier always contains a
positive integer. Lightning validates and applies `strategy`. Batch size and
optimizer settings remain in `TrainingConfig`, and architecture remains in
`ModelConfig`. See the [PyTorch Workflow](workflows/pytorch.md#43-programmatic)
for the complete programmatic call.

Historical execution-level topology and optimizer aliases are retired. Put
architecture values in Torch `ModelConfig`, and learning rate, scheduler,
gradient clipping, and accumulation values in the resolved Torch
`TrainingConfig`. Omitted CLI flags do not overwrite file or baseline values.

### Training Parameters (TrainingConfig)

These parameters control the training loop, data handling, and loss functions.

**Illustrative subset — full field list: `TrainingConfig` in `ptycho/config/config.py`.**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_data_file` | `Optional[Path]` | `None` | **Required.** Path to the training dataset (.npz file). |
| `test_data_file` | `Optional[Path]` | `None` | Path to the test dataset (.npz file). |
| `batch_size` | `int` | `16` | The number of samples per batch. Must be an exact built-in positive integer; it need not be a power of two. |
| `nepochs` | `int` | `50` | Number of training epochs. |
| `mae_weight` | `float` | `0.0` | Weight for the Mean Absolute Error loss in diffraction space. Range: [0, 1]. |
| `nll_weight` | `float` | `1.0` | Weight for the Negative Log-Likelihood (Poisson) loss. Recommended: 1.0. Range: [0, 1]. |
| `realspace_mae_weight` | `float` | `0.0` | Weight for the MAE loss in the object domain. |
| `realspace_weight` | `float` | `0.0` | General weight for all real-space losses. |
| `nphotons` | `float` | `1e9` | Legacy/runtime compatibility value used by existing training physics. It does not define generated dose; new datasets take photon count from `SimulationConfig.detector.photons_per_pattern`. |
| `training_groups` | `Optional[int]` | `None` (`512` after `TrainingConfig.__post_init__` when unset) | Exact number of groups and unique designated centers. Each group contains `C = gridsize²` distinct same-object rows, including the center in column zero. Cannot exceed the selected candidate-row count. **Replaces deprecated `n_images`.** |
| `n_images` | `Optional[int]` | `None` | **[DEPRECATED]** Legacy alias for `training_groups`. |
| `train_raw_selection` | `Optional[int]` | `None` | Candidate rows selected from the dataset before grouping. |
| `subsample_seed` | `Optional[int]` | `None` | Seed for reproducible candidate-row selection. |
| `sequential_sampling` | `bool` | `False` | If `False`, training uses a dedicated seeded, epoch-varying shuffle and validation remains sequential. If `True`, both selection and training order are sequential. Configure it as `training.sequential_sampling` for structured workflows; `ptycho_synthetic` exposes `--sequential-sampling` / `--no-sequential-sampling`, while `ptycho_train` exposes `--sequential_sampling` / `--no-sequential_sampling`. |
| `positions_provided` | `bool` | `True` | If True, use the provided scan positions. |
| `probe_trainable` | `bool` | `False` | If True, allows the model to learn and update the probe function during training. |
| `intensity_scale_trainable` | `bool` | `True` | If True, allows the model to learn the global intensity scaling factor. |
| `output_dir` | `Path` | `"training_outputs"` | The directory where training outputs (model, logs, images) will be saved. |
| `optimizer` | `Literal['adam','adamw','sgd']` | `'adam'` | Optimizer family. Torch learning rate, scheduler, clipping, and accumulation resolve through the Torch Training owner rather than execution configuration. |
| `weight_decay` | `float` | `0.0` | Optimizer weight decay. |
| `scheduler` | `str` | `'Default'` | Learning rate scheduler type: `'Default'`, `'Exponential'`, `'WarmupCosine'`, `'ReduceLROnPlateau'`. |
| `lr_warmup_epochs` | `int` | `0` | Warmup epochs for the WarmupCosine scheduler. |
| `lr_min_ratio` | `float` | `0.1` | Minimum LR ratio for WarmupCosine (eta_min = base_lr × ratio). |
| `plateau_factor` | `float` | `0.5` | ReduceLROnPlateau factor (multiplier applied when plateau detected). |
| `plateau_patience` | `int` | `2` | ReduceLROnPlateau patience (epochs without improvement before reducing LR). |
| `plateau_min_lr` | `float` | `5e-5` | ReduceLROnPlateau minimum learning rate. |
| `plateau_threshold` | `float` | `0.0` | ReduceLROnPlateau threshold for measuring improvement. |

### Inference Parameters (InferenceConfig)

These parameters control inference and evaluation workflows.

**Illustrative subset — full field list: `InferenceConfig` in `ptycho/config/config.py`.**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `Path` | **Required** | Path to the trained model directory containing `wts.h5.zip`. |
| `test_data_file` | `Path` | **Required** | Path to the test dataset (.npz file) for inference. |
| `output_dir` | `Path` | `"inference_outputs"` | Directory where inference results will be saved. |
| `inference_groups` | `Optional[int]` | `None` | Number of bounded centers to reconstruct. Each group contains `C = gridsize²` distinct same-object rows, including the center in column zero. If unset, uses every eligible bounded center. **Replaces deprecated `n_images`.** |
| `n_images` | `Optional[int]` | `None` | **[DEPRECATED]** Legacy alias for `inference_groups`. |
| `inference_raw_selection` | `Optional[int]` | `None` | Candidate rows selected from test data before grouping. |
| `subsample_seed` | `Optional[int]` | `None` | Seed for reproducible candidate-row selection. |
| `debug` | `bool` | `False` | Enable debug mode for additional logging. |

## Understanding Sampling Parameters

Candidate-pool selection and exact group count are independent:

- **`train_raw_selection` / `inference_raw_selection`** selects candidate rows.
- **`training_groups`** selects that many unique designated train centers.
- **`inference_groups`** bounds how many eligible centers are reconstructed.
- **`neighbor_count`** is the K nearest non-center candidate pool for each
  center and must be at least `gridsize² - 1`.
- **`subsample_seed`** makes candidate-row selection reproducible.

Every group has `C = gridsize²` distinct same-object members, with its
designated center in column zero. Neighbor rows may be shared across groups,
so `groups × C` is not a distinct-row count. Training group count cannot
exceed the selected candidate-row count.

The deprecated `n_images` input still aliases `training_groups` or
`inference_groups`, according to the workflow, and emits a warning.

#### Example Scenarios

```yaml
# Dense grouping: every selected candidate row is a designated center
train_raw_selection: 1200
training_groups: 1200
neighbor_count: 7
gridsize: 2

# Sparse grouping: use fewer centers than selected candidates
train_raw_selection: 10000
training_groups: 500
neighbor_count: 7
gridsize: 2

# Memory-constrained GS1 grouping
train_raw_selection: 5000
training_groups: 2000
neighbor_count: 1
gridsize: 1
```

## Example YAML Configuration

You can create a `.yaml` file to specify a set of parameters for a run. This is useful for managing and reproducing experiments.

```yaml
# File: configs/my_experiment_config.yaml

# Model Architecture Parameters
N: 64
gridsize: 2
n_filters_scale: 2
model_type: 'pinn'
amp_activation: 'swish'
object_layout: 'grouped_patches'
training_canvas: 'relative_overlap'
training_patch_weighting: 'central_mask'
probe_trainable: true

# Training Parameters
train_data_file: 'datasets/fly/fly001_prepared_train.npz'
test_data_file: 'datasets/fly/fly001_prepared_test.npz'
output_dir: 'results/my_experiment_run_1'
nepochs: 100
batch_size: 32
training_groups: 4096  # Exact group count and unique-center count

# Loss Function Weights
nll_weight: 1.0
mae_weight: 0.0

# Runtime/model compatibility parameters for already-materialized data
nphotons: 1e9
probe_scale: 4.0
gaussian_smoothing_sigma: 0.0
```

To use this configuration, you would run:

```bash
ptycho_train --config configs/my_experiment_config.yaml
```

You can still override any parameter from the command line:

```bash
# Use the config file but run for only 10 epochs
ptycho_train --config configs/my_experiment_config.yaml --nepochs 10
```

## Configuration Best Practices

1. Start from the project-recommended values in
   [docs/model_baselines.md](model_baselines.md); this catalog defines fields and
   raw defaults, not the best combination for a run.
2. **Use YAML files** for reproducible experiments and parameter sets you want to reuse.
3. **Use `training_groups` / `inference_groups`;** reserve deprecated `n_images` for migration tests.
4. **Use `object_layout`, `training_canvas`, and `training_patch_weighting`**
   instead of deprecated `object_big`.
5. **Override sparingly** from the command line - use it mainly for quick parameter tweaks.
6. **Document your configs** with comments explaining the experimental purpose.
7. **Version control** your configuration files alongside your code.
8. **Test configurations** with small datasets before running full experiments.

## Parameter Migration

For migrating existing configurations:

```yaml
# Old (deprecated but still works)
n_images: 1000

# New (recommended)
training_groups: 1000  # Exact group count and unique-center count
```

```yaml
# Old (deprecated but still accepted)
object_big: true

# New (recommended)
object_layout: grouped_patches
training_canvas: relative_overlap
training_patch_weighting: central_mask
```
