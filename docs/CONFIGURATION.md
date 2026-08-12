# PtychoPINN Configuration Guide

This guide has two layers:

- **Users and study authors:** start with [Which configuration should I use?](#which-configuration-should-i-use).
- **Developers:** see [Developer architecture](#developer-architecture) for the
  public/Torch split, `ModelSpec`, artifact versions, and the legacy bridge.

This document defines parameter ownership and records defaults from the owning
configuration types. Public training configuration is a nested Pydantic model;
simulation, model, and inference configuration remain validated standard-library
dataclasses. These defaults are valid construction values, not necessarily the
best scientific settings for every dataset. A governing study or run contract
may select different values explicitly.

## Which Configuration Should I Use?

Configure the stage where a choice first changes behavior:

| You want to change… | Configure… | What it owns |
|---|---|---|
| A synthetic dataset | `SimulationConfig` | Probe construction, synthetic object, scan, detector/noise, `N`, and generation seed |
| The model or differentiable physics | `ModelConfig` | Architecture, output representation, object grouping/assembly, and model-time probe behavior |
| Optimization | `TrainingConfig` | Loss, optimizer family, schedule, epochs, batch size, sampling, and training paths |
| Reconstruction/evaluation | `InferenceConfig` | Checkpoint, test data, grouping, and inference-only reconstruction behavior |
| Torch execution mechanics | `ExecutionRequest` or explicit CLI runtime values | Requested device, DDP strategy, workers, precision, logging, and Lightning `Trainer` mechanics; capability resolution returns a runtime-only `PyTorchExecutionConfig` |
| Measured diffraction, positions, or the actual probe | Dataset/acquisition data | Physical inputs such as `diff3d`, coordinates, `probeGuess`, and optional realized-probe fields; these are data, not model settings |

In normal CLI and study workflows, supply config-file values and explicit
overrides. The entry point constructs and validates the appropriate
configuration record. Do not manually construct both public and Torch
representations merely to keep shared fields synchronized.

The practical ownership rules are:

1. If changing a value changes generated arrays or their identity, it belongs
   to `SimulationConfig`.
2. If it changes the graph or differentiable forward model, it belongs to
   `ModelConfig`.
3. If it changes parameter updates, it belongs to `TrainingConfig`.
4. If it changes only reconstruction after training, it belongs to
   `InferenceConfig`.
5. If it changes Torch devices, processes, loaders, or Trainer mechanics, it
   belongs to the unresolved `ExecutionRequest`. The factory resolves that
   request into `PyTorchExecutionConfig`; users do not pass that resolved
   carrier back into a factory.
6. If it is measured or saved in an NPZ, it is data.

Fields such as `N` and grid size appear at multiple boundaries because they are
validated join keys. They are not independent choices: disagreement is an
error.

### Acquisition Data Is Not Configuration

`ptycho.acquisition.AcquisitionRecord` is the framework-neutral carrier for
measured or simulated arrays crossing backend boundaries. It snapshots
coordinates, diffraction, probe/object guesses, scan indices, metadata, and
sampling identity. It deliberately does not own loading, grouping, tensor
conversion, or backend behavior.

## Data Transport and Entry-Point Routing

The entry point and the type of input supplied by its caller select the data
route. Dataset size, DDP settings, and NPZ key inspection do not automatically
choose between in-memory, NPZ, and memory-mapped processing.

```text
caller supplies arrays or a data object
  └─ in-memory adapter ──► model-ready container / DataLoader

caller supplies one NPZ file
  └─ RawData loader ──► grouping in memory ──► model-ready container / DataLoader

caller supplies an NPZ directory to train_lightning_only
  └─ PtychoDataset ──► TensorDict memory map ──► PtychoDataModuleLightning

caller invokes the grid-lines study runner
  └─ grid-lines cached-NPZ adapter ──► dict container ──► ordinary DataLoader

caller supplies an existing memory map
  └─ PrebuiltPtychoDataModule ──► ordinary Lightning training lifecycle
```

There are therefore three different persistence/residency modes:

| Mode | Persistence boundary | Runtime behavior |
|---|---|---|
| End-to-end in memory | None | Simulation or caller-owned NumPy arrays become `RawData`, `PtychoDataset.from_np()`, or an existing container without an intermediate save/reload. |
| NPZ-backed, RAM-resident | An NPZ is written or supplied | The selected loader reads the file, after which grouping, adaptation, and training use in-memory arrays and ordinary DataLoaders. Grid-lines cached NPZs use this mode. |
| Disk-backed memory map | Standalone NPZs are supplied through the mmap entry point | `PtychoDataset` reads the NPZs to build a TensorDict memory map; later epochs and ranks open that map and fetch batches from it. The NPZ archive itself is not directly memory-mapped. |

### Current Routing by Entry Point

| Caller or entry point | Input boundary | Selected route |
|---|---|---|
| `RawData.from_simulation()` or `generate_simulated_data()` followed directly by a workflow call | In-memory object | Remains in memory unless the caller explicitly saves it. |
| `PtychoDataset.from_np()` and the in-memory API loaders | NumPy arrays | Bypass NPZ I/O and the on-disk memory map. |
| Unified/file-oriented training CLIs and `python -m ptycho_torch.train` | One standalone NPZ path | Load through `RawData`, group in memory, adapt to `PtychoDataContainerTorch`, then use ordinary DataLoaders. |
| `ptycho_torch.train_lightning_only.main(ptycho_dir=...)` | Directory containing standalone NPZ scans | Build or open the TensorDict mmap through `PtychoDataModuleLightning`. This is the established Lightning multi-device/DDP data rail. |
| `scripts/studies/grid_lines_torch_runner.py` | Grid-lines train/test cached NPZ paths | Load the specialized cache into dictionaries, select grid-lines probe/coordinate semantics, adapt to the dict-container batch contract, and call `_train_with_lightning`. This path currently constructs a single-device Trainer. |
| `PrebuiltPtychoDataModule` | Existing TensorDict mmap | Reopen the already-built map without reparsing source NPZs. |
| Default Torch inference CLI | One standalone NPZ path | Load through `RawData` and run inference in memory. |
| Barycentric or probe-weighted Torch inference | One standalone NPZ path | Stage the NPZ in an isolated directory and build a temporary `PtychoDataset` mmap because that reconstruction implementation consumes the grouped dataset representation. |

The factory-resolved `PyTorchExecutionConfig` controls devices, DDP strategy,
workers, and Lightning runtime mechanics after this routing decision. It does
not select a dataset schema or convert a grid-lines cache into the mmap schema.
In particular, requesting DDP does not cause a file-based or grid-lines entry
point to switch automatically to `PtychoDataModuleLightning`.

### Standalone NPZ Versus Grid-Lines Cached NPZ

These formats carry diffraction under consumer-specific keys and represent
different pipeline stages:

| Format | Typical contents | Consumer |
|---|---|---|
| Standalone scan NPZ through `RawData` | One ungrouped 3-D `diff3d` stack, `xcoords`, `ycoords`, `probeGuess`, and the other acquisition fields required by `RawData.from_file()` | Unified/file-oriented workflows and the default Torch inference route |
| Standalone scan NPZ through the Torch mmap writer | One ungrouped 3-D `diff3d` stack, or the accepted `diffraction` compatibility alias, plus scan coordinates and the writer-required acquisition fields | `PtychoDataModuleLightning` / `PtychoDataset` mmap route |
| Grid-lines cached NPZ | Pre-grouped/channelized `diffraction`, `Y_I`, `Y_phi`, `coords_nominal`, `coords_true`, `YY_full`, and optional `probe_simulated` | Grid-lines cached-dataset adapter |

`RawData.from_file()` requires the standalone key `diff3d`; it does not apply
the mmap loader's `diffraction` alias. Conversely, acceptance of both spellings
by the mmap route does not make a grid-lines cache interchangeable with a
standalone scan.
The mmap writer expects ungrouped diffraction plus scan coordinates and its
writer-required acquisition fields; a grid-lines cache is already grouped and
uses separate amplitude/phase labels. The grid-lines CI adapter also has
probe-provenance behavior that the generic standalone loader does not infer:
when both splits carry `probe_simulated`, it selects that realized simulation
probe instead of blindly using `probeGuess`.

There is currently no canonical `data_transport = memory | npz | mmap` setting
and no global schema dispatcher. To determine the route for a run, start from
the invoked CLI/function and follow the input type it accepts. Schema
validation occurs inside the already-selected loader.

Standalone NPZ shapes are described in the
[PtychoPINN Data Contracts](data_contracts.md). The routing description above
explains which current entry point consumes each input form without turning
historical runner choices into new format requirements.

## The Probe Lifecycle

Several fields contain the word “probe,” but they act at different stages:

```text
SimulationConfig.probe
  source + transform_pipeline + optional simulation mask
                         │
                         ▼
        generated dataset probeGuess
        + optional realized probe_simulated
                         │
                         ▼
             selected physics probe
                         │
                         ▼
          loader / selected scaling contract
             ├─ legacy: normalized probe carrier
             └─ CI: probe_physical + probe_training
                         │
                         ▼
              differentiable forward model
                         ▲
                         │
       optional ModelConfig.probe_mask support prior
```

| Name | Meaning |
|---|---|
| `SimulationConfig.probe.transform_pipeline` | Constructs the probe used to simulate the dataset. Extension from 64×64 to 128×128, for example, happens here. |
| `SimulationConfig.probe.mask_diameter` | Applies a simulation-time mask before diffraction is generated. Its result is baked into `probeGuess` and the dataset identity. |
| Dataset `probeGuess` | The stored/configured complex probe guess. For synthetic data it contains the declared simulation transforms and mask, but it is not universally the exact illumination that generated the recorded counts. |
| Dataset `probe_simulated` | Optional grid-lines field containing the realized simulator illumination after the simulator's internal probe normalization. |
| Selected CI probe | When both real train/test splits carry `probe_simulated`, grid-lines CI uses it. Otherwise CI falls back to `probeGuess`. A one-sided `probe_simulated` bundle fails closed. Non-CI arms use `probeGuess`. |
| CI `probe_physical` / `probe_training` | Physical and normalized training views derived from the selected CI probe. Legacy normalized-amplitude paths retain their generic normalized probe carrier. These are data representations, not independent configs. |
| `ModelConfig.probe_mask`, `probe_mask_diameter`, `probe_mask_sigma` | Apply an additional model-time support prior inside the differentiable forward model. They do not alter the saved dataset. |
| `ModelConfig.probe_big` | Historical name for the CNN decoder's learned complementary outer spatial support. It does **not** resize, pad, extrapolate, or construct the physical probe. |

For an exact matched synthetic replay, follow the dataset's recorded probe
provenance rather than assuming `probeGuess` is always the realized
illumination. `ModelConfig.probe_mask=False` then avoids applying a second
model-time mask. Enable a model-time mask only when the experiment intentionally
adds that support prior.

Simulation probe settings and model probe settings are not automatically
inherited from one another, and the factory does not infer a model mask from
simulation lineage. Canonically generated datasets record the simulation
recipe and probe hashes so the relationship can be audited.

See the [simulation tools guide](../scripts/simulation/README.md) for probe
construction and [Data Normalization](DATA_NORMALIZATION_GUIDE.md) for the
legacy and CI probe representations.

## Object Layout and Training Assembly

New code should use the three explicit public fields below:

| Field | Choices | Meaning |
|---|---|---|
| `object_layout` | `single_patch`, `grouped_patches` | Whether model components represent independent patches or a grouped set of neighboring patches |
| `training_canvas` | `independent`, `relative_overlap` | Whether training evaluates patches independently or places them on one relative-overlap canvas |
| `training_patch_weighting` | `central_mask`, `uniform`, `probe` | How overlapping grouped patches are combined in the training forward model |

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
supports all three training weighting modes; TensorFlow currently supports
`central_mask` only. Torch inference has a separate
`InferenceConfig.patch_weighting` choice for post-training reconstruction.

`object_big` is a deprecated compatibility alias:

- `object_big: false` maps to `single_patch` + `independent`.
- `object_big: true` maps to `grouped_patches` + `relative_overlap`.
- Supplying contradictory old and new fields is an error.
- When all object-policy fields are omitted, the resolved default is
  `grouped_patches` + `relative_overlap` + `central_mask`.

The raw `None` defaults are intentional: they preserve whether a caller omitted
a field, allowing the resolver to distinguish the canonical default from an
explicit legacy alias. After resolution, all fields, including the derived
`object_big` readback, are materialized.

`probe_big` and `pad_object` are independent choices. They are not implied by
the object layout.

### Training Assembly Versus Inference Reconstruction

These are separate policies:

- Resolved `ModelConfig.training_patch_weighting` produces a
  `TrainingAssemblySpec`. It controls the differentiable merge used during
  training and can affect gradients.
- Torch `InferenceConfig.patch_weighting` and `varpro_scaling` produce a
  versioned `ReconstructionPolicy`. It controls post-training placement,
  overlap weighting, optional VarPro calibration, and presentation.

Changing inference probe weighting or VarPro does not retroactively change the
training forward model. Conversely, training weighting is not an inference
default. Active CI inference is one intentional exception to general
optionality: when the saved model uses
`physics_forward_mode="rectangular_scaled"`,
`scale_contract_version="ci_intensity_v2"`, and
`measurement_domain="count_intensity"`, inference requires
`varpro_scaling=True` and fails closed if it is disabled.

## Developer Architecture

### One Meaning, Several Representations

The configuration system has several representations because it serves a
public API, two backends, checkpoint reconstruction, and legacy modules. These
representations are not co-equal sources of truth:

| Representation | Role | Should users edit it directly? |
|---|---|---|
| `ptycho.config.config.TrainingConfig` and its nested Pydantic models | Public training authoring contract and legacy projection | Yes, when using the Python API |
| `ptycho.config.config` simulation, model, and inference dataclasses | Public/shared non-training configuration contracts | Yes, when using the Python API |
| Factory-resolved `ptycho_torch.config_params` dataclasses | Torch data, topology, physics, training, and inference carriers after defaults, aliases, and object policy are materialized | Usually no; use the closed factory or a study wrapper |
| `TrainingPayload` / `InferencePayload` | Phase-local bundles returned by the factory | No; consume them |
| `ModelSpec("torch-model-spec-portable-v2")` | Derived, sealed Torch graph/state identity used for construction and reload | No |
| `ExecutionRequest` | Unresolved Torch runtime values plus explicit-input provenance; the CLI builds this from options the caller actually supplied | Yes, for programmatic runtime requests |
| `PyTorchExecutionConfig` | Capability-resolved Torch runtime output consumed by Lightning and DataLoader construction; never an optimization or model-topology owner | No |
| `ptycho.params.cfg` | Flat compatibility projection for legacy consumers | Never as a new configuration source |

The public and Torch training records overlap where backend entry points still
need a public compatibility carrier and a resolved Torch training carrier. The
factory resolves one effective configuration and records applied overrides;
callers should not independently maintain both records.

The `tf_training_config` member of `TrainingPayload` is historically named. In
a native Torch run it is the public compatibility projection used to update
`params.cfg`; it is not a second training plan and does not construct the Torch
model.

The normal Torch training flow is:

```text
canonical model/training values + unresolved ExecutionRequest
  ├─ resolve_training_payload() ───────────────────────────┐
  │    modern resolution; no params.cfg access             │
  └─ create_training_payload()                             │
       compatibility resolution                            │
       └─ CONFIG-001 projection ──► params.cfg             │
                                                           ▼
                                                    TrainingPayload
              │
              ├─ pt_data_config
              ├─ pt_model_config
              ├─ pt_training_config
              ├─ pt_inference_config
              ├─ tf_training_config
              ├─ model_spec
              ├─ execution_config
              └─ overrides_applied
              │
              ├─ shared model fields + Torch extensions + data joins
              │                         │
              │                         ▼
              │       ModelSpec("torch-model-spec-portable-v2")
              │                         │
              │                         ▼
              │                 application factory
              │                         │
              │                         ▼
              │                PtychoPINN_Lightning
              │
              ├─ pt_training_config ──► optimizer / scheduler / clipping / accumulation
              └─ execution_config ──► Trainer / DataLoader runtime
```

`resolve_training_payload()` neither reads nor writes `params.cfg`; it may emit
deprecation/runtime notices and observe capabilities when an unresolved runtime
value requires that observation. `create_training_payload()` is the
compatibility boundary: it performs the same owner/runtime resolution and then
projects the public compatibility record into `params.cfg`.

`InferencePayload` is smaller: it carries the public inference projection,
Torch data/inference settings, execution settings, and applied overrides.
Saved model structure comes from the validated checkpoint/artifact identity.

The public and Torch model records overlap only where the backends share a
public concept. Torch-only topology and physics fields remain in the Torch
carrier. `derive_model_spec()` checks shared fields rather than silently
choosing one representation.

### Model and Artifact Identity

`ModelSpec` is derived after configuration resolution. It freezes every Torch
structural field needed to reconstruct the model and makes checkpoint identity
independent of later mutable defaults.

Current Torch artifacts use:

- `torch-model-spec-portable-v2` for sealed model identity;
- `torch-artifact-portable-v2` for the enclosing
  data/model/training/inference identity.

The `portable` qualifier is deliberate: these identifiers describe the
five-architecture surface selected for the main-compatible port. They are not
aliases for older source schema identifiers with a different frozen field set.

Portable version 2 stores `object_layout`, `training_canvas`, and
`training_patch_weighting` as the structural object policy. It does not retain
deprecated `object_big` as a second owner. Frozen portable-v1 artifacts remain
readable and are deterministically upgraded during decoding. Unrelated schema
identifiers fail closed rather than having unknown fields discarded. In
particular, `torch-model-spec-v1` and `torch-artifact-v1` are not aliases for
the portable schemas and are rejected. The outer Torch bundle manifest version
remains `2.0-pytorch`.

### Validation Boundaries

Factories and bridges fail closed on ambiguous composition:

- `SimulationConfig.N` must agree with `ModelConfig.N`.
- `SimulationConfig.scan.grid_size` must agree with `ModelConfig.gridsize`.
- Torch `DataConfig.C`, `C_model`, and `C_forward` must agree.
- Object layout/canvas pairs must be complete and supported.
- Deprecated aliases may agree with canonical fields but may not contradict
  them.
- Unknown simulation keys and unknown flat Torch training overrides are errors.
- Neither execution record owns model topology or optimization.
  `ExecutionRequest` contains runtime requests only; a bare resolved
  `PyTorchExecutionConfig` is rejected as factory input. Topology enters
  through Torch `ModelConfig`, and optimization enters through Torch
  `TrainingConfig` or its explicit factory patch.

Public code materializes the object policy with
`resolve_model_object_policy()`. Torch code uses
`resolve_torch_model_object_policy()` at its boundary. Downstream model code
must consume the resolved fields instead of reinterpreting `object_big`.

### Legacy Compatibility

Some TensorFlow-era modules still read the process-local
`ptycho.params.cfg`. Supported entry points therefore perform a one-way bridge:

```text
resolved configuration ──► update_legacy_dict(params.cfg, config) ──► legacy consumer
```

New code must not read `params.cfg` as a source for structured configuration.
Generation bridges `SimulationConfig` before legacy simulation. Training and
inference bridge their resolved public config separately before any legacy
loader, helper, or model code uses it.

For the external bridge contract, see
[Ptychodus API Specification](../specs/ptychodus_api_spec.md). For the
backend workflow and CONFIG-001 ordering, see
[PyTorch Workflow](workflows/pytorch.md).

## Usage and Precedence

### Public Training and Inference Resolution

The supported public source boundaries are exported from `ptycho.config` as
`resolve_training_config()` and `resolve_inference_config()`. They share this
precedence rule but intentionally have different authoring shapes:

```text
type defaults < file mapping < explicitly supplied CLI patch
```

An argparse default is not an explicit CLI value. Supported entry points pass
only options that the caller actually supplied, so an omitted CLI option does
not overwrite a file value.

Training authoring is nested. `resolve_training_config()` deep-merges the file
mapping and explicit CLI patch, then validates one `TrainingConfig`
(`BaseSettings`) with these Pydantic submodels:

```text
model                   ModelConfig dataclass
data                    DataConfig
sampling                SamplingConfig
loss                    LossConfig
tf_loss                 TFLossConfig
gradient_clip           GradientClipConfig
optimizer               OptimizerConfig
scheduler               SchedulerConfig
```

Model and every training-component field therefore belong under their named
sections. Unknown or misplaced fields fail because every
Pydantic section uses `extra="forbid"`. Although `TrainingConfig` derives from
`BaseSettings`, its implicit environment, dotenv, and secrets sources are
disabled: entry points explicitly load the file and CLI mappings and pass the
merged result for validation.

One enumerated exception exists for backward compatibility: the historical
flat root spellings of `data`, `tf_loss`, and `sampling` fields (for example
`train_data_file`, `nll_weight`, `n_groups`) are accepted during
`TrainingConfig` validation and lifted into their nested sections with a
`DeprecationWarning`. An equal flat/nested duplicate is accepted once; an
unequal duplicate fails with both spellings identified. This applies wherever
`TrainingConfig` validation runs (direct construction and resolved file/CLI
mappings). New configurations should use the nested spellings
(`specs/ptychodus_api_spec.md` §2.1 lists the full alias set).

On the unified CLI, direct training fields retain plain flags such as
`--nepochs`, `--batch_size`, `--output_dir`, and `--backend`; nested fields use
dotted flags such as `--data.train_data_file`, `--sampling.n_groups`,
`--optimizer.algorithm`, and `--scheduler.kind`. The native
`python -m ptycho_torch.train` CLI is a separate interface and retains its
documented flat flags.

Current `refactor` limitation: the generated parser leaves numeric and Boolean
CLI values as strings, which the strict Pydantic validators reject during
`setup_configuration()`. Author those values in the nested YAML file until the
CLI decoder is corrected. Path and literal-string overrides remain usable; the
dotted names above describe the intended public surface rather than a claim
that every generated value type currently completes resolution.

Within `SamplingConfig`, `n_groups` is canonical and `n_images` remains a
deprecated alias. An alias-only value becomes `n_groups`; equal alias and
canonical values are accepted; unequal values fail. Because source mappings
are deep-merged before Pydantic validation, conflicting alias/canonical values
across file and CLI sources also fail rather than using one spelling to
silently override the other. Successful validation clears `n_images`; omitted
`n_groups` materializes the default of `512`.

Inference authoring remains flat except for `model`. Within each inference
source, model fields may be written either at the root or under `model`. An
equal flat/nested duplicate is accepted once; an unequal duplicate fails with
both locations identified. Across sources, normal precedence applies even when
one source uses the flat form and the other uses the nested form. Inference
resolves its `n_images` alias per source before applying CLI precedence.

Validation is deliberately layered:

- `validate_*_config_structure()` checks types, closed domains, local ranges,
  and cross-field coherence without filesystem access. The public resolvers
  run the corresponding structural validator before returning.
- `validate_runnable_training_config()` adds the requirements for starting a
  run, including positive execution values and an existing readable training
  dataset.
- `validate_inference_resources()` adds model-archive and test-data resource
  checks at the inference consumer boundary.

The older `validate_model_config()`, `validate_training_config()`, and
`validate_inference_config()` exports remain compatibility facades with their
historical predicates; they are not aliases for the new layers. Source
resolution and structural validation do not mutate global state or the
filesystem. Successful source resolution may emit deprecation warnings for
accepted legacy aliases. Supported workflow consumers apply runnable or
resource validation before the one-way `update_legacy_dict()` bridge.

Other configuration families retain their entry-point-specific source rules:

- Generation CLIs apply retained explicit CLI overrides over
  `--simulation-config` values. Simulation files may be TOML, YAML, or JSON;
  omitted file fields use dataclass defaults, while omitting the file invokes
  the entry point's historical compatibility defaults.
- Unknown simulation keys and conflicting compatibility aliases are errors.
- Not every configuration field has a CLI flag.

### Profiles Are Starting Bundles

This section is the conceptual authority for profile/preset semantics,
locks/defaults, and the training-only `ci` profile. The simulation guide owns
operational commands, public flags, and exact stage-reuse mechanics.

A profile is a resolver-registered named bundle of starting values. The
resolver expands it before ordinary configuration-object construction and
validation. "Preset" is informal and may also describe an unregistered
combination of runtime controls, such as the TF-parity preset. There is no
separate generic preset registry or resolver, and profiles do not bypass
downstream validation.

The synthetic runner registers these bundles:

| Profile | Recipe | Measurement path |
|---|---|---|
| `synthetic-lines` | `synthetic-lines-v1` | Default legacy normalized amplitude |
| `cnn-lines-ci` | `cnn-lines-ci-v1` | Count-intensity CNN |

With no selection, `ptycho_synthetic` uses `synthetic-lines`. A YAML, TOML, or
JSON workflow may set root `profile`; explicit `--profile` wins. The remaining
value precedence is selected profile, then file values, then explicit CLI
values for overrideable fields. A selected profile's locked fields may only be
restated equally; contradictions fail. A config filename or path never selects
a profile.

`cnn-lines-ci` locks `scale_contract_version=ci_intensity_v2`,
`measurement_domain=count_intensity`, `architecture=cnn`,
`physics_forward_mode=rectangular_scaled`, `cnn_output_mode=real_imag`,
`loss_function=Poisson`, `torch_loss_mode=poisson`, and `nll=true`. Matching
restatements are accepted; contradictions fail. Its
`rect_s1s2_init=dose_closure` and gradient-clipping settings are overrideable
defaults. `synthetic-lines` has no profile-specific locks, but its final
resolved values still pass all normal validators.

The resulting `ResolvedSyntheticWorkflow`, including `profile`,
`recipe_version`, and all resolved values, is written to
`resolved_workflow.json`. Persisted identity includes `measurement_domain` and
`scale_contract_version`, preventing detector-domain drift. Stage reuse consumes
the relevant resolved identity and verifies NPZ content separately; see
[Stage identity and reuse](../scripts/simulation/README.md#stage-identity-and-reuse)
for the exact namespace and digest mechanics.

This five-epoch invocation is a functional contract example:

```bash
ptycho_synthetic --profile cnn-lines-ci \
  --output-root outputs/synthetic-cnn-ci-contract \
  --epochs 5 \
  --rect-s1s2-init dose_closure
```

It demonstrates resolution and execution of the CI contract; it is not a
validated CNN quality threshold or baseline. See the
[simulation workflow guide](../scripts/simulation/README.md) for profile and
NPZ details.

### Torch Training-Only `ci` Profile

Use `resolve_training_payload(..., profile="ci")` on the modern programmatic
path, `create_training_payload(..., profile="ci")` at the compatibility
boundary, or the Torch training CLI's `--profile ci` instead of assembling a
partial count-intensity configuration by hand. This `ci` profile applies only
to Torch training with supplied NPZs; it is separate from the synthetic
runner's `cnn-lines-ci` profile. With `profile=None`, the factory follows
ordinary resolution without applying a named bundle.

The profile locks these coherent contract fields:

| Field | Required value | Meaning |
|---|---|---|
| `scale_contract_version` | `ci_intensity_v2` | Selects the versioned scaling and units contract persisted with the resolved model and artifact. It is an identity tag, not a numerical multiplier. |
| `measurement_domain` | `count_intensity` | Declares that NPZ diffraction contains detector counts/intensity rather than normalized amplitude. |
| `physics_forward_mode` | `rectangular_scaled` | Selects the real/imaginary intensity forward with per-dataset `s1`/`s2` gauge factors. |
| `torch_loss_mode` | `poisson` | Selects the primary Torch/Lightning Poisson objective that compares predicted intensity with measured counts. |
| `loss_function` | `Poisson` | Keeps the shared/legacy model loss identity aligned; it does not override `torch_loss_mode` in Lightning. |

It also supplies these non-contract defaults, which callers may override:

| Field | Default | Meaning |
|---|---|---|
| `amplitude_physics_gain` | `1.0` | Adds no legacy amplitude-forward gain. Normal validation requires exactly `1.0` while the rectangular forward is active. |
| `rect_s1s2_trainable` | `True` | Lets the optimizer update the per-dataset real/imaginary gauge factors after initialization. |
| `rect_s1s2_init` | `dose_closure` | Solves the shared startup gauge from the resolved training data. Explicit `ones` keeps exact unit initialization without inspecting the loader. |
| `cnn_output_mode` | `real_imag` | Makes CNN heads represent real and imaginary object components. Non-CNN generators use their `generator_output_mode` contract. |

`cnn_output_mode` and `physics_forward_mode` are coupled but not aliases: the
first chooses the CNN's object representation, while the second chooses the
downstream diffraction/scaling calculation and its prediction domain. The CI
profile selects both because `rectangular_scaled` requires an effective
real/imaginary generator output; real/imaginary output can also be used with the
amplitude forward. See the
[PyTorch output/forward compatibility matrix](workflows/pytorch.md#35-cnn-output-and-physics-forward-knobs).

An explicit contradiction of a locked contract field fails closed. Non-contract
profile defaults follow normal override precedence, then the downstream scaling
and model validators enforce coherence. Omitting an initialization override
therefore keeps the profile's `dose_closure` default; an explicit `ones` wins.
A bare `ModelConfig` still defaults to `ones`, because it has no resolved CI
training dataset. `dose_closure` requires the complete CI contract.

The profile name is an authoring convenience, not an inference selector. The
persisted resolved model, data, training, and bundle identity is authoritative
when the model is loaded; inference does not require selecting `ci` again.

#### Dose-closure initialization

For an existing count-intensity NPZ, select the training-only profile and omit
the initialization field to use its `dose_closure` default:

```python
from ptycho_torch.config_factory import resolve_training_payload

payload = resolve_training_payload(
    train_data_file=train_npz,      # count-intensity data (diffraction = counts)
    output_dir=output_dir,
    profile="ci",                   # locks the coherent CI contract set
    overrides={
        "gridsize": 1,
        "N": 128,
        "n_groups": 4489,
        "batch_size": 16,
    },
)
```

The non-obvious pieces:

| Piece | Meaning |
|---|---|
| `profile="ci"` | Locks `ci_intensity_v2` + `count_intensity` + `rectangular_scaled` + Poisson as an inseparable set; contradicting any locked field fails closed. |
| `rect_s1s2_init="dose_closure"` | Before fitting, one shared `s1=s2` startup gauge is solved from the actual forward with a unit object. Exactly 256 logical `(row, channel)` detector slots are selected uniformly without replacement across the complete resolved training dataset using fixed seed `20260806` and policy `splitmix64_rejection_v1`. It fixes startup conditioning when the stored probe's global scalar does not match the recorded counts; it does not calibrate the probe or identify physical object units. |
| `n_groups` / `gridsize` | Required grouping identity: number of sampled groups and frames per group axis. `gridsize=1` degenerates grouping to single-frame groups. |
| Startup record | Fresh training persists a strict `rect-s1s2-initialization-v2` record (`solved_gauge`, `method`, `mode`, `sampled_patterns`) in `training_summary.json`; the dose method is `dose_closure_seeded_uniform_unit_object`. Readers accept valid historical v1 prefix-era records without rewriting them. A solved gauge far from 1 signals that the data's probe/object decomposition convention disagrees with the forward model. |

The name *dose closure* refers to closing the aggregate count budget on the
fixed sample. With a unit object, the initializer computes

```text
c* = sum(measured detector counts) / sum(predicted unit-object intensity)
s1 = s2 = sqrt(c*)
```

so the sampled predicted and measured count totals agree at startup. This is a
conditioning step for the shared rectangular gauge, not a calibration of the
physical probe or an identification of absolute object units.

The [representative-sampling design](superpowers/specs/2026-08-06-ci-dose-closure-representative-sampling-design.md#sampling-contract)
defines the pinned draw and logical-slot mapping.

The sample size, seed, and policy are fixed rather than user-configurable. If
the resolved training population has fewer than 256 detector slots or cannot
produce a valid gauge, initialization fails; it does not use a prefix, reduce
the sample size, or fall back to `ones`. To request unit initialization, set
`overrides={"rect_s1s2_init": "ones", ...}` explicitly.

The only supported values are `ones` and `dose_closure`. Historical
`rect_s1s2_init="data"` configuration and artifacts are not migrated or
translated; use historical code or retrain them.

For end-to-end generation and training, use
`ptycho_synthetic --profile cnn-lines-ci`; that synthetic profile selects
`dose_closure` by default. The [simulation workflow guide](../scripts/simulation/README.md)
documents the runnable command and count-domain data meaning.

## Parameter Reference

The tables below are representative. The configuration definitions in
`ptycho/config/config.py` and `ptycho_torch/config_params.py` are the complete
field lists.

### Generated Data (`SimulationConfig`)

`SimulationConfig` is a frozen nested recipe with `probe`, `object`, `scan`, and
`detector` sections. Load TOML, YAML, or JSON with
`load_simulation_config()`; unknown keys are errors.

#### Validation boundary

The five simulation records remain standard frozen stdlib dataclasses: the
root recipe plus its four nested sections. Direct construction and
`dataclasses.replace()` retain normal dataclass behavior and do not validate.
Programmatic callers must call `validate_simulation_config()` before using,
serializing, hashing, or projecting a recipe. Raw mappings parsed from TOML,
YAML, or JSON go through `simulation_config_from_mapping()`;
`load_simulation_config()` applies that mapping boundary for files.

One cached Pydantic `TypeAdapter` performs structural and type validation at
those two boundaries only. It checks nested field shapes, strict scalar types,
unknown keys, and exact `Literal` membership. Pydantic is neither the domain
model nor the serializer: the stored records remain stdlib dataclasses, and
the existing explicit simulation serializers and canonical identity functions
remain authoritative.

Input is strict rather than relying on coercion. Booleans are not accepted as
integers, numeric strings are not converted to numbers, closed strings require
their exact `Literal` spellings, and boolean fields require exact `bool`
values. At the raw-mapping boundary, the documented structural normalizations
are a path string to `pathlib.Path` and a two-element list pair to a tuple.
Accepted numeric values retain their exact built-in kind, so
integer-versus-float diameter values remain distinct in canonical simulation
identity.

Semantic and cross-field rules remain explicit domain validation after the
TypeAdapter check. These include probe source/path coherence, probe-pipeline
grammar and terminal-operation constraints, agreement between the pipeline
output size and `SimulationConfig.N`, and square object and scan dimensions.

Supported probe pipeline operations are ordered and composable:

| Operation | Meaning |
|---|---|
| `smooth:0.5` | Smooth complex amplitude and unwrapped phase at the current resolution. |
| `pad_preserve:128` | Center-pad the prepared complex probe without changing its values. |
| `pad:128` | Accepted legacy alias for `pad_preserve:128`; prefer the explicit canonical spelling in new recipes. |
| `interp:128` | Cubic real/imaginary interpolation. |
| `pad_extrapolate:128` | Legacy behavior: fit and evaluate one quadratic phase over the entire target probe, including the center. |
| `pad_extrapolate_boundary_matched:128` | Preserve the prepared source exactly and construct a C0 boundary-matched outer phase that relaxes toward the fitted quadratic at the target perimeter. This operation must be last. |

The canonical outer-only form is
`smooth:0.5|pad_extrapolate_boundary_matched:128`: smoothing happens before
extension, and no later operation may alter the copied center. Changing a
pipeline changes simulation and dataset recipe digests; it cannot silently
reuse a dataset generated by another pipeline.

Grid-lines generation writes beneath
`<output_dir>/datasets/N<N>/gs<gridsize>/simulation-<simulation_config_sha256>/`.
Explicit-output simulation records both `simulation_config_sha256` and
`dataset_recipe_sha256` and rejects mismatched reuse. See the
[simulation tools guide](../scripts/simulation/README.md).

```toml
[simulation]
N = 128
seed = 3

[simulation.probe]
source = "custom"
source_path = "path/to/probe.npz"
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

### Model Architecture (`ModelConfig`)

To add a new selectable architecture—not merely tune one of the registered
values—follow the [Custom PyTorch CDI Architecture
Guide](workflows/custom_torch_architecture.md). It covers the additional Torch
config, construction, `ModelSpec`, training, and inference boundaries.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `N` | `Literal[64, 128, 256]` | `64` | Diffraction-pattern side length. This must agree with the data and simulation recipe. |
| `gridsize` | `int` | `1` | Neighbor-grid side length; a grouped sample has `gridsize²` patches. |
| `architecture` | `Literal['cnn','ffno','fno','fno_vanilla','neuralop_uno']` | `'cnn'` | Supported generator architecture. |
| `n_filters_scale` | `int` | `2` | CNN channel multiplier. |
| `fno_modes` | `int` | `12` | Spectral modes retained by supported Fourier operators. |
| `fno_width` | `int` | `32` | Hidden width for supported Fourier operators. |
| `fno_blocks` | `int` | `4` | Number of Fourier blocks. |
| `fno_cnn_blocks` | `int` | `2` | Number of local CNN refinement blocks used by the applicable Fourier generators. |
| `fno_input_transform` | `Literal['none','sqrt','log1p','instancenorm']` | `'none'` | Optional Torch input dynamic-range transform. |
| `amp_activation` | `Literal['sigmoid','swish','softplus','relu']` | `'sigmoid'` | Public amplitude activation spelling. |
| `object_layout` | `Optional[Literal['single_patch','grouped_patches']]` | `None` | Public component-layout policy. Must be supplied with `training_canvas`. |
| `training_canvas` | `Optional[Literal['independent','relative_overlap']]` | `None` | Public training-canvas policy paired with `object_layout`. |
| `training_patch_weighting` | `Optional[Literal['central_mask','uniform','probe']]` | `None` | Training-forward overlap weighting. The resolved default is `central_mask`; TensorFlow supports only that value. |
| `object_big` | `Optional[bool]` | `None` | **Deprecated alias.** Maps to the corresponding layout/canvas pair; contradictory dual input is rejected. |
| `probe_big` | `bool` | `True` | CNN learned complementary outer support. It does not resize or extend the physical probe. |
| `probe_mask` | `bool` | `False` | Applies an additional model-time circular support mask inside the forward model. |
| `pad_object` | `bool` | `True` | Controls object padding in the forward model. |
| `probe_scale` | `float` | `4.0` | Legacy/public probe-normalization factor. |
| `gaussian_smoothing_sigma` | `float` | `0.0` | TensorFlow exit-wave smoothing after probe-object multiplication; zero disables it. Canonical Torch records and seals the field but currently does not consume it, so changing it has no Torch numerical effect. |

### PyTorch Execution Request and Resolved Runtime

Users and CLIs express unresolved runtime intent as an `ExecutionRequest`. Its
`values` mapping contains primitive runtime values and `explicit_fields`
records which values the caller actually supplied. The factory validates
canonical model and training owners first, resolves any requested hardware
capabilities, and returns `PyTorchExecutionConfig` in the payload:

```text
CLI/runtime values ──► ExecutionRequest ──► capability resolution
                                             │
                                             ▼
                                  PyTorchExecutionConfig
                                  (Trainer/DataLoader only)
```

A bare `PyTorchExecutionConfig` is already resolved and is therefore rejected
as factory input. Construct `ExecutionRequest` directly for programmatic
runtime control, or let the CLI build it with
`build_execution_request_from_args()`.

| Runtime value | Request default | Resolution |
|---|---|---|
| `accelerator` | `'auto'` | Resolves to an available supported accelerator; the output carrier never stores `'auto'`. |
| `devices` | `1` | Accepts a positive integer or `'auto'`; resolves to a positive integer. |
| `strategy` | `'auto'` | Passed to Lightning after runtime resolution, including `ddp` when requested. |
| `precision` | `'32-true'` | Resolves the supported Torch precision for the selected accelerator. |
| `num_workers` | `0` | Controls DataLoader worker processes. |
| `logger_backend` | `'csv'` | Selects CSV, TensorBoard, MLflow, or disabled logging (`None` after resolving CLI `'none'`). |

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

Learning rate, optimizer, scheduler, gradient clipping, and gradient
accumulation are not execution fields. Configure them through Torch
`TrainingConfig` or an explicitly supplied canonical factory training patch.
Optimizer construction and Lightning's derived clipping/accumulation mechanics
read that single resolved training record.

### Training (`TrainingConfig`)

The public and resolved Torch training records share many fields. Public
training authoring is nested as shown below. Torch-only `learning_rate` and
`accum_steps` are not fields of the public Pydantic model; supported CLI
spellings become an explicit factory patch, and the resolved Torch
`TrainingConfig` owns their effective values.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data.train_data_file` | `Optional[Path]` | `None` | Training dataset path. Required by training entry points. |
| `data.test_data_file` | `Optional[Path]` | `None` | Optional validation/test dataset path. |
| `batch_size` | `int` | `16` | Samples per batch. |
| `nepochs` | `int` | `50` | Number of training epochs. |
| `tf_loss.mae_weight` | `float` | `0.0` | TensorFlow diffraction-space MAE weight. |
| `tf_loss.nll_weight` | `float` | `1.0` | TensorFlow Poisson negative-log-likelihood weight. |
| `tf_loss.realspace_mae_weight` | `float` | `0.0` | TensorFlow object-domain MAE weight. |
| `tf_loss.realspace_weight` | `float` | `0.0` | TensorFlow general real-space loss weight. |
| `loss.torch_loss_mode` | `Literal['poisson','mae']` | `'poisson'` | Primary Torch loss family. |
| `data.nphotons` | `float` | `1e9` | Legacy/runtime compatibility value. Generated dose belongs to `SimulationConfig.detector.photons_per_pattern`. |
| `sampling.n_groups` | `Optional[int]` | `512` after validation | Number of grouped samples used for training. |
| `sampling.n_images` | `Optional[int]` | `None` | **Deprecated** alias for `sampling.n_groups`; cleared after successful validation. |
| `sampling.n_subsample` | `Optional[int]` | `None` | Number of raw images selected before grouping. |
| `sampling.subsample_seed` | `Optional[int]` | `None` | Reproducible subsampling seed. |
| `positions_provided` | `bool` | `True` | Use provided scan positions. |
| `probe_trainable` | `bool` | `False` | Optimize the probe jointly with the object model. |
| `intensity_scale_trainable` | `bool` | `True` | Optimize the global intensity scale. |
| `optimizer.algorithm` | `Literal['adam','adamw','sgd']` | `'adam'` | Optimizer family. |
| `optimizer.weight_decay` | `float` | `0.0` | Optimizer weight decay. |
| `scheduler.kind` | `Literal['Default','Exponential','WarmupCosine','ReduceLROnPlateau']` | `'Default'` | Public scheduler choice. |
| `gradient_clip.val` | `Optional[float]` | `None` | Torch gradient-clipping threshold; `None` disables clipping. |
| `gradient_clip.algorithm` | `Literal['norm','value','agc']` | `'norm'` | Torch gradient-clipping algorithm. |
| `output_dir` | `Path` | `training_outputs` | Training output directory. |

### Inference (`InferenceConfig`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_path` | `Path` | Required | Trained model/checkpoint location. |
| `test_data_file` | `Path` | Required | Inference dataset path. |
| `output_dir` | `Path` | `inference_outputs` | Reconstruction output directory. |
| `n_groups` | `Optional[int]` | `None` | Number of groups to reconstruct; `None` uses all available groups. |
| `n_images` | `Optional[int]` | `None` | **Deprecated** alias for `n_groups`. |
| `n_subsample` | `Optional[int]` | `None` | Number of raw test images selected before grouping. |
| `subsample_seed` | `Optional[int]` | `None` | Reproducible inference subsampling seed. |
| `debug` | `bool` | `False` | Enable additional diagnostic output. |

## Understanding Sampling Parameters

When only deprecated `n_images` is supplied, it behaves as `n_groups`:

- `gridsize=1`: each group contains one image.
- `gridsize>1`: each group contains `gridsize²` neighboring images.

When `n_subsample` is supplied, the controls are independent:

- `n_subsample` selects raw images from the dataset.
- `n_groups` controls how many grouped samples are used.
- `subsample_seed` makes raw-image selection reproducible.

```yaml
# Select 10,000 raw images, then construct 500 groups of four patches.
model:
  gridsize: 2
sampling:
  n_subsample: 10000
  n_groups: 500
  subsample_seed: 3
```

## Example YAML Configuration

```yaml
model:
  N: 64
  gridsize: 2
  architecture: cnn
  n_filters_scale: 2
  amp_activation: swish
  object_layout: grouped_patches
  training_canvas: relative_overlap
  training_patch_weighting: central_mask

data:
  train_data_file: datasets/fly/fly001_prepared_train.npz
  test_data_file: datasets/fly/fly001_prepared_test.npz
  # Compatibility value for already-materialized data. Generated dose belongs
  # to SimulationConfig.detector.photons_per_pattern.
  nphotons: 1000000000.0

output_dir: results/my_experiment_run_1
nepochs: 100
batch_size: 32
probe_trainable: true

sampling:
  n_groups: 4096

tf_loss:
  nll_weight: 1.0
  mae_weight: 0.0
```

```bash
ptycho_train --config configs/my_experiment_config.yaml

# Literal-string CLI values override the file when explicitly supplied.
ptycho_train --config configs/my_experiment_config.yaml --backend pytorch
```
