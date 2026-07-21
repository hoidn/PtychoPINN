# PtychoPINN Configuration Guide

This guide has two layers:

- **Users and study authors:** start with [Which configuration should I use?](#which-configuration-should-i-use).
- **Developers:** see [Developer architecture](#developer-architecture) for the
  public/Torch split, `ModelSpec`, artifact versions, and the legacy bridge.

This document defines parameter ownership and records raw dataclass defaults.
Those defaults are valid construction values, not necessarily the best
scientific settings for every dataset. A governing study or run contract may
select different values explicitly.

## Which Configuration Should I Use?

Configure the stage where a choice first changes behavior:

| You want to change… | Configure… | What it owns |
|---|---|---|
| A synthetic dataset | `SimulationConfig` | Probe construction, synthetic object, scan, detector/noise, `N`, and generation seed |
| The model or differentiable physics | `ModelConfig` | Architecture, output representation, object grouping/assembly, and model-time probe behavior |
| Optimization | `TrainingConfig` | Loss, optimizer family, schedule, epochs, batch size, sampling, and training paths |
| Reconstruction/evaluation | `InferenceConfig` | Checkpoint, test data, grouping, and inference-only reconstruction behavior |
| Torch execution mechanics | `PyTorchExecutionConfig` | Device, DDP strategy, workers, precision, logging, Lightning `Trainer` controls, and the current Torch learning-rate/clipping/accumulation controls |
| Measured diffraction, positions, or the actual probe | Dataset/acquisition data | Physical inputs such as `diff3d`, coordinates, `probeGuess`, and optional realized-probe fields; these are data, not model settings |

In normal CLI and study workflows, supply config-file values and explicit
overrides. The entry point constructs and validates the dataclasses. Do not
manually construct both public and Torch representations merely to keep shared
fields synchronized.

The practical ownership rules are:

1. If changing a value changes generated arrays or their identity, it belongs
   to `SimulationConfig`.
2. If it changes the graph or differentiable forward model, it belongs to
   `ModelConfig`.
3. If it changes parameter updates, it belongs to `TrainingConfig`.
4. If it changes only reconstruction after training, it belongs to
   `InferenceConfig`.
5. If it changes Torch devices, processes, loaders, or Trainer mechanics, it
   belongs to `PyTorchExecutionConfig`.
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

`PyTorchExecutionConfig` controls devices, DDP strategy, workers, and Lightning
runtime mechanics after this routing decision. It does not select a dataset
schema or convert a grid-lines cache into the mmap schema. In particular,
requesting DDP does not cause a file-based or grid-lines entry point to switch
automatically to `PtychoDataModuleLightning`.

### Standalone NPZ Versus Grid-Lines Cached NPZ

These formats may both contain a field named `diffraction`, but they represent
different pipeline stages:

| Format | Typical contents | Consumer |
|---|---|---|
| Standalone scan NPZ | One ungrouped 3-D diffraction stack, `xcoords`, `ycoords`, `probeGuess`, and the other acquisition fields required by the selected loader | `RawData` or the native Torch mmap writer |
| Grid-lines cached NPZ | Pre-grouped/channelized `diffraction`, `Y_I`, `Y_phi`, `coords_nominal`, `coords_true`, `YY_full`, and optional `probe_simulated` | Grid-lines cached-dataset adapter |

The presence of the same field name does not make the formats interchangeable.
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

The normative standalone and batch shapes remain owned by
[PtychoPINN Core Contract](specs/spec-ptycho-core.md) and
[Torch Loader and Batch Contract](specs/spec-ptycho-interfaces.md). The routing
description above is an implementation guide: it explains which current
entry-point consumes each contract without turning historical runner choices
into new format requirements.

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
default.

## Developer Architecture

### One Meaning, Several Representations

The configuration system has several representations because it serves a
public API, two backends, checkpoint reconstruction, and legacy modules. These
representations are not co-equal sources of truth:

| Representation | Role | Should users edit it directly? |
|---|---|---|
| `ptycho.config.config` dataclasses | Public/shared configuration contract and legacy projection | Yes, when using the Python API |
| Factory-resolved `ptycho_torch.config_params` dataclasses | Torch data, topology, physics, training, and inference carriers after defaults, aliases, and object policy are materialized | Usually no; use the closed factory or a study wrapper |
| `TrainingPayload` / `InferencePayload` | Phase-local bundles returned by the factory | No; consume them |
| `ModelSpec("torch-model-spec-portable-v2")` | Derived, sealed Torch graph/state identity used for construction and reload | No |
| `PyTorchExecutionConfig` | Torch runtime/Trainer mechanics and current backend-specific optimizer controls; never a model-topology owner | Yes |
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
User / study / CLI values
              │
              ▼
    create_training_payload()
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
              ├─ tf_training_config ──► update_legacy_dict() ──► params.cfg
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
              └─ execution_config ──► Trainer / DataLoader / optimizer setup
```

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
- `PyTorchExecutionConfig` does not own model topology. Historical structural
  inputs accepted there are compatibility aliases mapped one-way into Torch
  `ModelConfig`; generators read only the resolved model config.

Public code materializes the object policy with
`resolve_model_object_policy()`. Torch code uses
`resolve_torch_model_object_policy()` at its boundary. Downstream model code
must consume the resolved fields instead of reinterpreting `object_big`.

### Legacy Compatibility

Some TensorFlow-era modules still read the process-local
`ptycho.params.cfg`. Supported entry points therefore perform a one-way bridge:

```text
resolved dataclass ──► update_legacy_dict(params.cfg, config) ──► legacy consumer
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

Configuration precedence is entry-point specific:

- Generation CLIs apply retained explicit CLI overrides over
  `--simulation-config` values. Simulation files may be TOML, YAML, or JSON;
  omitted file fields use dataclass defaults, while omitting the file invokes
  the entry point's historical compatibility defaults.
- Training and inference CLIs retain their documented `--config`/CLI
  precedence.
- Unknown simulation keys and conflicting compatibility aliases are errors.
- Not every dataclass field has a CLI flag.

### Named CI Profile

Use `create_training_payload(..., profile="ci")` or the Torch training CLI's
`--profile ci` instead of assembling a partial count-intensity configuration
by hand.

The profile locks these coherent contract fields:

| Field | Required value |
|---|---|
| `scale_contract_version` | `ci_intensity_v2` |
| `measurement_domain` | `count_intensity` |
| `physics_forward_mode` | `rectangular_scaled` |
| `torch_loss_mode` | `poisson` |
| `loss_function` | `Poisson` |

It also supplies these profile defaults:

| Field | Default |
|---|---|
| `amplitude_physics_gain` | `1.0` |
| `rect_s1s2_trainable` | `True` |
| `rect_s1s2_init` | `data` |
| `cnn_output_mode` | `real_imag` |

An explicit contradiction of a locked contract field fails closed. Non-contract
profile defaults follow normal override precedence, then the downstream scaling
and model validators enforce coherence.

## Parameter Reference

The tables below are representative. The dataclass definitions in
`ptycho/config/config.py` and `ptycho_torch/config_params.py` are the complete
field lists.

### Generated Data (`SimulationConfig`)

`SimulationConfig` is a frozen nested recipe with `probe`, `object`, `scan`, and
`detector` sections. Load TOML, YAML, or JSON with
`load_simulation_config()`; unknown keys are errors.

Supported probe pipeline operations are ordered and composable:

| Operation | Meaning |
|---|---|
| `smooth:0.5` | Smooth complex amplitude and unwrapped phase at the current resolution. |
| `pad_preserve:128` | Center-pad the prepared complex probe without changing its values. |
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
| `gaussian_smoothing_sigma` | `float` | `0.0` | Optional model-time probe smoothing; zero disables it. |

### PyTorch Execution (`PyTorchExecutionConfig`)

`PyTorchExecutionConfig` owns Torch runtime behavior. It also currently carries
the learning-rate, scheduler, clipping, and accumulation inputs consumed by
Torch entry points. Those optimizer-adjacent fields do not make it a
model-topology owner.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `accelerator` | `str` | `'auto'` | Resolves to CUDA when available, otherwise CPU. |
| `devices` | `Union[int, Literal['auto']]` | `1` | Number of devices supplied to Lightning. |
| `strategy` | `str` | `'auto'` | Lightning strategy, including `ddp` for multi-device execution. |
| `precision` | `Literal['32-true','16-mixed','bf16-mixed']` | `'32-true'` | Torch numerical precision policy. |
| `num_workers` | `int` | `0` | DataLoader worker-process count. |
| `learning_rate` | `float` | `1e-3` | Base learning rate used by current Torch training entry points. |
| `scheduler` | `str` | `'Default'` | Torch scheduler selection at the execution boundary. |
| `gradient_clip_val` | `Optional[float]` | `None` | Optional gradient-clipping threshold. |
| `gradient_clip_algorithm` | `Literal['norm','value','agc']` | `'norm'` | Gradient-clipping algorithm. |
| `accum_steps` | `int` | `1` | Gradient-accumulation steps. |
| `logger_backend` | `Optional[str]` | `'csv'` | CSV, TensorBoard, MLflow, or disabled logging. |

Historical structural fields may still be accepted as deprecated factory
inputs. New code must configure topology through Torch `ModelConfig`; equal
dual input is accepted, conflicting dual input fails, and generators never
read execution config for topology.

### Training (`TrainingConfig`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `train_data_file` | `Optional[Path]` | `None` | Training dataset path. Required by training entry points. |
| `test_data_file` | `Optional[Path]` | `None` | Optional validation/test dataset path. |
| `batch_size` | `int` | `16` | Samples per batch. |
| `nepochs` | `int` | `50` | Number of training epochs. |
| `mae_weight` | `float` | `0.0` | Diffraction-space MAE weight. |
| `nll_weight` | `float` | `1.0` | Poisson negative-log-likelihood weight. |
| `realspace_mae_weight` | `float` | `0.0` | Object-domain MAE weight. |
| `realspace_weight` | `float` | `0.0` | General real-space loss weight. |
| `nphotons` | `float` | `1e9` | Legacy/runtime compatibility value. Generated dose belongs to `SimulationConfig.detector.photons_per_pattern`. |
| `n_groups` | `Optional[int]` | `None` (`512` after public `TrainingConfig.__post_init__`) | Number of grouped samples used for training. |
| `n_images` | `Optional[int]` | `None` | **Deprecated** alias for `n_groups`. |
| `n_subsample` | `Optional[int]` | `None` | Number of raw images selected before grouping. |
| `subsample_seed` | `Optional[int]` | `None` | Reproducible subsampling seed. |
| `positions_provided` | `bool` | `True` | Use provided scan positions. |
| `probe_trainable` | `bool` | `False` | Optimize the probe jointly with the object model. |
| `intensity_scale_trainable` | `bool` | `True` | Optimize the global intensity scale. |
| `optimizer` | `Literal['adam','adamw','sgd']` | `'adam'` | Optimizer family. |
| `weight_decay` | `float` | `0.0` | Optimizer weight decay. |
| `scheduler` | `Literal['Default','Exponential','WarmupCosine','ReduceLROnPlateau']` | `'Default'` | Public scheduler choice. |
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
n_subsample: 10000
n_groups: 500
subsample_seed: 3
gridsize: 2
```

## Example YAML Configuration

```yaml
# Model
N: 64
gridsize: 2
architecture: cnn
n_filters_scale: 2
amp_activation: swish
object_layout: grouped_patches
training_canvas: relative_overlap
training_patch_weighting: central_mask

# Training
train_data_file: datasets/fly/fly001_prepared_train.npz
test_data_file: datasets/fly/fly001_prepared_test.npz
output_dir: results/my_experiment_run_1
nepochs: 100
batch_size: 32
n_groups: 4096
nll_weight: 1.0
mae_weight: 0.0
probe_trainable: true

# Compatibility values for already-materialized data
nphotons: 1e9
probe_scale: 4.0
gaussian_smoothing_sigma: 0.0
```

```bash
ptycho_train --config configs/my_experiment_config.yaml

# Explicit CLI values override the file for entry points that expose the flag.
ptycho_train --config configs/my_experiment_config.yaml --nepochs 10
```
