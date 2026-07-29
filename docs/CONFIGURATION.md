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
| The measured diffraction, positions, or actual probe | The dataset/acquisition record | Physical inputs such as `diff3d`, coordinates, and `probeGuess`; these are data, not model settings |

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

## The Probe Lifecycle

Several fields contain the word “probe,” but they act at different stages:

```text
SimulationConfig.probe
  source + transform_pipeline + optional simulation mask
                         │
                         ▼
             generated dataset probeGuess
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
| Dataset `probeGuess` | The actual stored complex probe supplied by the acquisition or simulator. For a synthetic dataset, it already contains the configured simulation transforms and mask. |
| CI `probe_physical` / `probe_training` | Named physical and normalized training views used by the count-intensity/CI contract. Legacy normalized-amplitude paths use their existing generic probe carrier instead. These are data representations, not independently chosen configs. |
| `ModelConfig.probe_mask`, `probe_mask_diameter`, `probe_mask_sigma` | Apply an additional model-time support prior inside the differentiable forward model. They do not alter the saved dataset. |
| `ModelConfig.probe_big` | Historical name for the CNN decoder’s learned complementary outer spatial support. It does **not** resize, pad, extrapolate, or construct the physical probe. |

For an exact matched synthetic replay, the stored `probeGuess` already embodies
the simulation recipe. Therefore `ModelConfig.probe_mask=False` normally uses
that probe without applying a second mask. Enable a model-time mask only when
the experiment intentionally adds that support prior.

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

The normal Torch training flow is:

```text
User / study / CLI values
              │
              ▼
    create_training_payload()
              │
              ├─ DataConfig
              ├─ Torch ModelConfig
              ├─ Torch TrainingConfig
              ├─ Torch InferenceConfig
              ├─ canonical TrainingConfig projection
              ├─ ExecutionRequest
              └─ applied-overrides audit
              │
              ├─ canonical projection ──► scoped legacy bridge
              │                              └─► params.cfg ──► named legacy leaves
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
              └─ ExecutionRequest ──► capability resolution
                                      └─► PyTorchExecutionConfig
                                           └─► Trainer / DataLoader setup
```

The canonical and Torch model records overlap only where the backends share a
public concept. Torch-only topology and physics fields remain in the Torch
carrier. `derive_model_spec()` checks the shared fields rather than silently
choosing one representation.

### Model and Artifact Identity

`ModelSpec` is derived after configuration resolution. It freezes every Torch
structural field needed to reconstruct the model and makes checkpoint identity
independent of later mutable defaults.

Current Torch artifacts use:

- `torch-model-spec-v2` for sealed model identity;
- `torch-artifact-v2` for the enclosing data/model/training/inference identity.

Version 2 stores `object_layout`, `training_canvas`, and
`training_patch_weighting` as the structural object policy. It does not retain
deprecated `object_big` as a second owner. Frozen v1 artifacts remain readable
and are deterministically upgraded during decoding. TensorFlow artifact formats
are unchanged by this Torch schema migration.

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

- Generation CLIs apply retained explicit CLI overrides over `--simulation-config` values. Simulation files may be TOML, YAML, or JSON; omitted file fields use dataclass defaults, while omitting the file invokes the entry point's historical compatibility defaults.
- Training and inference CLIs retain their documented `--config`/CLI precedence.
- Unknown simulation keys and conflicting legacy aliases are errors; not every dataclass field has a CLI flag.

## Parameter Reference

### Generated data (SimulationConfig)

`SimulationConfig` is a frozen nested recipe with `probe`, `object`, `scan`, and `detector` sections. Load TOML, YAML, or JSON with `load_simulation_config()`; unknown keys are errors. Generation CLIs use explicit CLI value over config-file value over the historical no-file default.

Supported probe pipeline operations are ordered and composable:

| Operation | Meaning |
|---|---|
| `smooth:0.5` | Smooth complex amplitude and unwrapped phase at the current resolution. |
| `pad_preserve:128` | Center-pad the prepared complex probe without changing its values. |
| `interp:128` | Cubic real/imaginary interpolation. |
| `pad_extrapolate:128` | Legacy behavior: fit and evaluate one quadratic phase over the entire target probe, including the center. |
| `pad_extrapolate_boundary_matched:128` | Center-copy the prepared source exactly and solve a C0 harmonic Dirichlet correction only outside it, relaxing to the fitted quadratic at the target perimeter. This operation must be last. |

The canonical new outer-only form is `smooth:0.5|pad_extrapolate_boundary_matched:128`: smoothing happens before extension, and no post-extension operation may alter the copied center. Changing a pipeline changes the simulation and dataset recipe digests; it cannot reuse a dataset generated by another pipeline.

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

**Illustrative subset — full field list: `ModelConfig` in `ptycho/config/config.py`.** The `architecture` field's full 14-value `Literal` is enumerated authoritatively in `docs/specs/spec-ptycho-config-bridge.md` §3.

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
| `pad_object` | `bool` | `True` | Controls padding behavior in the model. |
| `probe_scale` | `float` | `4.0` | A normalization factor for the probe's amplitude. |
| `gaussian_smoothing_sigma` | `float` | `0.0` | Standard deviation for the Gaussian filter applied to the probe. 0.0 means no smoothing. |

### PyTorch Execution Requests and Resolved Runtime

**Illustrative subset — full field list: `PyTorchExecutionConfig` in `ptycho/config/config.py`.**

Callers provide unresolved runtime values through `ExecutionRequest`, normally
via the request builder or explicit CLI flags. Capability resolution produces
`PyTorchExecutionConfig`, which owns only effective device, distributed
strategy, DataLoader, logging, checkpoint, and Trainer runtime mechanics. A
bare resolved carrier is not accepted as a new request.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `accelerator` | `str` | request: `'auto'` | The request accepts `auto`, `cpu`, `gpu`, `cuda`, or `mps`; the resolved carrier contains the selected concrete runtime. |
| `devices` | `Union[int, Literal['auto']]` | `1` | Number of devices supplied to Lightning. |
| `strategy` | `str` | `'auto'` | Lightning execution strategy, including `ddp` for multi-device execution. |
| `precision` | `Literal['32-true','16-mixed','bf16-mixed']` | `'32-true'` | Torch numerical precision policy. |
| `num_workers` | `int` | `0` | DataLoader worker-process count. |
| `logger_backend` | `Optional[str]` | `'csv'` | Logging backend: CSV, TensorBoard, MLflow, or disabled. |

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
| `n_groups` | `Optional[int]` | `None` (`512` after `TrainingConfig.__post_init__` when unset) | Number of groups to use from the dataset. Each group contains 1 image for gridsize=1, or gridsize² images for gridsize>1. **Replaces deprecated `n_images` parameter.** |
| `n_images` | `Optional[int]` | `None` | **[DEPRECATED]** Legacy parameter name for `n_groups`. Still supported for backward compatibility but will show deprecation warnings. New code should use `n_groups`. |
| `n_subsample` | `Optional[int]` | `None` | Number of images to subsample from the dataset before grouping (independent control). When provided, controls data selection separately from grouping. |
| `subsample_seed` | `Optional[int]` | `None` | Random seed for reproducible subsampling. Ensures consistent data selection across runs. |
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
| `n_groups` | `Optional[int]` | `None` | Number of groups to use for inference. If None, uses all available. Each group contains 1 image for gridsize=1, or gridsize² images for gridsize>1. **Replaces deprecated `n_images` parameter.** |
| `n_images` | `Optional[int]` | `None` | **[DEPRECATED]** Legacy parameter name for `n_groups`. Still supported for backward compatibility but will show deprecation warnings. New code should use `n_groups`. |
| `n_subsample` | `Optional[int]` | `None` | Number of images to subsample from test data (independent control). When provided, controls data selection separately from grouping. |
| `subsample_seed` | `Optional[int]` | `None` | Random seed for reproducible subsampling during inference. |
| `debug` | `bool` | `False` | Enable debug mode for additional logging. |

## Understanding Sampling Parameters

The project supports two modes for controlling data sampling:

### Legacy Mode (Backward Compatible)
When only the deprecated `n_images` parameter is used, it behaves as `n_groups`:
- **gridsize=1**: `n_images` specifies how many groups of 1 image each to use
- **gridsize>1**: `n_images` specifies how many neighbor groups to create (total patterns = n_images × gridsize²)

**Note**: New code should use `n_groups` instead of the deprecated `n_images` parameter.

### Independent Control Mode (New)
When `n_subsample` is provided, you get independent control:
- **`n_subsample`**: Controls how many images to randomly select from the dataset
- **`n_groups`**: Controls how many groups to use for training/inference
- **`subsample_seed`**: Ensures reproducible random selection

**Note**: The deprecated `n_images` parameter can still be used in place of `n_groups` but will show warnings.

#### Example Scenarios:
```yaml
# Dense grouping: Use almost all subsampled data in groups
n_subsample: 1200
n_groups: 1000  # Creates 1000 groups of 4 images each (gridsize=2)
gridsize: 2

# Sparse grouping: Subsample large dataset, use subset for groups  
n_subsample: 10000
n_groups: 500   # Creates 500 groups of 4 images each (gridsize=2)
gridsize: 2

# Memory-constrained: Limit data loading
n_subsample: 5000
n_groups: 2000  # Creates 2000 groups of 1 image each (gridsize=1)
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
n_groups: 4096  # Use 4096 groups for this training run

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
3. **Use `n_groups` instead of deprecated `n_images`** in new configurations.
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
n_groups: 1000  # Always means "number of groups" regardless of gridsize
```

```yaml
# Old (deprecated but still accepted)
object_big: true

# New (recommended)
object_layout: grouped_patches
training_canvas: relative_overlap
training_patch_weighting: central_mask
```
