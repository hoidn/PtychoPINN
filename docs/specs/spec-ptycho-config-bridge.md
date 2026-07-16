# Spec: Configuration Bridge (TensorFlow ↔ PyTorch)

This specification defines the normative mapping between TensorFlow configuration dataclasses and PyTorch configuration singletons, and the one‑way bridging flow to the legacy `params.cfg` dictionary. It is authoritative for cross‑backend configuration behavior.

## 1. Scope and Goals

- Canonical source: dataclasses under `ptycho/config/config.py` (`SimulationConfig`, `ModelConfig`, `TrainingConfig`, `InferenceConfig`).
- PyTorch inputs: Config singletons under `ptycho_torch/config_params.py` (e.g., `DataConfig`, `ModelConfig`, `TrainingConfig`, `InferenceConfig`).
- Bridge adapter: `ptycho_torch/config_bridge.py` functions translate PyTorch config objects into TensorFlow dataclasses.
- Legacy state: `update_legacy_dict(params.cfg, config)` must be called before any legacy module usage.

Non‑goals: Define training semantics, runtime constraints, or data contracts (see other specs).

`SimulationConfig` exclusively owns properties baked into generated data:
probe source/transforms, synthetic-object recipe, scan geometry, detector/noise
settings, and generation seed. Model and training configuration do not acquire
ownership of these fields merely because a runtime compatibility structure
repeats `N` or grid size.

## 2. Canonical Flow (CONFIG‑001)

```
PyTorch config_params → config_bridge.py → TensorFlow dataclasses → update_legacy_dict() → params.cfg
```

Generated-data entry points use a separate one-way flow before simulation:

```text
SimulationConfig → update_legacy_dict(params.cfg, simulation_config) → legacy simulation
```

Rules:
- One‑way flow only. Do not mutate PyTorch configs during translation.
- Call `update_legacy_dict(params.cfg, config)` before: data loading, model construction, and any legacy module imports that read `params.cfg`.
- A simulation entry point must bridge its resolved `SimulationConfig` before
  invoking legacy generation, independently of any later model/training bridge.
- If repeated shape fields disagree (`simulation.N` versus `model.N`, or
  `simulation.scan.grid_size` versus `model.gridsize`), composition fails with
  a field-specific error. Implementations must not choose one silently.
- Training-only values such as epochs, optimizer, learning rate, batch size,
  loss weights, architecture, and output directory are invalid under the
  `simulation` namespace.

References:
- TensorFlow configs and bridge: `ptycho/config/config.py`
- Bridge adapter implementation: `ptycho_torch/config_bridge.py`
- Tests (conformance): `tests/torch/test_config_bridge.py`

## 3. Field Mapping Matrix (normative)

The following mappings are normative. Where not listed, identical field names/types are copied unchanged.

- Model/Geometry
  - `DataConfig.grid_size: Tuple[int,int]` → `ModelConfig.gridsize: int`
    - Transform: require square; use first element; else error.
  - `DataConfig.N: int` → `ModelConfig.N: Literal[64,128,256]`
    - Validate against allowed set.
  - `ModelConfig.mode: {'Unsupervised','Supervised'}` → `ModelConfig.model_type: {'pinn','supervised'}`
    - Map: Unsupervised→pinn, Supervised→supervised.
  - `ModelConfig.architecture: {'cnn','ffno','fno','hybrid','stable_hybrid','fno_vanilla','neuralop_uno','hybrid_resnet','hybrid_resnet_ffno_ptychoblock_encoder','hybrid_resnet_ptychoblock_ffno_encoder','spectral_resnet_bottleneck_net','spectral_resnet_bottleneck_linear_decoder','hybrid_resnet_ffno_bottleneck','hybrid_resnet_convnext_bottleneck'}` → `ModelConfig.architecture: {same 14 values}`
    - Direct pass-through. Generator architecture for PINN models. Default: 'cnn'. Source of truth: the `architecture` `Literal` in `ptycho.config.config.ModelConfig` and `ptycho_torch.config_params.ModelConfig`.
  - `ModelConfig.resnet_width: Optional[int]` → `ModelConfig.resnet_width: Optional[int]`
    - Direct pass-through. Used by the hybrid_resnet generator to fix bottleneck width.
  - `ModelConfig.amp_activation: str = 'silu'` (PyTorch field is an unvalidated `str`) → `ModelConfig.amp_activation: Literal['sigmoid','swish','softplus','relu'] = 'sigmoid'`
    - Map: silu/SiLU→swish; sigmoid/swish/softplus/relu pass through unchanged. Any other string is accepted by the unvalidated PyTorch field but rejected by `to_model_config`'s `activation_mapping` lookup, which raises `ValueError` for values outside `{silu, SiLU, sigmoid, swish, softplus, relu}`.
  - Public object policy:
    - `object_layout: Literal['single_patch','grouped_patches']`
    - `training_canvas: Literal['independent','relative_overlap']`
    - `training_patch_weighting: Literal['central_mask','uniform','probe']`
    - The supported layout/canvas pairs are
      `single_patch`/`independent` and
      `grouped_patches`/`relative_overlap`. The bridge resolves these fields
      before legacy mutation and derives the deprecated `object_big` /
      `object.big` Boolean projection. Partial pairs and contradictory dual
      old/new input fail closed.
    - PyTorch supports all three weighting values. TensorFlow supports
      `central_mask` only and rejects `uniform` or `probe` before model
      construction.
    - Unset public fields resolve to grouped patches, relative-overlap canvas,
      and central-mask weighting. Legacy-only `object_big` input remains
      accepted with deprecation signaling.

- Training lifecycle
  - `TrainingConfig.epochs: int` → `TrainingConfig.nepochs: int` (rename)
  - `DataConfig.K: int` → `TrainingConfig.neighbor_count: int`
  - `DataConfig.nphotons: float` → `TrainingConfig.nphotons: float`
  - `TrainingConfig.output_dir: PathLike` → `TrainingConfig.output_dir: Path` (normalize to `Path`)
  - `TrainingConfig.debug: bool` → used by PyTorch only; optional carry‑over to TF ignored (no TF field).

- Data paths
  - `DataConfig.train_data_file: PathLike` → `TrainingConfig.train_data_file: Path`
  - `DataConfig.test_data_file: PathLike` → `TrainingConfig.test_data_file: Path`
  - `InferenceConfig.model_path: PathLike` → `InferenceConfig.model_path: Path`

- Grouping / sampling
  - `DataConfig.n_groups: Optional[int]` → `TrainingConfig.n_groups: Optional[int]`
  - `DataConfig.n_images: Optional[int]` → `TrainingConfig.n_images: Optional[int]` (deprecated; preserved for compatibility)
  - `TrainingConfig.subsample_seed: Optional[int]` → `TrainingConfig.subsample_seed: Optional[int]` (unchanged)
  - `TrainingConfig.sequential_sampling: bool` → `TrainingConfig.sequential_sampling: bool` (unchanged)

- Loss/weights (if present in PyTorch configs)
  - `TrainingConfig.nll: bool` → `TrainingConfig.nll_weight: float`
    - Transform: True→1.0, False→0.0

- Default-divergence caveats (informational — bridge adapter behavior, not a field-mapping transform)
  - `DataConfig.grid_size` default `(2,2)` (`ptycho_torch.config_params.DataConfig`) vs `ModelConfig.gridsize` default `1` (`ptycho.config.config.ModelConfig`): a bridged run that never supplies an explicit `grid_size` starts from PyTorch's 2×2-grouping default, not TensorFlow's single-patch default. Set `grid_size` explicitly when parity with TF defaults is required.
  - `positions_provided`: `to_training_config` unconditionally hardcodes `True`, matching the TensorFlow `TrainingConfig.positions_provided` default. This diverges from the legacy `params.cfg` module-level default `False`, which is only in effect before `update_legacy_dict()` has ever run. PyTorch `TrainingConfig` has no `positions_provided` field to translate — the bridge value is not derived from any PyTorch input.

- Non-bridged Torch-only knobs (no TensorFlow counterpart, or present-but-not-forwarded by the current adapter)
  - `ModelConfig.generator_output_mode: Literal['real_imag','amp_phase_logits','amp_phase']` exists identically on both sides, both default `'real_imag'`, but `to_model_config` does not include it in the `kwargs` it forwards — the bridged TF config takes the TF default unless supplied via `overrides`.
  - `ModelConfig.cnn_output_mode: Literal['amp_phase','real_imag']` — Torch-only CNN decoder output selector; no TF field.
  - `ModelConfig.physics_forward_mode: Literal['amplitude','rectangular_scaled']` — Torch-only forward-model physics selector; no TF field.
  - `ModelConfig.rect_s1s2_trainable: bool` — Torch-only; only consulted when `physics_forward_mode='rectangular_scaled'`; no TF field.
  - Loss-weight mapping asymmetry: TensorFlow expresses loss balance as continuous weights (`TrainingConfig.mae_weight`, `nll_weight`, `realspace_mae_weight`, `realspace_weight`); PyTorch instead selects a loss family via `ModelConfig.loss_function: Literal['MAE','Poisson']` plus independent regularization terms `amp_loss`/`phase_loss`/`amp_loss_coeff`/`phase_loss_coeff` and `TrainingConfig.nll: bool`. These are not equivalent parameterizations; only the single `nll`→`nll_weight` cast is field-mapped.

Defaults & Precedence:
- If a PyTorch field is missing, the bridge may accept an `overrides: Dict[str,Any]` to supply required values.
- Absent optional fields fall back to TensorFlow dataclass defaults.

## 4. Validation and Error Conditions

- Non‑square `grid_size` → error (TensorFlow backend assumes square grids).
- Unsupported activations (not in TF enum) → error.
- `N` outside allowed set → error.
- Type normalization failures (e.g., invalid `PathLike`) → error.
- Partial or unsupported object layout/canvas pairs → error.
- Contradictory `object_big` and canonical object policy → error before bridge mutation.
- TensorFlow with `training_patch_weighting='uniform'` or `'probe'` → error
  before model construction.

## 5. Bridging to Legacy (params.cfg)

- After translation, call `update_legacy_dict(params.cfg, config)` with the resulting TF dataclass.

### 5.1 Field ownership

Configuration ownership is semantic, not based on matching field names across
dataclasses:

| Concern | Owner | Mapping boundary |
|---|---|---|
| Synthetic object, scan, detector, and probe construction | `ptycho.config.config.SimulationConfig` and its nested records | Explicit simulation adapters only; these fields do not enter model or execution config |
| Acquired data shape, channels, photon/count profile, and grouping | Canonical workflow/data contract; current Torch carrier `ptycho_torch.config_params.DataConfig` | Declared `config_bridge` mappings plus factory data constructor |
| Shared model identity | `ptycho.config.config.ModelConfig` | `config_bridge.to_model_config` maps declared shared fields |
| Torch-only generator topology and structural physics/output choices | `ptycho_torch.config_params.ModelConfig` | The training factory's closed structural constructor; these fields determine graph/state-dict identity |
| Optimization and training schedule | Canonical `TrainingConfig`; current Torch carrier `ptycho_torch.config_params.TrainingConfig` | Declared training bridge and factory constructor |
| Devices, distributed strategy, loaders, logging, and trainer mechanics | `PyTorchExecutionConfig` | Passed to runtime orchestration only; it cannot be read by graph constructors |
| Deprecated flat compatibility state | `ptycho.params.cfg` | One-way `update_legacy_dict`; never a source for new structured configuration |

The Hybrid, FFNO-encoder, and spectral topology names historically exposed on
`PyTorchExecutionConfig` are deprecated input aliases. The training factory maps
an explicitly supplied alias one-way into Torch `ModelConfig`, warns, accepts an
equal structural value, and rejects a conflicting structural value. Omitted
execution aliases do not overwrite model defaults or explicit model input.
Downstream generator construction reads only Torch `ModelConfig`. Unknown flat
factory override names are rejected rather than silently dropped.
- This applies to both backends to keep legacy modules in sync (e.g., physics routines, helpers).
- `update_legacy_dict` performs KEY_MAPPINGS translation to legacy names and value serialization.

### 5.1 Simulation-owned legacy keys

The `SimulationConfig` bridge is deliberately narrower than the training and
inference bridges. It flattens only generated-data properties:

- `N` → `N`
- `scan.grid_size` (square) → `gridsize`
- `object.image_size` (square) → `size`
- `scan.offset`, outer offsets, train/test group counts, and buffer → the
  corresponding legacy scan fields
- `detector.photons_per_pattern` → `nphotons`
- `object.kind` and `object.set_phi` → legacy object-source fields
- `seed` → `npseed`
- probe source path, normalized transform pipeline, and simulation-time mask →
  probe-construction lineage fields

This bridge does not transfer model-time probe masks, architecture, optimizer,
epochs, or other training/runtime ownership. Optional `None` values remain
absent rather than clearing unrelated legacy state.

## 6. Conformance (Testing Requirements)

- Translation must satisfy the unit tests in `tests/torch/test_config_bridge.py` (mapping, overrides, and params.cfg population).
- Workflows must show `update_legacy_dict(params.cfg, config)` called before data operations (see `docs/workflows/pytorch.md`).

## 7. Compliance & Prohibitions (Normative)

To preserve a single source of truth between the dataclasses and `params.cfg`, the following practices are mandatory:

1. **No implicit initialization:** Modules SHALL NOT read from or mutate `ptycho.params` at module scope (import time). All `params` access MUST occur inside functions or methods after configuration synchronization.
2. **Bridge mandate:** Every execution entry point (CLI, notebook, script, API) MUST call `update_legacy_dict(params.cfg, config_object)` immediately after resolving configuration and BEFORE importing or executing modules that read from `ptycho.params`.

## 8. Examples

```python
from pathlib import Path
from ptycho_torch.config_params import DataConfig, ModelConfig as PTModel, TrainingConfig as PTTrain
from ptycho_torch import config_bridge
from ptycho.config.config import update_legacy_dict
import ptycho.params as params

pt_data = DataConfig(N=128, grid_size=(2,2), nphotons=1e9, K=7,
                     train_data_file=Path('train.npz'), test_data_file=Path('test.npz'))
pt_model = PTModel(mode='Unsupervised', amp_activation='silu')
pt_train = PTTrain(epochs=50, output_dir=Path('outputs/run1'))

tf_model = config_bridge.to_model_config(pt_data, pt_model)
tf_train = config_bridge.to_training_config(
    tf_model, pt_data, pt_model, pt_train,
    overrides=dict(n_groups=512)
)
update_legacy_dict(params.cfg, tf_train)
```

## 9. References

- TensorFlow configs: `ptycho/config/config.py`
- PyTorch configs: `ptycho_torch/config_params.py`
- Bridge adapter: `ptycho_torch/config_bridge.py`
- Workflows: `docs/workflows/pytorch.md`
- Tests: `tests/torch/test_config_bridge.py`
