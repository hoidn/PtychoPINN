## PtychoPINN Configuration Management API Specification

### 1. Overview

The `ptychopinn` configuration system is a hybrid architecture designed to support both modern, type-safe workflows and maintain backward compatibility with over 20 legacy modules. It consists of two primary components:

1.  **Modern Configuration Types (`config/config.py`):** The authoritative source of truth for all parameters. Public training uses a nested Pydantic `BaseSettings` model; simulation, model, and inference use validated standard-library dataclasses.
2.  **Legacy Global Dictionary (`params.py`):** A global, mutable dictionary (`ptycho.params.cfg`) that is used by older modules throughout the codebase.

The API is built around a **one-way data flow**: parameters are defined in the modern configuration records and then propagated to the legacy dictionary. Direct manipulation of the legacy dictionary is strongly discouraged in new code.

The central function that bridges these two systems is `ptycho.config.config.update_legacy_dict()`. Any external system, like `ptychodus`, **must** use this bridge to configure `ptychopinn` correctly.

**⚠️ PyTorch Requirement:** PyTorch `>= 2.2` is a **mandatory runtime dependency** for the PyTorch backend (`ptycho_torch/`). The project declares `torch>=2.2` in `pyproject.toml`. The TensorFlow backend (`ptycho/`) continues to function independently, but callers integrating the PyTorch stack **must** ensure a supported PyTorch installation is available. For installation guidance, see the PyTorch workflow guide at <doc-ref type="workflow">docs/workflows/pytorch.md</doc-ref>.

### 2. Core Components

#### 2.1. Modern Configuration Types

These types, defined in `config/config.py`, are the primary way to specify configuration.

- **`ModelConfig`**: Defines the neural network architecture and core physics parameters.
  - `N`: The size of the input diffraction patterns (e.g., 64, 128).
  - `gridsize`: The number of adjacent scan positions to process simultaneously (e.g., `gridsize=2` means a 2x2 group).
  - `model_type`: The type of model, either `'pinn'` or `'supervised'`.
  - `architecture`: `Literal` selecting the generator/model architecture — five values
    (`'cnn'`, `'ffno'`, `'fno'`, `'fno_vanilla'`, `'neuralop_uno'`), default `'cnn'`
    (`ptycho.config.config.ModelConfig`). Routes generator resolution on the PyTorch backend
    (`ptycho_torch.generators.registry.resolve_generator`); the TensorFlow backend does not branch on it.
    The public validator enforces the selected architecture domain and shared
    scalar/object-policy constraints. Architecture-specific construction
    constraints are enforced by the selected PyTorch generator and ModelSpec
    boundary.
  - `amp_activation`: The activation function for the amplitude decoder.
  - `object_layout`: `'single_patch'` or `'grouped_patches'`; owns the
    reconstructed component layout.
  - `training_canvas`: `'independent'` or `'relative_overlap'`; must be supplied
    with `object_layout`. The supported pairs are
    `single_patch`/`independent` and
    `grouped_patches`/`relative_overlap`.
  - `training_patch_weighting`: `'central_mask'`, `'uniform'`, or `'probe'`;
    selects grouped-patch training assembly. TensorFlow supports
    `'central_mask'` only, while PyTorch supports all three values.
  - `object_big`: Deprecated optional compatibility alias. When supplied alone,
    `False` maps to `single_patch`/`independent` and `True` maps to
    `grouped_patches`/`relative_overlap`. New callers should use the three
    canonical fields.
  - `probe_big`, `pad_object`: Independent booleans controlling probe support
    and object padding; neither is implied by the object layout.
  - Additional fields consumed by `ptychodus`, including `n_filters_scale`, `probe_mask`, `probe_scale`, and `gaussian_smoothing_sigma` 
    (`ptycho.config.config.ModelConfig`, `ptychodus.model.ptychopinn.reconstructor`). These must be honoured by any
    alternative backend because they drive grouping geometry, probe handling, and image smoothing downstream.

- **`TrainingConfig`**: A Pydantic `BaseSettings` model defining the training workflow.
  - `model`: A nested `ModelConfig` instance.
  - `data`: A nested `DataConfig` containing `train_data_file`,
    `test_data_file`, and `nphotons`.
  - `sampling`: A nested `SamplingConfig` containing `n_groups`, deprecated
    `n_images`, raw-row selection, neighbor grouping, and oversampling controls.
  - `loss` and `tf_loss`: Nested Torch loss selection and TensorFlow loss
    weights, respectively.
  - `gradient_clip`, `optimizer`, and `scheduler`: Nested optimization-policy
    records.
  - `batch_size`: The number of samples per training step.
  - `nepochs`: The total number of training epochs.
  - Direct controls include `positions_provided`, `probe_trainable`,
    `intensity_scale_trainable`, `output_dir`, and `backend`.
  - All nested models forbid extra fields and revalidate instances. Although
    the root type is `BaseSettings`, implicit environment, dotenv, and secrets
    sources are disabled. File mappings and explicit CLI patches are loaded by
    the entry point, deep-merged, and validated through
    `resolve_training_config()`.
  - Deprecated compatibility: the historical flat root spellings of the
    `data`, `tf_loss`, and `sampling` fields (`train_data_file`,
    `test_data_file`, `nphotons`, `mae_weight`, `nll_weight`,
    `realspace_mae_weight`, `realspace_weight`, `n_groups`, `n_images`,
    `n_subsample`, `subsample_seed`, `neighbor_count`, `enable_oversampling`,
    `neighbor_pool_size`, `sequential_sampling`) are accepted at the root and
    lifted into their nested owners during validation, emitting
    `DeprecationWarning`. An equal flat/nested duplicate is accepted once; an
    unequal duplicate is rejected with both spellings identified. All other
    root extras remain forbidden. New callers must use the nested fields.

- **`InferenceConfig`**: Defines parameters for the reconstruction (inference) workflow.
  - `model`: A nested `ModelConfig` instance.
  - `model_path`: `pathlib.Path` to the trained model directory.
  - `test_data_file`: `pathlib.Path` to the data to be reconstructed.
  - Extended options used by `ptychodus`, including `n_groups`, `n_subsample`, `neighbor_count`, `debug`, and `output_dir`
    (`ptycho.config.config.InferenceConfig`, `ptychodus.model.ptychopinn.reconstructor`).

#### 2.2. Legacy Global Dictionary (`params.py`)

**⚠️ WARNING:** This component is a global mutable state and is maintained for backward compatibility only. Do not directly modify `ptycho.params.cfg` in new code.

- **`ptycho.params.cfg`**: A global dictionary that holds all configuration parameters in a flat key-value structure.
- **`ptycho.params.get(key)`**: The function used by legacy modules to retrieve a parameter. It includes logic for derived parameters (e.g., `bigN`).
- **`ptycho.params.set(key, value)`**: The function to update a value in the global dictionary.

#### 2.3. The Compatibility Bridge (`config/config.py`)

This is the most critical part of the configuration API. It translates modern configuration records into the legacy format.

- **`update_legacy_dict(cfg: dict, config_obj: Any)`**:
  - **Purpose**: To populate the legacy `ptycho.params.cfg` dictionary from a modern configuration record (`TrainingConfig` or `InferenceConfig`).
  - **Mechanism**: It calls `dataclass_to_legacy_dict()` to perform the translation and then updates the global `cfg` dictionary. This is the **only supported way** to configure `ptychopinn` from an external caller like `ptychodus`.

- **`dataclass_to_legacy_dict(obj: Any)`**:
  - **Purpose**: Translates a supported configuration instance into a flat dictionary with legacy key names. The historical function name is retained even though public training is now Pydantic.
  - **Mechanism**:
    1.  Converts a supported dataclass or Pydantic training record to primitive values.
    2.  Flattens the nested training sections and `ModelConfig` into their legacy keys.
    3.  It resolves the public object policy, rejecting partial, unsupported,
        or contradictory old/new representations before mutating the legacy
        dictionary.
    4.  It applies the `KEY_MAPPINGS` dictionary to translate modern,
        snake_case field names to legacy dot-separated names. The deprecated
        `object_big` alias is materialized from `object_layout` and projected as
        the exact legacy `object.big` Boolean.
    5.  It automatically converts `pathlib.Path` objects to strings, as the legacy system expects string paths.

- **PyTorch Configuration Adapters (`ptycho_torch.config_bridge`):**
  - **Purpose**: Translate Torch-side configuration objects to public configuration instances, enabling PyTorch workflows to populate `params.cfg` through `update_legacy_dict`.
  - **Key Functions**:
    - `to_model_config(data: DataConfig, model: ModelConfig, overrides=None) -> TFModelConfig`: Converts PyTorch `DataConfig` and `ModelConfig` to TensorFlow `ModelConfig`, handling critical transformations such as `grid_size` tuple → `gridsize` int, `mode` enum → `model_type` enum, and activation name normalization.
    - `to_training_config(model: TFModelConfig, data: DataConfig, pt_model: ModelConfig, training: TrainingConfig, overrides=None) -> TFTrainingConfig`: Translates PyTorch training parameters to TensorFlow `TrainingConfig`, converting `epochs` → `nepochs`, `K` → `neighbor_count`, `nll` bool → `nll_weight` float, and requiring explicit `overrides` for fields missing in PyTorch configs (e.g., `train_data_file`, `n_groups`).
    - `to_inference_config(model: TFModelConfig, data: DataConfig, inference: InferenceConfig, overrides=None) -> TFInferenceConfig`: Converts PyTorch inference parameters to TensorFlow `InferenceConfig`, mapping `K` → `neighbor_count` and requiring `overrides` for `model_path` and `test_data_file`.
  - **Contract**: These adapters MUST produce public configuration records compatible with `update_legacy_dict` and maintain behavioral parity with direct public construction. Consumers (e.g., `ptychodus` PyTorch integration) MUST call these adapters before invoking `update_legacy_dict` to ensure correct `params.cfg` population. Implementation details and field mappings are documented in `ptycho_torch.config_bridge` and tested via `tests/torch/test_config_bridge.py`.

### 3. API Specification and Data Flow

The correct and only supported way for an external system to configure `ptychopinn` is as follows:

1.  **Instantiate a Configuration Record**: Create an instance of `TrainingConfig` or `InferenceConfig` with the desired parameters.
2.  **Call the Bridge Function**: Pass the legacy `cfg` dictionary and the newly created record to `update_legacy_dict`.

This one-way data flow ensures that modern configuration remains the single source of truth while correctly populating the state required by legacy modules.

```mermaid
graph TD
    A[External Caller, e.g., ptychodus] --> B{1. Instantiate<br/>TrainingConfig};
    B --> C{2. update_legacy_dict(params.cfg, config)};
    C --> D{3. dataclass_to_legacy_dict(config)};
    D --> E{4. Resolve object policy};
    E --> F{5. Apply KEY_MAPPINGS<br/>(derived object_big -> object.big)};
    F --> G{6. Update ptycho.params.cfg};
    G --> H[Legacy Modules<br/>(e.g., model.py, diffsim.py)];
    H --> I{7. Access config via<br/>params.get('key')};

    style C fill:#f9f,stroke:#333,stroke-width:2px
    style F fill:#ccf,stroke:#333,stroke-width:2px
```

**Example Implementation (as used by `ptychodus`):**

```python
# In ptychodus.model.ptychopinn.reconstructor.py

# Import necessary components from ptychopinn
from ptycho.config.config import InferenceConfig, ModelConfig, update_legacy_dict
import ptycho.params

# 1. Create a ModelConfig from ptychodus settings
model_config = ModelConfig(
    N=model_size,
    gridsize=self._model_settings.gridsize.get_value(),
    # ... other parameters
)

# Create an InferenceConfig for reconstruction
inference_config = InferenceConfig(
    model=model_config,
    model_path=Path(),  # Not used directly here, but required
    test_data_file=Path(), # Not used directly here, but required
    # ... other parameters
)

# 2. Call the bridge function to update ptychopinn's global state
update_legacy_dict(ptycho.params.cfg, inference_config)

# Now, any subsequent calls to ptychopinn functions that rely on
# ptycho.params.get() will use the correct configuration.
```

### 4. PtychoPINN Reconstructor Contract

`ptychodus` integrates with `ptychopinn` through `PtychoPINNTrainableReconstructor`
(`ptychodus.model.ptychopinn.reconstructor`). A replacement backend must honor the
following behavioural contract in addition to the configuration bridge.

#### 4.1. Entry Points and Lifecycle

- **Instantiation**: Two reconstructor instances (`PINN`, `Supervised`) are created by
  `PtychoPINNReconstructorLibrary` when the `ptychopinn` package can be imported
  (`ptychodus.model.ptychopinn.core`). Both share a settings registry and must support
  runtime updates to model/training/inference knobs.
- **Capability Flags**: `MODEL_FILE_NAME = 'wts.h5.zip'` and file filters exposed through
  `get_model_file_filter()` / `get_training_data_file_filter()` dictate the on-disk formats the UI offers.

#### 4.2. Configuration Handshake

- `reconstruct()` and `train()` assemble `ModelConfig`, `InferenceConfig`, and `TrainingConfig` instances from
  live settings (`ptychodus.model.ptychopinn.reconstructor`). Every field in
  these records must be respected because they directly feed downstream modules.
- After source resolution and structural, runnable, and resource validation,
  `update_legacy_dict(ptycho.params.cfg, validated_config)` is called immediately
  before backend dispatch or another legacy consumer. The backend must continue
  to populate `ptycho.params.cfg` so that legacy consumers (`ptycho.raw_data`,
  `ptycho.loader`, `ptycho.model`) observe consistent validated values.
- Loaded models overwrite `params.cfg` via `load_inference_bundle`, so a backend must either replicate that side
  effect or provide an alternative hook (`ptycho.workflows.components.load_inference_bundle`).
- **PyTorch Import Requirement**: The PyTorch backend (`ptycho_torch/`) raises an actionable `RuntimeError` with installation guidance if `torch` cannot be imported. All modules in `ptycho_torch/` assume PyTorch availability and fail fast with clear error messages directing users to install `torch>=2.2`.

#### 4.3. Data Ingestion and Grouping

- `create_raw_data()` converts `ReconstructInput` objects to `RawData` by collecting NumPy arrays for
  coordinates, diffraction patterns, probe guess, and a single object layer
  (`ptychodus.model.ptychopinn.reconstructor.create_raw_data`). A backend must accept the same layout.
- Grouped data is generated through `RawData.generate_grouped_data`, which expects
  `params.cfg['gridsize']`, `neighbor_count`, and optional sampling controls to be already populated
  (`ptycho.raw_data.RawData.generate_grouped_data`). The function returns a dictionary with keys `diffraction`, `Y`,
  `coords_offsets`, `coords_relative`, `coords_start_offsets`, `coords_start_relative`, `coords_nn`,
  `coords_start_nn`, `nn_indices`, `objectGuess`, and (after normalization) `X_full`.
  There is no `local_offsets` key in this dict; `local_offsets` is a
  `PtychoDataContainer` attribute, populated from the `coords_relative` key when `ptycho.loader.load`
  constructs the container. `ptycho.loader.load` consumes this dictionary. Any replacement must either continue producing this dictionary or adapt
  `ptychodus` accordingly.
- `PtychoDataContainer` instances expose TensorFlow tensors (`X`, `local_offsets`, `global_offsets`, complex
  ground truth) that are passed straight into the model (`ptycho.loader.PtychoDataContainer`). Shapes depend on
  `N` and `gridsize` and must match the TensorFlow model signature described below.

#### 4.4. TensorFlow Inference Behaviour

- Inference uses `tf.keras.Model.predict` with two inputs: scaled diffraction data and local position offsets
  (`ptychodus.model.ptychopinn.reconstructor`). The backend model must expose the same
  call signature and support eager execution.
- `ptycho.model.params()['intensity_scale']` supplies the multiplicative scale factor. Backends must provide the
  `intensity_scale` parameter (and optionally `intensity_scale.trainable`) inside `params.cfg`.
- Output tensors are stitched into a 2D array via `ptycho.tf_helper.reassemble_position`, which expects
  `global_offsets` and model output channels to follow the existing TensorFlow helper contract.

#### 4.4.1. Inference Model I/O Shapes (Authoritative)

The inference bundle (`wts.h5.zip`) MUST contain two models: `autoencoder` and
`diffraction_to_obj`. External callers SHALL invoke `diffraction_to_obj`. Shapes use:
- B: batch size; N: patch size (e.g., 64, 128); C: channels = `gridsize²` (1 for gs=1; 4 for gs=2).
All diffraction inputs are amplitude (sqrt intensity) per DATA‑001.

TensorFlow — `diffraction_to_obj`
- Inputs (channels‑last):
  - `input`: tf.float32 amplitude, `[B, N, N, C]`
  - `input_positions`: tf.float32 relative offsets, `[B, 1, 2, C]`
- Output:
  - `trimmed_obj`: tf.complex64 complex object patch, `[B, N, N, 1]`
- Notes: SavedModel signature uses named inputs `input` and `input_positions` and returns
  `trimmed_obj`. Stitching is performed by helper functions outside the model.

PyTorch — `diffraction_to_obj`
- Inputs (channels‑first):
  - `x`: torch.float32 amplitude, `[B, C, N, N]`
  - `positions`: torch.float32 relative offsets, `[B, C, 1, 2]`
  - `probe`: complex64 probe (implementation‑specific tensor; typically `[N, N]`)
  - `input_scale_factor`: torch.float32 per‑sample scale, `[B]` or `[B,1,1,1]`
- Output:
  - Complex object patches, `torch.complex64` `[B, C, N, N]`. Internally, generator modules (FNO/Hybrid
    heads) produce raw real/imag tensors `[B, H, W, C, 2]` (`ptycho_torch.model._predict_complex_patches`), which
    are converted to complex `[B, C, N, N]` before `forward_predict` returns. The public inference output is always complex — there
    is no real/imag-stacked output variant. Reassembly is performed by workflow helpers.

Autoencoder (both backends)
- Internal sub‑model; accepts the same diffraction input and produces amplitude/phase per channel that
  are combined into complex object patches. External inference SHOULD call `diffraction_to_obj`.

Channel semantics
- `C = gridsize²`. Position tensors align with the same channel order as diffraction inputs.

Scaling
- Backends MUST ensure positive intensities for Poisson losses. TF applies `IntensityScaler` layers;
  Torch uses `IntensityScalerModule` with per‑sample factors.

#### 4.5. Training Workflow and NPZ Interfaces

- `export_training_data()` writes NPZ archives with keys `xcoords`, `ycoords`, `diff3d`, `probeGuess`,
  `objectGuess`, and `scan_index` (`ptychodus.model.ptychopinn.reconstructor.export_training_data`). Any
  alternate implementation must emit the same schema so that `RawData.from_file` and downstream code can
  reload the data (`ptycho/raw_data.py`).
- NPZ diffraction content MUST be amplitude (sqrt of intensity), not raw intensity, to avoid downstream
  shape/scale mismatches (see `docs/TROUBLESHOOTING.md`). Callers are responsible
  for converting intensity to amplitude prior to packaging NPZ inputs.
- `train()` expects a directory containing `train_data.npz` and `test_data.npz` with the same schema and runs the
  full TensorFlow pipeline via `run_cdi_example` (`ptychodus.model.ptychopinn.reconstructor.train`,
  `ptycho.workflows.components.run_cdi_example`). The backend must either call into those workflows or provide
  equivalent functionality (data grouping, model training, optional image stitching). Return values are expected
  to be compatible with `save_outputs()` and `Product` reconstruction.

#### 4.6. Model Persistence Contract

- `open_model()` delegates to `load_inference_bundle`, which reads a directory containing `wts.h5.zip` and loads
  multiple TensorFlow models via `ModelManager.load_multiple_models` (`ptychodus.model.ptychopinn.reconstructor.open_model`,
  `ptycho.workflows.components.load_inference_bundle`, `ptycho.model_manager.ModelManager`). A replacement backend must define an
  equivalent archival format or provide adapters that keep the UI workflow unchanged.
- `ModelManager` serializes `params.cfg` alongside weights, relies on `tf.keras.config.enable_unsafe_deserialization`,
  and preserves custom TensorFlow layers. Custom backends must retain these behaviours or supply compatible
  save/load routines.
- `save_model()` calls `ptycho.model_manager.save`, which ultimately produces the same archive layout expected by
  `open_model()`.

Archive identification and backend tagging
- File name: Model archives SHALL use the canonical base name `wts.h5` with a zip extension, i.e. `wts.h5.zip`.
- Manifest: Archives SHALL include a `manifest.dill` at the root with, at minimum, `{'models': [...], 'version': 'X.Y'}`.
  PyTorch archives MUST additionally include `backend: 'pytorch'`; TensorFlow MAY omit this field and defaults to `'tensorflow'`.
- Contents: TensorFlow archives contain Keras/SavedModel payloads and serialized custom objects; PyTorch archives contain Lightning
  `.ckpt` payload(s) and serialized hyperparameters required for state-free reload. The outer archive structure remains identical.
- PyTorch object-policy identity: newly written PyTorch archives use
  `artifact_schema_version='torch-artifact-portable-v2'` and a nested
  `torch-model-spec-portable-v2`. The v2 structural model payload stores
  `object_layout`, `training_canvas`, and `training_patch_weighting`; it does
  not treat `object_big` as a second structural owner. The outer archive
  version remains `2.0-pytorch` and the exact model roles remain
  `autoencoder` and `diffraction_to_obj`.
- Compatibility decoding: `torch-artifact-portable-v1` and
  `torch-model-spec-portable-v1` are
  immutable historical schemas. New loaders require their frozen exact field
  sets and deterministically upgrade the persisted `object_big` representation
  to the v2 in-memory identity before model construction or state loading.
  TensorFlow archive version `1.0` and its flat derived `object.big` value are
  unchanged. Old installed binaries are not required to read new v2 PyTorch
  artifacts.
- Cross-backend loading: Not required. When unsupported, loaders MUST raise a descriptive error stating the archived backend and
  the active loader backend.

#### 4.7. Backend-Specific Runtime Requirements

**TensorFlow Path:**
- The pipeline assumes TensorFlow tensors throughout (`ptycho.loader`, `ptycho.tf_helper`, `ptycho.model`).
  Replacing TensorFlow with another framework requires shims for dtype conversion, TensorFlow-specific custom
  layers, and helper utilities invoked during load/save.
- Lambda layers and custom layers (e.g., `CombineComplexLayer`, `ReassemblePatchesLayer`) are registered through
  `ModelManager`. Alternative backends must either emulate these layers or
  rewrite the orchestration modules that depend on them.

**PyTorch Path:**
- The PyTorch backend (`ptycho_torch/`) MUST use PyTorch Lightning (`lightning.pytorch.Trainer`) for training orchestration and checkpoint management. Implementations SHALL instantiate `PtychoPINN_Lightning` modules from resolved owner records and the resolved runtime carrier defined in §4.9.
- Checkpoint persistence MUST produce `wts.h5.zip` archives compatible with the TensorFlow persistence contract (§4.6), containing both Lightning `.ckpt` state and bundled hyperparameters for state-free reload.
- CLI entrypoints (`ptycho_torch/train.py`, `ptycho_torch/inference.py`) MUST delegate to shared helper functions (`ptycho_torch/cli/shared.py`) for path validation and pure execution-request construction. Helpers SHALL emit deprecation warnings for legacy flags (`--device`, `--disable_mlflow`) and map them to modern equivalents (`--accelerator`, `--logger none`, and `--quiet`) without inspecting hardware.
- Execution config objects (`PyTorchExecutionConfig`, see §4.9) MUST NOT populate `params.cfg` via `update_legacy_dict`; they control runtime behavior only. Canonical configs (`TrainingConfig`, `InferenceConfig`) continue to bridge via CONFIG-001.
- Runtime failures SHALL raise actionable errors: `RuntimeError` if PyTorch >=2.2 unavailable (POLICY-001), `ValueError` for invalid execution config fields, `FileNotFoundError` for missing data/checkpoint paths (Phase C2 evidence: `ptycho_torch/cli/shared.py:validate_paths`).
 - Activating MLflow logging is OPTIONAL. The `mlflow` package is a direct
   project dependency, while the default logger backend remains `'csv'`
   (`logger_backend='csv'`). The resolved configuration uses `None` to disable
   logging; the CLI accepts `'none'` and canonicalizes it to `None`.

#### 4.8. Backend Selection & Dispatch

- **Configuration Field**: `TrainingConfig.backend` and `InferenceConfig.backend` MUST accept the literals `'tensorflow'` or `'pytorch'` and SHALL default to `'tensorflow'` to maintain backward compatibility. Callers MAY override this field when invoking PtychoPINN through Ptychodus.
- **CONFIG-001 Compliance**: Source resolution and structural, runnable, or resource validation MAY inspect `config.backend` and MUST complete before mutating legacy state. Implementations MUST then call `update_legacy_dict(ptycho.params.cfg, validated_config)` immediately before importing or dispatching either backend, or before invoking any other legacy consumer. During inference loading, the validated bootstrap projection occurs first and the archive-restored configuration remains authoritative afterward.
- **Execution Request Boundary**: For PyTorch paths, dispatchers MUST accept an optional provenance-carrying `ExecutionRequest` or build one via `build_execution_request_from_args(...)`. A bare `PyTorchExecutionConfig` is a resolved output carrier and MUST be rejected as a factory input before capability observation, payload creation, or global mutation. Optimizer and topology inputs enter through their canonical `TrainingConfig` and Torch `ModelConfig` patches, respectively. See §4.9.
- **Routing Guarantees**:
  - When `config.backend == 'tensorflow'`, the dispatcher SHALL delegate to `ptycho.workflows.components` entry points without attempting PyTorch imports.
  - When `config.backend == 'pytorch'`, the dispatcher SHALL delegate to `ptycho_torch.workflows.components` entry points and return the same `(amplitude, phase, results_dict)` structure expected by TensorFlow workflows.
- **Torch Unavailability**: Selecting `'pytorch'` MUST raise an actionable `RuntimeError` if the PyTorch stack cannot be imported. The message SHALL include the phrases "PyTorch backend selected" and installation guidance (e.g., `pip install torch>=2.2`). Silent fallbacks to TensorFlow are prohibited (POLICY-001).
- **Result Metadata**: Dispatchers MUST annotate the returned `results_dict` with `results['backend'] = config.backend` to aid downstream logging and regression harnesses.
- **Persistence Parity**: Backends MUST persist archives in formats compatible with their load paths. Cross-backend artifact loading is OPTIONAL but, when unsupported, the dispatcher MUST raise a descriptive error (covered by `tests/torch/test_model_manager.py`).
- **Validation Errors**: Dispatcher MUST raise `ValueError` if `config.backend` is not one of the supported literals, guiding callers to correct usage. Factories MUST raise `ValueError` for invalid execution config fields and `FileNotFoundError` for missing paths (Phase C2 validation evidence).
- **Inference Symmetry**: The same guarantees apply to `load_inference_bundle_with_backend()` to ensure train/save/load/infer workflows remain symmetric.

Routing surface
- Acceptable entrypoints for the PyTorch path include either `ptycho_torch.workflows.components` or the high-level API
  in `ptycho_torch/api/base_api.py`, provided the exposed functions conform to the same signatures and return values as
  the TensorFlow `ptycho.workflows.components` functions. The dispatcher MUST ensure signature parity and identical
  result semantics regardless of the chosen surface.
- The high-level API accepts exact resolved configuration records and the
  versioned checkpoint loader. Generic in-place updates and reconstruction
  from unversioned JSON or MLflow scalar dictionaries are not supported
  configuration boundaries.

#### 4.9. PyTorch Execution Configuration Contract

The PyTorch backend has two execution-boundary records:

- `ptycho_torch.execution_request.ExecutionRequest` is the sole supported
  unresolved factory input. It contains copied primitive runtime values and an
  explicit-field set. Structural and owner validation is pure and completes
  before any hardware capability is observed.
- `ptycho.config.config.PyTorchExecutionConfig` is the resolved runtime output
  carrier consumed by Lightning and DataLoader construction. It contains only
  runtime fields, performs no capability observation, and MUST NOT be accepted
  as a factory request.

Neither record populates `params.cfg`, enters `ModelSpec`, or contributes to
portable artifact identity. Model topology is owned only by Torch
`ModelConfig`; optimizer, scheduler, clipping, and accumulation semantics are
owned only by Torch `TrainingConfig`. A CLI or API default is not explicit
provenance merely because argparse or a configuration type materialized it.

Resolution order is:

1. resolve and validate canonical scientific, topology, and optimizer owners;
2. validate the primitive runtime request;
3. observe capabilities only if an unresolved runtime value requires them;
4. produce the resolved runtime carrier and requested/resolved audit.

A bare resolved carrier MUST fail at step 1. The old
`build_execution_config_from_args()` and standalone `resolve_accelerator()`
helpers are not supported APIs.

**Field Categories and Validation Rules:**

1. **Lightning Trainer Knobs:**
   - `accelerator`: requests accept `'auto'`, `'cpu'`, `'gpu'`, `'cuda'`, or `'mps'`; TPU is unsupported. The resolved carrier accepts no `'auto'` value. `'auto'` prefers CUDA and otherwise resolves to CPU with the POLICY-001 notice.
   - `devices`: requests accept a positive integer or `'auto'`; the resolved carrier stores a positive integer. CUDA `'auto'` resolves to the observed CUDA device count, while CPU/MPS resolves to one.
   - `strategy` (str, default `'auto'`): Distributed strategy. Validated downstream; future CLI exposure planned (Phase E.B2).
   - `deterministic` (bool, default `True`): Enforce reproducibility. Controlled via `--deterministic` / `--no-deterministic` flags.
   - `precision`: request values are resolved before carrier construction; CPU
     `'16-mixed'` becomes `'bf16-mixed'`.

2. **DataLoader Knobs:**
   - `num_workers` (int, default `0`): Worker process count. MUST be ≥ 0. Exposed via `--num-workers`.
   - `pin_memory` (bool, default `False`): Enable CUDA pinned memory. GPU-specific; safe default for CPU.
   - `persistent_workers` (bool, default `False`): Keep workers alive between epochs. Only valid when `num_workers > 0`.
   - `prefetch_factor` (int|None, default `None`): Batches to prefetch per worker. Not yet exposed via CLI.

3. **Canonical optimization owners:**
   - Learning rate, optimizer family and parameters, scheduler family and
     settings, gradient clipping, and gradient accumulation are fields of
     Torch `TrainingConfig`, not either execution record.
   - Explicit optimizer CLI spellings are converted to a canonical training
     patch before factory resolution. Omitted flags do not overwrite file or
     baseline values.
   - Optimizer construction reads the resolved `TrainingConfig`. Trainer
     clipping and accumulation arguments are derived from that same record
     (with manual-optimization mechanics applied explicitly) and never from a
     second execution owner.

4. **Checkpoint/Logging Knobs:**
   - `enable_progress_bar` (bool, default `False`): Show training progress. Derived from `--quiet` flag inversion.
   - `enable_checkpointing` (bool, default `True`): Enable Lightning automatic checkpointing during training. Exposed via `--enable-checkpointing` / `--disable-checkpointing`.
   - `checkpoint_save_top_k` (int, default `1`): Number of best checkpoints to retain. MUST be ≥ 0; `0` disables saving. The save-all spelling `-1` is not supported. Exposed via `--checkpoint-save-top-k`.
   - `checkpoint_monitor_metric` (str, default `'val_loss'`): Metric for best checkpoint selection. The literal `'val_loss'` is dynamically mapped to `model.val_loss_name` (typically `'poisson_val_loss'` for PINN models) during Lightning configuration, ensuring compatibility with the model's actual metric names. Falls back to `model.train_loss_name` when validation data is unavailable. Exposed via `--checkpoint-monitor`.
   - `checkpoint_mode` (str, default `'min'`): Mode for checkpoint metric optimization. MUST be `'min'` (lower metric is better) or `'max'` (higher metric is better). Exposed via `--checkpoint-mode`.
   - `early_stop_patience` (int, default `100`): Early stopping patience epochs. MUST be > 0. Training stops if monitored metric doesn't improve for this many epochs. Exposed via `--early-stop-patience`.
   - `logger_backend` (`str|None`, default `'csv'`): Resolved experiment tracking backend. MUST be one of `'csv'`, `'tensorboard'`, `'mlflow'`, or `None`. The raw CLI spelling `'none'` resolves to `None`; direct programmatic `None` has the same disabled meaning. Controls Lightning logger selection for capturing training/validation metrics:
     - `'csv'`: CSVLogger (default) — zero dependencies, stores metrics as CSV files in `{output_dir}/lightning_logs/`. Recommended for CI/automated workflows.
     - `'tensorboard'`: TensorBoardLogger — requires tensorboard (auto-installed via TensorFlow dependency), enables rich visualization via `tensorboard --logdir {output_dir}/lightning_logs/`.
     - `'mlflow'`: MLFlowLogger — uses the project-installed MLflow dependency and integrates with an MLflow tracking server. Server URI must be configured separately.
     - `None`: Disable logging — metrics from `self.log()` calls are discarded. Use the raw CLI spelling `'none'` with `--quiet` to suppress all output.
     Omission uses the configuration default `'csv'`; explicit `None` disables logging. Exposed via `--logger` CLI flag. **Note:** MLflow backend currently uses legacy `mlflow.pytorch.autolog()` in `ptycho_torch.train`; migration to Lightning `MLFlowLogger` tracked as Phase EB3.C4 backlog. **Deprecation:** `--disable_mlflow` emits `DeprecationWarning` and resolves to the same disabled `None` value.

5. **Inference Knobs:**
   - `inference_batch_size` (int|None, default `None`): Override batch size for inference. MUST be > 0 if set. Exposed via `--inference-batch-size`. When `None`, reuses training `batch_size`.
   - `middle_trim` (int, default `0`): Inference trimming parameter. Not yet implemented (documented as TODO).
   - `pad_eval` (bool, default `False`): Padding for evaluation. Not yet implemented.

**CLI Integration:**
- Shared helpers in `ptycho_torch/cli/shared.py` build `ExecutionRequest`
  values and canonical TrainingConfig patches from raw-option suppliedness.
- Factory functions accept `ExecutionRequest | None`, resolve the environment
  after owner validation, and return `PyTorchExecutionConfig` in the payload.
- CLI scripts MUST NOT instantiate the resolved carrier or inspect hardware.

**Reference Implementation:** See `ptycho_torch.execution_request`,
`ptycho.config.config.PyTorchExecutionConfig`,
`ptycho_torch/cli/shared.py`, and `ptycho_torch/config_factory.py`.

### 5. Configuration Field Reference

The tables below enumerate every configuration field surfaced through `ModelConfig`, `TrainingConfig`, and
`InferenceConfig`, the legacy key it populates inside `params.cfg`, and the primary consumers in the existing
`ptychopinn` implementation. Use these references when adding new settings to ensure downstream dependencies are
updated in lockstep.

#### 5.1. `ModelConfig` fields

| Field | Legacy `params.cfg` key | Primary consumers | Notes |
| :----- | :---------------------- | :----------------- | :----- |
| `N` | `N` | `RawData.generate_grouped_data`, `PtychoDataContainer`, model constructors | Controls crop size for grouping, tensor shapes, and network input resolution. |
| `gridsize` | `gridsize` | `RawData.generate_grouped_data`, `PtychoDataContainer`, model constructors | Determines group cardinality (`gridsize²`), tensor channel layout, and model input signature. |
| `n_filters_scale` | `n_filters_scale` | model constructors | Scales convolution filter widths throughout encoder/decoder stacks. |
| `model_type` | `model_type` | training/export workflows | Selects physics-informed vs supervised workflows and annotates saved artifacts. |
| `architecture` | `architecture` | `resolve_generator`, `to_model_config` | Selects one of the five supported generator architectures (`'cnn'`, `'ffno'`, `'fno'`, `'fno_vanilla'`, `'neuralop_uno'`; default `'cnn'`). The PyTorch registry and ModelSpec boundary enforce architecture-specific construction requirements; TensorFlow ignores this routing field. |
| `amp_activation` | `amp_activation` | model amplitude head | Chooses activation function for the reconstructed amplitude head. |
| `object_layout` | `object_layout` | model construction and Torch structural identity | Canonical component layout: `'single_patch'` or `'grouped_patches'`. |
| `training_canvas` | `training_canvas` | model construction and Torch structural identity | Canonical canvas policy paired with `object_layout`: `'independent'` or `'relative_overlap'`. |
| `training_patch_weighting` | `training_patch_weighting` | training-forward assembly | Canonical assembly policy. PyTorch accepts `central_mask`, `uniform`, and `probe`; TensorFlow accepts `central_mask` only. |
| `object_big` *(deprecated)* | `object.big` | declared legacy consumers | Derived compatibility projection of `object_layout`; contradictory dual input is rejected. |
| `probe_big` | `probe.big` | model probe branch | Enables large-probe decoding branches for extended field-of-view. |
| `probe_mask` | `probe.mask` | probe illumination module | Applies optional circular masking inside the learned probe module. |
| `pad_object` | `pad_object` | model reconstruction geometry | Independent padding policy retained across both object layouts. |
| `probe_scale` | `probe_scale` | `ptycho.probe` | Sets normalization applied to the complex probe guess. |
| `gaussian_smoothing_sigma` | `gaussian_smoothing_sigma` | `ProbeIllumination` | Controls Gaussian smoothing performed by `ProbeIllumination`. |

#### 5.2. `TrainingConfig` fields (excluding nested `model`)

The nested spellings below are canonical. The historical flat root spellings
of `data.*`, `tf_loss.*`, and `sampling.*` fields remain accepted as
deprecated aliases (see §2.1) so existing external callers such as
`ptychodus` continue to validate.

| Field | Legacy `params.cfg` key | Primary consumers | Notes |
| :----- | :---------------------- | :----------------- | :----- |
| `data.train_data_file` | `train_data_file_path` | Ptychodus reconstructor, workflow components | Provides the NPZ source for training data and for diagnostics during grouping. |
| `data.test_data_file` | `test_data_file_path` | Ptychodus reconstructor, workflow components | Optional NPZ path used for validation/inference data preparation. |
| `data.nphotons` | `nphotons` | model/train scaling paths | Photon-count compatibility value for already-materialized data. |
| `batch_size` | `batch_size` | legacy training loops via `params.cfg` | Maintained for compatibility with legacy CLI pipelines; current PINN training reads it from `params.cfg` when constructing datasets. |
| `nepochs` | `nepochs` | legacy training scripts | Number of optimizer epochs; propagated to legacy CLI workflows. |
| `sampling.n_groups` | `n_groups` | workflow components, `RawData.generate_grouped_data` | Determines grouped samples requested from the dataset; omitted input validates to 512. |
| `sampling.n_images` *(deprecated)* | `n_images` | Pydantic alias validator and compatibility paths | Alias converted to `sampling.n_groups` and cleared during validation; unequal alias/canonical values fail. |
| `sampling.n_subsample` | `n_subsample` | workflow sampling paths | Optional independent raw-row selection count before grouping. |
| `sampling.subsample_seed` | `subsample_seed` | workflow sampling paths | Ensures reproducible subsampling when provided. |
| `sampling.neighbor_count` | `neighbor_count` | workflow components, `RawData.generate_grouped_data` | Sets nearest-neighbor query width. |
| `sampling.enable_oversampling` | `enable_oversampling` | workflow sampling paths | Explicitly enables combination-based oversampling. |
| `sampling.neighbor_pool_size` | `neighbor_pool_size` | workflow sampling paths | Candidate-pool size used for oversampling combinations. |
| `sampling.sequential_sampling` | `sequential_sampling` | `RawData.generate_grouped_data` | Uses deterministic first-N grouping anchors within the already selected raw-row pool; it does not change raw-row subsampling. |
| `loss.torch_loss_mode` | `torch_loss_mode` | Torch loss construction | Selects `poisson` or amplitude-only `mae`. |
| `loss.torch_mae_pred_l2_match_target` | `torch_mae_pred_l2_match_target` | Torch MAE loss construction | Enables prediction-L2 matching on the Torch MAE path. |
| `tf_loss.mae_weight` | `mae_weight` | TensorFlow model loss configuration | Weight applied to diffraction MAE. |
| `tf_loss.nll_weight` | `nll_weight` | TensorFlow model loss configuration | Weight applied to Poisson NLL. |
| `tf_loss.realspace_mae_weight` | `realspace_mae_weight` | TensorFlow real-space helpers | Optional real-space MAE coefficient. |
| `tf_loss.realspace_weight` | `realspace_weight` | TensorFlow model loss configuration | General real-space consistency weight. |
| `gradient_clip.val` | `gradient_clip_val` | Torch training loop | Clipping threshold; `None` disables clipping. |
| `gradient_clip.algorithm` | `gradient_clip_algorithm` | Torch training loop | Selects `norm`, `value`, or `agc`. |
| `optimizer.algorithm` | `optimizer` | Torch optimizer construction | Selects `adam`, `adamw`, or `sgd`. |
| `optimizer.weight_decay` | `weight_decay` | Torch optimizer construction | Optimizer weight decay. |
| `optimizer.sgd.momentum` | `momentum` | Torch SGD construction | SGD momentum. |
| `optimizer.adam.beta1` | `adam_beta1` | Torch Adam/AdamW construction | First Adam beta. |
| `optimizer.adam.beta2` | `adam_beta2` | Torch Adam/AdamW construction | Second Adam beta. |
| `scheduler.kind` | `scheduler` | Torch scheduler construction | Public scheduler selection. |
| `scheduler.lr_warmup_epochs` | `lr_warmup_epochs` | Torch scheduler construction | Warmup duration. |
| `scheduler.lr_min_ratio` | `lr_min_ratio` | Torch scheduler construction | Cosine minimum learning-rate ratio. |
| `scheduler.plateau_factor` | `plateau_factor` | Torch scheduler construction | Reduce-on-plateau factor. |
| `scheduler.plateau_patience` | `plateau_patience` | Torch scheduler construction | Reduce-on-plateau patience. |
| `scheduler.plateau_min_lr` | `plateau_min_lr` | Torch scheduler construction | Reduce-on-plateau minimum learning rate. |
| `scheduler.plateau_threshold` | `plateau_threshold` | Torch scheduler construction | Reduce-on-plateau threshold. |
| `positions_provided` | `positions.provided` | legacy training paths | Maintained for backwards compatibility with legacy simulation scripts. |
| `probe_trainable` | `probe.trainable` | model probe configuration | Enables joint optimization of probe parameters. |
| `intensity_scale_trainable` | `intensity_scale.trainable` | model scaling configuration | Toggles learnable diffraction intensity normalization. |
| `output_dir` | `output_prefix` | Ptychodus reconstructor, workflow components | Targets directory for saved weights, plots, and metadata. |
| `backend` | `backend` | backend selector | Selects `tensorflow` or `pytorch`. |

#### 5.3. `InferenceConfig` fields (excluding nested `model`)

| Field | Legacy `params.cfg` key | Primary consumers | Notes |
| :----- | :---------------------- | :----------------- | :----- |
| `model_path` | `model_path` | Ptychodus reconstructor, `load_inference_bundle` | Directory containing `wts.h5.zip`; consumed by `load_inference_bundle` / `ModelManager`. |
| `test_data_file` | `test_data_file_path` | workflow components | Optional NPZ path for inference data preparation. |
| `n_groups` | `n_groups` | workflow components | Controls requested grouped samples during inference workflows. |
| `n_images` *(deprecated)* | `n_images` | config `__post_init__` and compatibility workflow paths | Legacy alias; converted to `n_groups` by `InferenceConfig.__post_init__`. |
| `n_subsample` | `n_subsample` | workflow sampling paths | Optional inference-time subsampling before grouping. |
| `subsample_seed` | `subsample_seed` | workflow sampling paths | Seed for reproducible inference subsampling. |
| `neighbor_count` | `neighbor_count` | workflow components | Sets K-nearest-neighbor search width during inference grouping. |
| `debug` | `debug` | logging/debug decorators | Enables verbose debug logging decorators throughout the pipeline. |
| `output_dir` | `output_prefix` | workflow components | Destination directory for inference exports and plots. |

### 6. `KEY_MAPPINGS` Specification

The `KEY_MAPPINGS` dictionary in `config/config.py` defines the translation rules. Below is a specification of these mappings:

| Modern Configuration Field    | Legacy `params.cfg` Key     | Description                                                                                              |
| :---------------------------- | :-------------------------- | :------------------------------------------------------------------------------------------------------- |
| `object_big` *(deprecated)*   | `object.big`                | Derived Boolean compatibility projection: `grouped_patches` → `True`, `single_patch` → `False`.          |
| `probe_big`                   | `probe.big`                 | If `True`, enables a low-resolution reconstruction of the outer region of the real-space grid.           |
| `probe_mask`                  | `probe.mask`                | If `True`, applies a circular mask to the probe function.                                                |
| `probe_trainable`             | `probe.trainable`           | If `True`, optimizes the probe function during training. (Experimental)                                  |
| `intensity_scale_trainable`   | `intensity_scale.trainable` | If `True`, optimizes the model's internal amplitude scaling factor during training.                      |
| `positions_provided`          | `positions.provided`        | A legacy flag indicating whether scan positions are available.                                           |
| `output_dir`                  | `output_prefix`             | The directory path for saving outputs. `pathlib.Path` is automatically converted to `str`.               |
| `data.train_data_file` (training) | `train_data_file_path` | The nested training path is flattened and converted to `str`. |
| `data.test_data_file` (training) / `test_data_file` (inference) | `test_data_file_path` | The owning path is flattened and converted to `str`. |

### 7. CLI Reference — Execution Configuration Flags

The PyTorch backend (`ptycho_torch/train.py`, `ptycho_torch/inference.py`)
exposes runtime and canonical optimization knobs through command-line flags.
Runtime flags form an `ExecutionRequest`; optimization flags form an explicit
Torch `TrainingConfig` patch. See §4.9.

#### 7.1. Training CLI Execution Flags

Runtime flags map to `ExecutionRequest`; optimization rows below map directly
to Torch `TrainingConfig`. Raw-option suppliedness determines whether a
canonical optimization override exists.

| CLI Flag | Type | Default | Config Field | Description |
|----------|------|---------|--------------|-------------|
| `--accelerator` | str | `'auto'` | `ExecutionRequest.accelerator` | Hardware accelerator request: `'auto'` (detect GPU, default), `'cpu'`, `'gpu'`/`'cuda'`, or `'mps'`. TPU is unsupported. |
| `--deterministic` / `--no-deterministic` | bool | `True` | `ExecutionRequest.deterministic` | Enable reproducible training with fixed RNG seeds. Use `--no-deterministic` to disable for potential performance gains (results become non-reproducible). |
| `--num-workers` | int | `0` | `ExecutionRequest.num_workers` | DataLoader worker process count (0 = main thread only, CPU-safe). Typical values: 2-8 for multi-core systems. |
| `--learning-rate` | float | `1e-3` | `TrainingConfig.learning_rate` | Explicit optimizer learning-rate override. Must be > 0; omission preserves the resolved baseline. |
| `--scheduler` | str | `'Default'` | `TrainingConfig.scheduler` | Canonical Torch scheduler. Native Torch accepts `Default`, `Exponential`, `MultiStage`, `Adaptive`, `WarmupCosine`, and `ReduceLROnPlateau`; public/unified entry points expose their declared public subset. |
| `--accumulate-grad-batches` | int | `1` | `TrainingConfig.accum_steps` | Explicit gradient-accumulation override. Must be positive; omission preserves the resolved baseline. |
| `--quiet` | flag | `False` | `ExecutionRequest.enable_progress_bar` | Suppress progress bars and reduce console logging. Inverted to populate `enable_progress_bar` (`--quiet` → `False`). |
| `--enable-checkpointing` / `--disable-checkpointing` | bool | `True` | `ExecutionRequest.enable_checkpointing` | Enable automatic model checkpointing during training (default: enabled). Checkpoints are saved based on monitored metric performance. Use `--disable-checkpointing` to turn off. |
| `--checkpoint-save-top-k` | int | `1` | `ExecutionRequest.checkpoint_save_top_k` | Number of best checkpoints to keep (default: 1). Must be non-negative; `0` disables saving. |
| `--checkpoint-monitor` | str | `'val_loss'` | `ExecutionRequest.checkpoint_monitor_metric` | Metric to monitor for checkpoint selection (default: `'val_loss'`). The literal `'val_loss'` is dynamically aliased to `model.val_loss_name` (e.g., `'poisson_val_loss'`) during Lightning configuration. Falls back to `model.train_loss_name` when validation data is unavailable. Common choices: val_loss, train_loss, val_accuracy. |
| `--checkpoint-mode` | str | `'min'` | `ExecutionRequest.checkpoint_mode` | Mode for checkpoint metric optimization (default: min). Use 'min' when lower metric values are better (e.g., loss), 'max' when higher values are better (e.g., accuracy). |
| `--early-stop-patience` | int | `100` | `ExecutionRequest.early_stop_patience` | Early stopping patience epochs. Must be positive. |
| `--logger` | str | `'csv'` | `ExecutionRequest.logger_backend` | Experiment tracking backend. The boundary spelling `'none'` canonicalizes to `None`. |

**Deprecated Flags:**
- `--device` (str): Superseded by `--accelerator`. Using `--device` emits a deprecation warning and maps to `--accelerator`. Will be removed in Phase E post-ADR acceptance. Use `--accelerator` instead.
- `--disable_mlflow` (flag): **DEPRECATED.** Emits DeprecationWarning directing users to `--logger none` for disabling experiment tracking and `--quiet` for suppressing progress bars. This flag will be removed in a future release. Current behavior: maps to `--logger none` internally.

**Factory Integration:** CLI scripts call `create_training_payload()` with
canonical explicitly supplied owner overrides plus an `ExecutionRequest`.
The returned payload contains resolved owner records and the resolved
`PyTorchExecutionConfig` runtime carrier.

**CONFIG-001 Compliance:** The factory ensures `update_legacy_dict(ptycho.params.cfg, tf_config)` is called before any data loading or model construction, guaranteeing legacy subsystems observe synchronized parameters regardless of execution config values.

**Planned Exposure (Phase E.B Backlog):**
The following runtime request fields are not yet exposed by every CLI but are
available through `ExecutionRequest`: `strategy`, `prefetch_factor`,
`pin_memory`, and `persistent_workers`. Gradient clipping is canonical
`TrainingConfig` state, not an execution field.

#### 7.2. Inference CLI Execution Flags

| CLI Flag | Type | Default | Config Field | Description |
|----------|------|---------|--------------|-------------|
| `--accelerator` | str | `'auto'` | `ExecutionRequest.accelerator` | Hardware accelerator request: `'auto'`, `'cpu'`, `'gpu'`/`'cuda'`, or `'mps'`. TPU is unsupported. |
| `--num-workers` | int | `0` | `ExecutionRequest.num_workers` | DataLoader worker process count (0 = synchronous, CPU-safe). Typical values: 2-8 for multi-core systems. |
| `--inference-batch-size` | int | `None` | `ExecutionRequest.inference_batch_size` | Batch size for inference DataLoader. When `None` (default), reuses training `batch_size` from checkpoint. |
| `--quiet` | flag | `False` | `ExecutionRequest.enable_progress_bar` | Native inference only: suppress progress bars and reduce console logging. |

**Deprecated Flags:**
- `--device` (str): Superseded by `--accelerator`. Using `--device` emits a deprecation warning and maps to `--accelerator`. Will be removed in Phase E post-ADR acceptance.

**Reference Implementation:** See `ptycho_torch.train` (training flags),
`ptycho_torch.inference` (inference flags),
`ptycho_torch/cli/shared.py` (`build_execution_request_from_args`,
canonical optimizer-option handling, and `validate_paths`), and
`ptycho_torch/config_factory.py`.

**Note:** For programmatic runtime requests not exposed by a CLI, construct
`ExecutionRequest` with explicit provenance. `PyTorchExecutionConfig` is a
resolved output and is not a factory input.

### 8. Usage Guidelines for Developers

- **DO** instantiate `ModelConfig`, `TrainingConfig`, or `InferenceConfig` to define your parameters.
- **DO** call `update_legacy_dict(ptycho.params.cfg, ...)` at the documented compatibility boundary after resolving and validating the configuration and before a legacy consumer runs.
- **DO NOT** modify `ptycho.params.cfg` directly (e.g., `ptycho.params.cfg['N'] = 128`). This breaks the one-way data flow and can lead to inconsistent state.
- **DO NOT** create new dependencies on `ptycho.params.get()` in new code. Instead, pass configuration records as arguments.

### 9. Architectural Rationale

This hybrid system supports modernization of a large existing codebase. The
legacy `params.cfg` dictionary enabled rapid prototyping but created tight
coupling and global-state hazards. The modern Pydantic/dataclass configuration
types introduce structure, type safety, and validation. The
`update_legacy_dict` bridge keeps legacy modules operational while new code and
external systems such as `ptychodus` use explicit configuration records.

Terminology note
- “Model archive” refers to the training/inference weights bundle (`wts.h5.zip`).
- “Product file” refers to the Ptychodus HDF5 product (`*.h5`, `*.hdf5`) defined in `specs/data_contracts.md`.
