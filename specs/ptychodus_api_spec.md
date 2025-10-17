## PtychoPINN Configuration Management API Specification

### 1. Overview

The `ptychopinn` configuration system is a hybrid architecture designed to support both modern, type-safe workflows and maintain backward compatibility with over 20 legacy modules. It consists of two primary components:

1.  **Modern Dataclass System (`config/config.py`):** The authoritative source of truth for all parameters. It uses Python dataclasses for structured, type-hinted, and validated configuration.
2.  **Legacy Global Dictionary (`params.py`):** A global, mutable dictionary (`ptycho.params.cfg`) that is used by older modules throughout the codebase.

The API is built around a **one-way data flow**: parameters are defined in the modern dataclasses and then propagated to the legacy dictionary. Direct manipulation of the legacy dictionary is strongly discouraged in new code.

The central function that bridges these two systems is `ptycho.config.config.update_legacy_dict()`. Any external system, like `ptychodus`, **must** use this bridge to configure `ptychopinn` correctly.

**⚠️ PyTorch Requirement:** As of Phase F (INTEGRATE-PYTORCH-001), PyTorch `>= 2.2` is a **mandatory runtime dependency** for the PyTorch backend (`ptycho_torch/`). The package specifies `torch>=2.2` in `setup.py` install_requires. The TensorFlow backend (`ptycho/`) continues to function independently, but callers integrating the PyTorch stack **must** ensure PyTorch is available; the system will raise an actionable `RuntimeError` if torch cannot be imported. This policy is documented in <doc-ref type="findings">docs/findings.md#policy-001</doc-ref> and reflects the governance decision archived at `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T184624Z/governance_decision.md`. For installation guidance, see the PyTorch workflow guide at <doc-ref type="workflow">docs/workflows/pytorch.md</doc-ref>.

### 2. Core Components

#### 2.1. Modern Configuration Dataclasses

These dataclasses, defined in `config/config.py`, are the primary way to specify configuration.

- **`ModelConfig`**: Defines the neural network architecture and core physics parameters.
  - `N`: The size of the input diffraction patterns (e.g., 64, 128).
  - `gridsize`: The number of adjacent scan positions to process simultaneously (e.g., `gridsize=2` means a 2x2 group).
  - `model_type`: The type of model, either `'pinn'` or `'supervised'`.
  - `amp_activation`: The activation function for the amplitude decoder.
  - `object_big`, `probe_big`, `pad_object`: Booleans controlling the reconstruction geometry and padding strategy.
  - Additional fields consumed by `ptychodus`, including `n_filters_scale`, `probe_mask`, `probe_scale`, and `gaussian_smoothing_sigma` 
    (`ptycho/config/config.py:72-105`, `ptychodus/src/ptychodus/model/ptychopinn/reconstructor.py:87-103`). These must be honoured by any
    alternative backend because they drive grouping geometry, probe handling, and image smoothing downstream.

- **`TrainingConfig`**: Defines parameters for the training workflow.
  - `model`: A nested `ModelConfig` instance.
  - `train_data_file`: `pathlib.Path` to the training dataset.
  - `batch_size`: The number of samples per training step.
  - `nepochs`: The total number of training epochs.
  - `nll_weight`, `mae_weight`, `realspace_weight`: Loss function weights.
  - `nphotons`: The expected number of photons, crucial for the Poisson noise model in the PINN loss.
  - Additional controls surfaced to `ptychodus`, such as `n_groups`, `n_subsample`, `subsample_seed`, `neighbor_count`,
    `positions_provided`, `probe_trainable`, `intensity_scale_trainable`, `sequential_sampling`, and `output_dir` 
    (`ptycho/config/config.py:87-133`, `ptychodus/src/ptychodus/model/ptychopinn/reconstructor.py:229-259`). These values
    propagate to data sampling (`RawData.generate_grouped_data`) and training orchestration (`ptycho.workflows.components`).

- **`InferenceConfig`**: Defines parameters for the reconstruction (inference) workflow.
  - `model`: A nested `ModelConfig` instance.
  - `model_path`: `pathlib.Path` to the trained model directory.
  - `test_data_file`: `pathlib.Path` to the data to be reconstructed.
  - Extended options used by `ptychodus`, including `n_groups`, `n_subsample`, `neighbor_count`, `debug`, and `output_dir`
    (`ptycho/config/config.py:129-140`, `ptychodus/src/ptychodus/model/ptychopinn/reconstructor.py:119-158`).

#### 2.2. Legacy Global Dictionary (`params.py`)

**⚠️ WARNING:** This component is a global mutable state and is maintained for backward compatibility only. Do not directly modify `ptycho.params.cfg` in new code.

- **`ptycho.params.cfg`**: A global dictionary that holds all configuration parameters in a flat key-value structure.
- **`ptycho.params.get(key)`**: The function used by legacy modules to retrieve a parameter. It includes logic for derived parameters (e.g., `bigN`).
- **`ptycho.params.set(key, value)`**: The function to update a value in the global dictionary.

#### 2.3. The Compatibility Bridge (`config/config.py`)

This is the most critical part of the configuration API. It translates modern dataclasses into the legacy format.

- **`update_legacy_dict(cfg: dict, dataclass_obj: Any)`**:
  - **Purpose**: To populate the legacy `ptycho.params.cfg` dictionary from a modern configuration dataclass (`TrainingConfig` or `InferenceConfig`).
  - **Mechanism**: It calls `dataclass_to_legacy_dict()` to perform the translation and then updates the global `cfg` dictionary. This is the **only supported way** to configure `ptychopinn` from an external caller like `ptychodus`.

- **`dataclass_to_legacy_dict(obj: Any)`**:
  - **Purpose**: Translates a dataclass instance into a flat dictionary with legacy key names.
  - **Mechanism**:
    1.  Converts the dataclass to a standard dictionary.
    2.  If the dataclass contains a nested `model` field, it flattens the structure by merging the `ModelConfig` parameters into the main dictionary.
    3.  It applies the `KEY_MAPPINGS` dictionary to translate modern, snake_case field names to legacy dot.separated names (e.g., `object_big` -> `object.big`).
    4.  It automatically converts `pathlib.Path` objects to strings, as the legacy system expects string paths.

- **PyTorch Configuration Adapters (`ptycho_torch.config_bridge`):**
  - **Purpose**: Translate PyTorch singleton configuration objects to TensorFlow dataclass instances, enabling PyTorch workflows to populate `params.cfg` via the standard `update_legacy_dict` function.
  - **Key Functions**:
    - `to_model_config(data: DataConfig, model: ModelConfig, overrides=None) -> TFModelConfig`: Converts PyTorch `DataConfig` and `ModelConfig` to TensorFlow `ModelConfig`, handling critical transformations such as `grid_size` tuple → `gridsize` int, `mode` enum → `model_type` enum, and activation name normalization.
    - `to_training_config(model: TFModelConfig, data: DataConfig, pt_model: ModelConfig, training: TrainingConfig, overrides=None) -> TFTrainingConfig`: Translates PyTorch training parameters to TensorFlow `TrainingConfig`, converting `epochs` → `nepochs`, `K` → `neighbor_count`, `nll` bool → `nll_weight` float, and requiring explicit `overrides` for fields missing in PyTorch configs (e.g., `train_data_file`, `n_groups`).
    - `to_inference_config(model: TFModelConfig, data: DataConfig, inference: InferenceConfig, overrides=None) -> TFInferenceConfig`: Converts PyTorch inference parameters to TensorFlow `InferenceConfig`, mapping `K` → `neighbor_count` and requiring `overrides` for `model_path` and `test_data_file`.
  - **Contract**: These adapters MUST produce dataclasses compatible with `update_legacy_dict` and maintain behavioral parity with direct TensorFlow dataclass instantiation. Consumers (e.g., `ptychodus` PyTorch integration) MUST call these adapters before invoking `update_legacy_dict` to ensure correct params.cfg population. Implementation details and field mappings are documented in `ptycho_torch/config_bridge.py:1-380` and tested via `tests/torch/test_config_bridge.py`.

### 3. API Specification and Data Flow

The correct and only supported way for an external system to configure `ptychopinn` is as follows:

1.  **Instantiate a Configuration Dataclass**: Create an instance of `TrainingConfig` or `InferenceConfig` with the desired parameters.
2.  **Call the Bridge Function**: Pass the legacy `cfg` dictionary and the newly created dataclass instance to `update_legacy_dict`.

This one-way data flow ensures that the modern dataclasses remain the single source of truth while correctly populating the state required by legacy modules.

```mermaid
graph TD
    A[External Caller, e.g., ptychodus] --> B{1. Instantiate<br/>TrainingConfig};
    B --> C{2. update_legacy_dict(params.cfg, config)};
    C --> D{3. dataclass_to_legacy_dict(config)};
    D --> E{4. Apply KEY_MAPPINGS<br/>(e.g., object_big -> object.big)};
    E --> F{5. Update ptycho.params.cfg};
    F --> G[Legacy Modules<br/>(e.g., model.py, diffsim.py)];
    G --> H{6. Access config via<br/>params.get('key')};

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
(`ptychodus/src/ptychodus/model/ptychopinn/reconstructor.py:61-273`). A replacement backend must honor the
following behavioural contract in addition to the configuration bridge.

#### 4.1. Entry Points and Lifecycle

- **Instantiation**: Two reconstructor instances (`PINN`, `Supervised`) are created by
  `PtychoPINNReconstructorLibrary` when the `ptychopinn` package can be imported
  (`ptychodus/src/ptychodus/model/ptychopinn/core.py:22-59`). Both share a settings registry and must support
  runtime updates to model/training/inference knobs.
- **Capability Flags**: `MODEL_FILE_NAME = 'wts.h5.zip'` and file filters exposed through
  `get_model_file_filter()` / `get_training_data_file_filter()` dictate the on-disk formats the UI offers.

#### 4.2. Configuration Handshake

- `reconstruct()` and `train()` assemble `ModelConfig`, `InferenceConfig`, and `TrainingConfig` instances from
  live settings (`ptychodus/src/ptychodus/model/ptychopinn/reconstructor.py:87-158`, `229-259`). Every field in
  these dataclasses must be respected because they directly feed downstream modules.
- `update_legacy_dict(ptycho.params.cfg, config)` is called immediately after instantiation. The backend must
  continue to populate `ptycho.params.cfg` so that legacy consumers (`ptycho.raw_data`, `ptycho.loader`,
  `ptycho.model`) observe consistent values.
- Loaded models overwrite `params.cfg` via `load_inference_bundle`, so a backend must either replicate that side
  effect or provide an alternative hook (`ptycho/workflows/components.py:94-174`).
- **PyTorch Import Requirement (Phase F)**: The PyTorch backend (`ptycho_torch/`) **must** raise an actionable `RuntimeError` with installation guidance if `torch` cannot be imported. Silent fallbacks or optional import guards are prohibited per <doc-ref type="findings">docs/findings.md#policy-001</doc-ref>. All modules in `ptycho_torch/` assume PyTorch availability and will fail fast with clear error messages directing users to install `torch>=2.2`. Test suites automatically skip `tests/torch/` in TensorFlow-only CI environments via directory-based pytest collection rules (`tests/conftest.py`), but local development expects PyTorch to be present.

#### 4.3. Data Ingestion and Grouping

- `create_raw_data()` converts `ReconstructInput` objects to `RawData` by collecting NumPy arrays for
  coordinates, diffraction patterns, probe guess, and a single object layer
  (`ptychodus/src/ptychodus/model/ptychopinn/reconstructor.py:40-58`). A backend must accept the same layout.
- Grouped data is generated through `RawData.generate_grouped_data`, which expects
  `params.cfg['gridsize']`, `neighbor_count`, and optional sampling controls to be already populated
  (`ptycho/raw_data.py:365-438`). The function returns a dictionary with keys such as `diffraction`, `coords_offsets`,
  `coords_relative`, and `local_offsets`; `ptycho.loader.load` consumes this structure verbatim
  (`ptycho/loader.py:1-186`). Any replacement must either continue producing this dictionary or adapt
  `ptychodus` accordingly.
- `PtychoDataContainer` instances expose TensorFlow tensors (`X`, `local_offsets`, `global_offsets`, complex
  ground truth) that are passed straight into the model (`ptycho/loader.py:93-200`). Shapes depend on
  `N` and `gridsize` and must match the TensorFlow model signature described below.

#### 4.4. TensorFlow Inference Behaviour

- Inference uses `tf.keras.Model.predict` with two inputs: scaled diffraction data and local position offsets
  (`ptychodus/src/ptychodus/model/ptychopinn/reconstructor.py:113-158`). The backend model must expose the same
  call signature and support eager execution.
- `ptycho.model.params()['intensity_scale']` supplies the multiplicative scale factor. Backends must provide the
  `intensity_scale` parameter (and optionally `intensity_scale.trainable`) inside `params.cfg`
  (`ptycho/model.py:217-560`).
- Output tensors are stitched into a 2D array via `ptycho.tf_helper.reassemble_position`, which expects
  `global_offsets` and model output channels to follow the existing TensorFlow helper contract
  (`ptychodus/src/ptychodus/model/ptychopinn/reconstructor.py:153-159`, `ptycho/tf_helper.py`).

#### 4.5. Training Workflow and NPZ Interfaces

- `export_training_data()` writes NPZ archives with keys `xcoords`, `ycoords`, `diff3d`, `probeGuess`,
  `objectGuess`, and `scan_index` (`ptychodus/src/ptychodus/model/ptychopinn/reconstructor.py:200-224`). Any
  alternate implementation must emit the same schema so that `RawData.from_file` and downstream code can
  reload the data (`ptycho/raw_data.py`).
- `train()` expects a directory containing `train_data.npz` and `test_data.npz` with the same schema and runs the
  full TensorFlow pipeline via `run_cdi_example` (`ptychodus/src/ptychodus/model/ptychopinn/reconstructor.py:229-269`,
  `ptycho/workflows/components.py:676-732`). The backend must either call into those workflows or provide
  equivalent functionality (data grouping, model training, optional image stitching). Return values are expected
  to be compatible with `save_outputs()` and `Product` reconstruction.

#### 4.6. Model Persistence Contract

- `open_model()` delegates to `load_inference_bundle`, which reads a directory containing `wts.h5.zip` and loads
  multiple TensorFlow models via `ModelManager.load_multiple_models` (`ptychodus/src/ptychodus/model/ptychopinn/reconstructor.py:183-195`,
  `ptycho/workflows/components.py:94-174`, `ptycho/model_manager.py:1-360`). A replacement backend must define an
  equivalent archival format or provide adapters that keep the UI workflow unchanged.
- `ModelManager` serializes `params.cfg` alongside weights, relies on `tf.keras.config.enable_unsafe_deserialization`,
  and preserves custom TensorFlow layers. Custom backends must retain these behaviours or supply compatible
  save/load routines.
- `save_model()` calls `ptycho.model_manager.save`, which ultimately produces the same archive layout expected by
  `open_model()` (`ptychodus/src/ptychodus/model/ptychopinn/reconstructor.py:194-195`).

#### 4.7. TensorFlow-Specific Requirements

- The pipeline assumes TensorFlow tensors throughout (`ptycho.loader`, `ptycho.tf_helper`, `ptycho.model`).
  Replacing TensorFlow with another framework requires shims for dtype conversion, TensorFlow-specific custom
  layers, and helper utilities invoked during load/save.
- Lambda layers and custom layers (e.g., `CombineComplexLayer`, `ReassemblePatchesLayer`) are registered through
  `ModelManager` (`ptycho/model_manager.py:90-209`). Alternative backends must either emulate these layers or
  rewrite the orchestration modules that depend on them.

### 5. Configuration Field Reference

The tables below enumerate every configuration field surfaced through `ModelConfig`, `TrainingConfig`, and
`InferenceConfig`, the legacy key it populates inside `params.cfg`, and the primary consumers in the existing
`ptychopinn` implementation. Use these references when adding new settings to ensure downstream dependencies are
updated in lockstep.

#### 5.1. `ModelConfig` fields

| Field | Legacy `params.cfg` key | Primary consumers | Notes |
| :----- | :---------------------- | :----------------- | :----- |
| `N` | `N` | ptycho/raw_data.py:365<br>ptycho/loader.py:178<br>ptycho/model.py:280 | Controls crop size for grouping, tensor shapes, and network input resolution. |
| `gridsize` | `gridsize` | ptycho/raw_data.py:365<br>ptycho/loader.py:181<br>ptycho/model.py:264 | Determines group cardinality (`gridsize²`), tensor channel layout, and model input signature. |
| `n_filters_scale` | `n_filters_scale` | ptycho/model.py:280 | Scales convolution filter widths throughout encoder/decoder stacks. |
| `model_type` | `model_type` | ptycho/train.py:98<br>ptycho/export.py:20 | Selects physics-informed vs supervised workflows and annotates saved artifacts. |
| `amp_activation` | `amp_activation` | ptycho/model.py:406 | Chooses activation function for the reconstructed amplitude head. |
| `object_big` | `object.big` | ptycho/model.py:445 | Toggles whole-object stitching vs centered crop in reconstruction. |
| `probe_big` | `probe.big` | ptycho/model.py:445 | Enables large-probe decoding branches for extended field-of-view. |
| `probe_mask` | `probe.mask` | ptycho/model.py:195 | Applies optional circular masking inside the learned probe module. |
| `pad_object` | `pad_object` | ptycho/model.py:345 | Chooses between padded output and stitch recomposition when `object_big` is false. |
| `probe_scale` | `probe_scale` | ptycho/probe.py:63 | Sets normalization applied to the complex probe guess. |
| `gaussian_smoothing_sigma` | `gaussian_smoothing_sigma` | ptycho/model.py:176 | Controls Gaussian smoothing performed by `ProbeIllumination`. |

#### 5.2. `TrainingConfig` fields (excluding nested `model`)

| Field | Legacy `params.cfg` key | Primary consumers | Notes |
| :----- | :---------------------- | :----------------- | :----- |
| `train_data_file` | `train_data_file_path` | ptychodus/src/ptychodus/model/ptychopinn/reconstructor.py:240<br>ptycho/workflows/components.py:560 | Provides the NPZ source for training data and for diagnostics during grouping. |
| `test_data_file` | `test_data_file_path` | ptychodus/src/ptychodus/model/ptychopinn/reconstructor.py:240<br>ptycho/workflows/components.py:560 | Optional NPZ path used for validation/inference data preparation. |
| `batch_size` | `batch_size` | ptycho/train.py:77<br>legacy training loops via `params.cfg` | Maintained for compatibility with legacy CLI pipelines; current PINN training reads it from `params.cfg` when constructing datasets. |
| `nepochs` | `nepochs` | ptycho/train.py:81<br>legacy training scripts | Number of optimizer epochs; propagated to legacy CLI workflows. |
| `mae_weight` | `mae_weight` | ptycho/model.py:541 | Weight applied to diffraction MAE term in composite loss. |
| `nll_weight` | `nll_weight` | ptycho/model.py:541 | Weight applied to Poisson NLL loss component. |
| `realspace_mae_weight` | `realspace_mae_weight` | ptycho/tf_helper.py:1434 | Coefficient for optional real-space MAE alignment term. |
| `realspace_weight` | `realspace_weight` | ptycho/model.py:526 | Controls weighting of real-space consistency branch. |
| `nphotons` | `nphotons` | ptycho/model.py:213<br>ptycho/train_pinn.py:162 | Sets photon-count prior for scaling and loss normalization. |
| `n_groups` | `n_groups` | ptycho/workflows/components.py:560<br>ptycho/raw_data.py:365 | Determines number of grouped samples requested from the dataset (replaces `n_images`). |
| `n_images` *(deprecated)* | `n_images` | ptycho/workflows/components.py:226 | Legacy alias converted to `n_groups` during `TrainingConfig.__post_init__`. |
| `n_subsample` | `n_subsample` | ptycho/workflows/components.py:226-337 | Optional independent subsampling count before grouping. |
| `subsample_seed` | `subsample_seed` | ptycho/workflows/components.py:316 | Ensures reproducible subsampling when provided. |
| `neighbor_count` | `neighbor_count` | ptycho/workflows/components.py:563<br>ptycho/raw_data.py:365 | Sets K-nearest-neighbor search width for grouping / oversampling. |
| `positions_provided` | `positions.provided` | ptycho/train.py:100 | Maintained for backwards compatibility with legacy simulation scripts. |
| `probe_trainable` | `probe.trainable` | ptycho/model.py:164 | Enables joint optimization of probe parameters. |
| `intensity_scale_trainable` | `intensity_scale.trainable` | ptycho/model.py:217 | Toggles learnable diffraction intensity normalization. |
| `output_dir` | `output_prefix` | ptychodus/src/ptychodus/model/ptychopinn/reconstructor.py:263<br>ptycho/workflows/components.py:726 | Targets directory for saved weights, plots, and metadata. |
| `sequential_sampling` | `sequential_sampling` | ptycho/workflows/components.py:566<br>ptycho/raw_data.py:365 | Forces deterministic sequential grouping instead of random sampling. |

#### 5.3. `InferenceConfig` fields (excluding nested `model`)

| Field | Legacy `params.cfg` key | Primary consumers | Notes |
| :----- | :---------------------- | :----------------- | :----- |
| `model_path` | `model_path` | ptychodus/src/ptychodus/model/ptychopinn/reconstructor.py:183<br>ptycho/workflows/components.py:94 | Directory containing `wts.h5.zip`; consumed by `load_inference_bundle` / `ModelManager`. |
| `test_data_file` | `test_data_file_path` | ptycho/workflows/components.py:226 | Optional NPZ path for inference data preparation. |
| `n_groups` | `n_groups` | ptycho/workflows/components.py:560 | Controls requested grouped samples during inference workflows. |
| `n_images` *(deprecated)* | `n_images` | ptycho/workflows/components.py:226 | Legacy alias; converted to `n_groups` by `InferenceConfig.__post_init__`. |
| `n_subsample` | `n_subsample` | ptycho/workflows/components.py:226-337 | Optional inference-time subsampling before grouping. |
| `subsample_seed` | `subsample_seed` | ptycho/workflows/components.py:316 | Seed for reproducible inference subsampling. |
| `neighbor_count` | `neighbor_count` | ptycho/workflows/components.py:563 | Sets K-nearest-neighbor search width during inference grouping. |
| `debug` | `debug` | ptycho/logging.py:52 | Enables verbose debug logging decorators throughout the pipeline. |
| `output_dir` | `output_prefix` | ptycho/workflows/components.py:726 | Destination directory for inference exports and plots. |

### 6. `KEY_MAPPINGS` Specification

The `KEY_MAPPINGS` dictionary in `config/config.py` defines the translation rules. Below is a specification of these mappings:

| Modern Dataclass Field        | Legacy `params.cfg` Key     | Description                                                                                              |
| :---------------------------- | :-------------------------- | :------------------------------------------------------------------------------------------------------- |
| `object_big`                  | `object.big`                | If `True`, enables a separate real-space reconstruction for each input diffraction image.                |
| `probe_big`                   | `probe.big`                 | If `True`, enables a low-resolution reconstruction of the outer region of the real-space grid.           |
| `probe_mask`                  | `probe.mask`                | If `True`, applies a circular mask to the probe function.                                                |
| `probe_trainable`             | `probe.trainable`           | If `True`, optimizes the probe function during training. (Experimental)                                  |
| `intensity_scale_trainable`   | `intensity_scale.trainable` | If `True`, optimizes the model's internal amplitude scaling factor during training.                      |
| `positions_provided`          | `positions.provided`        | A legacy flag indicating whether scan positions are available.                                           |
| `output_dir`                  | `output_prefix`             | The directory path for saving outputs. `pathlib.Path` is automatically converted to `str`.               |
| `train_data_file`             | `train_data_file_path`      | The path to the training data file. `pathlib.Path` is automatically converted to `str`.                  |
| `test_data_file`              | `test_data_file_path`       | The path to the test data file. `pathlib.Path` is automatically converted to `str`.                      |

### 7. Usage Guidelines for Developers

- **DO** instantiate `ModelConfig`, `TrainingConfig`, or `InferenceConfig` to define your parameters.
- **DO** call `update_legacy_dict(ptycho.params.cfg, ...)` immediately after creating your configuration dataclass and before calling any other `ptychopinn` functions.
- **DO NOT** modify `ptycho.params.cfg` directly (e.g., `ptycho.params.cfg['N'] = 128`). This breaks the one-way data flow and can lead to inconsistent state.
- **DO NOT** create new dependencies on `ptycho.params.get()` in new code. Instead, pass configuration dataclasses as arguments.

### 8. Architectural Rationale

This hybrid system was intentionally designed to facilitate the modernization of a large, existing codebase. The legacy `params.cfg` dictionary allowed for rapid prototyping but created tight coupling and global state issues. The modern dataclass system introduces structure, type safety, and validation. The `update_legacy_dict` bridge allows legacy modules to continue functioning without immediate refactoring, while enabling new code and external systems like `ptychodus` to use a clean, modern API.
