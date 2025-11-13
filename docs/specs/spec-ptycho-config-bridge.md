# Spec: Configuration Bridge (TensorFlow ↔ PyTorch)

This specification defines the normative mapping between TensorFlow configuration dataclasses and PyTorch configuration singletons, and the one‑way bridging flow to the legacy `params.cfg` dictionary. It is authoritative for cross‑backend configuration behavior.

## 1. Scope and Goals

- Canonical source: TensorFlow dataclasses under `ptycho/config/config.py` (`ModelConfig`, `TrainingConfig`, `InferenceConfig`).
- PyTorch inputs: Config singletons under `ptycho_torch/config_params.py` (e.g., `DataConfig`, `ModelConfig`, `TrainingConfig`, `InferenceConfig`).
- Bridge adapter: `ptycho_torch/config_bridge.py` functions translate PyTorch config objects into TensorFlow dataclasses.
- Legacy state: `update_legacy_dict(params.cfg, config)` must be called before any legacy module usage.

Non‑goals: Define training semantics, runtime constraints, or data contracts (see other specs).

## 2. Canonical Flow (CONFIG‑001)

```
PyTorch config_params → config_bridge.py → TensorFlow dataclasses → update_legacy_dict() → params.cfg
```

Rules:
- One‑way flow only. Do not mutate PyTorch configs during translation.
- Call `update_legacy_dict(params.cfg, config)` before: data loading, model construction, and any legacy module imports that read `params.cfg`.

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
  - `ModelConfig.amp_activation: {'silu','SiLU','swish','relu','sigmoid','softplus'}` → `ModelConfig.amp_activation: {'swish','relu','sigmoid','softplus'}`
    - Map: silu/SiLU→swish; others must be supported by TF enum.

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

Defaults & Precedence:
- If a PyTorch field is missing, the bridge may accept an `overrides: Dict[str,Any]` to supply required values.
- Absent optional fields fall back to TensorFlow dataclass defaults.

## 4. Validation and Error Conditions

- Non‑square `grid_size` → error (TensorFlow backend assumes square grids).
- Unsupported activations (not in TF enum) → error.
- `N` outside allowed set → error.
- Type normalization failures (e.g., invalid `PathLike`) → error.

## 5. Bridging to Legacy (params.cfg)

- After translation, call `update_legacy_dict(params.cfg, config)` with the resulting TF dataclass.
- This applies to both backends to keep legacy modules in sync (e.g., physics routines, helpers).
- `update_legacy_dict` performs KEY_MAPPINGS translation to legacy names and value serialization.

## 6. Conformance (Testing Requirements)

- Translation must satisfy the unit tests in `tests/torch/test_config_bridge.py` (mapping, overrides, and params.cfg population).
- Workflows must show `update_legacy_dict(params.cfg, config)` called before data operations (see `docs/workflows/pytorch.md`).

## 7. Examples

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

## 8. References

- TensorFlow configs: `ptycho/config/config.py`
- PyTorch configs: `ptycho_torch/config_params.py`
- Bridge adapter: `ptycho_torch/config_bridge.py`
- Workflows: `docs/workflows/pytorch.md`
- Tests: `tests/torch/test_config_bridge.py`

