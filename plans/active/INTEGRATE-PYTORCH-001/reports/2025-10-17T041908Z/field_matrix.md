# Configuration Field Parity Matrix

**Initiative:** INTEGRATE-PYTORCH-001 Phase B.B4
**Timestamp:** 2025-10-17T041908Z
**Purpose:** Canonical field-by-field mapping for parameterized parity tests

---

## Matrix Legend

- **direct**: Field exists with same name/type in PyTorch and TensorFlow; no transformation needed
- **transform**: Field exists but requires name/type conversion (e.g., tuple→int, enum mapping, bool→float)
- **override_required**: Field missing from PyTorch configs; must be provided via `overrides` dict
- **unsupported**: Spec-required field with no PyTorch equivalent and no clear mapping path
- **default_diverge**: Field exists but with incompatible default values between PyTorch and TensorFlow

---

## ModelConfig Fields (11 total)

| Field Name | Spec §Ref | Classification | PyTorch Source | Transformation | Test Strategy | Notes |
|------------|-----------|----------------|----------------|----------------|---------------|-------|
| `N` | 5.1:1 | direct | `DataConfig.N` | None | Assert equality | int, both default=64 |
| `gridsize` | 5.1:2 | transform | `DataConfig.grid_size` | Tuple[int,int]→int: extract `grid_size[0]`, validate square | Assert equality + non-square error case | **CRITICAL:** Assumes square grids |
| `n_filters_scale` | 5.1:3 | direct | `ModelConfig.n_filters_scale` | None | Assert equality | int, both default=1 |
| `model_type` | 5.1:4 | transform | `ModelConfig.mode` | Enum mapping: `'Unsupervised'→'pinn'`, `'Supervised'→'supervised'` | Assert enum value + invalid enum error | **CRITICAL:** Different enum values |
| `amp_activation` | 5.1:5 | transform | `ModelConfig.amp_activation` | String normalization: `'silu'→'swish'`, `'SiLU'→'swish'`, validate against Literal | Assert normalized value + unknown activation error | PyTorch default='silu', TF default='sigmoid' |
| `object_big` | 5.1:6 | direct | `ModelConfig.object_big` | None (KEY_MAPPINGS handles dot notation) | Assert equality | bool, both default=True |
| `probe_big` | 5.1:7 | direct | `ModelConfig.probe_big` | None (KEY_MAPPINGS handles dot notation) | Assert equality | bool, both default=True |
| `probe_mask` | 5.1:8 | transform | `ModelConfig.probe_mask` | Optional[Tensor]→bool: convert `None→False`, `Tensor→True` | **xfail (not MVP)**: Complex type mismatch | PyTorch stores actual tensor mask |
| `pad_object` | 5.1:9 | override_required | *(missing in PyTorch)* | Default value: True | Assert default applied when not in overrides | Missing from PyTorch ModelConfig |
| `probe_scale` | 5.1:10 | default_diverge | `DataConfig.probe_scale` | None, but defaults differ | Assert non-default value used (catch silent default bug) | PyTorch=1.0, TF=4.0 (4x difference!) |
| `gaussian_smoothing_sigma` | 5.1:11 | override_required | *(missing in PyTorch)* | Default value: 0.0 | Assert default applied when not in overrides | Missing from PyTorch ModelConfig |

---

## TrainingConfig Fields (18 total, excluding nested `model`)

| Field Name | Spec §Ref | Classification | PyTorch Source | Transformation | Test Strategy | Notes |
|------------|-----------|----------------|----------------|----------------|---------------|-------|
| `train_data_file` | 5.2:1 | override_required | *(missing in PyTorch)* | Must come from overrides, convert str→Path | **MVP**: Assert override value propagates | PyTorch uses `training_directories: List[str]` instead |
| `test_data_file` | 5.2:2 | override_required | *(missing in PyTorch)* | Must come from overrides, convert str→Path | **MVP**: Assert override value propagates | Missing from PyTorch TrainingConfig |
| `batch_size` | 5.2:3 | direct | `TrainingConfig.batch_size` | None | Assert equality | int, both default=16 (legacy field) |
| `nepochs` | 5.2:4 | transform | `TrainingConfig.epochs` | Field rename: `epochs→nepochs` | Assert equality | int, both default=50 |
| `mae_weight` | 5.2:5 | override_required | *(missing in PyTorch)* | Default value: 0.0 (PyTorch uses categorical `loss_function` instead) | Assert default applied | PyTorch `loss_function='MAE'` vs TF weight-based loss |
| `nll_weight` | 5.2:6 | transform | `TrainingConfig.nll` | Bool→float: `True→1.0`, `False→0.0` | Assert conversion + boundary values | **CRITICAL:** Type mismatch |
| `realspace_mae_weight` | 5.2:7 | override_required | *(missing in PyTorch)* | Default value: 0.0 | Assert default applied | Missing from PyTorch TrainingConfig |
| `realspace_weight` | 5.2:8 | override_required | *(missing in PyTorch)* | Default value: 0.0 | Assert default applied | Missing from PyTorch TrainingConfig |
| `nphotons` | 5.2:9 | default_diverge | `DataConfig.nphotons` | None, but defaults differ | **MVP**: Assert non-default value used | PyTorch=1e5, TF=1e9 (4 orders of magnitude!) |
| `n_groups` | 5.2:10 | override_required | *(missing in PyTorch)* | Must come from overrides | **MVP**: Assert override value propagates | **CRITICAL:** Missing from PyTorch, required for grouping |
| `n_subsample` | 5.2:12 | override_required | *(DataConfig.n_subsample exists but different semantics)* | Must come from overrides (semantic collision) | Assert override value propagates | PyTorch `n_subsample=7` is coordinate subsampling, not sample count |
| `subsample_seed` | 5.2:13 | override_required | *(missing in PyTorch)* | Must come from overrides | Assert override value propagates | Missing from PyTorch TrainingConfig |
| `neighbor_count` | 5.2:14 | transform | `DataConfig.K` | Semantic rename: `K→neighbor_count` | **MVP**: Assert equality | **CRITICAL:** PyTorch also has `K_quadrant` (different param) |
| `positions_provided` | 5.2:15 | override_required | *(missing in PyTorch)* | Default value: True | Assert default applied | Legacy simulation flag, missing from PyTorch |
| `probe_trainable` | 5.2:16 | override_required | *(missing in PyTorch)* | Default value: False | Assert default applied | Missing from PyTorch TrainingConfig |
| `intensity_scale_trainable` | 5.2:17 | direct | `ModelConfig.intensity_scale_trainable` | None (KEY_MAPPINGS handles dot notation) | Assert equality | bool, both default=False (but lives in ModelConfig in PyTorch, TrainingConfig in TF) |
| `output_dir` | 5.2:18 | override_required | *(missing in PyTorch)* | Must come from overrides, convert str→Path | Assert override value propagates | Missing from PyTorch TrainingConfig |
| `sequential_sampling` | 5.2:19 | override_required | *(missing in PyTorch)* | Default value: False | Assert default applied | Missing from PyTorch TrainingConfig |

---

## InferenceConfig Fields (9 total, excluding nested `model`)

| Field Name | Spec §Ref | Classification | PyTorch Source | Transformation | Test Strategy | Notes |
|------------|-----------|----------------|----------------|----------------|---------------|-------|
| `model_path` | 5.3:1 | override_required | *(missing in PyTorch)* | Must come from overrides, convert str→Path | **MVP**: Assert override value propagates | **CRITICAL:** Missing from PyTorch, required for loading |
| `test_data_file` | 5.3:2 | override_required | *(missing in PyTorch)* | Must come from overrides, convert str→Path | **MVP**: Assert override value propagates | **CRITICAL:** Missing from PyTorch |
| `n_groups` | 5.3:3 | override_required | *(missing in PyTorch)* | Must come from overrides | **MVP**: Assert override value propagates | **CRITICAL:** Missing from PyTorch |
| `n_subsample` | 5.3:5 | override_required | *(DataConfig.n_subsample exists but different semantics)* | Must come from overrides (semantic collision) | Assert override value propagates | Same collision as TrainingConfig |
| `subsample_seed` | 5.3:6 | override_required | *(missing in PyTorch)* | Must come from overrides | Assert override value propagates | Missing from PyTorch InferenceConfig |
| `neighbor_count` | 5.3:7 | transform | `DataConfig.K` | Semantic rename: `K→neighbor_count` | **MVP**: Assert equality | Same as TrainingConfig |
| `debug` | 5.3:8 | override_required | *(missing in PyTorch)* | Default value: False | Assert default applied | Missing from PyTorch InferenceConfig |
| `output_dir` | 5.3:9 | override_required | *(missing in PyTorch)* | Must come from overrides, convert str→Path | Assert override value propagates | Missing from PyTorch InferenceConfig |

---

## Summary Statistics

| Classification | ModelConfig | TrainingConfig | InferenceConfig | Total |
|----------------|-------------|----------------|-----------------|-------|
| **direct** | 4 | 2 | 0 | 6 |
| **transform** | 4 | 3 | 1 | 8 |
| **override_required** | 2 | 11 | 8 | 21 |
| **default_diverge** | 1 | 1 | 0 | 2 |
| **unsupported** | 0 | 0 | 0 | 0 |
| **TOTAL** | 11 | 18 | 9 | 38 |

---

## Critical Transformation Details

### 1. grid_size → gridsize (transform)
```python
# PyTorch: grid_size: Tuple[int, int] = (2, 2)
# TensorFlow: gridsize: int = 2

# Adapter logic:
grid_h, grid_w = data.grid_size
if grid_h != grid_w:
    raise ValueError("Non-square grids not supported")
gridsize = grid_h
```

### 2. mode → model_type (transform)
```python
# PyTorch: mode: Literal['Supervised', 'Unsupervised']
# TensorFlow: model_type: Literal['pinn', 'supervised']

# Adapter logic:
mode_to_model_type = {
    'Unsupervised': 'pinn',
    'Supervised': 'supervised'
}
model_type = mode_to_model_type[model.mode]
```

### 3. amp_activation normalization (transform)
```python
# PyTorch: amp_activation: str = 'silu' (freeform)
# TensorFlow: amp_activation: Literal['sigmoid', 'swish', 'softplus', 'relu'] = 'sigmoid'

# Adapter logic:
activation_mapping = {
    'silu': 'swish',
    'SiLU': 'swish',
    'sigmoid': 'sigmoid',
    'swish': 'swish',
    # ... etc
}
amp_activation = activation_mapping[model.amp_activation]
```

### 4. nll → nll_weight (transform)
```python
# PyTorch: nll: bool = True
# TensorFlow: nll_weight: float = 1.0

# Adapter logic:
nll_weight = 1.0 if training.nll else 0.0
```

### 5. epochs → nepochs (transform)
```python
# PyTorch: epochs: int = 50
# TensorFlow: nepochs: int = 50

# Adapter logic: trivial rename
nepochs = training.epochs
```

### 6. K → neighbor_count (transform)
```python
# PyTorch: K: int = 6 (in DataConfig)
# TensorFlow: neighbor_count: int = 4

# Adapter logic: semantic mapping
neighbor_count = data.K
```

---

## Test Parameterization Strategy

### MVP Fields (9 fields - current test coverage)
These fields are already tested in `tests/torch/test_config_bridge.py::TestConfigBridgeMVP`:
- ModelConfig: `N`, `gridsize`, `model_type`
- TrainingConfig: `train_data_file`, `n_groups`, `neighbor_count`, `nphotons`
- InferenceConfig: `model_path`, `test_data_file`

### Phase B.B4 Extension (29 additional fields)
New parameterized tests should cover:

**ModelConfig (8 fields):**
- `n_filters_scale` (direct)
- `amp_activation` (transform)
- `object_big`, `probe_big` (direct, KEY_MAPPINGS)
- `probe_mask` (xfail - complex transform)
- `pad_object`, `gaussian_smoothing_sigma` (override_required)
- `probe_scale` (default_diverge)

**TrainingConfig (16 fields):**
- `test_data_file` (override_required)
- `batch_size` (direct)
- `nepochs` (transform)
- `mae_weight`, `realspace_mae_weight`, `realspace_weight` (override_required)
- `nll_weight` (transform)
- `n_subsample`, `subsample_seed` (override_required)
- `positions_provided`, `probe_trainable`, `sequential_sampling` (override_required)
- `intensity_scale_trainable` (direct)
- `output_dir` (override_required)

**InferenceConfig (5 fields):**
- `n_subsample`, `subsample_seed` (override_required)
- `debug` (override_required)
- `output_dir` (override_required)

---

## Default Divergence Risks

These fields have different defaults between PyTorch and TensorFlow and must be explicitly set in tests to catch silent fallback bugs:

| Field | PyTorch Default | TensorFlow Default | Risk Level |
|-------|-----------------|---------------------|------------|
| `nphotons` | 1e5 | 1e9 | **HIGH** - 4 orders of magnitude difference affects physics scaling |
| `probe_scale` | 1.0 | 4.0 | **MEDIUM** - 4x difference affects probe normalization |
| `amp_activation` | 'silu' | 'sigmoid' | **MEDIUM** - Different activation changes reconstruction quality |
| `neighbor_count` | 6 (K) | 4 | **LOW** - Small difference in grouping locality |

---

## Next Steps (Phase B)

1. **B1 - Design pytest parameters**: Generate `pytest.param` structures from this matrix
2. **B2 - Author failing tests**: Implement parameterized test cases using canonical fixtures
3. **B3 - Encode xfail**: Mark `probe_mask` and future gaps with `pytest.mark.xfail(strict=True)`
4. **Run red phase**: Execute `pytest tests/torch/test_config_bridge.py -k parity -v` and capture output

---

## References

- Canonical baseline values: `fixtures.py` in this directory
- PyTorch config schema: `ptycho_torch/config_params.py:1-171`
- TensorFlow config schema: `ptycho/config/config.py:72-154`
- Spec field tables: `specs/ptychodus_api_spec.md:220-273`
- Prior mapping analysis: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T032218Z/config_schema_map.md`
