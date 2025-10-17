# Configuration Schema Mapping: PyTorch ↔ TensorFlow ↔ params.cfg

**Initiative:** INTEGRATE-PYTORCH-001 Phase B.B1
**Date:** 2025-10-17
**Purpose:** Field-by-field audit mapping PyTorch singleton config → TensorFlow dataclasses → legacy params.cfg keys

---

## Mapping Methodology

This document maps every configuration field from the PyTorch singleton pattern (`ptycho_torch/config_params.py`) to:
1. TensorFlow dataclass fields (`ptycho/config/config.py`: ModelConfig, TrainingConfig, InferenceConfig)
2. Legacy `params.cfg` dictionary keys (via KEY_MAPPINGS where applicable)
3. Spec-mandated fields (`specs/ptychodus_api_spec.md` §5.1-5.3)

**Sources:**
- PyTorch: `ptycho_torch/config_params.py:1-171`
- TensorFlow: `ptycho/config/config.py:72-154`
- Spec: `specs/ptychodus_api_spec.md:213-273`
- KEY_MAPPINGS: `ptycho/config/config.py:231-241`

---

## Legend

- ✅ **Direct Match:** Field exists with compatible name/type
- 🔄 **Transformation Required:** Field exists but needs name/type conversion
- ❌ **Missing in PyTorch:** Spec-required field absent from PyTorch config
- ➕ **PyTorch-Only:** Field unique to PyTorch, not in TensorFlow spec
- ⚠️ **Type Mismatch:** Field exists but with incompatible types

---

## ModelConfig Fields

| Status | PyTorch Field | TensorFlow Field | params.cfg Key | Transformation Notes |
|--------|---------------|------------------|----------------|---------------------|
| ✅ | `N` | `N` | `N` | Direct match (int) |
| 🔄 | `grid_size: Tuple[int, int]` | `gridsize: int` | `gridsize` | **CRITICAL:** PyTorch uses tuple (2D grid), TensorFlow uses single int. Requires conversion logic: `gridsize = grid_size[0]` (assuming square grids). Non-square grids unsupported in TensorFlow spec. |
| ✅ | `n_filters_scale` | `n_filters_scale` | `n_filters_scale` | Direct match (int) |
| 🔄 | `mode: Literal['Supervised', 'Unsupervised']` | `model_type: Literal['pinn', 'supervised']` | `model_type` | **CRITICAL:** Different enum values. Mapping: `'Unsupervised' → 'pinn'`, `'Supervised' → 'supervised'`. |
| 🔄 | `amp_activation: str = 'silu'` | `amp_activation: Literal['sigmoid', 'swish', 'softplus', 'relu'] = 'sigmoid'` | `amp_activation` | **Type mismatch:** PyTorch uses freeform string (default 'silu'), TensorFlow uses constrained Literal (default 'sigmoid'). Need validation layer. |
| ✅ | `object_big` | `object_big` | `object.big` | Direct match (bool). KEY_MAPPINGS required: `object_big → object.big` |
| ✅ | `probe_big` | `probe_big` | `probe.big` | Direct match (bool). KEY_MAPPINGS required: `probe_big → probe.big` |
| 🔄 | `probe_mask: Optional[torch.Tensor]` | `probe_mask: bool` | `probe.mask` | **Type mismatch:** PyTorch stores actual tensor, TensorFlow stores boolean flag. Requires adapter logic. |
| ❌ | *(missing)* | `pad_object: bool = True` | `pad_object` | **Missing in PyTorch.** Spec-required field (§5.1 row 8). Default behavior unknown. |
| ✅ | `probe_scale: float = 1.0` | `probe_scale: float = 4.0` | `probe_scale` | Match with **different defaults**. PyTorch=1.0, TensorFlow=4.0. Reconcile default in harmonization. |
| ❌ | *(missing)* | `gaussian_smoothing_sigma: float = 0.0` | `gaussian_smoothing_sigma` | **Missing in PyTorch.** Spec-required field (§5.1 row 11). Used by ProbeIllumination module. |
| ✅ | `intensity_scale_trainable` | `intensity_scale_trainable` | `intensity_scale.trainable` | Direct match (bool). KEY_MAPPINGS required: `intensity_scale_trainable → intensity_scale.trainable` |
| ➕ | `intensity_scale: float = 10000.0` | *(not in dataclass)* | `intensity_scale` | **PyTorch-only.** Stored separately in TensorFlow model layer, not config. |
| ➕ | `C_model: int = 1` | *(not in dataclass)* | *(none)* | **PyTorch-only.** Model architecture detail, not exposed in TensorFlow config. |
| ➕ | `batch_norm: bool = False` | *(not in dataclass)* | *(none)* | **PyTorch-only.** Architecture choice not in TensorFlow. |
| ➕ | `max_position_jitter: int = 10` | *(not in dataclass)* | *(none)* | **PyTorch-only.** Data augmentation parameter. |
| ➕ | `edge_pad: int = 10` | *(not in dataclass)* | *(none)* | **PyTorch-only.** Decoder architecture detail. |
| ➕ | `decoder_last_c_outer_fraction: float` | *(not in dataclass)* | *(none)* | **PyTorch-only.** Decoder architecture detail. |
| ➕ | `decoder_last_amp_channels: int` | *(not in dataclass)* | *(none)* | **PyTorch-only.** Decoder architecture detail. |
| ➕ | `eca_encoder: bool` | *(not in dataclass)* | *(none)* | **PyTorch-only.** Attention mechanism toggle. |
| ➕ | `cbam_encoder: bool` | *(not in dataclass)* | *(none)* | **PyTorch-only.** Attention mechanism toggle. |
| ➕ | `cbam_bottleneck: bool` | *(not in dataclass)* | *(none)* | **PyTorch-only.** Attention mechanism toggle. |
| ➕ | `cbam_decoder: bool` | *(not in dataclass)* | *(none)* | **PyTorch-only.** Attention mechanism toggle. |
| ➕ | `eca_decoder: bool` | *(not in dataclass)* | *(none)* | **PyTorch-only.** Attention mechanism toggle. |
| ➕ | `spatial_decoder: bool` | *(not in dataclass)* | *(none)* | **PyTorch-only.** Attention mechanism toggle. |
| ➕ | `decoder_spatial_kernel: int` | *(not in dataclass)* | *(none)* | **PyTorch-only.** Attention mechanism parameter. |
| ➕ | `offset: int = 6` | *(not in dataclass)* | `offset` | **PyTorch-only.** May correspond to legacy `params.cfg['offset']`. Needs investigation. |
| ➕ | `C_forward: int` | *(not in dataclass)* | *(none)* | **PyTorch-only.** Forward model channel count. |
| ➕ | `loss_function: Literal['MAE', 'Poisson']` | *(not in dataclass)* | *(none)* | **PyTorch-only.** Loss choice is implicit in TensorFlow (via `nll_weight` / `mae_weight` ratios). |
| ➕ | `amp_loss: Literal[...]` | *(not in dataclass)* | *(none)* | **PyTorch-only.** Regularization term. |
| ➕ | `phase_loss: Literal[...]` | *(not in dataclass)* | *(none)* | **PyTorch-only.** Regularization term. |
| ➕ | `amp_loss_coeff: float` | *(not in dataclass)* | *(none)* | **PyTorch-only.** Regularization weight. |
| ➕ | `phase_loss_coeff: float` | *(not in dataclass)* | *(none)* | **PyTorch-only.** Regularization weight. |

---

## TrainingConfig Fields

| Status | PyTorch Field | TensorFlow Field | params.cfg Key | Transformation Notes |
|--------|---------------|------------------|----------------|---------------------|
| ❌ | *(missing)* | `train_data_file: Optional[Path]` | `train_data_file_path` | **Missing in PyTorch.** Spec-required field (§5.2 row 1). PyTorch uses `training_directories: List[str]` instead. |
| ❌ | *(missing)* | `test_data_file: Optional[Path]` | `test_data_file_path` | **Missing in PyTorch.** Spec-required field (§5.2 row 2). |
| 🔄 | `batch_size: int = 16` | `batch_size: int = 16` | `batch_size` | Direct match with **legacy note**: TensorFlow marks as "maintained for compatibility" (§5.2 row 3). |
| 🔄 | `epochs: int = 50` | `nepochs: int = 50` | `nepochs` | **Name mismatch:** `epochs` vs `nepochs`. Direct semantic match. |
| ❌ | *(missing)* | `mae_weight: float = 0.0` | `mae_weight` | **Missing in PyTorch.** Spec-required field (§5.2 row 5). PyTorch uses categorical `loss_function` instead. |
| 🔄 | `nll: bool = True` | `nll_weight: float = 1.0` | `nll_weight` | **Type mismatch:** PyTorch boolean toggle vs TensorFlow float weight. Requires conversion: `nll_weight = 1.0 if nll else 0.0`. |
| ❌ | *(missing)* | `realspace_mae_weight: float` | `realspace_mae_weight` | **Missing in PyTorch.** Spec field (§5.2 row 7). |
| ❌ | *(missing)* | `realspace_weight: float` | `realspace_weight` | **Missing in PyTorch.** Spec field (§5.2 row 8). |
| ✅ | `nphotons: float = 1e5` | `nphotons: float = 1e9` | `nphotons` | Match with **different defaults**. PyTorch=1e5, TensorFlow=1e9 (4 orders of magnitude!). |
| ❌ | *(missing)* | `n_groups: Optional[int]` | `n_groups` | **Missing in PyTorch.** Spec-required field (§5.2 row 10). Critical for data grouping. |
| ❌ | *(missing)* | `n_images: Optional[int]` | `n_images` | **Missing in PyTorch** (deprecated field in TensorFlow). |
| ❌ | *(missing)* | `n_subsample: Optional[int]` | `n_subsample` | **Ambiguous:** TensorFlow has different `n_subsample` (§5.2 row 12) vs PyTorch `DataConfig.n_subsample: int = 7`. Need semantic clarification. |
| ❌ | *(missing)* | `subsample_seed: Optional[int]` | `subsample_seed` | **Missing in PyTorch.** Spec field (§5.2 row 13). |
| ❌ | *(missing)* | `neighbor_count: int = 4` | `neighbor_count` | **Semantic collision:** TensorFlow `neighbor_count: int = 4` (K-NN search width) vs PyTorch `K: int = 6` + `K_quadrant: int = 30` (multiple neighbor parameters). Requires clarification. |
| ❌ | *(missing)* | `positions_provided: bool = True` | `positions.provided` | **Missing in PyTorch.** Spec field (§5.2 row 15). |
| ❌ | *(missing)* | `probe_trainable: bool = False` | `probe.trainable` | **Missing in PyTorch.** Spec field (§5.2 row 16). |
| ❌ | *(missing)* | `output_dir: Path` | `output_prefix` | **Missing in PyTorch.** Spec field (§5.2 row 18). |
| ❌ | *(missing)* | `sequential_sampling: bool = False` | `sequential_sampling` | **Missing in PyTorch.** Spec field (§5.2 row 19). |
| ➕ | `training_directories: List[str]` | *(not in dataclass)* | *(none)* | **PyTorch-only.** Replaces single `train_data_file` with multiple directories. |
| ➕ | `device: str = 'cuda'` | *(not in dataclass)* | *(none)* | **PyTorch-only.** Runtime device selection. |
| ➕ | `strategy: Optional[str] = 'ddp'` | *(not in dataclass)* | *(none)* | **PyTorch-only.** Lightning distributed strategy. |
| ➕ | `n_devices: int = 1` | *(not in dataclass)* | *(none)* | **PyTorch-only.** Multi-GPU configuration. |
| ➕ | `learning_rate: float = 1e-3` | *(not in dataclass)* | *(none)* | **PyTorch-only.** Optimizer parameter. |
| ➕ | `epochs_fine_tune: int = 0` | *(not in dataclass)* | *(none)* | **PyTorch-only.** Multi-stage training. |
| ➕ | `fine_tune_gamma: float = 0.1` | *(not in dataclass)* | *(none)* | **PyTorch-only.** Multi-stage training. |
| ➕ | `scheduler: Literal[...]` | *(not in dataclass)* | *(none)* | **PyTorch-only.** Learning rate scheduling. |
| ➕ | `num_workers: int = 4` | *(not in dataclass)* | *(none)* | **PyTorch-only.** DataLoader parameter. |
| ➕ | `accum_steps: int = 1` | *(not in dataclass)* | *(none)* | **PyTorch-only.** Gradient accumulation. |
| ➕ | `gradient_clip_val: Union[float,None]` | *(not in dataclass)* | *(none)* | **PyTorch-only.** Gradient clipping. |
| ➕ | `stage_1/2/3_epochs: int` | *(not in dataclass)* | *(none)* | **PyTorch-only.** Multi-stage training logic (3 fields). |
| ➕ | `physics_weight_schedule: str` | *(not in dataclass)* | *(none)* | **PyTorch-only.** Multi-stage physics weighting. |
| ➕ | `stage_3_lr_factor: float` | *(not in dataclass)* | *(none)* | **PyTorch-only.** Multi-stage LR reduction. |
| ➕ | `experiment_name: str` | *(not in dataclass)* | *(none)* | **PyTorch-only.** MLflow tracking. |
| ➕ | `notes: str` | *(not in dataclass)* | *(none)* | **PyTorch-only.** MLflow tracking. |
| ➕ | `model_name: str` | *(not in dataclass)* | *(none)* | **PyTorch-only.** MLflow tracking. |

---

## InferenceConfig Fields

| Status | PyTorch Field | TensorFlow Field | params.cfg Key | Transformation Notes |
|--------|---------------|------------------|----------------|---------------------|
| ❌ | *(missing)* | `model_path: Path` | `model_path` | **Missing in PyTorch.** Spec-required field (§5.3 row 1). Critical for model loading. |
| ❌ | *(missing)* | `test_data_file: Path` | `test_data_file_path` | **Missing in PyTorch.** Spec-required field (§5.3 row 2). |
| ❌ | *(missing)* | `n_groups: Optional[int]` | `n_groups` | **Missing in PyTorch.** Spec field (§5.3 row 3). |
| ❌ | *(missing)* | `n_images: Optional[int]` | `n_images` | **Missing in PyTorch** (deprecated field). |
| ❌ | *(missing)* | `n_subsample: Optional[int]` | `n_subsample` | **Missing in PyTorch.** Spec field (§5.3 row 5). |
| ❌ | *(missing)* | `subsample_seed: Optional[int]` | `subsample_seed` | **Missing in PyTorch.** Spec field (§5.3 row 6). |
| ❌ | *(missing)* | `neighbor_count: int = 4` | `neighbor_count` | **Missing in PyTorch.** Spec field (§5.3 row 7). |
| ❌ | *(missing)* | `debug: bool = False` | `debug` | **Missing in PyTorch.** Spec field (§5.3 row 8). |
| ❌ | *(missing)* | `output_dir: Path` | `output_prefix` | **Missing in PyTorch.** Spec field (§5.3 row 9). |
| ➕ | `middle_trim: int = 32` | *(not in dataclass)* | *(none)* | **PyTorch-only.** Inference trimming parameter. |
| ➕ | `batch_size: int = 1000` | *(not in dataclass)* | *(none)* | **PyTorch-only.** Inference batch size. |
| ➕ | `experiment_number: int = 0` | *(not in dataclass)* | *(none)* | **PyTorch-only.** MLflow tracking. |
| ➕ | `pad_eval: bool = True` | *(not in dataclass)* | *(none)* | **PyTorch-only.** Evaluation padding toggle. |
| ➕ | `window: int = 20` | *(not in dataclass)* | *(none)* | **PyTorch-only.** Edge error windowing. |

---

## DataConfig Fields (PyTorch-Only Group)

These fields exist only in PyTorch's `DataConfig` and have no TensorFlow dataclass equivalents:

| PyTorch Field | Type | Purpose | params.cfg Candidate |
|---------------|------|---------|---------------------|
| `C: int = 4` | int | Number of channels | Potentially maps to `params.cfg['C']` if exists |
| `K: int = 6` | int | Number of nearest neighbors | **Semantic collision** with `neighbor_count` |
| `K_quadrant: int = 30` | int | Quadrant neighbor count | PyTorch-specific |
| `n_subsample: int = 7` | int | Coordinate subsampling factor | **Name collision** with TensorFlow `n_subsample` (different semantics) |
| `neighbor_function: Literal[...]` | enum | Neighbor selection algorithm | PyTorch-specific |
| `min_neighbor_distance: float` | float | Distance bounds | PyTorch-specific |
| `max_neighbor_distance: float` | float | Distance bounds | PyTorch-specific |
| `scan_pattern: Literal[...]` | enum | Scan geometry | PyTorch-specific |
| `normalize: Literal['Group', 'Batch']` | enum | Normalization strategy | PyTorch-specific |
| `probe_normalize: bool` | bool | Probe normalization toggle | PyTorch-specific |
| `data_scaling: Literal[...]` | enum | Scaling strategy | PyTorch-specific |
| `phase_subtraction: bool` | bool | Supervised training flag | PyTorch-specific |
| `x_bounds: Tuple[float, float]` | tuple | Scan position bounds | PyTorch-specific |
| `y_bounds: Tuple[float, float]` | tuple | Scan position bounds | PyTorch-specific |

---

## DatagenConfig Fields (PyTorch-Only Group)

These fields support synthetic data generation and have no TensorFlow equivalents:

| PyTorch Field | Type | Purpose |
|---------------|------|---------|
| `objects_per_probe: int` | int | Synthetic generation parameter |
| `diff_per_object: int` | int | Synthetic generation parameter |
| `object_class: str` | str | Synthetic object type |
| `image_size: Tuple[int, int]` | tuple | Synthetic image dimensions |
| `probe_paths: List[str]` | list | Probe file sources |
| `beamstop_diameter: int` | int | Forward model parameter |

---

## Critical Gaps Summary

### 1. Missing Spec-Required Fields in PyTorch

These fields are **mandatory** per `specs/ptychodus_api_spec.md` but absent from PyTorch config:

**ModelConfig:**
- `pad_object: bool` (§5.1 row 8)
- `gaussian_smoothing_sigma: float` (§5.1 row 11)

**TrainingConfig:**
- `train_data_file: Optional[Path]` (§5.2 row 1) — **CRITICAL**
- `test_data_file: Optional[Path]` (§5.2 row 2)
- `mae_weight: float` (§5.2 row 5)
- `realspace_mae_weight: float` (§5.2 row 7)
- `realspace_weight: float` (§5.2 row 8)
- `n_groups: Optional[int]` (§5.2 row 10) — **CRITICAL**
- `n_subsample: Optional[int]` (§5.2 row 12)
- `subsample_seed: Optional[int]` (§5.2 row 13)
- `neighbor_count: int` (§5.2 row 14) — **CRITICAL**
- `positions_provided: bool` (§5.2 row 15)
- `probe_trainable: bool` (§5.2 row 16)
- `output_dir: Path` (§5.2 row 18)
- `sequential_sampling: bool` (§5.2 row 19)

**InferenceConfig:**
- `model_path: Path` (§5.3 row 1) — **CRITICAL**
- `test_data_file: Path` (§5.3 row 2) — **CRITICAL**
- `n_groups: Optional[int]` (§5.3 row 3)
- `n_subsample: Optional[int]` (§5.3 row 5)
- `subsample_seed: Optional[int]` (§5.3 row 6)
- `neighbor_count: int` (§5.3 row 7)
- `debug: bool` (§5.3 row 8)
- `output_dir: Path` (§5.3 row 9)

### 2. Type/Semantic Mismatches Requiring Translation

- **`grid_size` vs `gridsize`:** Tuple[int, int] vs int (conversion required)
- **`mode` vs `model_type`:** Different enum values (mapping required)
- **`amp_activation`:** freeform string vs constrained Literal (validation required)
- **`probe_mask`:** Tensor vs bool (adapter logic required)
- **`nll` vs `nll_weight`:** bool vs float (conversion required)
- **`epochs` vs `nepochs`:** name mismatch (trivial rename)
- **`K` vs `neighbor_count`:** semantic collision (clarification required)
- **`DataConfig.n_subsample` vs `TrainingConfig.n_subsample`:** name collision, different semantics

### 3. Default Value Conflicts

- **`nphotons`:** PyTorch=1e5 vs TensorFlow=1e9 (4 orders of magnitude!)
- **`probe_scale`:** PyTorch=1.0 vs TensorFlow=4.0
- **`amp_activation`:** PyTorch='silu' vs TensorFlow='sigmoid'

---

## Recommendations for Phase B.B3

### Strategy Decision (Q1): Refactor vs Dual-Schema

**Option A: Refactor PyTorch to Shared Dataclasses (Recommended)**
- Migrate `ptycho_torch/config_params.py` to import and extend `ptycho.config.config.{ModelConfig, TrainingConfig, InferenceConfig}`
- Add PyTorch-specific fields as optional extensions (e.g., `attention_config`, `training_strategy`)
- Benefits: Single source of truth, automatic KEY_MAPPINGS, reduced maintenance
- Risks: Breaks existing PyTorch code, requires extensive refactoring

**Option B: Dual-Schema with Translation Bridge**
- Keep PyTorch singleton pattern
- Implement `pytorch_to_tf_config()` translation functions
- Add PyTorch-specific KEY_MAPPINGS dictionary
- Benefits: Minimal disruption, preserves PyTorch architecture choices
- Risks: Increased maintenance, potential translation bugs, duplicate logic

### Priority Field Implementation Order (Q2: MVP vs Full Parity)

**Phase 1 (MVP — Minimum Viable Integration):**
1. Core model fields: `N`, `gridsize`, `model_type`, `object_big`, `probe_big`
2. Data paths: `train_data_file`, `test_data_file`, `model_path`
3. Grouping essentials: `n_groups`, `neighbor_count`
4. Loss weights: `mae_weight`, `nll_weight`, `nphotons`

**Phase 2 (Feature Parity):**
5. Sampling controls: `n_subsample`, `subsample_seed`, `sequential_sampling`
6. Training flags: `probe_trainable`, `intensity_scale_trainable`, `positions_provided`
7. Model refinements: `n_filters_scale`, `amp_activation`, `probe_mask`, `pad_object`, `probe_scale`, `gaussian_smoothing_sigma`
8. Output controls: `output_dir`, `debug`

**Phase 3 (Full Parity + PyTorch Extensions):**
9. Real-space loss: `realspace_weight`, `realspace_mae_weight`
10. PyTorch-specific: attention toggles, multi-stage training, MLflow integration

### KEY_MAPPINGS Extension Required

PyTorch config bridge must extend `ptycho/config/config.py:231-241` with:

```python
KEY_MAPPINGS = {
    # Existing TensorFlow mappings
    'object_big': 'object.big',
    'probe_big': 'probe.big',
    'probe_mask': 'probe.mask',
    'probe_trainable': 'probe.trainable',
    'intensity_scale_trainable': 'intensity_scale.trainable',
    'positions_provided': 'positions.provided',
    'output_dir': 'output_prefix',
    'train_data_file': 'train_data_file_path',
    'test_data_file': 'test_data_file_path',

    # New PyTorch-specific mappings (if dual-schema approach)
    'mode': 'model_type',  # Requires enum value translation
    'grid_size': 'gridsize',  # Requires tuple→int conversion
    'epochs': 'nepochs',
    'nll': 'nll_weight',  # Requires bool→float conversion
    # ... (extend as needed)
}
```

---

## Next Steps (Phase B.B2)

1. **Decision Required:** Choose Option A (refactor) or Option B (dual-schema) — impacts all subsequent Phase B tasks
2. **Write Failing Test:** Implement `tests/torch/test_config_bridge.py` demonstrating current bridge failure
3. **Implement Translation:** Based on chosen strategy, implement schema harmonization
4. **Extend Tests:** Parameterized tests for all MVP fields (Phase 1 priority list above)

---

## References

- TensorFlow dataclasses: `ptycho/config/config.py:72-154`
- PyTorch singletons: `ptycho_torch/config_params.py:8-171`
- Spec contract: `specs/ptychodus_api_spec.md:213-273`
- KEY_MAPPINGS: `ptycho/config/config.py:231-241`
- Stakeholder brief: `plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T031500Z/stakeholder_brief.md`
