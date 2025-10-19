# ADR-003 Phase B1 — Configuration Override Matrix

**Date:** 2025-10-19
**Purpose:** Comprehensive mapping of every configuration field to its data source, default value, and factory handling strategy.

---

## 1. Reading This Matrix

**Column Definitions:**
- **Field:** Configuration parameter name (PyTorch naming shown; TensorFlow equivalent in notes)
- **Source:** Where the value originates (CLI Flag / TF Dataclass / PT Dataclass / Execution Config / Inferred)
- **Default:** Default value if not overridden
- **Priority:** Override precedence level (1=highest, 5=lowest)
- **Factory Role:** How factory functions handle this field
- **File:Line:** Code location where field is defined or used

**Priority Levels:**
1. **Explicit Overrides** (user-provided dict parameter)
2. **Execution Config** (PyTorchExecutionConfig fields)
3. **CLI Defaults** (argparse default values)
4. **PyTorch Config Defaults** (DataConfig, ModelConfig, TrainingConfig)
5. **TensorFlow Config Defaults** (canonical dataclasses)

---

## 2. Core Model Configuration Fields

| Field | Source | Default | Priority | Factory Role | File:Line | Notes |
|-------|--------|---------|----------|--------------|-----------|-------|
| **N** | Inferred from NPZ | 64 (fallback) | 1 (inferred) | Call `infer_probe_size()`, use override if present | `ptycho_torch/train.py:468` | TF equivalent: `ModelConfig.N` |
| **gridsize** | CLI Flag `--gridsize` | 2 | 3 (CLI) | Pass to `DataConfig.grid_size` as `(gridsize, gridsize)` | `ptycho_torch/train.py:366` | TF: `ModelConfig.gridsize` (int); PT: `DataConfig.grid_size` (tuple) |
| **grid_size** | Derived from gridsize | `(2, 2)` | 1 (derived) | Convert `gridsize` → `(gridsize, gridsize)` | `ptycho_torch/train.py:476` | PyTorch-specific tuple format |
| **K** | Hardcoded | 7 | 5 (PT default) | Factory sets default 7, bridge maps to `neighbor_count` | `ptycho_torch/train.py:477` | TF equivalent: `TrainingConfig.neighbor_count` |
| **nphotons** | Hardcoded | 1e9 (PT: 1e5) | 5 (PT default) | Factory sets 1e9, warns if PT default differs | `ptycho_torch/train.py:478` | **Divergence:** PT default 1e5 vs TF 1e9 (`config_bridge.py:259-269`) |
| **model_type** | CLI `--mode` | 'pinn' | 3 (CLI) | Map PT enum 'Unsupervised'→TF 'pinn', 'Supervised'→'supervised' | `ptycho_torch/train.py:483` | Bridge handles enum conversion (`config_bridge.py:116-124`) |
| **amp_activation** | Hardcoded | 'silu' | 5 (PT default) | Factory sets 'silu', TF default is 'sigmoid' | `ptycho_torch/train.py:484` | **Parity gap:** TF uses sigmoid by default |
| **n_filters_scale** | TF Dataclass | 1 | 5 (TF default) | Pass through to TF ModelConfig | `ptycho/config/config.py:72` | Not exposed in PyTorch CLI |
| **object_big** | TF Dataclass | False | 5 (TF default) | Pass through to TF ModelConfig | `ptycho/config/config.py:95` | Not exposed in PyTorch CLI |
| **probe_big** | TF Dataclass | False | 5 (TF default) | Pass through to TF ModelConfig | `ptycho/config/config.py:96` | Not exposed in PyTorch CLI |
| **pad_object** | TF Dataclass | True | 5 (TF default) | Pass through to TF ModelConfig | `ptycho/config/config.py:97` | Not exposed in PyTorch CLI |
| **probe_mask** | TF Dataclass | False | 5 (TF default) | Pass through to TF ModelConfig | `ptycho/config/config.py:98` | Not exposed in PyTorch CLI |
| **probe_scale** | TF Dataclass | 1.0 | 5 (TF default) | Pass through to TF ModelConfig | `ptycho/config/config.py:99` | Not exposed in PyTorch CLI |
| **gaussian_smoothing_sigma** | TF Dataclass | 0.0 | 5 (TF default) | Pass through to TF ModelConfig | `ptycho/config/config.py:100` | Not exposed in PyTorch CLI |

---

## 3. Training Configuration Fields

| Field | Source | Default | Priority | Factory Role | File:Line | Notes |
|-------|--------|---------|----------|--------------|-----------|-------|
| **train_data_file** | CLI Flag `--train_data_file` | Required | 3 (CLI) | Validate path exists, pass to TF TrainingConfig | `ptycho_torch/train.py:366` | No default; raises error if missing |
| **test_data_file** | CLI Flag `--test_data_file` | None (optional) | 3 (CLI) | Validate path if provided, warn if missing | `ptycho_torch/train.py:366` | Optional validation data |
| **n_groups** | CLI Flag `--n_groups` | Required | 1 (override) | **CRITICAL:** Must be in overrides dict; no default | `ptycho_torch/train.py:518` | Bridge raises error if missing (`config_bridge.py:271-278`) |
| **n_images** | Legacy CLI `--n_images` | Deprecated | 3 (CLI) | Convert to `n_groups` via `__post_init__` | `ptycho/config/config.py:124` | Deprecated alias for `n_groups` |
| **n_subsample** | TF Dataclass | None | 5 (TF default) | Pass through to TF TrainingConfig | `ptycho/config/config.py:118` | **Missing CLI flag** in PyTorch |
| **subsample_seed** | TF Dataclass | None | 5 (TF default) | Pass through to TF TrainingConfig | `ptycho/config/config.py:119` | **Missing CLI flag** in PyTorch |
| **sequential_sampling** | TF Dataclass | False | 5 (TF default) | Pass through to TF TrainingConfig | `ptycho/config/config.py:129` | **Missing CLI flag** in PyTorch |
| **batch_size** | CLI Flag `--batch_size` | 4 | 3 (CLI) | Pass to PT TrainingConfig, TF TrainingConfig | `ptycho_torch/train.py:366` | Shared across backends |
| **nepochs** | CLI `--max_epochs` | 10 | 3 (CLI) | Map `max_epochs`→`nepochs` via bridge | `ptycho_torch/train.py:487` | **Naming divergence:** PT uses `epochs`, TF uses `nepochs` |
| **epochs** | PT TrainingConfig | 10 | 4 (PT default) | Bridge converts to `nepochs` | `ptycho_torch/train.py:487` | PyTorch-specific naming |
| **nll_weight** | TF Dataclass | 1.0 | 5 (TF default) | Pass through to TF TrainingConfig | `ptycho/config/config.py:111` | PT uses bool `nll`, bridge converts |
| **mae_weight** | TF Dataclass | 1.0 | 5 (TF default) | Pass through to TF TrainingConfig | `ptycho/config/config.py:110` | Not exposed in PyTorch CLI |
| **realspace_weight** | TF Dataclass | 0.0 | 5 (TF default) | Pass through to TF TrainingConfig | `ptycho/config/config.py:113` | Not exposed in PyTorch CLI |
| **realspace_mae_weight** | TF Dataclass | 0.0 | 5 (TF default) | Pass through to TF TrainingConfig | `ptycho/config/config.py:112` | Not exposed in PyTorch CLI |
| **neighbor_count** | Derived from K | 7 | 1 (derived) | Bridge maps `K`→`neighbor_count` | `config_bridge.py:279` | TF naming; PT uses `K` |
| **positions_provided** | TF Dataclass | True | 5 (TF default) | Pass through to TF TrainingConfig | `ptycho/config/config.py:120` | Legacy compatibility flag |
| **probe_trainable** | TF Dataclass | False | 5 (TF default) | Pass through to TF TrainingConfig | `ptycho/config/config.py:121` | Not exposed in PyTorch CLI |
| **intensity_scale_trainable** | TF Dataclass | True | 5 (TF default) | Pass through to TF TrainingConfig | `ptycho/config/config.py:122` | Not exposed in PyTorch CLI |
| **output_dir** | CLI Flag `--output_dir` | Required | 3 (CLI) | Validate directory, create if missing, pass to TF | `ptycho_torch/train.py:366` | TF naming: `output_prefix` |

---

## 4. Inference Configuration Fields

| Field | Source | Default | Priority | Factory Role | File:Line | Notes |
|-------|--------|---------|----------|--------------|-----------|-------|
| **model_path** | CLI Flag `--model_path` | Required | 3 (CLI) | Validate directory exists, contains `wts.h5.zip` | `ptycho_torch/inference.py:293` | No default; raises error if missing |
| **test_data_file** | CLI Flag `--test_data_file` | Required | 3 (CLI) | Validate path exists | `ptycho_torch/inference.py:293` | Used for inference data |
| **n_groups** | CLI Flag `--n_groups` | Required | 1 (override) | Same as training; must be in overrides | `ptycho_torch/inference.py:293` | Bridge requires explicit value |
| **debug** | TF Dataclass | False | 5 (TF default) | Controls progress bar, logging verbosity | `ptycho/config/config.py:137` | Maps to `enable_progress_bar` |
| **output_dir** | CLI Flag `--output_dir` | Required | 3 (CLI) | Validate directory, create if missing | `ptycho_torch/inference.py:293` | Used for saving reconstructions |

---

## 5. PyTorch Execution Configuration Fields (Proposed)

**Note:** These fields do NOT exist in TensorFlow canonical configs. They control PyTorch-specific runtime behavior.

| Field | Source | Default | Priority | Factory Role | File:Line | Notes |
|-------|--------|---------|----------|--------------|-----------|-------|
| **accelerator** | Execution Config | 'auto' | 2 (exec) | Pass to Lightning Trainer | `ptycho_torch/workflows/components.py:567` | cpu/gpu/tpu/mps/auto |
| **strategy** | Execution Config | 'auto' | 2 (exec) | Pass to Lightning Trainer | N/A (not implemented) | auto/ddp/fsdp/deepspeed |
| **n_devices** | CLI `--device` (derived) | 1 | 3 (CLI) | Convert 'cpu'→1, 'gpu'→`torch.cuda.device_count()` | `ptycho_torch/train.py:493` | Lightning `devices` parameter |
| **deterministic** | Execution Config | True | 2 (exec) | Pass to Lightning Trainer | `ptycho_torch/workflows/components.py:571` | Reproducibility flag |
| **num_workers** | Execution Config | 0 | 2 (exec) | Pass to DataLoader | `ptycho_torch/workflows/components.py:361` | Hardcoded 0; should be configurable |
| **pin_memory** | Execution Config | False | 2 (exec) | Pass to DataLoader | `ptycho_torch/workflows/components.py:362` | Hardcoded False |
| **persistent_workers** | Execution Config | False | 2 (exec) | Pass to DataLoader | N/A (not used) | Performance knob |
| **prefetch_factor** | Execution Config | None | 2 (exec) | Pass to DataLoader | N/A (not used) | Performance knob |
| **learning_rate** | Execution Config | 1e-3 | 2 (exec) | Pass to optimizer | `ptycho_torch/workflows/components.py:538` | Hardcoded; **missing CLI flag** |
| **scheduler** | Execution Config | 'Default' | 2 (exec) | Select LR scheduler type | N/A (not implemented) | Default/Exponential/MultiStage |
| **gradient_clip_val** | Execution Config | None | 2 (exec) | Pass to Lightning Trainer | N/A (not used) | Gradient clipping |
| **accum_steps** | Execution Config | 1 | 2 (exec) | Gradient accumulation steps | N/A (not used) | For large batch simulation |
| **enable_checkpointing** | Execution Config | True | 2 (exec) | Lightning checkpoint callback | `ptycho_torch/workflows/components.py:570` | Implicit enabled |
| **checkpoint_save_top_k** | Execution Config | 1 | 2 (exec) | How many checkpoints to keep | N/A (uses Lightning default) | Top-K callback config |
| **checkpoint_monitor_metric** | Execution Config | 'val_loss' | 2 (exec) | Metric for best checkpoint | N/A (not implemented) | Checkpoint selection |
| **early_stop_patience** | Execution Config | 100 | 2 (exec) | Early stopping patience | `ptycho_torch/train.py:247` | Hardcoded; **missing CLI flag** |
| **enable_progress_bar** | CLI `--debug` (derived) | False | 3 (CLI) | Controls progress bar visibility | `ptycho_torch/workflows/components.py:570` | Derived from `config.debug` |
| **logger_backend** | Execution Config | None | 2 (exec) | tensorboard/wandb/mlflow | N/A (MLflow in legacy API only) | Experiment tracking |
| **disable_mlflow** | CLI Flag (legacy) | False | 3 (CLI) | Disables MLflow autologging | `ptycho_torch/api/example_train.py:119` | Legacy API only |
| **inference_batch_size** | Execution Config | None | 2 (exec) | Override batch_size for inference | N/A (uses training batch_size) | Inference-specific |
| **middle_trim** | Execution Config | 0 | 2 (exec) | Inference trimming parameter | N/A (not used) | Inference-specific |
| **pad_eval** | Execution Config | False | 2 (exec) | Padding for evaluation | N/A (not used) | Inference-specific |

---

## 6. Override Conflict Resolution Examples

### Example 1: n_groups Specified Multiple Ways

**Scenario:**
```python
# CLI: --n_groups 512
# Overrides dict: {'n_groups': 1024}
# Execution config: (no n_groups field)
```

**Resolution:**
- Priority 1 (Overrides dict) wins: `n_groups = 1024`
- Factory logs warning: "Overriding CLI --n_groups=512 with overrides dict value 1024"

### Example 2: Probe Size Inference vs Override

**Scenario:**
```python
# NPZ metadata: probeGuess.shape = (128, 128) → inferred N=128
# Overrides dict: {'N': 64}
```

**Resolution:**
- Priority 1 (Overrides dict) wins: `N = 64`
- Factory logs warning: "Overriding inferred probe size N=128 with explicit override N=64"

### Example 3: Execution Config vs CLI Default

**Scenario:**
```python
# CLI: --batch_size not specified (default=4)
# Execution config: PyTorchExecutionConfig(batch_size=8)  # Hypothetical if added
```

**Resolution:**
- Priority 2 (Execution config) wins: `batch_size = 8`
- No warning (expected behavior)

### Example 4: PyTorch vs TensorFlow Defaults (nphotons)

**Scenario:**
```python
# PyTorch default: nphotons = 1e5 (DataConfig)
# TensorFlow default: nphotons = 1e9 (TrainingConfig)
# Overrides dict: (empty)
```

**Resolution:**
- Factory uses TensorFlow default: `nphotons = 1e9`
- Factory logs warning: "nphotons divergence: PyTorch default 1e5 vs TensorFlow 1e9; using TensorFlow value"
- **Rationale:** TensorFlow config is canonical source of truth

---

## 7. Missing CLI Flags (Identified)

**The following TensorFlow config fields are NOT exposed as PyTorch CLI flags:**

| TF Field | Current Status | Impact | Recommendation |
|----------|----------------|--------|----------------|
| `n_subsample` | Not in CLI | Cannot control independent subsampling | **HIGH:** Add `--n_subsample` flag |
| `subsample_seed` | Not in CLI | Cannot reproduce subsampling | **HIGH:** Add `--subsample_seed` flag |
| `sequential_sampling` | Not in CLI | Cannot enforce deterministic grouping | **MEDIUM:** Add `--sequential_sampling` flag |
| `n_filters_scale` | Not in CLI | Cannot tune model capacity | **LOW:** Advanced parameter |
| `object_big` | Not in CLI | Cannot enable per-position reconstruction | **LOW:** Experimental feature |
| `probe_big` | Not in CLI | Cannot enable large-probe mode | **LOW:** Experimental feature |
| `pad_object` | Not in CLI | Cannot control padding strategy | **LOW:** Default sufficient |
| `probe_mask` | Not in CLI | Cannot enable probe masking | **LOW:** Experimental feature |
| `probe_scale` | Not in CLI | Cannot control probe normalization | **LOW:** Default 1.0 sufficient |
| `gaussian_smoothing_sigma` | Not in CLI | Cannot control probe smoothing | **LOW:** Default 0.0 (disabled) |
| `nll_weight` | Not in CLI | Cannot tune NLL loss weight | **MEDIUM:** Physics-informed training |
| `mae_weight` | Not in CLI | Cannot tune MAE loss weight | **MEDIUM:** Physics-informed training |
| `realspace_weight` | Not in CLI | Cannot tune realspace loss | **LOW:** Advanced feature |
| `probe_trainable` | Not in CLI | Cannot enable probe optimization | **MEDIUM:** Experimental feature |
| `intensity_scale_trainable` | Not in CLI | Cannot toggle scale learning | **LOW:** Default True works |
| `learning_rate` | Hardcoded 1e-3 | Cannot tune optimizer | **HIGH:** Should be CLI flag |
| `early_stop_patience` | Hardcoded 100 | Cannot control early stopping | **MEDIUM:** Should be CLI flag |

**Phase C/D Recommendation:** Add HIGH and MEDIUM priority flags to PyTorch CLI for parity.

---

## 8. Factory Responsibilities Summary

| Factory Function | Inputs | Outputs | Key Operations |
|------------------|--------|---------|----------------|
| `create_training_payload()` | train_data_file, output_dir, overrides, execution_config | TrainingPayload | 1. Infer probe size<br>2. Construct PyTorch configs<br>3. Apply overrides<br>4. Translate via bridge<br>5. Populate params.cfg<br>6. Validate n_groups |
| `create_inference_payload()` | model_path, test_data_file, output_dir, overrides, execution_config | InferencePayload | 1. Load checkpoint config<br>2. Construct PyTorch configs<br>3. Apply overrides<br>4. Translate via bridge<br>5. Populate params.cfg |
| `infer_probe_size()` | data_file | int (N value) | 1. Load NPZ probeGuess<br>2. Extract shape[0]<br>3. Validate square<br>4. Fallback to 64 on error |
| `populate_legacy_params()` | tf_config, force | None (side effect) | 1. Call update_legacy_dict<br>2. Log params.cfg snapshot<br>3. Validate CONFIG-001 compliance |

---

## 9. Validation Matrix

| Field | Validation Type | Error Condition | Action |
|-------|-----------------|-----------------|--------|
| `train_data_file` | Path existence | File not found | Raise `FileNotFoundError` |
| `test_data_file` | Path existence | File not found | Raise `FileNotFoundError` |
| `model_path` | Directory existence | Directory not found | Raise `FileNotFoundError` |
| `output_dir` | Directory creation | Cannot create dir | Raise `PermissionError` |
| `n_groups` | Required field | Missing in overrides | Raise `ValueError("n_groups required")` |
| `gridsize` | Square grid | Non-square (future) | Raise `ValueError("Non-square grids not supported")` |
| `N` | Positive integer | N <= 0 | Raise `ValueError("N must be positive")` |
| `batch_size` | Positive integer | batch_size <= 0 | Raise `ValueError("batch_size must be positive")` |
| `nepochs` | Positive integer | nepochs <= 0 | Raise `ValueError("nepochs must be positive")` |
| `nphotons` | Divergence warning | PT ≠ TF default | Log warning, use TF value |
| `probe_size` | Inference failure | NPZ invalid | Log warning, fallback to N=64 |
| `test_data_file` (training) | Optional field | Missing for training | Log info (validation data skipped) |

---

## 10. References

**Config Bridge Translation:**
- `ptycho_torch/config_bridge.py:79-124` — `to_model_config()`
- `ptycho_torch/config_bridge.py:185-309` — `to_training_config()`
- `ptycho_torch/config_bridge.py:309-380` — `to_inference_config()`

**CLI Argument Parsing:**
- `ptycho_torch/train.py:366-458` — Training CLI flags
- `ptycho_torch/inference.py:293-410` — Inference CLI flags

**TensorFlow Canonical Configs:**
- `ptycho/config/config.py:42-105` — ModelConfig
- `ptycho/config/config.py:107-133` — TrainingConfig
- `ptycho/config/config.py:135-150` — InferenceConfig

**PyTorch Singleton Configs:**
- `ptycho_torch/config_params.py:12-57` — DataConfig
- `ptycho_torch/config_params.py:59-89` — ModelConfig
- `ptycho_torch/config_params.py:91-127` — TrainingConfig
- `ptycho_torch/config_params.py:129-154` — InferenceConfig

**Workflow Integration:**
- `ptycho_torch/workflows/components.py:150` — update_legacy_dict call
- `ptycho_torch/workflows/components.py:343-373` — DataLoader creation
- `ptycho_torch/workflows/components.py:565-574` — Lightning Trainer config

**Phase A Inventories:**
- `plans/active/ADR-003-BACKEND-API/reports/2025-10-19T225905Z/phase_a_inventory/execution_knobs.md` — 54 execution-only knobs
- `plans/active/ADR-003-BACKEND-API/reports/2025-10-19T225905Z/phase_a_inventory/cli_inventory.md` — CLI surface analysis
