# CLI Flag Inventory — ADR-003 Phase C4.A1

**Date:** 2025-10-20
**Purpose:** Consolidated mapping of current PyTorch CLI flags (training + inference) with TensorFlow equivalents and execution config targets.

---

## Reading This Inventory

**Columns:**
- **Flag:** CLI argument name (--kebab-case)
- **Current Default:** Default value in current implementation
- **Type:** Argument type (str/int/float/bool/choice)
- **Destination Field:** Which dataclass field receives the value
- **TF Equivalent:** Corresponding TensorFlow CLI flag (if any)
- **File:Line:** Current argparse definition location
- **Notes:** Implementation details, gaps, or deviations

---

## Training CLI Flags (ptycho_torch/train.py)

### Current Implementation (12 flags)

| Flag | Current Default | Type | Destination Field | TF Equivalent | File:Line | Notes |
|------|----------------|------|------------------|---------------|-----------|-------|
| `--train_data_file` | required | str | TrainingConfig.train_data_file | `--train_data_file` | train.py:380 | New CLI interface, required flag |
| `--test_data_file` | None (optional) | str | TrainingConfig.test_data_file | `--test_data_file` | train.py:382 | Optional validation data |
| `--output_dir` | required | str | TrainingConfig.output_dir | `--output_dir` | train.py:384 | Required for checkpoint storage |
| `--max_epochs` | 100 | int | TrainingConfig.epochs | `--nepochs` | train.py:386 | **Naming divergence:** PT uses max_epochs, TF uses nepochs |
| `--n_images` | 512 | int | bridge override `n_groups` | `--n_groups` | train.py:388 | **Naming divergence:** PT uses n_images, TF uses n_groups |
| `--gridsize` | 2 | int | DataConfig.grid_size | `--gridsize` | train.py:390 | Convert to tuple (gridsize, gridsize) |
| `--batch_size` | 4 | int | TrainingConfig.batch_size | `--batch_size` | train.py:392 | Shared across backends |
| `--device` | 'cpu' | choice | TrainingConfig.n_devices | (none) | train.py:394 | **PT-specific:** Converts to n_devices (1 for cpu, torch.cuda.device_count() for cuda) |
| `--disable_mlflow` | False | bool | (legacy API) | (none) | train.py:396 | Legacy flag; not part of canonical config |
| `--ptycho_dir` | None | str | (legacy) | `--ptycho_dir` | train.py:400 | Legacy interface only |
| `--config` | None | str | (legacy) | `--config` | train.py:402 | Legacy JSON config path |
| (inferred) | N/A | N/A | ModelConfig.N | `--N` | train.py:467-473 | **Not a flag:** Inferred from NPZ probeGuess |

**Total Flags:** 9 active new interface flags, 2 legacy interface flags, 1 inferred parameter

---

### Hardcoded Values (Exposed in Code, Not CLI)

| Parameter | Hardcoded Value | Location | Destination Field | Notes |
|-----------|----------------|----------|------------------|-------|
| `nphotons` | 1e9 | train.py:530 | bridge override | **Divergence:** PT default 1e5 vs TF 1e9; hardcoded override to match TF |
| `K` | 7 | train.py:477 | DataConfig.K | Neighbor count; no CLI flag |
| `experiment_name` | 'ptychopinn_pytorch' | train.py:494 | TrainingConfig.experiment_name | No CLI flag |
| `amp_activation` | 'silu' | train.py:484 | ModelConfig.activate | **Naming divergence:** TF uses amp_activation, PT uses activate |
| `mode` | 'Unsupervised' | train.py:483 | ModelConfig.mode | Maps to TF model_type='pinn' |
| `nll` | (not set) | N/A | TrainingConfig.nll | PT bool flag; TF uses nll_weight float |
| `early_stop_patience` | 100 | (legacy code) | (execution config target) | Not exposed in new CLI |
| `learning_rate` | 1e-3 | (workflow) | (execution config target) | Hardcoded in _train_with_lightning; **missing CLI flag** |

**Total Hardcoded:** 8 values that should be configurable

---

## Inference CLI Flags (ptycho_torch/inference.py)

### Current Implementation (10 flags)

| Flag | Current Default | Type | Destination Field | TF Equivalent | File:Line | Notes |
|------|----------------|------|------------------|---------------|-----------|-------|
| `--model_path` | required | str | InferenceConfig.model_path | `--model_path` | inference.py:344 | Directory containing checkpoint |
| `--test_data` | required | str | InferenceConfig.test_data_file | `--test_data_file` | inference.py:350 | **Naming divergence:** PT uses test_data, TF uses test_data_file |
| `--output_dir` | required | str | InferenceConfig.output_dir | `--output_dir` | inference.py:356 | Required for reconstruction outputs |
| `--n_images` | 32 | int | bridge override `n_groups` | `--n_groups` | inference.py:362 | Same naming divergence as training |
| `--device` | 'cpu' | choice | (execution config target) | (none) | inference.py:368 | PT-specific device selection |
| `--quiet` | False | bool | InferenceConfig.debug (inverted) | `--debug` (inverted) | inference.py:375 | **Logic inversion:** --quiet → debug=False |

**Total Flags:** 6 flags

---

### Missing Inference Execution Config Flags

| Missing Flag | Target Field | Default | Rationale |
|--------------|-------------|---------|-----------|
| `--inference-batch-size` | PyTorchExecutionConfig.inference_batch_size | None (uses training batch_size) | Override batch size for inference DataLoader |
| `--num-workers` | PyTorchExecutionConfig.num_workers | 0 | DataLoader parallelism (shared with training) |

**Total Missing:** 2 execution config knobs

---

## Proposed Execution Config Flags (High Priority for C4)

### Training Execution Config Flags (5 new flags)

| Proposed Flag | Type | Default | Destination Field | Rationale | Priority |
|---------------|------|---------|------------------|-----------|----------|
| `--accelerator` | choice | 'auto' | PyTorchExecutionConfig.accelerator | cpu/gpu/tpu/mps selection | **HIGH** |
| `--deterministic` | bool | True | PyTorchExecutionConfig.deterministic | Reproducibility control | **HIGH** |
| `--num-workers` | int | 0 | PyTorchExecutionConfig.num_workers | DataLoader parallelism | **HIGH** |
| `--learning-rate` | float | 1e-3 | PyTorchExecutionConfig.learning_rate | Optimizer learning rate | **HIGH** |
| `--inference-batch-size` | int | None | PyTorchExecutionConfig.inference_batch_size | Inference DataLoader batch size | **MEDIUM** |

**Total Proposed:** 5 flags for Phase C4

---

## Deferred Execution Config Flags (Phase D)

Per `override_matrix.md` §5 and Phase C4 plan §"Deferred", the following knobs are **intentionally out of scope** for C4:

| Deferred Flag | Target Field | Default | Deferred Reason |
|---------------|-------------|---------|-----------------|
| `--checkpoint-save-top-k` | PyTorchExecutionConfig.checkpoint_save_top_k | 1 | Requires Lightning ModelCheckpoint callback config |
| `--checkpoint-monitor-metric` | PyTorchExecutionConfig.checkpoint_monitor_metric | 'val_loss' | Requires metric name validation |
| `--early-stop-patience` | PyTorchExecutionConfig.early_stop_patience | 100 | Currently hardcoded to 100 in legacy code |
| `--logger-backend` | PyTorchExecutionConfig.logger_backend | None | MLflow vs TensorBoard governance decision pending |
| `--scheduler` | PyTorchExecutionConfig.scheduler | 'Default' | LR scheduler selection (StepLR/ReduceLROnPlateau/CosineAnnealing) |
| `--prefetch-factor` | PyTorchExecutionConfig.prefetch_factor | None | DataLoader performance knob; not yet critical |
| `--persistent-workers` | PyTorchExecutionConfig.persistent_workers | False | DataLoader performance knob; not yet critical |

**Total Deferred:** 7 execution config knobs

---

## TensorFlow CLI Gaps (Not Exposed in PyTorch)

The following TensorFlow canonical config fields are **not exposed** as PyTorch CLI flags (per `override_matrix.md` §7):

### Core Model Config Gaps

| TF Field | Impact | Recommendation |
|----------|--------|----------------|
| `n_filters_scale` | Cannot tune model capacity | **LOW:** Advanced parameter |
| `object_big` | Cannot enable per-position reconstruction | **LOW:** Experimental feature |
| `probe_big` | Cannot enable large-probe mode | **LOW:** Experimental feature |
| `pad_object` | Cannot control padding strategy | **LOW:** Default sufficient |
| `probe_mask` | Cannot enable probe masking | **LOW:** Experimental feature |
| `probe_scale` | Cannot control probe normalization | **LOW:** Default 1.0 sufficient |
| `gaussian_smoothing_sigma` | Cannot control probe smoothing | **LOW:** Default 0.0 (disabled) |

### Training Config Gaps

| TF Field | Impact | Recommendation |
|----------|--------|----------------|
| `n_subsample` | Cannot control independent subsampling | **HIGH:** Add flag |
| `subsample_seed` | Cannot reproduce subsampling | **HIGH:** Add flag |
| `sequential_sampling` | Cannot enforce deterministic grouping | **MEDIUM:** Add flag |
| `nll_weight` | Cannot tune NLL loss weight | **MEDIUM:** Physics-informed training |
| `mae_weight` | Cannot tune MAE loss weight | **MEDIUM:** Physics-informed training |
| `realspace_weight` | Cannot tune realspace loss | **LOW:** Advanced feature |
| `probe_trainable` | Cannot enable probe optimization | **MEDIUM:** Experimental feature |
| `intensity_scale_trainable` | Cannot toggle scale learning | **LOW:** Default True works |

**Total TF Gaps:** 15 fields (6 model, 9 training)

---

## Naming Divergence Summary

| PyTorch Flag | TensorFlow Flag | Mapping Strategy |
|--------------|----------------|------------------|
| `--max_epochs` | `--nepochs` | Bridge maps `epochs`→`nepochs` |
| `--n_images` | `--n_groups` | Bridge override maps `n_images`→`n_groups` |
| `--test_data` | `--test_data_file` | CLI rename to `--test_data_file` recommended |
| `--device` (choice) | (none) | PT-specific; converts to `n_devices` |
| `--quiet` | `--debug` | Logic inversion; `quiet=True` → `debug=False` |
| `ModelConfig.activate` | `ModelConfig.amp_activation` | Bridge handles name conversion |
| `DataConfig.grid_size` (tuple) | `ModelConfig.gridsize` (int) | Bridge converts int→tuple |

**Total Divergences:** 7 naming/logic differences

---

## References

**Current CLI Definitions:**
- Training: `ptycho_torch/train.py:366-404` (argparse setup)
- Inference: `ptycho_torch/inference.py:319-379` (argparse setup)

**Factory Design:**
- `plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/factory_design.md`

**Override Matrix:**
- `plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/override_matrix.md` §2-5

**Config Bridge:**
- `ptycho_torch/config_bridge.py:79-124` (to_model_config)
- `ptycho_torch/config_bridge.py:185-309` (to_training_config)

**Execution Config Definition:**
- `ptycho/config/config.py:72-90` (PyTorchExecutionConfig dataclass)

---

**Summary:**
- **Training:** 9 active flags + 8 hardcoded values + 5 proposed execution config flags = 22 total parameters
- **Inference:** 6 active flags + 2 missing execution config flags = 8 total parameters
- **Gaps:** 15 TF canonical fields not exposed, 7 naming divergences documented
- **Phase C4 Scope:** Add 5 high-priority execution config flags (3 training, 2 inference)
