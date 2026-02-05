# PyTorch CLI Entry Points: Comprehensive Flag Inventory & Config Mapping

**Status**: Analysis Complete  
**Last Updated**: 2025-10-19  
**Target Files**:
- `/home/ollie/Documents/PtychoPINN2/ptycho_torch/train.py` (lines 366-570)
- `/home/ollie/Documents/PtychoPINN2/ptycho_torch/inference.py` (lines 293-400)
- `/home/ollie/Documents/PtychoPINN2/ptycho_torch/config_params.py`
- `/home/ollie/Documents/PtychoPINN2/ptycho_torch/config_bridge.py`

---

## Part 1: Training CLI Flags (train.py)

### Overview
Two distinct training interfaces are supported:
1. **New CLI Interface** (Phase E2.C1): Modern, user-friendly flags
2. **Legacy Interface**: `--ptycho_dir` + `--config` for backward compatibility

### New CLI Interface Flags (Phase E2.C1)

#### Data Input Flags

| Flag Name | Type | Default | Required | Help Text | Line | Maps to Config Field | Category |
|-----------|------|---------|----------|-----------|------|----------------------|----------|
| `--train_data_file` | str | None | YES | Path to training NPZ dataset | 380-381 | `DataConfig` + overrides | Data Input |
| `--test_data_file` | str | None | NO | Path to validation NPZ dataset | 382-383 | `DataConfig` + overrides | Data Input |
| `--output_dir` | str | None | YES | Directory for checkpoint outputs | 384-385 | `overrides['output_dir']` | Execution |

#### Model & Architecture Flags

| Flag Name | Type | Default | Required | Help Text | Line | Maps to Config Field | Category |
|-----------|------|---------|----------|-----------|------|----------------------|----------|
| `--n_images` | int | 512 | NO | Number of diffraction groups to process | 388-389 | `overrides['n_groups']` | Data |
| `--gridsize` | int | 2 | NO | Grid size for spatial grouping | 390-391 | `DataConfig.grid_size = (gridsize, gridsize)` | Model |

#### Training Hyperparameters

| Flag Name | Type | Default | Required | Help Text | Line | Maps to Config Field | Category |
|-----------|------|---------|----------|-----------|------|----------------------|----------|
| `--max_epochs` | int | 100 | NO | Maximum training epochs | 386-387 | `TrainingConfig.epochs` | Training |
| `--batch_size` | int | 4 | NO | Training batch size | 392-393 | `TrainingConfig.batch_size` | Training |

#### Execution Knobs

| Flag Name | Type | Default | Required | Help Text | Line | Maps to Config Field | Category |
|-----------|------|---------|----------|-----------|------|----------------------|----------|
| `--device` | str ('cpu'\|'cuda') | 'cpu' | NO | Compute device: cpu or cuda | 394-395 | `TrainingConfig.n_devices` → computed as: `1 if args.device == 'cpu' else torch.cuda.device_count()` | Execution |
| `--disable_mlflow` | bool (flag) | False | NO | Disable MLflow experiment tracking | 396-397 | `main(..., disable_mlflow=True)` | Execution |

---

### Legacy Interface Flags (Backward Compatible)

| Flag Name | Type | Default | Required | Help Text | Line | Behavior |
|-----------|------|---------|----------|-----------|------|----------|
| `--ptycho_dir` | str | None | YES (if legacy) | Path to ptycho directory | 400-401 | Passed directly to `main(args.ptycho_dir, ...)` |
| `--config` | str | None | NO (if legacy) | Path to JSON configuration file | 402-403 | Passed as `config_path` to `main()` |

---

### Flag Conflict Detection (train.py lines 407-414)

The CLI implementation enforces **mutual exclusivity**:
- **Cannot mix**: Legacy flags + New CLI flags
- **Must choose**: Either `(--ptycho_dir, --config)` OR `(--train_data_file, --output_dir)`

```python
# Line 408
legacy_interface = args.ptycho_dir is not None or args.config is not None

# Line 409
new_interface = args.train_data_file is not None or args.output_dir is not None

# Line 411-414: Fail if both specified
if legacy_interface and new_interface:
    sys.exit(1)  # ERROR: Cannot mix interfaces
```

---

## Part 2: Inference CLI Flags (inference.py)

### Overview
Two distinct inference modes:
1. **Lightning Checkpoint Inference** (Phase E2.C2): Direct model loading
2. **MLflow-based Inference** (Legacy): Loads from MLflow tracking server

### Lightning Checkpoint Interface Flags (Phase E2.C2)

Triggered when first arg contains `--model_path` or `--help` (line 576)

#### Input/Output Flags

| Flag Name | Type | Default | Required | Help Text | Line | Behavior | Category |
|-----------|------|---------|----------|-----------|------|----------|----------|
| `--model_path` | str | None | YES | Path to training output directory (expects `checkpoints/last.ckpt` or `wts.pt`) | 343-348 | Converted to Path, validated for existence, searched for checkpoint | Input |
| `--test_data` | str | None | YES | Path to test data NPZ file (must conform to specs/data_contracts.md) | 349-354 | Converted to Path, validated for existence, loaded via `np.load()` | Input |
| `--output_dir` | str | None | YES | Directory to save reconstruction outputs (amplitude/phase PNGs) | 355-360 | Created with `mkdir(parents=True, exist_ok=True)` | Output |

#### Inference Configuration Flags

| Flag Name | Type | Default | Required | Help Text | Line | Behavior | Category |
|-----------|------|---------|----------|-----------|------|----------|----------|
| `--n_images` | int | 32 | NO | Number of images to use for reconstruction | 361-366 | Limits diffraction array: `diffraction[:args.n_images]` (line 504) | Inference |
| `--device` | str ('cpu'\|'cuda') | 'cpu' | NO | Device to run inference on (cpu or cuda) | 367-373 | Passed to `torch.load(..., map_location=args.device)` (line 444) and tensor `.to()` calls | Execution |

#### Output Verbosity

| Flag Name | Type | Default | Required | Help Text | Line | Behavior | Category |
|-----------|------|---------|----------|-----------|------|----------|----------|
| `--quiet` | bool (flag) | False | NO | Suppress progress output | 374-378 | Controls print statements (lines 432-436, 449-451, 464-465, 527-528, 553-562) | Execution |

### Legacy MLflow Inference Flags

Triggered when first arg NOT in `['--model_path', '--help', '-h']` (line 576)

| Flag Name | Type | Default | Required | Help Text | Line | Behavior |
|-----------|------|---------|----------|-----------|------|----------|
| `--run_id` | str | None | YES (if MLflow) | Unique MLflow run id from training | 582 | Used to locate model via `mlflow.set_tracking_uri()` and `f"runs:/{run_id}/model"` |
| `--infer_dir` | str | None | YES (if MLflow) | Inference directory with ptycho files | 583 | Passed to `load_and_predict()` as `ptycho_files_dir` |
| `--file_index` | int | 0 | NO | File index if more than one file in infer_dir | 584 | Used to select which NPZ to process |
| `--config` | str | None | NO | Config to override loaded values | 585 | Optional: Passed as `config_override_path` to `load_and_predict()` |

---

## Part 3: Config Dataclass Field Mapping

### DataConfig Fields (ptycho_torch/config_params.py lines 17-44)

| PyTorch Field | Type | Default | CLI Source | Notes |
|---------------|------|---------|-----------|-------|
| `N` | int | 64 | Inferred from `probeGuess.shape[0]` (line 468) | Probe size detection via `_infer_probe_size()` |
| `grid_size` | Tuple[int, int] | (2, 2) | `--gridsize` → `(args.gridsize, args.gridsize)` | Converted to tuple from scalar |
| `K` | int | 6 | Hardcoded in config creation (line 479) | Not exposed in CLI; set to 7 in training setup |
| `nphotons` | float | 1e5 | Override in bridge (line 480, 530) | Explicitly set to 1e9 to match TensorFlow default |
| `probe_scale` | float | 1.0 | Not exposed in CLI | Uses PyTorch default |
| `probe_normalize` | bool | True | Not exposed in CLI | Uses PyTorch default |
| `normalize` | Literal | 'Batch' | Not exposed in CLI | Uses PyTorch default |
| `data_scaling` | Literal | 'Parseval' | Not exposed in CLI | Uses PyTorch default |

### ModelConfig Fields (ptycho_torch/config_params.py lines 47-90)

| PyTorch Field | Type | Default | CLI Source | Notes |
|---------------|------|---------|-----------|-------|
| `mode` | Literal['Unsupervised', 'Supervised'] | 'Unsupervised' | Hardcoded (line 485) | PINN mode for training |
| `amp_activation` | str | 'silu' | Hardcoded (line 486) | Set to 'silu' in training config creation |
| `intensity_scale_trainable` | bool | False | Not exposed | Uses default; moved to TrainingConfig in bridge |
| `object_big` | bool | False | Not exposed | Grid-lines Torch runner forces `object_big=False` for TF parity; other workflows use the PyTorch default unless overridden in config |
| `probe_big` | bool | True | Not exposed | Grid-lines Torch runner forces `probe_big=False` for TF parity; other workflows use the PyTorch default unless overridden in config |
| `loss_function` | Literal | 'Poisson' | Not exposed | Uses PyTorch default |

### TrainingConfig Fields (ptycho_torch/config_params.py lines 94-130)

| PyTorch Field | Type | Default | CLI Source | Notes |
|---------------|------|---------|-----------|-------|
| `epochs` | int | 50 | `--max_epochs` | Mapped to TensorFlow `nepochs` |
| `batch_size` | int | 16 | `--batch_size` | Direct 1:1 mapping |
| `learning_rate` | float | 1e-3 | Not exposed | Computed via `find_learning_rate()` (train_utils.py:80-88) |
| `n_devices` | int | 1 | `--device` → computed | `1` if cpu, else `torch.cuda.device_count()` |
| `nll` | bool | True | Not exposed | Converted to `nll_weight: float` in bridge |
| `device` | str | 'cuda' | `--device` | Set to 'cpu' or 'cuda' based on CLI |
| `experiment_name` | str | 'Synthetic_Runs' | Hardcoded (line 494) | Set to 'ptychopinn_pytorch' in training |
| `num_workers` | int | 4 | Not exposed | Uses PyTorch default |
| `accum_steps` | int | 1 | Not exposed | Uses PyTorch default |

### InferenceConfig Fields (ptycho_torch/config_params.py lines 132-138)

| PyTorch Field | Type | Default | CLI Source | Notes |
|---------------|------|---------|-----------|-------|
| `batch_size` | int | 1000 | Not exposed | Uses PyTorch default |
| `experiment_number` | int | 0 | Not exposed | Can be overridden for file selection |
| `window` | int | 20 | Not exposed | Used for edge trimming in reconstruction visualization |

---

## Part 4: Config Bridge Transformations

### Critical Field Transformations (config_bridge.py)

#### Grid Size Transformation
```python
# Line 107-113 (config_bridge.py)
grid_h, grid_w = data.grid_size
if grid_h != grid_w:
    raise ValueError("Non-square grids not supported")
gridsize = grid_h
# RESULT: Tuple[int, int] → int (extracts first element)
```

#### Mode Enum Mapping
```python
# Line 116-124 (config_bridge.py)
mode_to_model_type = {
    'Unsupervised': 'pinn',
    'Supervised': 'supervised'
}
model_type = mode_to_model_type[model.mode]
# RESULT: 'Unsupervised' → 'pinn' (for TensorFlow backend)
```

#### Activation Normalization
```python
# Line 127-140 (config_bridge.py)
activation_mapping = {
    'silu': 'swish',
    'SiLU': 'swish',
    'sigmoid': 'sigmoid',
    'swish': 'swish',
    'softplus': 'softplus',
    'relu': 'relu'
}
amp_activation = activation_mapping[model.amp_activation]
# RESULT: PyTorch 'silu' → TensorFlow 'swish' normalization
```

#### NLL Weight Conversion
```python
# Line 218 (config_bridge.py)
nll_weight = 1.0 if training.nll else 0.0
# RESULT: bool → float (True→1.0, False→0.0)
```

#### Nphotons Default Divergence Handling
```python
# Line 261-269 (config_bridge.py)
# PyTorch default: 1e5, TensorFlow default: 1e9
if 'nphotons' not in overrides and data.nphotons == pytorch_default_nphotons:
    raise ValueError(...)  # CRITICAL: Require explicit override
# RESULT: Validation forces explicit override to prevent silent divergence
```

---

## Part 5: Execution-Only Knobs (Not in Config Dataclasses)

### Training Execution-Only Parameters

| Parameter | CLI Flag | Value Source | Used In | Purpose |
|-----------|----------|---------------|---------|---------|
| Learning Rate Scaling | `--device`, `--batch_size` | `find_learning_rate()` (train_utils.py:80-88) | `model.lr` (line 222) | Scales base LR via sqrt(EBS/baseline) |
| Training Strategy | `--device` | `get_training_strategy()` (train_utils.py:62-78) | `L.Trainer(..., strategy=...)` (line 272) | Selects 'auto' for ≤1 GPU, DDPStrategy for >1 GPU |
| Checkpoint Root | `--output_dir` | `Path(output_dir)` (line 447) | `L.Trainer(..., default_root_dir=...)` (line 266) | Directory for Lightning checkpoints |
| MLflow Experiment | `--disable_mlflow` | Boolean flag (line 396) | `mlflow.set_experiment()` (line 289) | Controls experiment tracking |
| Validation Split | (Hardcoded) | `0.05` (line 210) | `PtychoDataModule()` | 5% of data reserved for validation |

### Inference Execution-Only Parameters

| Parameter | CLI Flag | Value Source | Used In | Purpose |
|-----------|----------|---------------|---------|---------|
| Checkpoint Search Path | `--model_path` | Candidate list (lines 412-416) | Loaded via `PtychoPINN_Lightning.load_from_checkpoint()` (line 442) | Finds checkpoint: `checkpoints/last.ckpt` → `wts.pt` → `model.pt` |
| Data Dtype Casting | `--test_data` | `torch.float32`, `torch.complex64` (lines 494-495) | Forward pass (line 533-538) | Prevents Conv2d dtype mismatch (Phase D1d hardening) |
| Diffraction Shape Permutation | `--test_data` | Detected at runtime (lines 499-501) | `diffraction.permute(2, 0, 1)` (line 501) | Handles (H, W, n) → (n, H, W) transposition |

---

## Part 6: Current Manual Wiring Patterns

### Pattern 1: Flag-to-Config Instantiation (train.py lines 475-501)

**Current (Manual):**
```python
# Line 475-487
data_config = DataConfig(
    N=inferred_N,
    grid_size=(args.gridsize, args.gridsize),
    K=7,
    nphotons=1e9,
)

model_config = ModelConfig(
    mode='Unsupervised',
    amp_activation='silu',
)

training_config = TrainingConfig(
    epochs=args.max_epochs,
    batch_size=args.batch_size,
    n_devices=1 if args.device == 'cpu' else ...,
    experiment_name='ptychopinn_pytorch',
)
```

**Factory Pattern Alternative:**
```python
# Proposed factory approach
config_set = PyTorchConfigFactory.from_cli_args(
    train_data_file=args.train_data_file,
    max_epochs=args.max_epochs,
    batch_size=args.batch_size,
    gridsize=args.gridsize,
    device=args.device,
)
data_config, model_config, training_config, ... = config_set
```

### Pattern 2: Bridge Transformation with Overrides (train.py lines 515-532)

**Current (Manual):**
```python
# Lines 520-532
tf_training_config = to_training_config(
    tf_model_config,
    data_config,
    model_config,
    training_config,
    overrides=dict(
        train_data_file=train_data_file,
        test_data_file=test_data_file,
        output_dir=output_dir,
        n_groups=args.n_images,
        nphotons=1e9,
    )
)
```

**Factory Pattern Alternative:**
```python
# Proposed factory approach
tf_training_config = TensorFlowConfigFactory.from_pytorch_configs(
    pytorch_configs=config_set,
    cli_overrides={
        'train_data_file': train_data_file,
        'test_data_file': test_data_file,
        'output_dir': output_dir,
        'n_groups': args.n_images,
        'nphotons': 1e9,
    },
    validate_strict=True,
)
```

### Pattern 3: Device Resolution (train.py line 493)

**Current (Manual):**
```python
# Line 493
n_devices=1 if args.device == 'cpu' else (
    torch.cuda.device_count() if torch.cuda.is_available() else 1
)
```

**Factory Pattern Alternative:**
```python
# Proposed factory helper
n_devices = DeviceResolver.compute_n_devices(
    device_arg=args.device,
    torch_available=torch.cuda.is_available()
)
```

### Pattern 4: Probe Size Inference (train.py lines 468-473)

**Current (Manual):**
```python
# Lines 468-473
inferred_N = _infer_probe_size(train_data_file)
if inferred_N is None:
    print(f"WARNING: Could not infer...")
    inferred_N = 64
else:
    print(f"✓ Inferred probe size...")
```

**Factory Pattern Alternative:**
```python
# Proposed factory helper
inferred_N = ProbeMetadataFactory.infer_N_from_npz(
    npz_path=train_data_file,
    fallback=64,
    verbose=True
)
```

### Pattern 5: Checkpoint Candidate Search (inference.py lines 412-429)

**Current (Manual):**
```python
# Lines 412-429
checkpoint_candidates = [
    model_path / "checkpoints" / "last.ckpt",
    model_path / "wts.pt",
    model_path / "model.pt",
]

checkpoint_path = None
for candidate in checkpoint_candidates:
    if candidate.exists():
        checkpoint_path = candidate
        break

if checkpoint_path is None:
    raise FileNotFoundError(...)
```

**Factory Pattern Alternative:**
```python
# Proposed factory helper
checkpoint_path = CheckpointResolver.find_lightning_checkpoint(
    model_path=model_path,
    search_order=['checkpoints/last.ckpt', 'wts.pt', 'model.pt'],
    strict=True  # Raise if not found
)
```

---

## Part 7: Summary of Gaps & Factory Opportunities

### Current State
- CLI flags → config instantiation is **fully manual** in `cli_main()` functions
- Bridge transformations hardcoded with `to_model_config()`, `to_training_config()` functions
- Execution-only knobs scattered: device resolution, checkpoint search, dtype casting
- No centralized validation or error messaging

### Factory Replacement Opportunities

| Category | Location | Current Pattern | Factory Candidate |
|----------|----------|------------------|-------------------|
| **Config Instantiation** | train.py:475-501 | Manual `DataConfig(N=..., grid_size=...)` | `PyTorchConfigFactory.from_cli_args()` |
| **Bridge Transformation** | train.py:515-532 | Manual `to_training_config(..., overrides=dict(...))` | `TensorFlowConfigFactory.from_pytorch_configs()` |
| **Device Resolution** | train.py:493 | Inline ternary operator | `DeviceResolver.compute_n_devices()` |
| **Probe Inference** | train.py:468-473 | Manual `_infer_probe_size()` + fallback | `ProbeMetadataFactory.infer_N_from_npz()` |
| **Checkpoint Search** | inference.py:412-429 | Manual loop + None check | `CheckpointResolver.find_lightning_checkpoint()` |
| **Data Type Casting** | inference.py:494-495 | Inline `torch.from_numpy(...).to(dtype=torch.float32)` | `TorchDataFactory.cast_diffraction_data()` |
| **Diffraction Shape Handling** | inference.py:499-501 | Inline shape detection + permute | `DiffractionShapeFactory.normalize_shape()` |

---

## Part 8: Reference: Complete CLI Invocation Examples

### Training (New CLI)
```bash
# Basic training with defaults
python -m ptycho_torch.train \
  --train_data_file data/train.npz \
  --output_dir ./outputs \
  --max_epochs 10 \
  --device cpu

# With validation data and custom grid
python -m ptycho_torch.train \
  --train_data_file data/train.npz \
  --test_data_file data/test.npz \
  --output_dir ./outputs \
  --max_epochs 50 \
  --batch_size 8 \
  --gridsize 4 \
  --n_images 256 \
  --device cuda \
  --disable_mlflow
```

### Inference (Lightning Checkpoint)
```bash
# Basic inference
python -m ptycho_torch.inference \
  --model_path training_outputs \
  --test_data test.npz \
  --output_dir inference_outputs

# With custom device and n_images
python -m ptycho_torch.inference \
  --model_path training_outputs \
  --test_data datasets/Run1084.npz \
  --output_dir inference_outputs \
  --n_images 64 \
  --device cuda \
  --quiet
```

---

## Appendix: File:Line Citation Index

### train.py (ptycho_torch/train.py)

| Element | Lines | Description |
|---------|-------|-------------|
| `cli_main()` function | 353-570 | Main CLI entrypoint |
| argparse setup (new interface) | 366-403 | Flag definitions |
| Flag conflict detection | 407-414 | Mutual exclusivity check |
| Legacy interface dispatch | 416-430 | Backward compatibility path |
| New interface dispatch | 432-562 | Phase E2.C1 new CLI path |
| Probe size inference | 468-473 | Inferred N from NPZ |
| DataConfig instantiation | 475-481 | Manual config creation |
| ModelConfig instantiation | 484-487 | Manual config creation |
| TrainingConfig instantiation | 490-495 | Manual config creation |
| Config bridge call | 515-532 | `to_training_config()` with overrides |
| params.cfg population | 535 | CONFIG-001 compliance |
| `main()` delegation | 549-555 | Call to training execution |

### inference.py (ptycho_torch/inference.py)

| Element | Lines | Description |
|---------|-------|-------------|
| `cli_main()` function | 293-571 | Main CLI entrypoint |
| Mode detection | 576-578 | Route to new vs. legacy path |
| argparse setup (Lightning) | 319-380 | Flag definitions |
| Lightning checkpoint loading | 438-457 | Model load from checkpoint |
| Test data loading | 459-480 | NPZ file validation |
| Checkpoint search | 412-429 | Candidate list iteration |
| Dtype casting | 494-495 | float32 + complex64 conversion |
| Diffraction shape normalization | 499-501 | (H,W,n) → (n,H,W) transposition |
| Forward pass | 531-538 | Model inference |
| Save outputs | 559 | `save_individual_reconstructions()` |

### config_params.py (ptycho_torch/config_params.py)

| Element | Lines | Description |
|---------|-------|-------------|
| `DataConfig` dataclass | 17-44 | Data pipeline configuration |
| `ModelConfig` dataclass | 47-90 | Model architecture configuration |
| `TrainingConfig` dataclass | 94-130 | Training hyperparameters |
| `InferenceConfig` dataclass | 132-138 | Inference parameters |
| `DatagenConfig` dataclass | 141-148 | Data generation parameters |
| `update_existing_config()` | 154-159 | Helper to update config instances |

### config_bridge.py (ptycho_torch/config_bridge.py)

| Element | Lines | Description |
|---------|-------|-------------|
| `to_model_config()` | 79-182 | PyTorch → TensorFlow ModelConfig translation |
| Grid size transformation | 107-113 | Tuple[int, int] → int extraction |
| Mode enum mapping | 116-124 | 'Unsupervised' → 'pinn' |
| Activation normalization | 127-140 | 'silu' → 'swish' mapping |
| `to_training_config()` | 185-306 | PyTorch → TensorFlow TrainingConfig translation |
| NLL weight conversion | 218 | bool → float conversion |
| Nphotons validation | 261-269 | Default divergence enforcement |
| n_groups validation | 271-278 | Required override check |
| `to_inference_config()` | 309-360 | PyTorch → TensorFlow InferenceConfig translation |

---

**End of Inventory**
