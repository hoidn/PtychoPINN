# PyTorch CLI Flags: Quick Reference Guide

## Training CLI

### Command Structure
```bash
python -m ptycho_torch.train [OPTIONS]
```

### All Available Flags

```
Data Input:
  --train_data_file PATH              (REQUIRED) Training NPZ dataset
  --test_data_file PATH               (optional) Validation NPZ dataset
  --output_dir PATH                   (REQUIRED) Checkpoint output directory

Model Configuration:
  --gridsize INT                      (default: 2) Grid size for grouping
  --n_images INT                      (default: 512) Number of diffraction groups

Training Hyperparameters:
  --max_epochs INT                    (default: 100) Training epochs
  --batch_size INT                    (default: 16) Training batch size

Execution Control:
  --device {cpu|cuda}                 (default: cpu) Compute device
  --disable_mlflow                    (flag) Disable MLflow tracking

Legacy Interface (backward compatible):
  --ptycho_dir PATH                   (legacy) Ptycho directory
  --config PATH                       (legacy) JSON config file
```

### Example Commands

**Minimal**
```bash
python -m ptycho_torch.train \
  --train_data_file data/train.npz \
  --output_dir ./outputs
```

**Full-featured**
```bash
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

---

## Inference CLI

### Command Structure
```bash
python -m ptycho_torch.inference [OPTIONS]
```

### All Available Flags (Lightning Mode)

```
Input/Output:
  --model_path PATH                   (REQUIRED) Training output directory
  --test_data PATH                    (REQUIRED) Test data NPZ file
  --output_dir PATH                   (REQUIRED) Output directory

Inference Configuration:
  --n_images INT                      (default: 32) Images to reconstruct
  --device {cpu|cuda}                 (default: cpu) Compute device
  --quiet                             (flag) Suppress progress output

Legacy MLflow Mode (auto-detected):
  --run_id ID                         (REQUIRED) MLflow run id
  --infer_dir PATH                    (REQUIRED) Inference directory
  --file_index INT                    (default: 0) File index
  --config PATH                       (optional) Config override
```

### Example Commands

**Minimal**
```bash
python -m ptycho_torch.inference \
  --model_path training_outputs \
  --test_data test.npz \
  --output_dir inference_outputs
```

**Full-featured**
```bash
python -m ptycho_torch.inference \
  --model_path training_outputs \
  --test_data datasets/Run1084.npz \
  --output_dir inference_outputs \
  --n_images 64 \
  --device cuda \
  --quiet
```

---

## Configuration Mapping Summary

### How CLI Flags Map to Config Objects

```
train.py CLI Flags
        ↓
PyTorch Config Dataclasses (DataConfig, ModelConfig, TrainingConfig, InferenceConfig)
        ↓
config_bridge Functions (to_model_config, to_training_config)
        ↓
TensorFlow Config Dataclasses (ModelConfig, TrainingConfig, InferenceConfig)
        ↓
params.cfg (Legacy TensorFlow backend)
```

### Key Field Mappings

| CLI Flag | PyTorch Config | TensorFlow Config | Type Transform |
|----------|---|---|---|
| `--train_data_file` | overrides dict | TrainingConfig.train_data_file | str → Path |
| `--max_epochs` | TrainingConfig.epochs | TrainingConfig.nepochs | int (rename) |
| `--batch_size` | TrainingConfig.batch_size | TrainingConfig.batch_size | direct |
| `--gridsize` | DataConfig.grid_size | ModelConfig.gridsize | int → Tuple[int, int] |
| `--n_images` | overrides dict | TrainingConfig.n_groups | int (rename) |
| `--device` | TrainingConfig.n_devices | n_devices (computed) | str → int |
| `--n_images` (inference) | inference slice | forward pass limit | int (slice) |

---

## Critical Implementation Details

### 1. Interface Selection (train.py line 407-414)

Cannot mix legacy and new CLI interfaces:
```python
# ERROR: These are mutually exclusive
python -m ptycho_torch.train --ptycho_dir DIR --train_data_file FILE.npz
```

### 2. Probe Size Inference (train.py line 468-473)

Automatically extracts `N` from `probeGuess.shape[0]`:
```python
inferred_N = _infer_probe_size(train_data_file)
if inferred_N is None:
    inferred_N = 64  # fallback
```

### 3. Device Resolution (train.py line 493)

Maps `--device` to `n_devices`:
- `'cpu'` → `n_devices = 1`
- `'cuda'` → `n_devices = torch.cuda.device_count()`

### 4. Nphotons Default Divergence (config_bridge.py line 261-269)

Explicit override required to prevent silent divergence:
```python
# PyTorch default: 1e5
# TensorFlow default: 1e9
# MUST specify: overrides=dict(nphotons=1e9)
```

### 5. Checkpoint Search (inference.py line 412-429)

Tries multiple checkpoint locations in order:
1. `checkpoints/last.ckpt` (Lightning default)
2. `wts.pt` (custom format)
3. `model.pt` (alternative naming)

### 6. Data Type Casting (inference.py line 494-495)

Enforces dtype consistency:
```python
diffraction = torch.from_numpy(test_data['diffraction']).to(
    args.device, dtype=torch.float32
)
probe = torch.from_numpy(test_data['probeGuess']).to(
    args.device, dtype=torch.complex64
)
```

---

## Factory Refactoring Roadmap

Current state: **Manual wiring in cli_main() functions**

Proposed improvements:

| Location | Current | Proposed Factory |
|----------|---------|------------------|
| train.py:475-501 | Manual config creation | `PyTorchConfigFactory.from_cli_args()` |
| train.py:515-532 | Manual bridge call | `TensorFlowConfigFactory.from_pytorch_configs()` |
| train.py:493 | Inline device resolution | `DeviceResolver.compute_n_devices()` |
| train.py:468-473 | Manual probe inference | `ProbeMetadataFactory.infer_N_from_npz()` |
| inference.py:412-429 | Manual checkpoint search | `CheckpointResolver.find_lightning_checkpoint()` |
| inference.py:494-495 | Manual dtype casting | `TorchDataFactory.cast_diffraction_data()` |
| inference.py:499-501 | Manual shape handling | `DiffractionShapeFactory.normalize_shape()` |

---

## Testing Checklist

- [ ] Training with defaults: `--train_data_file data.npz --output_dir out`
- [ ] Training with all flags set
- [ ] Training on GPU: `--device cuda`
- [ ] Training with validation: `--test_data_file test.npz`
- [ ] Inference with defaults: `--model_path out --test_data data.npz --output_dir infer`
- [ ] Inference on GPU: `--device cuda`
- [ ] Probe size inference works
- [ ] MLflow tracking disabled: `--disable_mlflow`
- [ ] Output files created correctly
- [ ] Legacy interface still works: `--ptycho_dir dir --config file.json`

---

**Document Status**: Complete CLI flag analysis (2025-10-19)  
**Source Code References**: See `pytorch_cli_inventory.md` for detailed line citations
