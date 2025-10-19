# PyTorch CLI to Config Dataflow Architecture

## Overview

This document maps the complete data flow from CLI arguments through config transformations to final CONFIG-001 compliance.

---

## 1. Training Workflow: CLI → Config → TensorFlow → Legacy

### Phase 1A: CLI Argument Parsing (train.py:366-405)

```
USER INPUT
    ↓
argparse.ArgumentParser.parse_args()
    ↓
argparse.Namespace object with 12 fields:
    - train_data_file: str
    - test_data_file: str | None
    - output_dir: str
    - max_epochs: int
    - n_images: int
    - gridsize: int
    - batch_size: int
    - device: {'cpu' | 'cuda'}
    - disable_mlflow: bool
    - ptycho_dir: str | None
    - config: str | None
```

**Source**: train.py lines 380-403

### Phase 1B: Interface Selection (train.py:407-442)

```
                         ┌─── legacy_interface check (line 408)
args.Namespace ──┬───────┤
                 │       └─── ptycho_dir is not None OR config is not None
                 │
                 └───── new_interface check (line 409)
                        └─── train_data_file is not None OR output_dir is not None

                MUTUAL EXCLUSIVITY VALIDATION
                if legacy_interface AND new_interface → FAIL (line 411-414)
```

### Phase 2A: PyTorch Config Instantiation (train.py:463-501)

```
CLI Arguments (args.*)
    ↓
    ├─ train_data_file
    │  └─→ Path(args.train_data_file)
    │      └─→ _infer_probe_size(train_data_file)
    │          └─→ READ NPZ METADATA: probeGuess.shape[0]
    │              └─→ inferred_N: int (or fallback 64)
    │                  └─→ DataConfig(N=inferred_N, ...)
    │
    ├─ gridsize
    │  └─→ args.gridsize
    │      └─→ DataConfig(grid_size=(args.gridsize, args.gridsize), ...)
    │
    ├─ max_epochs, batch_size, device
    │  └─→ TrainingConfig(epochs=args.max_epochs, batch_size=args.batch_size, ...)
    │
    └─ ModelConfig (hardcoded)
       └─→ ModelConfig(mode='Unsupervised', amp_activation='silu', ...)

RESULT: (DataConfig, ModelConfig, TrainingConfig, InferenceConfig, DatagenConfig)
```

**Source**: train.py lines 468-501

### Phase 2B: PyTorch Config Field Assignments

```
DataConfig Fields (line 476-481):
  N                = inferred_N (from NPZ probeGuess.shape[0])
  grid_size        = (args.gridsize, args.gridsize)  [Tuple[int,int]]
  K                = 7  [hardcoded]
  nphotons         = 1e9  [hardcoded override]
  probe_scale      = 1.0  [default]
  probe_normalize  = True  [default]
  normalize        = 'Batch'  [default]
  data_scaling     = 'Parseval'  [default]
  [other fields use defaults]

ModelConfig Fields (line 484-487):
  mode             = 'Unsupervised'  [hardcoded]
  amp_activation   = 'silu'  [hardcoded]
  [other fields use defaults]

TrainingConfig Fields (line 490-495):
  epochs           = args.max_epochs
  batch_size       = args.batch_size
  n_devices        = COMPUTE_N_DEVICES(args.device)
  experiment_name  = 'ptychopinn_pytorch'  [hardcoded]
  device           = 'cpu' | 'cuda'  [from args.device]
  [other fields use defaults]

InferenceConfig, DatagenConfig:
  [all default values]
```

### Phase 3A: Device Resolution (train.py:493, train_utils.py:62-78)

```
args.device: str ('cpu' or 'cuda')
    ↓
if args.device == 'cpu':
    n_devices = 1
else:
    n_devices = torch.cuda.device_count()  [if torch.cuda.is_available() else 1]
    ↓
TrainingConfig.n_devices = n_devices
    ↓
get_training_strategy(n_devices)
    ├─ if n_devices <= 1: return 'auto'
    └─ if n_devices >= 2: return DDPStrategy(...)
        ↓
        Used in L.Trainer(strategy=..., devices=n_devices)
```

### Phase 3B: Config Bridge Transformation (train.py:515-532, config_bridge.py)

```
PyTorch Configs (DataConfig, ModelConfig, TrainingConfig)
    ↓
    ├─→ to_model_config(data_config, model_config, overrides=None)
    │   ├─ Grid size: (int, int) → int
    │   │  │ grid_h, grid_w = data.grid_size
    │   │  │ if grid_h != grid_w: raise ValueError (line 109-112)
    │   │  │ gridsize = grid_h
    │   │  └─→ TFModelConfig(gridsize=gridsize, ...)
    │   │
    │   ├─ Mode mapping: 'Unsupervised' | 'Supervised' → 'pinn' | 'supervised'
    │   │  │ mode_to_model_type = {'Unsupervised': 'pinn', 'Supervised': 'supervised'}
    │   │  │ model_type = mode_to_model_type[model.mode]
    │   │  └─→ TFModelConfig(model_type=model_type, ...)
    │   │
    │   └─ Activation: 'silu' | 'SiLU' | 'sigmoid' | ... → 'swish' | 'sigmoid' | ...
    │      │ activation_mapping = {'silu': 'swish', ...}
    │      │ amp_activation = activation_mapping[model.amp_activation]
    │      └─→ TFModelConfig(amp_activation=amp_activation, ...)
    │
    └─→ to_training_config(tf_model, data, model, training, overrides={...})
        ├─ NLL bool → float: nll_weight = 1.0 if training.nll else 0.0  (line 218)
        │
        ├─ Field mappings:
        │  │ batch_size        = training.batch_size  [direct]
        │  │ nepochs           = training.epochs  [field rename]
        │  │ nll_weight        = 1.0 | 0.0  [bool → float]
        │  │ neighbor_count    = data.K  [semantic rename]
        │  │ nphotons          = data.nphotons  [direct, but validated]
        │  │ backend           = 'pytorch'  [hardcoded flag]
        │  │ model             = tf_model  [from bridge]
        │  │ train_data_file   = overrides['train_data_file']  [required]
        │  │ test_data_file    = overrides['test_data_file']  [optional]
        │  │ n_groups          = overrides['n_groups']  [required]
        │  │ output_dir        = overrides['output_dir']  [required]
        │  │
        │  └─ [other fields use defaults or overrides]
        │
        ├─ NPHOTONS VALIDATION (line 261-269):
        │  │ pytorch_default_nphotons = 1e5
        │  │ tensorflow_default_nphotons = 1e9
        │  │ if 'nphotons' NOT in overrides AND data.nphotons == 1e5:
        │  │     raise ValueError("CRITICAL: Require explicit override")
        │  │ else if 'nphotons' in overrides:
        │  │     nphotons = overrides['nphotons']
        │  │
        │  └─→ TFTrainingConfig(nphotons=nphotons, ...)
        │
        ├─ N_GROUPS VALIDATION (line 271-278):
        │  │ if kwargs['n_groups'] is None:
        │  │     raise ValueError("n_groups required in overrides")
        │  │
        │  └─→ TFTrainingConfig(n_groups=n_groups, ...)
        │
        └─ REQUIRED OVERRIDES DICT (line 525-531):
           ├─ train_data_file = Path(args.train_data_file)  [CRITICAL]
           ├─ test_data_file = Path(args.test_data_file) if provided
           ├─ output_dir = Path(args.output_dir)  [CRITICAL]
           ├─ n_groups = args.n_images  [CRITICAL, renamed from n_images]
           └─ nphotons = 1e9  [CRITICAL, explicit override to match TF default]

RESULT: TFTrainingConfig (with all 50+ fields populated)
```

**Source**: 
- config_bridge.py lines 79-306 (transformations)
- train.py lines 515-532 (call with overrides)

### Phase 4: Legacy params.cfg Population (train.py:535)

```
TFTrainingConfig (from bridge)
    ↓
update_legacy_dict(params.cfg, tf_training_config)
    ├─ For each field in tf_training_config:
    │  └─ KEY_MAPPINGS = { 'nepochs': 'epochs', 'gridsize': 'gridsize', ... }
    │     └─ params.cfg[KEY_MAPPINGS.get(field, field)] = value
    │
    └─ RESULT: params.cfg is now populated for CONFIG-001 compliance
       ├─ params.cfg['N'] = 64 (or inferred from NPZ)
       ├─ params.cfg['gridsize'] = 2 (or args.gridsize)
       ├─ params.cfg['n_groups'] = 512 (or args.n_images)
       ├─ params.cfg['epochs'] = 100 (or args.max_epochs)
       ├─ params.cfg['batch_size'] = 4 (or args.batch_size)
       ├─ params.cfg['backend'] = 'pytorch'
       └─ [40+ additional fields from TFTrainingConfig]
```

**Source**: 
- train.py line 535
- ptycho/config/config.py (update_legacy_dict implementation)

### Phase 5: Training Execution (train.py:549-555)

```
params.cfg (now populated ✓ CONFIG-001 compliant)
    ↓
main(
    ptycho_dir=str(train_data_file.parent),
    config_path=None,
    existing_config=(data_config, model_config, training_config, ...),
    disable_mlflow=args.disable_mlflow,
    output_dir=str(output_dir)
)
    ├─ Legacy modules can now read:
    │  └─ from ptycho import params
    │     └─ params.cfg['N'], params.cfg['gridsize'], params.cfg['n_groups'], ...
    │
    └─→ Training proceeds with CONFIG-001 guarantees
```

---

## 2. Inference Workflow: CLI → Checkpoint → Forward Pass

### Phase 1: CLI Argument Parsing (inference.py:319-380)

```
USER INPUT
    ↓
argparse.ArgumentParser.parse_args()
    ↓
argparse.Namespace object:
    - model_path: str (REQUIRED)
    - test_data: str (REQUIRED)
    - output_dir: str (REQUIRED)
    - n_images: int (default: 32)
    - device: {'cpu' | 'cuda'} (default: 'cpu')
    - quiet: bool (default: False)
```

### Phase 2: Input Validation (inference.py:399-409)

```
argparse.Namespace
    ├─ Path(args.model_path)
    │  └─ exists()? → YES
    │      └─ Store as model_path: Path
    │          └─ if NOT exists: raise FileNotFoundError
    │
    ├─ Path(args.test_data)
    │  └─ exists()? → YES
    │      └─ Store as test_data_path: Path
    │          └─ if NOT exists: raise FileNotFoundError
    │
    └─ Path(args.output_dir)
       └─ mkdir(parents=True, exist_ok=True)
           └─ Store as output_dir: Path
```

### Phase 3: Checkpoint Discovery (inference.py:412-429)

```
model_path: Path
    ↓
checkpoint_candidates = [
    model_path / "checkpoints" / "last.ckpt",  [Lightning default]
    model_path / "wts.pt",                      [Custom bundle]
    model_path / "model.pt",                    [Alternative]
]
    ↓
checkpoint_path = None
for candidate in checkpoint_candidates:
    if candidate.exists():
        checkpoint_path = candidate
        break

if checkpoint_path is None:
    raise FileNotFoundError(f"No checkpoint found in {model_path}")
    ├─ Searched: [list of candidates]
    ├─ Ensure training completed successfully
    └─ ERROR EXIT
```

### Phase 4A: Lightning Model Loading (inference.py:442-451)

```
checkpoint_path: Path
    ↓
PtychoPINN_Lightning.load_from_checkpoint(
    str(checkpoint_path),
    map_location=args.device
)
    ├─ Loads PyTorch Lightning module state_dict
    ├─ Maps to device: 'cpu' or 'cuda'
    ├─ Returns: PtychoPINN_Lightning instance
    │
    └─→ model.eval()  [Set to evaluation mode]
        └→ model.to(args.device)  [Ensure on correct device]
```

### Phase 4B: Test Data Loading (inference.py:462-480)

```
test_data_path: Path
    ↓
np.load(test_data_path)
    └─ Opens NPZ archive
        ├─ Validates required fields:
        │  ├─ 'diffraction': REQUIRED
        │  ├─ 'probeGuess': REQUIRED
        │  └─ 'objectGuess': REQUIRED
        │      └─ if missing: raise ValueError
        │
        └─ Returns: numpy.lib.npyio.NpzFile (dict-like)
           ├─ test_data['diffraction']: ndarray
           ├─ test_data['probeGuess']: ndarray
           └─ test_data['objectGuess']: ndarray
```

### Phase 5: Data Type Casting & Shape Normalization (inference.py:494-517)

```
test_data['diffraction']
    ├─ torch.from_numpy(...)  [Convert numpy → tensor]
    ├─ .to(args.device, dtype=torch.float32)  [CRITICAL: Cast to float32]
    │  └─ Prevents Conv2d dtype mismatch (Phase D1d hardening)
    │
    └─ Shape detection:
       ├─ if diffraction.ndim == 3 and diffraction.shape[-1] < diffraction.shape[0]:
       │  └─ Transpose (H, W, n) → (n, H, W): diffraction.permute(2, 0, 1)
       │      └─ Result: (n, H, W)
       │
       └─ Limit to n_images:
          ├─ diffraction[:args.n_images]  [Slice to n_images samples]
          └─ Result: (n, H, W) where n = min(original_n, args.n_images)

test_data['probeGuess']
    ├─ torch.from_numpy(...)
    ├─ .to(args.device, dtype=torch.complex64)  [Complex tensor]
    │
    └─ Add batch dimensions:
       ├─ if probe.ndim == 2:
       │  └─ probe.unsqueeze(0).unsqueeze(0).unsqueeze(0)
       │      └─ Result: (1, 1, 1, H, W)
```

### Phase 6: Forward Pass (inference.py:531-538)

```
diffraction: (n, H, W) or (n, 1, H, W)
probe: (1, 1, 1, H, W)
positions: (n, 1, 1, 2)  [dummy, all zeros]
input_scale_factor: (n, 1, 1, 1)  [dummy, all ones]
    ↓
with torch.no_grad():
    reconstruction = model.forward_predict(
        diffraction,
        positions,
        probe,
        input_scale_factor
    )
    └─ Returns: complex tensor (n, H, W) or (n, 1, H, W)
```

### Phase 7: Reconstruction Processing (inference.py:540-551)

```
reconstruction: torch.Tensor (complex, on device)
    ├─ reconstruction.cpu().numpy()  [Move to CPU, convert to numpy]
    │  └─ Result: ndarray (n, H, W) complex64
    │
    ├─ np.mean(..., axis=0)  [Average across batch]
    │  └─ Result: ndarray (H, W) complex64
    │
    ├─ Remove channel dimension if present:
    │  ├─ if ndim == 3: result_avg = result_avg[0]
    │  └─ Result: ndarray (H, W) complex64
    │
    ├─ result_amp = np.abs(reconstruction_avg)  [Amplitude]
    │  └─ Result: ndarray (H, W) float32
    │
    └─ result_phase = np.angle(reconstruction_avg)  [Phase]
       └─ Result: ndarray (H, W) float32
```

### Phase 8: Output Saving (inference.py:559)

```
result_amp: ndarray (H, W)
result_phase: ndarray (H, W)
output_dir: Path
    ↓
save_individual_reconstructions(result_amp, result_phase, output_dir)
    ├─ Create figure (amplitude)
    │  ├─ ax.imshow(result_amp, cmap='gray')
    │  ├─ plt.colorbar(...)
    │  └─ plt.savefig(output_dir / "reconstructed_amplitude.png")
    │
    └─ Create figure (phase)
       ├─ ax.imshow(result_phase, cmap='gray')
       ├─ plt.colorbar(...)
       └─ plt.savefig(output_dir / "reconstructed_phase.png")

RESULT:
    ├─ output_dir/reconstructed_amplitude.png (PNG image)
    └─ output_dir/reconstructed_phase.png (PNG image)
```

---

## 3. Config Transformation Summary Table

### Training Path Transformations

| Step | Input | Transformation | Output | Location |
|------|-------|----------------|--------|----------|
| 1 | CLI args | argparse | argparse.Namespace | train.py:366-405 |
| 2 | argparse.Namespace | interface select | legacy OR new path | train.py:407-414 |
| 3 | new path args | config instantiation | (DataConfig, ..., TrainingConfig, ...) | train.py:475-501 |
| 3a | train_data_file | NPZ metadata read | inferred_N | train.py:468-473 |
| 3b | args.device | device resolution | n_devices: int | train.py:493 |
| 4 | PyTorch configs | to_model_config() | TFModelConfig | config_bridge.py:79-182 |
| 4a | grid_size: (int, int) | extract first | gridsize: int | config_bridge.py:107-113 |
| 4b | mode: str | enum map | model_type: str | config_bridge.py:116-124 |
| 4c | amp_activation: str | normalize | amp_activation: str | config_bridge.py:127-140 |
| 5 | PyTorch configs + overrides | to_training_config() | TFTrainingConfig | config_bridge.py:185-306 |
| 5a | nll: bool | convert | nll_weight: float | config_bridge.py:218 |
| 5b | overrides dict | validate | nphotons: float (explicit) | config_bridge.py:261-269 |
| 5c | overrides dict | validate | n_groups: int (explicit) | config_bridge.py:271-278 |
| 6 | TFTrainingConfig | update_legacy_dict() | params.cfg populated | train.py:535 |
| 7 | params.cfg + configs | main() | training execution | train.py:549-555 |

### Inference Path Transformations

| Step | Input | Transformation | Output | Location |
|------|-------|----------------|--------|----------|
| 1 | CLI args | argparse | argparse.Namespace | inference.py:319-380 |
| 2 | argparse.Namespace | mode detect | Lightning or MLflow path | inference.py:576-578 |
| 3 | Lightning args | path validate | model_path, test_data_path validated | inference.py:399-409 |
| 4 | model_path | checkpoint search | checkpoint_path found | inference.py:412-429 |
| 5 | checkpoint_path | Lightning load | PtychoPINN_Lightning model | inference.py:442-451 |
| 6 | test_data_path | NPZ load | test_data dict-like | inference.py:462-480 |
| 7 | test_data + device | dtype cast | float32, complex64 tensors | inference.py:494-495 |
| 7a | diffraction shape | normalize | (H, W, n) → (n, H, W) if needed | inference.py:499-501 |
| 8 | tensors | forward_predict | reconstruction (complex) | inference.py:533-538 |
| 9 | reconstruction | numpy convert | (H, W) amplitude/phase | inference.py:540-551 |
| 10 | amplitude/phase | matplotlib save | PNG files | inference.py:559 |

---

## 4. Key Design Patterns

### Pattern: Overrides Dictionary

Used in both training and inference to inject execution-only parameters:

```python
# training:
overrides = dict(
    train_data_file=train_data_file,  # CLI → Path
    test_data_file=test_data_file,    # CLI → Path
    output_dir=output_dir,            # CLI → Path
    n_groups=args.n_images,           # CLI → semantic rename
    nphotons=1e9,                     # Explicit → force alignment
)

# Bridge function accepts overrides:
tf_training_config = to_training_config(
    tf_model,
    data,
    model,
    training,
    overrides=overrides  # Merges with defaults
)
```

### Pattern: Validation at Bridge

Critical validations happen during config bridge phase:

```python
# Nphotons default divergence detection (config_bridge.py:261-269)
# N_groups required field check (config_bridge.py:271-278)
# Test_data_file optional but warned (config_bridge.py:282-290)
```

### Pattern: Execution-Only Knobs

Parameters NOT in config dataclasses:

```python
# Device resolution (train_utils.py:62-88)
n_devices = get_strategy(args.device)

# Learning rate scaling (train_utils.py:80-88)
lr_scaled = find_learning_rate(base_lr, n_devices, batch_size)

# Checkpoint search (inference.py:412-429)
checkpoint_path = search_candidates([...])

# Data type casting (inference.py:494-495)
diffraction = tensor.to(dtype=torch.float32)
```

---

## 5. Critical Checkpoints (CONFIG-001 Compliance)

```
TRAIN WORKFLOW:
Step 3 ✓ PyTorch configs created
    ↓
Step 4 ✓ Bridge transformation to TensorFlow configs
    ↓
Step 5 ✓ Overrides dict with required fields
    ↓
Step 6 ✓✓✓ CRITICAL: params.cfg populated by update_legacy_dict()
    ↓
Step 7 ✓ Legacy modules can now access params.cfg safely
    ↓
✓ CONFIG-001 COMPLIANCE ACHIEVED
```

If Step 6 is skipped or fails:
- Legacy modules will fail with "AttributeError: params.cfg['field'] is None"
- Shape mismatch errors in model construction
- Silent failures in data loading

---

**Document Status**: Complete dataflow analysis (2025-10-19)  
**Architecture**: 3-layer config bridge (CLI → PyTorch → TensorFlow → Legacy)  
**Compliance**: CONFIG-001 validation enforced at bridge layer
