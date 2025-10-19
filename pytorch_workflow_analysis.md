# PyTorch Workflow Config Construction Analysis

## Executive Summary

This document analyzes how PyTorch workflow components currently construct and use configurations, identifying integration points where factory patterns would improve maintainability. The analysis covers inline config construction, CONFIG-001 compliance patterns, and execution parameters specific to PyTorch.

---

## 1. Current Config Construction Patterns

### 1.1 Inline Construction in CLI Entry Point (train.py)

**File:** `/home/ollie/Documents/PtychoPINN2/ptycho_torch/train.py:464-504`

Current pattern shows **direct instantiation** of PyTorch config singletons:

```python
# DataConfig: Configure data pipeline parameters
data_config = DataConfig(
    N=inferred_N,  # Derived from probeGuess.shape[0] in NPZ
    grid_size=(args.gridsize, args.gridsize),
    K=7,  # Default neighbor count
    nphotons=1e9,  # Use TensorFlow default to avoid divergence
)

# ModelConfig: Configure model architecture
model_config = ModelConfig(
    mode='Unsupervised',  # PINN mode
    amp_activation='silu',
)

# TrainingConfig: Configure training hyperparameters
training_config = TrainingConfig(
    epochs=args.max_epochs,
    batch_size=args.batch_size,
    n_devices=1 if args.device == 'cpu' else (torch.cuda.device_count() if torch.cuda.is_available() else 1),
    experiment_name='ptychopinn_pytorch',
)

# InferenceConfig and DatagenConfig (minimal instantiation)
inference_config = InferenceConfig()
datagen_config = DatagenConfig()
```

**Issues with inline pattern:**
1. **Scattered construction logic** — Config creation split across ~40 lines with domain-specific logic (e.g., probe size inference) embedded inline
2. **No reusability** — Each caller must duplicate probe size inference, device selection, and override logic
3. **Implicit defaults** — Constants like `K=7`, `mode='Unsupervised'`, `amp_activation='silu'` appear without documentation
4. **Validation absent** — No centralized place to validate interdependencies (e.g., gridsize vs N compatibility)

### 1.2 Bridge Translation Pattern (config_bridge.py)

**File:** `/home/ollie/Documents/PtychoPINN2/ptycho_torch/config_bridge.py:79-306`

Config bridge provides three translation functions that **receive inline-constructed objects**:

```python
# Line 517-532 in train.py: Calling the bridge
tf_model_config = to_model_config(data_config, model_config)

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
        nphotons=1e9,  # Explicit override to match TensorFlow default
    )
)
```

**Bridge transformation sequence:**

| Function | Input | Output | Key Transformations |
|----------|-------|--------|---------------------|
| `to_model_config()` (line 79) | `DataConfig`, `ModelConfig` + overrides | `TFModelConfig` | grid_size tuple→int, mode enum mapping, activation normalization |
| `to_training_config()` (line 185) | `TFModelConfig`, `DataConfig`, `ModelConfig`, `TrainingConfig` + overrides | `TFTrainingConfig` | epochs→nepochs, K→neighbor_count, nll bool→float, validation of nphotons/n_groups |
| `to_inference_config()` (line 309) | `TFModelConfig`, `DataConfig`, `InferenceConfig` + overrides | `TFInferenceConfig` | K→neighbor_count, backend='pytorch' flag |

**Critical transformations in to_model_config (lines 79-182):**

```python
# Line 106-113: Grid size extraction (assumes square grids)
grid_h, grid_w = data.grid_size
if grid_h != grid_w:
    raise ValueError(f"Non-square grids not supported...")
gridsize = grid_h

# Line 115-124: Mode enum mapping
mode_to_model_type = {
    'Unsupervised': 'pinn',
    'Supervised': 'supervised'
}
model_type = mode_to_model_type[model.mode]

# Line 126-140: Activation normalization
activation_mapping = {
    'silu': 'swish',
    'SiLU': 'swish',
    'sigmoid': 'sigmoid',
    'swish': 'swish',
    'softplus': 'softplus',
    'relu': 'relu'
}
amp_activation = activation_mapping[model.amp_activation]

# Line 142-148: probe_mask translation (Optional[Tensor] → bool)
probe_mask_value = False
if model.probe_mask is not None:
    probe_mask_value = True
```

**Critical validations in to_training_config (lines 185-306):**

```python
# Line 259-269: nphotons divergence check (PyTorch 1e5 vs TensorFlow 1e9)
pytorch_default_nphotons = 1e5
tensorflow_default_nphotons = 1e9
if 'nphotons' not in overrides and data.nphotons == pytorch_default_nphotons:
    raise ValueError(
        f"nphotons default divergence detected: PyTorch default ({pytorch_default_nphotons}) "
        f"differs from TensorFlow default ({tensorflow_default_nphotons})..."
    )

# Line 271-278: n_groups validation (phase B.B5.D3)
if kwargs['n_groups'] is None:
    raise ValueError(
        "n_groups is required in overrides for TrainingConfig. "
        "Missing override leaves params.cfg['n_groups'] = None, breaking downstream workflows..."
    )

# Line 300-304: train_data_file requirement
if kwargs['train_data_file'] is None:
    raise ValueError(
        "train_data_file is required in overrides for TrainingConfig..."
    )
```

### 1.3 Lightning Trainer Config Construction (components.py)

**File:** `/home/ollie/Documents/PtychoPINN2/ptycho_torch/workflows/components.py:459-614`

The `_train_with_lightning()` function shows **secondary config construction** for PyTorch Lightning execution:

```python
# Line 516-541: Inline construction of PyTorch config objects from TensorFlow TrainingConfig
mode_map = {'pinn': 'Unsupervised', 'supervised': 'Supervised'}

pt_data_config = PTDataConfig(
    N=config.model.N,
    grid_size=(config.model.gridsize, config.model.gridsize),
    nphotons=config.nphotons,
    K=config.neighbor_count,
)

pt_model_config = PTModelConfig(
    mode=mode_map.get(config.model.model_type, 'Unsupervised'),
    amp_activation=config.model.amp_activation or 'silu',
    n_filters_scale=config.model.n_filters_scale,
)

pt_training_config = PTTrainingConfig(
    epochs=config.nepochs,
    learning_rate=1e-4,  # Default; can expose via config later
    device=getattr(config, 'device', 'cpu'),
)

pt_inference_config = PTInferenceConfig()
```

**Issues with this pattern:**
1. **Bidirectional conversion** — Already converted TensorFlow→PyTorch, now converting back
2. **Lossy reconstruction** — Information already translated may be incomplete (e.g., learning_rate hardcoded to 1e-4)
3. **Scattered mode mapping** — Mode enum mapping appears in both config_bridge AND here (line 521 vs config_bridge line 116-124)

### 1.4 Model Manager Config Persistence (model_manager.py)

**File:** `/home/ollie/Documents/PtychoPINN2/ptycho_torch/model_manager.py:60-185`

Config is captured **after training** via `dataclass_to_legacy_dict()`:

```python
# Line 121-122: Snapshot captured at save time
params_snapshot = dataclass_to_legacy_dict(config)

# Line 135-136: Version tag for backend detection
params_snapshot['_version'] = '2.0-pytorch'
```

Archive structure preserves params per model:
```
wts.h5.zip/
├── manifest.dill
├── autoencoder/
│   ├── model.pth
│   └── params.dill  # Full params.cfg snapshot (CONFIG-001)
└── diffraction_to_obj/
    ├── model.pth
    └── params.dill
```

---

## 2. CONFIG-001 Compliance Patterns

### 2.1 Mandatory update_legacy_dict Call Sites

**Pattern:** All PyTorch workflows MUST call `update_legacy_dict(params.cfg, config)` before delegating to legacy modules.

**Call Sites:**

1. **Train CLI entry point (train.py:535)**
   ```python
   # Line 535 in cli_main()
   update_legacy_dict(params.cfg, tf_training_config)
   ```
   Status: ✅ IMPLEMENTED (CONFIG-001 compliance verified)

2. **Workflow orchestration (components.py:150)**
   ```python
   # Line 150 in run_cdi_example_torch()
   ptycho_config.update_legacy_dict(params.cfg, config)
   ```
   Status: ✅ IMPLEMENTED (CONFIG-001 compliance verified)

3. **Data bridge (raw_data_bridge.py)**
   ```python
   # Line ~250 in RawDataTorch.__init__()
   if config is provided:
       update_legacy_dict(p.cfg, config)
   ```
   Status: ✅ IMPLEMENTED (optional, triggered when config passed)

4. **Model loading (model_manager.py:262)**
   ```python
   # Line 262 in load_torch_bundle()
   params.cfg.update(params_dict)  # Direct update instead of update_legacy_dict
   ```
   Status: ⚠️ PARTIAL (uses dict.update instead of update_legacy_dict - no validation)

### 2.2 Configuration State Flow

```
CLI Args
   ↓
[1] PyTorch singletons created (DataConfig, ModelConfig, TrainingConfig, InferenceConfig)
   ↓
[2] Config bridge translation (PyTorch → TensorFlow dataclasses)
   ↓
[3] update_legacy_dict(params.cfg, tf_config) — CONFIG-001 GATE
   ↓
[4] Legacy modules (loader, model, etc.) access params.cfg
   ↓
[5] Training execution via Lightning
   ↓
[6] Models saved with params snapshot (wts.h5.zip)
   ↓
[7] On inference: params.cfg restored from params.dill
```

---

## 3. PyTorch-Specific Execution Parameters

### 3.1 Device Selection (train.py:493)

```python
n_devices=1 if args.device == 'cpu' else (
    torch.cuda.device_count() if torch.cuda.is_available() else 1
)
```

**Factory integration point:** Device selection logic should be extracted to reusable helper.

### 3.2 Lightning Trainer Configuration (components.py:565-574)

```python
trainer = L.Trainer(
    max_epochs=config.nepochs,
    accelerator='auto',
    devices=1,  # Single device for MVP; multi-GPU later
    log_every_n_steps=1,
    default_root_dir=str(output_dir),
    enable_progress_bar=debug_mode,  # Suppress progress bar unless debug
    deterministic=True,  # Enforce reproducibility
    logger=False,  # Disable default logger for now; MLflow added in B3
)
```

**Issues:**
- Trainer config params hardcoded (devices=1, accelerator='auto')
- No abstraction for different execution strategies

### 3.3 Dataloader Configuration (components.py:266-373)

```python
# Training loader
train_loader = DataLoader(
    train_dataset,
    batch_size=getattr(config, 'batch_size', 4),
    shuffle=shuffle,
    num_workers=0,
    pin_memory=False
)

# Validation loader
val_loader = DataLoader(
    test_dataset,
    batch_size=getattr(config, 'batch_size', 4),
    shuffle=False,  # Never shuffle validation
    num_workers=0,
    pin_memory=False
) if test_container is not None else None
```

**Inline parameters:**
- `num_workers=0` (hardcoded)
- `pin_memory=False` (hardcoded)
- Shuffle logic based on `config.sequential_sampling`

### 3.4 Probe Size Inference (train.py:96-140)

```python
def _infer_probe_size(npz_file):
    """Infer probe size (N) from NPZ metadata without loading full arrays."""
    import zipfile
    import numpy as np
    
    try:
        with zipfile.ZipFile(npz_file) as archive:
            for name in archive.namelist():
                if name.startswith('probeGuess') and name.endswith('.npy'):
                    npy = archive.open(name)
                    version = np.lib.format.read_magic(npy)
                    shape, _, _ = np.lib.format._read_array_header(npy, version)
                    return shape[0]
        return None
    except (zipfile.BadZipFile, FileNotFoundError, KeyError) as e:
        return None
```

**Status:** Utility function exists but inline in train.py — good candidate for factory extraction.

---

## 4. Integration Touchpoints for Factories

### 4.1 Primary Integration Points

| Touchpoint | Current Location | Pattern | Factory Candidate |
|------------|------------------|---------|------------------|
| **CLI Argument Parsing** | train.py:405-405 | argparse → direct instantiation | `PyTorchConfigFactory.from_cli_args()` |
| **PyTorch Config Creation** | train.py:464-501 | Direct DataConfig/ModelConfig instantiation | `PyTorchConfigFactory.from_cli_args()` or `PyTorchConfigFactory.from_datafile()` |
| **Probe Size Inference** | train.py:96-140 | Utility function + inline call (line 468) | `ProbeInferenceFactory.from_npz()` |
| **Config Bridge Translation** | train.py:517-532 | Manual to_model_config() + to_training_config() calls | `TensorFlowConfigFactory.from_pytorch()` |
| **update_legacy_dict Gate** | train.py:535 | Direct call | Wrapped in factory or bridge |
| **Lightning Trainer Setup** | components.py:565-574 | Direct L.Trainer instantiation | `LightningTrainerFactory.create()` |
| **DataLoader Creation** | components.py:343-373 | Direct DataLoader instantiation | `DataLoaderFactory.create_train()` + `DataLoaderFactory.create_val()` |
| **Model Instantiation** | components.py:546-551 | Direct PtychoPINN_Lightning instantiation | `LightningModelFactory.create()` |
| **Bundle Persistence** | model_manager.py:60-185 | Manual save_torch_bundle call | Already has wrapper; enhance with factory |

### 4.2 Factory Integration Flow (Proposed)

```
┌─────────────────────────────────────────────────────────────┐
│ CLI Entry Point (cli_main)                                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ PyTorchConfigFactory.from_cli_args(args)             │   │
│  │  - Infer probe size via ProbeInferenceFactory        │   │
│  │  - Create DataConfig/ModelConfig/TrainingConfig      │   │
│  │  - Return 5-tuple (data, model, train, infer, dgen)  │   │
│  └──────────────────────────────────────────────────────┘   │
│                            ↓                                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ TensorFlowConfigFactory.from_pytorch(...)            │   │
│  │  - Call to_model_config(data, model)                 │   │
│  │  - Call to_training_config(..., overrides)           │   │
│  │  - Return TF dataclass configs                       │   │
│  └──────────────────────────────────────────────────────┘   │
│                            ↓                                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ ConfigBridge.populate_legacy_dict(params.cfg, config)│   │
│  │  - Call update_legacy_dict(CONFIG-001 gate)          │   │
│  │  - Validate params.cfg populated                     │   │
│  └──────────────────────────────────────────────────────┘   │
│                            ↓                                 │
│  Call main(ptycho_dir, existing_config, ...)               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 Nested Factory Integration (Workflow Components)

```
┌────────────────────────────────────────────────────────┐
│ train_cdi_model_torch(train_data, test_data, config)  │
├────────────────────────────────────────────────────────┤
│                                                        │
│  _ensure_container(train_data, config)                │
│   - Already handles RawData → RawDataTorch conversion  │
│   - Could delegate to DataContainerFactory            │
│                            ↓                          │
│  _build_lightning_dataloaders(...)                    │
│   - Delegate to DataLoaderFactory.create_train()      │
│   - Delegate to DataLoaderFactory.create_val()        │
│                            ↓                          │
│  _train_with_lightning(...)                           │
│   - Delegate to LightningModelFactory.create(...)     │
│   - Delegate to LightningTrainerFactory.create(...)   │
│   - Execute trainer.fit()                             │
│                                                        │
└────────────────────────────────────────────────────────┘
```

---

## 5. Key Design Patterns Identified

### 5.1 Mode Enum Mapping (Bidirectional)

**Forward mapping (PyTorch → TensorFlow):**
- Line config_bridge.py:116-124
- `'Unsupervised'` → `'pinn'`
- `'Supervised'` → `'supervised'`

**Reverse mapping (TensorFlow → PyTorch):**
- Line components.py:521
- `'pinn'` → `'Unsupervised'`
- `'supervised'` → `'Supervised'`

**Factory opportunity:** Extract to `ModeEnumFactory.pytorch_to_tensorflow()` and `ModeEnumFactory.tensorflow_to_pytorch()`

### 5.2 Activation Normalization

**Pattern (config_bridge.py:126-140):**
```python
activation_mapping = {
    'silu': 'swish',
    'SiLU': 'swish',
    'sigmoid': 'sigmoid',
    'swish': 'swish',
    'softplus': 'softplus',
    'relu': 'relu'
}
```

**Factory opportunity:** Extract to `ActivationFactory.normalize()`

### 5.3 Validation Patterns

Three levels of validation:

1. **Type validation:** config_bridge.py:108-123 (non-square grid check)
2. **Default divergence check:** config_bridge.py:259-269 (nphotons divergence)
3. **Required field validation:** config_bridge.py:271-304 (n_groups, train_data_file)

**Factory opportunity:** Extract to `ConfigValidator` class with pluggable validators

---

## 6. File-by-File Integration Summary

### ptycho_torch/train.py
- **Lines 96-140:** `_infer_probe_size()` utility — extract to factory
- **Lines 464-504:** Inline PyTorch config creation — factory candidate
- **Lines 511-532:** Config bridge translation + overrides — wrap in factory
- **Lines 535:** `update_legacy_dict()` call — already CONFIG-001 compliant

### ptycho_torch/config_bridge.py
- **Lines 79-182:** `to_model_config()` — validated transformation
- **Lines 185-306:** `to_training_config()` — validated transformation with detailed checks
- **Lines 309-380:** `to_inference_config()` — inference path (less used)

### ptycho_torch/workflows/components.py
- **Lines 150:** `update_legacy_dict()` call — CONFIG-001 compliant
- **Lines 266-373:** `_build_lightning_dataloaders()` — factory candidate (DataLoaderFactory)
- **Lines 459-614:** `_train_with_lightning()` — contains secondary config construction
  - Lines 516-542: Inline PyTorch config reconstruction (redundant conversion)
  - Lines 546-551: Model instantiation (factory candidate)
  - Lines 565-574: Trainer instantiation (factory candidate)

### ptycho_torch/model_manager.py
- **Lines 104-122:** Config snapshot capture via `dataclass_to_legacy_dict()` — already wrapped
- **Lines 259-262:** params.cfg restoration (partial CONFIG-001 compliance)

---

## 7. Implicit Dependencies and Assumptions

### 7.1 Grid Square Assumption
- config_bridge.py:106-113: Assumes `grid_size` is always square
- Error message clear but fails at runtime

**Factory improvement:** Validate at config construction time

### 7.2 Probe Size Inference Fallback
- train.py:468-471: Defaults to N=64 if inference fails
- No logging of why inference failed

**Factory improvement:** Add debug logging and retry logic

### 7.3 Device Selection Logic
- train.py:493: Complex ternary for device count
- Assumes CUDA availability check

**Factory improvement:** Extract to `DeviceFactory.select()`

### 7.4 Learning Rate Default
- components.py:538: Hardcoded `learning_rate=1e-4`
- Comment says "can expose via config later" (line 538)

**Factory improvement:** Make configurable through factory parameters

---

## Summary of Factory Integration Opportunities

| Factory | Location | Benefit | Priority |
|---------|----------|---------|----------|
| **PyTorchConfigFactory** | train.py:464-504 | Centralize probe inference + config creation | HIGH |
| **ProbeInferenceFactory** | train.py:96-140 | Reusable probe size extraction | HIGH |
| **TensorFlowConfigFactory** | train.py:511-532 | Wrap config bridge translation with validation | MEDIUM |
| **LightningModelFactory** | components.py:546-551 | Decouple model instantiation from orchestration | MEDIUM |
| **LightningTrainerFactory** | components.py:565-574 | Centralize trainer configuration strategy | MEDIUM |
| **DataLoaderFactory** | components.py:343-373 | Consistent DataLoader creation patterns | MEDIUM |
| **ModeEnumFactory** | config_bridge.py + components.py | Eliminate bidirectional mapping duplication | LOW |
| **ActivationFactory** | config_bridge.py:126-140 | Reusable activation normalization | LOW |
| **ConfigValidator** | config_bridge.py:259-304 | Pluggable validation rules | LOW |
| **DeviceFactory** | train.py:493 | Centralized device selection logic | LOW |

