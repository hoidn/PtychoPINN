# Phase D1.B: PyTorch Workflow Asset Inventory

**Goal**: Assess `ptycho_torch/` module readiness for Ptychodus reconstructor integration.

**Cross-reference**: TensorFlow callchain (`phase_d_callchain.md`), Phase B config bridge, Phase C data adapters.

---

## 1. Executive Summary

**Overall Assessment**: PyTorch training/inference infrastructure exists but is **MLflow-coupled** and **lacks orchestration wrapper** for Ptychodus reconstructor contract.

**Reusability Score**: 65/100
- Training loop (Lightning): 95% ready ✅
- Config translation: 100% ready but NEVER CALLED ⚠️
- Data loading: 85% ready ⚠️
- Model persistence: 20% ready (MLflow-only) ❌
- Inference: 60% ready (tight coupling) ⚠️

**Critical Blockers**:
1. No `run_cdi_example_torch` orchestrator (reconstruct contract §4.2-4.5 unmet)
2. `config_bridge` never invoked → `params.cfg` unpopulated → CONFIG-001 violation
3. Persistence format incompatible with TensorFlow `wts.h5.zip` archives

---

## 2. Module-by-Module Inventory

### 2.1. `ptycho_torch/train.py` (Training Entry Point)

**Purpose**: Main training script using PyTorch Lightning.

**Entry Point**:
```python
def main(ptycho_dir, config_path=None, existing_config=None):
    # Signature: directory of NPZ files → MLflow run_id
```

**Lines**: 255 (train loop ~170 lines)

**Configuration Handling**:
- Loads JSON config → singleton `DataConfig`, `ModelConfig`, `TrainingConfig`, etc. (lines 66-92)
- **CRITICAL GAP**: Never calls `config_bridge` or `update_legacy_dict` → `params.cfg` remains empty
- Direct instantiation of PyTorch singletons instead of TensorFlow dataclasses

**Data Pipeline**:
- Uses `PtychoDataModule` (Lightning) wrapping `PtychoDataset` (memory-mapped) (lines 100-108)
- **GAP**: Expects memory-mapped NPZ structure, NOT grouped-data dict from Phase C adapters
- `remake_map=True` flag forces memmap rebuild (line 105)

**Training Loop**:
- Lightning `Trainer.fit(model, datamodule)` (line 190)
- Callbacks: `ModelCheckpoint`, `EarlyStopping` (lines 123-141)
- MLflow autologging enabled (line 169)
- Multi-stage training support (lines 144-148)

**Outputs**:
- MLflow run_id (line 194) - used for `inference.py` loading
- Lightning checkpoint (via `ModelCheckpoint` callback)
- **NO** `wts.h5.zip` equivalent

**Reusability for Ptychodus**:
- ✅ Lightning training loop works
- ⚠️ Needs config_bridge call at entry (trivial fix)
- ⚠️ Data module expects different format than Phase C adapters
- ❌ No NPZ export path for Ptychodus `export_training_data()` contract

**Effort to Adapt**: 1-2 days (add config bridge, NPZ export shim, optional MLflow flag)

---

### 2.2. `ptycho_torch/inference.py` (Inference Entry Point)

**Purpose**: Load trained model from MLflow and run inference.

**Entry Point**:
```python
def load_and_predict(run_id, ptycho_files_dir, relative_mlflow_path='mlruns', ...):
    # Signature: MLflow run_id + data_dir → reconstructed image
```

**Lines**: 211 (inference loop ~100 lines)

**Configuration Handling**:
- Loads configs from MLflow run (line 82-85) OR JSON override (line 85)
- **SAME GAP**: No `config_bridge` call → incompatible with TensorFlow config system
- PyTorch singleton configs only

**Model Loading**:
- `mlflow.pytorch.load_model(model_uri)` (line 94) - requires MLflow server
- **CRITICAL**: Hardcoded MLflow dependency, no fallback to checkpoint files
- No compatibility with TensorFlow `wts.h5.zip` archives

**Data Pipeline**:
- `PtychoDataset` for memory-mapped loading (lines 101-102)
- **SAME GAP**: Incompatible with Phase C `PtychoDataContainerTorch` output

**Inference**:
- `reconstruct_image_barycentric(model, dataset, ...)` (lines 111-113)
- Saves amplitude/phase plots (lines 127-129)
- **Good**: Reuses PyTorch reassembly (barycentric interpolation)

**Reusability for Ptychodus**:
- ⚠️ Tight MLflow coupling blocks standalone usage
- ❌ Cannot load TensorFlow archives
- ✅ Reconstruction logic reusable

**Effort to Adapt**: 1-2 days (decouple MLflow, add checkpoint loader, config bridge)

---

### 2.3. `ptycho_torch/config_bridge.py` (Phase B Deliverable)

**Purpose**: Translate PyTorch singleton configs → TensorFlow dataclasses.

**Entry Points**:
```python
def to_model_config(pt_model: ModelConfig, overrides: dict) -> ptycho.config.ModelConfig
def to_training_config(pt_model: ModelConfig, pt_training: TrainingConfig, overrides: dict) -> ptycho.config.TrainingConfig
def to_inference_config(...) -> ptycho.config.InferenceConfig
```

**Lines**: 377 (complete implementation)

**Status**: ✅ **COMPLETE** (Phase B.B5) but **NEVER INVOKED**

**Integration State**:
- All 38 spec-required fields mapped
- Activation translation (`silu` → `swish`)
- Probe_mask tensor→bool conversion
- nphotons divergence validation
- **PROBLEM**: No caller in `train.py` or `inference.py` - bridge is orphaned

**Reusability**: 100% - just needs to be wired into orchestration layer

**Effort to Integrate**: < 1 day (add calls to train/inference entry points)

---

### 2.4. `ptycho_torch/raw_data_bridge.py` + `data_container_bridge.py` (Phase C Deliverables)

**Purpose**: Adapt PyTorch data loading to TensorFlow `RawData` → `PtychoDataContainer` pipeline.

**Entry Points**:
```python
class RawDataTorch:
    def __init__(self, config: TrainingConfig, npz_path: Path)
    def generate_grouped_data(self, ...) -> dict  # Delegates to TensorFlow RawData

class PtychoDataContainerTorch:
    def __init__(self, grouped_data: dict, config: TrainingConfig)
    # Exposes: X, Y, Y_I, Y_phi, coords_nominal, probe, nn_indices, ...
```

**Lines**: 324 (raw_data_bridge) + 280 (data_container_bridge) = 604

**Status**: ✅ **COMPLETE** (Phase C.C3) with torch-optional tests

**Integration State**:
- Delegation to TensorFlow `RawData.generate_grouped_data` (zero reimplementation)
- `update_legacy_dict` called automatically in constructor (CONFIG-001 compliant)
- Torch-optional: returns NumPy arrays if torch unavailable
- **PROBLEM**: Not used by `train.py` or `inference.py` - they use `PtychoDataset` (memory-mapped) instead

**Reusability**: 100% - but requires orchestration refactor to use adapters

**Effort to Integrate**: 1 day (replace `PtychoDataModule` with adapter-based loading)

---

### 2.5. `ptycho_torch/memmap_bridge.py` (Phase C.C3 Deliverable)

**Purpose**: Load memory-mapped NPZ files → grouped data via RawDataTorch delegation.

**Entry Point**:
```python
class MemmapDatasetBridge:
    def __init__(self, npz_path: Path, config: TrainingConfig)
    def generate_grouped_data(self, ...) -> dict
```

**Lines**: 213

**Status**: ✅ **COMPLETE** (Phase C.C3) with deterministic generation tests

**Integration State**:
- Delegates to `RawDataTorch.generate_grouped_data` (CONFIG-001 compliant)
- Cache-free (inherits TensorFlow sample-then-group)
- **PROBLEM**: Also not used by existing `train.py` / `inference.py`

**Reusability**: 100% - ready for integration

**Effort to Integrate**: < 1 day (add to data loading path)

---

### 2.6. `ptycho_torch/dataloader.py` (Existing PyTorch Data Loading)

**Purpose**: Memory-mapped dataset + TensorDict dataloader for Lightning.

**Key Classes**:
```python
class PtychoDataset(torch.utils.data.Dataset):
    # Memory-maps diffraction NPZ, loads probe separately
    # Returns: TensorDict with (images, coords, probe, scale)
```

**Lines**: 783 (large module)

**Integration Issues**:
- **Different format**: Emits `TensorDict` tuples, NOT grouped-data dict
- **No Y patches**: Missing ground truth loading (supervised training broken)
- **No RawDataTorch delegation**: Reimplements grouping logic (duplication risk)
- **Config bypass**: Uses singleton `DataConfig`, never calls `update_legacy_dict`

**Reusability**: 40% - works for unsupervised training but incompatible with adapter stack

**Effort to Replace**: 2 days (refactor to use Phase C adapters)

---

### 2.7. `ptycho_torch/model.py` (PyTorch Model Architecture)

**Purpose**: U-Net + physics layers implementation in PyTorch.

**Key Class**:
```python
class PtychoPINN_Lightning(L.LightningModule):
    # Architecture mirrors TensorFlow model.py
    # Losses: Poisson NLL, MAE, realspace consistency
```

**Lines**: 1268 (large, complex)

**Status**: ✅ **MATURE** - physics parity with TensorFlow achieved

**Integration**: No changes needed - model is decoupled from data/config issues

**Reusability**: 95%

---

### 2.8. `ptycho_torch/api/` (High-Level API Layer)

**Files**:
- `api/base_api.py` (995 lines) - abstract base classes
- `api/trainer_api.py` (50 lines) - training wrappers
- `api/example_use.py`, `api/example_train.py` - usage examples

**Purpose**: Attempt at high-level API abstraction (experimental).

**Status**: 🚧 **INCOMPLETE** / **UNUSED**

**Issues**:
- `base_api.py` defines abstract interfaces but no concrete implementations
- `trainer_api.py` provides thin wrappers but not integrated with `train.py`
- Example scripts use `train.main()` directly, bypassing API layer

**Reusability**: 20% - architectural blueprint but no production code

**Decision Impact**: If we choose **Option A** (API wrapper), must complete this layer.

---

### 2.9. `ptycho_torch/train_utils.py` (Training Utilities)

**Purpose**: Helper functions for Lightning training (LR scheduling, device management, MLflow logging).

**Key Components**:
- `PtychoDataModule(L.LightningDataModule)` - wraps `PtychoDataset`
- `set_seed()` - deterministic training
- `find_learning_rate()` - LR scaling for distributed training
- `log_parameters_mlflow()` - MLflow parameter logging

**Lines**: 441

**Status**: ✅ **MATURE** - well-tested utility functions

**Reusability**: 90% - `PtychoDataModule` needs adapter refactor, rest is reusable

---

## 3. Gap Analysis: PyTorch vs TensorFlow Workflow

| Workflow Stage | TensorFlow | PyTorch | Gap Severity | Fix Effort |
|----------------|------------|---------|--------------|------------|
| **Entry Orchestration** | `run_cdi_example` | ❌ Missing | CRITICAL | 1-2 days |
| **Config Bridge Call** | `update_legacy_dict` at entry | ❌ Never called | CRITICAL | < 1 day |
| **Data Format** | `RawData` → grouped dict → `PtychoDataContainer` | `TensorDict` tuples (incompatible) | HIGH | 1 day |
| **Ground Truth Loading** | `Y` patches in NPZ | ❌ Missing from `PtychoDataset` | MEDIUM | 1 day |
| **Probe Init** | `probe.set_probe_guess()` (global) | ❌ No equivalent | MEDIUM | 1 day |
| **Model Persistence** | `wts.h5.zip` (TensorFlow + params.cfg) | MLflow-only | HIGH | 1-2 days |
| **Inference Loading** | `ModelManager.load_multiple_models` | MLflow run_id only | HIGH | 1 day |
| **NPZ Export** | `export_training_data()` (reconstructor §4.5) | ❌ Missing | MEDIUM | 1 day |
| **Stitching** | `reassemble_position` (TF tensors) | `reconstruct_image_barycentric` (torch) | LOW | Done ✅ |

**Total Estimated Effort**: 4-5 days for critical path (orchestrator + persistence + config bridge integration)

---

## 4. Reusability Matrix

### 4.1. Modules Ready to Reuse (Minimal/No Changes)

| Module | Readiness | Integration Effort |
|--------|-----------|-------------------|
| `ptycho_torch/model.py` | 95% | None (architecture complete) |
| `ptycho_torch/config_bridge.py` | 100% | Wire into train/inference entry (< 1 day) |
| `ptycho_torch/raw_data_bridge.py` | 100% | Replace `PtychoDataset` usage (1 day) |
| `ptycho_torch/data_container_bridge.py` | 100% | Replace `PtychoDataset` usage (1 day) |
| `ptycho_torch/memmap_bridge.py` | 100% | Add to data loading path (< 1 day) |
| `ptycho_torch/reassembly.py` | 95% | None (stitching logic complete) |

### 4.2. Modules Needing Refactor

| Module | Issue | Refactor Scope |
|--------|-------|---------------|
| `ptycho_torch/train.py` | No config bridge call, MLflow-coupled | Add config bridge, optional MLflow flag (1 day) |
| `ptycho_torch/inference.py` | MLflow-only loading, no TF archive support | Add checkpoint loader, config bridge (1-2 days) |
| `ptycho_torch/dataloader.py` | TensorDict format, no Y patches, no RawDataTorch delegation | Replace with adapter-based loading (2 days) |
| `ptycho_torch/train_utils.py` | `PtychoDataModule` uses old dataloader | Refactor to use Phase C adapters (1 day) |

### 4.3. Modules Missing Entirely

| Missing Component | Purpose | Creation Effort |
|-------------------|---------|----------------|
| `ptycho_torch/workflows/components.py` | Orchestrator matching TensorFlow `run_cdi_example` | 1-2 days (D2) |
| `ptycho_torch/model_manager_torch.py` | Persistence matching TensorFlow archive format | 1-2 days (D3) |
| Export training data shim | `export_training_data()` for NPZ archives | 1 day |

---

## 5. Integration Recommendations

### 5.1. Immediate Actions (< 1 Day Each)

1. **Wire config_bridge into train.py**:
   ```python
   # ptycho_torch/train.py:45 (after loading configs)
   from ptycho_torch.config_bridge import to_training_config, to_model_config
   from ptycho.config.config import update_legacy_dict
   from ptycho import params

   tf_config = to_training_config(pt_model, pt_training, overrides={...})
   update_legacy_dict(params.cfg, tf_config)  # Populate params.cfg
   ```

2. **Add MLflow optionality flag**:
   ```python
   # ptycho_torch/train.py
   parser.add_argument('--disable_mlflow', action='store_true')
   if not args.disable_mlflow:
       mlflow.pytorch.autolog(...)
   ```

3. **Export `ptycho_torch.workflows` namespace**:
   ```python
   # ptycho_torch/__init__.py
   from .workflows import run_cdi_example_torch, train_cdi_model_torch
   ```

### 5.2. Short-Term Refactor (1-2 Days Each)

1. **Replace `PtychoDataset` with adapter stack**:
   - Refactor `PtychoDataModule` to use `RawDataTorch` → `PtychoDataContainerTorch`
   - Add Y patch loading support
   - Maintain Lightning DataModule interface

2. **Build `run_cdi_example_torch` orchestrator**:
   - Mirror TensorFlow signature from `phase_d_callchain.md` §7
   - Call config_bridge at entry
   - Wrap Lightning training
   - Optional stitching via `reconstruct_image_barycentric`

3. **Implement cross-system persistence**:
   - `TorchModelManager.save()` → produce `wts.h5.zip`-compatible archives
   - `TorchModelManager.load()` → read Lightning checkpoints OR TensorFlow archives
   - Bundle `params.cfg` snapshot (CONFIG-001 compliance)

### 5.3. Medium-Term Harmonization (2-3 Days)

1. **Unified config system**: Decide between (A) PyTorch extends TensorFlow dataclasses, OR (B) config_bridge remains translation layer (see Phase B Open Question Q1)
2. **Test coverage**: Extend `tests/torch/test_integration_workflow.py` to match TensorFlow parity (Phase D4)
3. **Documentation**: Update `docs/workflows/pytorch.md` with new orchestration patterns

---

## 6. Cross-Reference to Integration Plan

**From `plans/ptychodus_pytorch_integration_plan.md` Delta-2 (Integration Strategy)**:

| Delta-2 Item | PyTorch Asset Status | Action Required |
|--------------|---------------------|-----------------|
| Config schema harmonization | ✅ `config_bridge.py` complete | Wire into train/inference |
| API layer decision | ⚠️ `api/` incomplete | D1.C decision needed |
| Datagen package | ✅ Exists in `ptycho_torch/datagen/` | Not blocking (separate concern) |
| Barycentric reassembly | ✅ `reassembly.py` complete | Already integrated |
| Lightning/MLflow persistence | ⚠️ MLflow-only, no TF compat | D3 priority |

---

## 7. Summary: Ready vs Missing

### Ready to Use (Phase B/C Complete)
- ✅ Config translation (`config_bridge.py`)
- ✅ Data adapters (`raw_data_bridge.py`, `data_container_bridge.py`, `memmap_bridge.py`)
- ✅ Model architecture (`model.py`)
- ✅ Reassembly/stitching (`reassembly.py`)

### Needs Integration (Exists but Not Wired)
- ⚠️ Training loop (`train.py`) - add config_bridge call
- ⚠️ Inference (`inference.py`) - decouple MLflow, add checkpoint loader

### Missing Critical Components
- ❌ `run_cdi_example_torch` orchestrator
- ❌ TensorFlow-compatible persistence (`TorchModelManager`)
- ❌ NPZ export for `export_training_data()` contract
- ❌ Probe initialization shim

**Next Steps**: D1.C decision on orchestration approach (API wrapper vs low-level shims) to guide D2 implementation.
