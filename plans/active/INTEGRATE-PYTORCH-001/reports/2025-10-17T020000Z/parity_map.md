# TensorFlow ↔ PyTorch Parity Map

**Initiative:** INTEGRATE-PYTORCH-001
**Phase:** A — Refresh Parity Baseline
**Date:** 2025-10-17
**Purpose:** Complete inventory of TensorFlow touchpoints used by Ptychodus and corresponding PyTorch implementation status

---

## 1. Executive Summary

This document catalogs every integration point between Ptychodus and PtychoPINN's TensorFlow backend, then maps each to the current PyTorch implementation status. It identifies **7 critical gaps** that must be addressed before PyTorch can serve as a drop-in backend replacement.

### Gap Categories

| Category | Count | Criticality |
|:---------|:------|:------------|
| Configuration Bridge | 1 | **CRITICAL** — Blocks all workflows |
| Data Pipeline | 2 | **CRITICAL** — Required for training/inference |
| Workflow Orchestration | 2 | **HIGH** — Required for Ptychodus integration |
| Persistence Layer | 2 | **HIGH** — Required for model save/load |

---

## 2. Configuration & Legacy Bridge

### 2.1 TensorFlow Implementation

**Entry Point:** `ptychodus/src/ptychodus/model/ptychopinn/reconstructor.py:87-122`

```python
# Ptychodus creates dataclass instances
model_config = ModelConfig(N=64, gridsize=2, ...)
inference_config = InferenceConfig(model=model_config, ...)

# CRITICAL: Bridges to legacy params.cfg
from ptycho.config.config import update_legacy_dict
import ptycho.params
update_legacy_dict(ptycho.params.cfg, inference_config)
```

**Purpose:** Synchronize modern typed dataclasses → legacy global dictionary
**Consumers:** `ptycho/raw_data.py:365`, `ptycho/loader.py:178-181`, `ptycho/model.py:280`
**Contract:** `specs/ptychodus_api_spec.md:59-79` (update_legacy_dict specification)

**Key Fields Propagated:**
- `N` (diffraction pattern size)
- `gridsize` (group cardinality)
- `n_groups`, `n_subsample`, `neighbor_count` (sampling controls)
- `probe.mask`, `probe.scale`, `probe.trainable` (probe configuration)
- `intensity_scale_trainable` (physics scaling)
- `output_dir` → `output_prefix` (path mapping)

### 2.2 PyTorch Implementation

**Current State:** `ptycho_torch/config_params.py:40-112`

```python
# Singleton pattern WITHOUT dataclass ingestion
ModelConfig().set_settings(model_config_default)
ModelConfig().add("loss_function", "Poisson")
```

**Status:** ⚠️ **GAP — No dataclass bridge exists**

**Missing Components:**
1. **Dataclass ingestion:** No code path to consume `ModelConfig`/`TrainingConfig`/`InferenceConfig` dataclasses
2. **Legacy bridge:** No equivalent `update_legacy_dict` call to populate `ptycho.params.cfg`
3. **KEY_MAPPINGS:** No translation layer for dotted legacy keys (`object.big`, `probe.trainable`)
4. **Field validation:** Singleton accepts arbitrary keys without schema enforcement

**Impact:**
- Ptychodus cannot configure PyTorch backend using existing API
- Legacy modules (e.g., `ptycho/raw_data.py`) will fail when called by PyTorch workflows
- Configuration drift between TensorFlow and PyTorch backends

**Required Work (Phase B):**
- Design adapter: `ptycho_torch/config_bridge.py` to translate dataclasses → singleton settings
- Ensure `update_legacy_dict(ptycho.params.cfg, config)` is called by PyTorch orchestration
- Map all fields from `specs/ptychodus_api_spec.md §5` (tables 5.1-5.3)

---

## 3. Data Pipeline & Tensor Packaging

### 3.1 TensorFlow: RawData → PtychoDataContainer

**Flow:**
NPZ file → `RawData.from_coords_without_pc()` → `generate_grouped_data()` → `loader.load()` → `PtychoDataContainer`

**Key Touchpoints:**

#### A. Raw Data Grouping
**Source:** `ptycho/raw_data.py:365-438`

```python
def generate_grouped_data(N, K, gridsize, nsamples, ...):
    # 1. Discover all valid K-choose-C groups (cached)
    # 2. Randomly sample nsamples groups
    # 3. Return dict with keys: 'diffraction', 'coords_offsets', 'coords_relative', 'Y', 'nn_indices'
```

**Purpose:** Group-then-sample strategy with neighbor-aware caching
**Output Contract:** `specs/ptychodus_api_spec.md §4.3` (data ingestion)
**Cache Format:** `.g2k4.groups_cache.npz` (gridsize=2, K=4 example)

#### B. Tensor Conversion
**Source:** `ptycho/loader.py:93-341`

```python
class PtychoDataContainer:
    # TensorFlow tensors (converted from NumPy)
    X: tf.float32  # Diffraction patterns (n_images, N, N, n_channels)
    Y: tf.complex64  # Ground truth patches
    coords_nominal: tf.float32  # Scan coordinates
    local_offsets: tf.float32  # Patch offsets
    probe: tf.complex64  # Probe function
```

**Purpose:** NumPy→TensorFlow dtype conversion + multi-channel reshaping
**Consumers:** `ptycho/model.py`, `ptycho/workflows/components.py:543-571`

### 3.2 PyTorch Implementation

**Current State:** `ptycho_torch/dset_loader_pt_mmap.py:46-429`

```python
class PtychoDataset(Dataset):
    def memory_map_data(self, image_paths):
        # Creates MemoryMappedTensor with TensorDict
        # Returns: mmap_ptycho['images'], mmap_ptycho['coords_relative'], ...
```

**Status:** ⚠️ **PARTIAL — Data loading exists but lacks RawData compatibility**

**What Exists:**
- Memory-mapped tensor storage (`tensordict.MemoryMappedTensor`)
- Coordinate grouping via `patch_generator.group_coords()`
- Probe loading and scaling

**Missing Components:**
1. **RawData compatibility:** No equivalent to `RawData.from_coords_without_pc()` or `RawData.generate_grouped_data()`
2. **Caching parity:** PyTorch uses in-memory maps; TensorFlow uses `.groups_cache.npz` files
3. **PtychoDataContainer equivalent:** No named container matching TensorFlow's tensor layout
4. **NPZ round-trip:** Cannot consume NPZ files generated by TensorFlow workflows or Ptychodus exports

**Impact:**
- Ptychodus cannot call `create_raw_data()` → `RawDataTorch.generate_grouped_data()` workflow
- Training data prepared for TensorFlow backend is incompatible
- No shared fixture strategy for dual-backend tests

**Required Work (Phase C):**
- Implement `RawDataTorch` wrapper delegating to existing `ptycho/raw_data.py`
- Create `PtychoDataContainerTorch` exposing same tensor attributes as TensorFlow version
- Ensure `.groups_cache.npz` files are reused across backends

---

## 4. Workflow Orchestration

### 4.1 TensorFlow: run_cdi_example

**Entry Point:** `ptycho/workflows/components.py:676-723`

```python
def run_cdi_example(train_data, test_data, config, ...):
    # 1. Update params.cfg from config
    update_legacy_dict(params.cfg, config)

    # 2. Create containers
    train_container = create_ptycho_data_container(train_data, config)

    # 3. Train model
    results = train_cdi_model(train_container, test_container, config)

    # 4. Optional stitching
    if do_stitching:
        recon_amp, recon_phase, reassemble_results = reassemble_cdi_image(...)

    return recon_amp, recon_phase, results
```

**Purpose:** End-to-end training → reconstruction → visualization pipeline
**Called By:** `ptychodus/src/ptychodus/model/ptychopinn/reconstructor.py:229-269` (training workflow)
**Contract:** `specs/ptychodus_api_spec.md §4.5` (training workflow)

### 4.2 PyTorch Implementation

**Current State:** `ptycho_torch/train.py:193-273`

```python
def main(ptycho_dir, probe_dir):
    # 1. Seed singletons (NOT dataclasses)
    ModelConfig().set_settings(model_config_default)

    # 2. Create dataset
    ptycho_dataset = PtychoDataset(ptycho_dir, probe_dir)

    # 3. Train with Lightning
    trainer = L.Trainer(...)
    with mlflow.start_run():
        trainer.fit(model, train_loader)
```

**Status:** ⚠️ **GAP — No Ptychodus-compatible orchestration**

**What Exists:**
- Lightning training loop with MLflow autologging
- Dataset → Dataloader → Trainer pipeline

**Missing Components:**
1. **Unified entry point:** No `run_cdi_example_torch()` accepting `RawData`/`PtychoDataContainer` inputs
2. **Config-driven execution:** Current implementation hardcodes settings; needs dataclass-driven configuration
3. **Return value contract:** Lightning trainer returns different objects than TensorFlow `train_cdi_model()`
4. **Image stitching:** No equivalent to `reassemble_cdi_image()` for reconstruction visualization

**Impact:**
- Ptychodus `train()` method cannot call PyTorch backend
- No programmatic API for notebook/script integration
- Missing reconstruction outputs expected by Ptychodus UI

**Required Work (Phase D):**
- Design `ptycho_torch/workflows.py` with `run_cdi_example_torch()` signature matching TensorFlow
- Adapt Lightning return values to match `results` dictionary contract
- Implement or delegate to TensorFlow's `reassemble_cdi_image()` for stitching

---

## 5. Model Persistence

### 5.1 TensorFlow: ModelManager

**Entry Point:** `ptycho/model_manager.py:47-467`

```python
# Save
models_dict = {'autoencoder': model1, 'diffraction_to_obj': model2}
ModelManager.save_multiple_models(models_dict, 'output/wts.h5', custom_objects, scale)

# Load
from ptycho.workflows.components import load_inference_bundle
model, config = load_inference_bundle(Path("output_dir"))
```

**File Format:** `.h5.zip` archive containing:
- `manifest.dill` (model inventory)
- `{model_name}/model.keras` or `{model_name}/saved_model.pb` (TensorFlow formats)
- `{model_name}/params.dill` (serialized `ptycho.params.cfg` state)
- `{model_name}/custom_objects.dill` (Lambda layers, custom classes)

**Purpose:** Persist model + configuration + custom layers for cross-process reuse
**Contract:** `specs/ptychodus_api_spec.md §4.6` (model persistence contract)

### 5.2 PyTorch Implementation

**Current State:** `ptycho_torch/train.py:238-240`

```python
# Lightning autologging creates checkpoints
mlflow.pytorch.autolog(checkpoint_monitor="mae_train_loss")
with mlflow.start_run():
    trainer.fit(model, train_loader)  # Creates .ckpt files
```

**Status:** ⚠️ **GAP — No Ptychodus-compatible persistence**

**What Exists:**
- Lightning checkpoint saving (`.ckpt` format)
- MLflow artifact logging

**Missing Components:**
1. **Archive format:** Lightning `.ckpt` ≠ TensorFlow `.h5.zip`
2. **Config bundling:** No equivalent to serializing `ptycho.params.cfg` alongside weights
3. **Custom object registry:** No mechanism to save/load PyTorch equivalents of TensorFlow Lambda layers
4. **ModelManager API:** No `save_multiple_models_torch()` or `load_multiple_models_torch()`
5. **Backward compatibility:** Ptychodus expects `wts.h5.zip`; cannot load Lightning checkpoints

**Impact:**
- Ptychodus `open_model()` cannot load PyTorch-trained models
- No cross-session model reuse in PyTorch workflows
- Missing `MODEL_FILE_NAME = 'wts.h5.zip'` contract satisfaction

**Required Work (Phase D):**
- Design persistence shim: wrap Lightning checkpoints in `.h5.zip`-compatible archives
- Bundle `ptycho.params.cfg` snapshot alongside checkpoint
- Implement `load_pytorch_bundle()` reading Lightning checkpoints + config
- Update Ptychodus `MODEL_FILE_NAME` to support dual formats or unified shim

---

## 6. Inference & Reconstruction

### 6.1 TensorFlow: Model Predict

**Entry Point:** `ptychodus/src/ptychodus/model/ptychopinn/reconstructor.py:113-158`

```python
# Load model
model, config = load_inference_bundle(model_dir)

# Prepare data
raw_data = create_raw_data(...)
container = create_ptycho_data_container(raw_data, config)

# Inference
predictions = model.predict([container.X, container.local_offsets])

# Stitch
from ptycho.tf_helper import reassemble_position
reconstructed = reassemble_position(predictions, container.global_offsets)
```

**Purpose:** Load trained model → run inference → stitch patches → return reconstruction
**Contract:** `specs/ptychodus_api_spec.md §4.4` (TensorFlow inference behaviour)

### 6.2 PyTorch Implementation

**Current State:** `ptycho_torch/train.py:116-122`

```python
def forward_predict(self, x, positions, probe, scale_factor):
    x_amp, x_phase = self.autoencoder(x)
    x_combined = self.combine_complex(x_amp, x_phase)
    return x_combined  # No physics forward pass
```

**Status:** ⚠️ **PARTIAL — Inference exists but lacks Ptychodus integration**

**What Exists:**
- `LightningModule.predict_step()` for batch inference
- Complex reconstruction (`forward_predict`)

**Missing Components:**
1. **Standalone inference script:** No `ptycho_torch/inference.py` equivalent to TensorFlow's `scripts/inference/inference.py`
2. **Model loading workflow:** No programmatic API to load checkpoint + config without Lightning `Trainer`
3. **Patch stitching:** No PyTorch equivalent to `reassemble_position()` or delegation to TensorFlow helper
4. **Ptychodus reconstruct() integration:** No code path for Ptychodus to call PyTorch inference

**Impact:**
- Ptychodus `reconstruct()` cannot use PyTorch backend
- No way to run inference outside training context
- Missing visualization outputs for GUI

**Required Work (Phase D/E):**
- Create `ptycho_torch/inference.py` with Lightning-free loading
- Implement or delegate stitching to `ptycho.tf_helper.reassemble_position`
- Wire into Ptychodus `reconstruct()` method with backend selection

---

## 7. Ptychodus Integration Points

### 7.1 TensorFlow Reconstructor API

**Source:** `ptychodus/src/ptychodus/model/ptychopinn/reconstructor.py:61-273`

**Key Methods:**

| Method | Purpose | TensorFlow Implementation | PyTorch Status |
|:-------|:--------|:--------------------------|:---------------|
| `__init__()` | Initialize settings | Reads `PtychoPINNModelSettings` | ✅ Reusable |
| `ingest(input)` | Convert Ptychodus data | `create_raw_data()` → `RawData` | ⚠️ Needs `RawDataTorch` |
| `reconstruct()` | Run inference | `model.predict()` + stitching | ❌ Missing |
| `train()` | Train model | `run_cdi_example()` | ❌ Missing |
| `export_training_data()` | Save NPZ | Writes canonical NPZ schema | ✅ Reusable |
| `open_model()` | Load trained model | `load_inference_bundle()` | ❌ Missing |
| `save_model()` | Save trained model | `ModelManager.save()` | ❌ Missing |

### 7.2 Integration Strategy

**Option A:** Dual Reconstructor (Recommended for Phase E)
- Create `PtychoPINNTorchReconstructor` class implementing same interface
- Add backend selection in `PtychoPINNReconstructorLibrary`
- Share data ingestion (`create_raw_data`) and NPZ export

**Option B:** Unified Reconstructor with Backend Dispatch
- Single `PtychoPINNTrainableReconstructor` with runtime backend selection
- Dispatch to `ptycho_torch/workflows.py` or `ptycho/workflows/components.py`
- Requires careful state management to avoid config drift

**Recommendation:** Option A for Phase E to minimize regression risk

---

## 8. Dependency & Tooling Differences

### 8.1 Unique PyTorch Dependencies

| Dependency | Purpose | TensorFlow Equivalent | Integration Notes |
|:-----------|:--------|:----------------------|:------------------|
| `lightning` | Training orchestration | Direct `model.fit()` | Must suppress Lightning outputs in Ptychodus GUI |
| `mlflow.pytorch` | Experiment tracking | `tf.keras.callbacks` | Optional; disable in production |
| `tensordict` | Memory-mapped tensors | NumPy memmap | Ensure cache compatibility |

### 8.2 Configuration Tooling

**TensorFlow:** Dataclass-based (`ptycho/config/config.py`) with typed fields
**PyTorch:** Singleton dictionaries (`ptycho_torch/config_params.py`) with untyped keys

**Unification Strategy (Phase B):**
- PyTorch backend MUST accept dataclasses as inputs
- Internal conversion: dataclass → singleton settings
- Keep `ptycho.params.cfg` as shared state for legacy modules

---

## 9. Test Coverage Gaps

### 9.1 Existing Tests

**TensorFlow:** `tests/test_integration_workflow.py` validates train → save → load → infer cycle
**PyTorch:** No equivalent integration test

### 9.2 Required Test Coverage (from TEST-PYTORCH-001)

| Test Scope | Purpose | Status |
|:-----------|:--------|:-------|
| Config bridge | Dataclass → singleton → `params.cfg` | ❌ Missing |
| Data pipeline | NPZ → `RawDataTorch` → `PtychoDataContainerTorch` | ❌ Missing |
| Training workflow | `run_cdi_example_torch()` end-to-end | ❌ Missing |
| Persistence | Save → load → infer with Lightning checkpoints | ❌ Missing |
| Ptychodus integration | Dual-backend reconstructor selection | ❌ Missing |

**Test Plan Reference:** `plans/pytorch_integration_test_plan.md`

---

## 10. Priority-Ordered Gap Summary

| # | Gap | Phase | Blocking | Estimated Complexity |
|:--|:----|:------|:---------|:---------------------|
| 1 | Configuration dataclass bridge | B | ALL | Medium (2-3 loops) |
| 2 | RawDataTorch + PtychoDataContainerTorch | C | Training/Inference | High (4-5 loops) |
| 3 | Workflow orchestration (run_cdi_example_torch) | D | Ptychodus integration | Medium (3-4 loops) |
| 4 | Model persistence (Lightning checkpoint shim) | D | Model reuse | Medium (2-3 loops) |
| 5 | Inference script + stitching | D | Ptychodus reconstruct() | Low (1-2 loops) |
| 6 | Ptychodus dual-backend wiring | E | Production use | Medium (2-3 loops) |
| 7 | Integration test suite | E | CI/regression prevention | Low (1-2 loops) |

**Critical Path:** Phases B → C → D → E (serial dependencies)
**Parallelizable:** Test fixture creation (TEST-PYTORCH-001) can proceed alongside Phase B

---

## 11. Cross-References

### Specifications
- **API Contract:** `specs/ptychodus_api_spec.md` (§4 reconstructor contract, §5 config fields)
- **Data Format:** `specs/data_contracts.md` (NPZ schema, normalization requirements)

### Implementation Guides
- **TensorFlow Architecture:** `docs/architecture.md` (component diagram, data flow)
- **PyTorch Workflow:** `docs/workflows/pytorch.md` (current usage patterns)
- **Config Bridge Pattern:** `docs/DEVELOPER_GUIDE.md` §2.3 (update_legacy_dict example)

### Planning Documents
- **Implementation Phases:** `plans/active/INTEGRATE-PYTORCH-001/implementation.md` (Phases B-E tasks)
- **Test Strategy:** `plans/pytorch_integration_test_plan.md` (fixture requirements, test shape)
- **Glossary:** `plans/active/INTEGRATE-PYTORCH-001/glossary_and_ownership.md` (terminology reference)

---

**Next Steps:** Proceed to Phase B (Configuration Bridge) per `implementation.md`. Use this parity map as the authoritative reference for all integration decisions.
