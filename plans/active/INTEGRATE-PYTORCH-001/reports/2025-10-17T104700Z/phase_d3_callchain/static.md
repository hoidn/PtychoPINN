# Phase D3 TensorFlow Persistence Callchain (Static Analysis)

**Timestamp:** 2025-10-17T104700Z
**Initiative:** INTEGRATE-PYTORCH-001
**Focus:** Phase D3.A — TensorFlow persistence flow documentation
**Analysis Question:** "How does the TensorFlow workflow produce and consume `wts.h5.zip` bundles (train → `ModelManager.save_multiple_models` → archive layout → `load_inference_bundle`), and which touchpoints must the PyTorch backend replicate to stay spec-compliant? Contrast with Lightning checkpoint + MLflow outputs in `ptycho_torch.train`."

---

## 1. Candidate Entry Points

Based on documentation analysis (`specs/ptychodus_api_spec.md` §4.5–4.6, `docs/workflows/pytorch.md`, `ptycho/workflows/components.py` module docstring):

| Candidate | Relevance Signals | Confidence | Expected Code Region |
|-----------|------------------|------------|---------------------|
| **Training orchestration → ModelManager save** | "save", "wts.h5.zip", "multi-model archives", "training" | High | `ptycho/workflows/components.py::run_cdi_example`, `ptycho/train_pinn.py`, `ptycho/model_manager.py::save_multiple_models` |
| **Inference loading → ModelManager restore** | "load", "inference bundle", "parameter restoration", "model_path" | High | `ptycho/workflows/components.py::load_inference_bundle`, `ptycho/model_manager.py::load_multiple_models` |
| **PyTorch Lightning checkpoint creation** | "Lightning", "MLflow", "checkpoint", "autolog" | Medium | `ptycho_torch/train.py::main`, Lightning `ModelCheckpoint` callback, MLflow autolog hooks |
| **Legacy save() wrapper** | "save", "custom_objects", "intensity_scale" | Medium | `ptycho/model_manager.py::save` (wrapper calling save_multiple_models) |

**Selected Entrypoints:**
1. **Primary (Training):** `ptycho/workflows/components.py::run_cdi_example` → `train_cdi_model` → training loop → archive creation
2. **Primary (Inference):** `ptycho/workflows/components.py::load_inference_bundle` → `ModelManager.load_multiple_models`
3. **Contrast (PyTorch):** `ptycho_torch/train.py::main` Lightning + MLflow persistence

**Rationale:** These cover the complete lifecycle (train→save→load→infer) and provide direct contrast with PyTorch outputs.

---

## 2. Config Flow (TensorFlow)

### Entry Point Configuration Bridge
- **Where:** `ptycho/workflows/components.py:706`
- **How:** `update_legacy_dict(params.cfg, config)` at entry to `run_cdi_example`
- **Purpose:** Synchronize modern `TrainingConfig` dataclass → legacy `params.cfg` global dict
- **Side Effect:** **CRITICAL** — ModelManager save captures `params.cfg.copy()` snapshot (`ptycho/model_manager.py:74`), so all config must be populated before training

### Config Keys Persisted in Archive
From `ptycho/model_manager.py:73-78`:
```python
params_dict = params.cfg.copy()
params_dict['intensity_scale'] = intensity_scale
params_dict['_version'] = '1.0'
# Serialized via dill to <model_name>/params.dill
```

**Critical Fields** (from `params.cfg` at save time):
- `N` — diffraction pattern size
- `gridsize` — grouping parameter (e.g., 2 for 2×2 groups)
- `intensity_scale` — runtime-computed normalization factor
- `nphotons` — physics prior
- `model_type` — 'pinn' vs 'supervised'
- All other fields from TrainingConfig/ModelConfig (via `update_legacy_dict`)

### Config Restoration Path
- **Where:** `ptycho/model_manager.py:104-119`
- **How:** Load `params.dill` → extract `gridsize` & `N` → `params.cfg.update(loaded_params)`
- **Purpose:** **CONFIG-001 compliance** — restore exact training-time configuration before model reconstruction

---

## 3. Core Pipeline Stages (Training → Save)

### Stage 1: Training Orchestration
- **Entry:** `ptycho/workflows/components.py:676-723::run_cdi_example`
- **Purpose:** Top-level workflow coordinating train → optional stitching
- **Delegation:** Calls `train_cdi_model(train_data, test_data, config)` at line 709

### Stage 2: Training Execution
- **Entry:** `ptycho/workflows/components.py:573-609::train_cdi_model`
- **Purpose:** Data preparation + probe initialization + model training
- **Key Steps:**
  1. Convert `RawData` → `PtychoDataContainer` via `create_ptycho_data_container` (lines 591-595)
  2. Initialize probe: `probe.set_probe_guess(None, train_container.probe)` (line 598)
  3. Train via `train_pinn.train_eval(PtychoDataset(...))` (line 604)
  4. Return results dict including `train_container`, `test_container`, and training artifacts

### Stage 3: Model Save Trigger
- **Where:** Training scripts call `ptycho/model_manager.py:425-467::save(out_prefix)`
- **Purpose:** Wrapper collecting custom objects + intensity_scale, then delegating to `ModelManager.save_multiple_models`
- **Key Artifacts Assembled:**
  - `models_to_save = {'autoencoder': model.autoencoder, 'diffraction_to_obj': model.diffraction_to_obj}` (lines 462-465)
  - `custom_objects` dict with ~25 custom layers/functions (lines 440-460)
  - `intensity_scale` from `params.get('intensity_scale')` (line 467)

### Stage 4: Archive Creation
- **Entry:** `ptycho/model_manager.py:346-378::ModelManager.save_multiple_models`
- **Purpose:** Serialize multi-model archive to `{base_path}.zip`
- **Process:**
  1. Create temp directory (line 359)
  2. Save manifest: `{'models': ['autoencoder', 'diffraction_to_obj'], 'version': '1.0'}` → `manifest.dill` (lines 361-364)
  3. For each model:
     - Create subdirectory `{temp_dir}/{model_name}/`
     - Call `ModelManager.save_model(model, model_subdir, custom_objects, intensity_scale)` (line 370)
  4. Zip entire temp directory → `{base_path}.zip` (lines 373-378)

### Stage 5: Individual Model Persistence
- **Entry:** `ptycho/model_manager.py:49-86::ModelManager.save_model`
- **Purpose:** Save single model with full context
- **Files Created per Model:**
  1. **`model.keras`** — Keras 3 format (TensorFlow SavedModel internally) (line 67)
  2. **`custom_objects.dill`** — Dill-serialized custom objects dict (lines 70-71)
  3. **`params.dill`** — Dill-serialized `params.cfg` snapshot + `intensity_scale` + `_version` (lines 73-78)
  4. **`model.h5`** — Metadata-only HDF5 with `intensity_scale` attribute (lines 81-82)

---

## 4. Core Pipeline Stages (Load → Inference)

### Stage 1: Inference Bundle Loading
- **Entry:** `ptycho/workflows/components.py:102-184::load_inference_bundle`
- **Purpose:** Standard entry point for loading trained model for inference
- **Validation Steps:**
  1. Check `model_dir` is valid directory (lines 134-141)
  2. Verify `wts.h5.zip` exists (lines 144-152)
- **Delegation:** Calls `ModelManager.load_multiple_models(str(model_zip))` (line 160)
- **Return:** `(model, config)` tuple where `model = models_dict['diffraction_to_obj']` (line 171), `config = params.cfg.copy()` (line 175)

### Stage 2: Multi-Model Archive Extraction
- **Entry:** `ptycho/model_manager.py:381-422::ModelManager.load_multiple_models`
- **Purpose:** Extract zip archive, validate manifest, load requested models
- **Process:**
  1. Verify `{base_path}.zip` exists (lines 392-394)
  2. Extract to temp directory (lines 396-399)
  3. Load `manifest.dill` → validate requested models exist (lines 401-414)
  4. For each requested model:
     - Call `ModelManager.load_model(os.path.join(temp_dir, model_name))` (line 420)
  5. Return `Dict[str, tf.keras.Model]` (line 422)

### Stage 3: Individual Model Loading
- **Entry:** `ptycho/model_manager.py:89-343::ModelManager.load_model`
- **Purpose:** Architecture-aware model loading with parameter restoration
- **Process:**
  1. **Load params.dill** (lines 105-106)
  2. **Extract `gridsize` & `N`** — required for architecture reconstruction (lines 112-116)
  3. **Update `params.cfg`** with loaded parameters → **CONFIG-001 side effect** (line 119)
  4. **Load custom_objects.dill** + add missing objects (lines 122-167)
  5. **Enable unsafe deserialization** for Lambda layers (line 173)
  6. **Reconstruct blank model** via `create_model_with_gridsize(gridsize, N)` (line 176)
  7. **Load weights:**
     - Priority 1: `model.keras` (Keras 3 format) (lines 199-203)
     - Priority 2: `saved_model.pb` (SavedModel format, complex wrapper logic) (lines 221-291)
     - Fallback: Legacy weights-only (line 337)
  8. **Return loaded model** (line 339)

---

## 5. Archive Layout (wts.h5.zip Structure)

```
wts.h5.zip
├── manifest.dill              # {'models': ['autoencoder', 'diffraction_to_obj'], 'version': '1.0'}
├── autoencoder/
│   ├── model.keras           # TensorFlow Keras 3 SavedModel
│   ├── custom_objects.dill   # Dill-serialized custom layers/functions dict
│   ├── params.dill           # params.cfg snapshot + intensity_scale + _version
│   └── model.h5              # Metadata-only HDF5 (intensity_scale attribute)
└── diffraction_to_obj/
    ├── model.keras
    ├── custom_objects.dill
    ├── params.dill
    └── model.h5
```

**Key Observations:**
1. **Dual-model bundle:** Both `autoencoder` and `diffraction_to_obj` saved together
2. **Spec compliance:** §4.6 mandates `wts.h5.zip` naming and `diffraction_to_obj` key
3. **Params snapshot:** Each model subdirectory carries full `params.cfg` copy (enables independent loading)
4. **Custom objects:** Serialized via `dill` to handle Lambda layers and custom TensorFlow classes

---

## 6. Callgraph Edge List (Training Flow)

| From | To | Why | Anchors |
|------|-----|-----|---------|
| `scripts/training/train.py` (or equivalent) | `run_cdi_example` | Entry point for training workflow | `ptycho/workflows/components.py:676` |
| `run_cdi_example` | `update_legacy_dict` | **CONFIG-001 gate:** populate `params.cfg` | `ptycho/workflows/components.py:706` |
| `run_cdi_example` | `train_cdi_model` | Delegate to training orchestration | `ptycho/workflows/components.py:709` |
| `train_cdi_model` | `create_ptycho_data_container` | Convert RawData → PtychoDataContainer | `ptycho/workflows/components.py:591,593` |
| `train_cdi_model` | `probe.set_probe_guess` | Initialize probe from data | `ptycho/workflows/components.py:598` |
| `train_cdi_model` | `train_pinn.train_eval` | Execute training loop | `ptycho/workflows/components.py:604` |
| Training loop (user code) | `model_manager.save` | Trigger persistence after training | `ptycho/model_manager.py:425` |
| `model_manager.save` | `ModelManager.save_multiple_models` | Delegate to archive creation | `ptycho/model_manager.py:467` |
| `save_multiple_models` | `save_model` (×2) | Save each model to subdirectory | `ptycho/model_manager.py:370` |
| `save_model` | `model.save('model.keras')` | Keras 3 SavedModel serialization | `ptycho/model_manager.py:67` |
| `save_model` | `dill.dump(custom_objects, ...)` | Serialize custom objects | `ptycho/model_manager.py:70-71` |
| `save_model` | `dill.dump(params_dict, ...)` | Serialize params.cfg snapshot | `ptycho/model_manager.py:77-78` |
| `save_multiple_models` | `zipfile.ZipFile.write` | Create final `.zip` archive | `ptycho/model_manager.py:373-378` |

## 7. Callgraph Edge List (Inference Flow)

| From | To | Why | Anchors |
|------|-----|-----|---------|
| `scripts/inference/inference.py` (or ptychodus) | `load_inference_bundle` | Entry point for inference | `ptycho/workflows/components.py:102` |
| `load_inference_bundle` | Path validation | Verify `wts.h5.zip` exists | `ptycho/workflows/components.py:134-152` |
| `load_inference_bundle` | `ModelManager.load_multiple_models` | Delegate to archive loader | `ptycho/workflows/components.py:160` |
| `load_multiple_models` | `zipfile.ZipFile.extractall` | Extract archive to temp dir | `ptycho/model_manager.py:398-399` |
| `load_multiple_models` | `dill.load(manifest.dill)` | Read archive manifest | `ptycho/model_manager.py:402-404` |
| `load_multiple_models` | `load_model` (×N) | Load each requested model | `ptycho/model_manager.py:420` |
| `load_model` | `dill.load(params.dill)` | Restore params snapshot | `ptycho/model_manager.py:105-106` |
| `load_model` | `params.cfg.update(loaded_params)` | **CONFIG-001:** restore global state | `ptycho/model_manager.py:119` |
| `load_model` | `dill.load(custom_objects.dill)` | Restore custom objects | `ptycho/model_manager.py:122-123` |
| `load_model` | `create_model_with_gridsize(gridsize, N)` | Reconstruct blank model | `ptycho/model_manager.py:176` |
| `load_model` | `tf.keras.models.load_model('model.keras', ...)` | Load weights into blank model | `ptycho/model_manager.py:201` |
| `load_inference_bundle` | Return `(model, config)` | Provide model + params to caller | `ptycho/workflows/components.py:180` |

---

## 8. PyTorch Persistence Delta (Contrast Analysis)

### Current PyTorch Outputs (ptycho_torch/train.py)

#### Lightning Checkpoint
- **Where:** `ptycho_torch/train.py:123-128` (ModelCheckpoint callback)
- **Format:** PyTorch `.ckpt` file (Lightning-managed)
- **Location:** `{default_root_dir}/checkpoints/best-checkpoint.ckpt` (inferred from callback config line 127)
- **Contents:**
  - Model state_dict (PyTorch weights)
  - Optimizer state
  - Learning rate scheduler state
  - Training epoch counter
  - Lightning-specific metadata
- **Gap:** ❌ **No `params.cfg` snapshot**; ❌ **No custom objects serialization**; ❌ **Not zip archive format**

#### MLflow Artifacts
- **Where:** `ptycho_torch/train.py:169` (MLflow autolog enabled)
- **Autolog Behavior:**
  - Logs hyperparameters from `log_parameters_mlflow(...)` (line 179)
  - Logs training metrics automatically via Lightning integration
  - **MAY** log checkpoint as artifact (MLflow autolog behavior)
  - **MAY** log final model as MLflow model flavor (not guaranteed without explicit `mlflow.pytorch.log_model()`)
- **Location:** MLflow tracking URI (configurable, defaults to `mlruns/` local directory or remote server)
- **Gap:** ❌ **Persistence is MLflow-dependent** (spec requires standalone archives); ❌ **No `wts.h5.zip` equivalent**; ❌ **No dual-model bundle support**

### Critical Deltas (TensorFlow → PyTorch)

| Aspect | TensorFlow (ModelManager) | PyTorch (Current) | Gap Severity |
|--------|--------------------------|-------------------|--------------|
| **Archive Format** | `wts.h5.zip` (spec-mandated) | Lightning `.ckpt` + MLflow artifacts | 🔴 **CRITICAL** — spec §4.6 violation |
| **Params Snapshot** | `params.dill` per model | ❌ Not captured | 🔴 **CRITICAL** — CONFIG-001 violation on load |
| **Custom Objects** | `custom_objects.dill` | ❌ Not serialized | 🟡 **MODERATE** — PyTorch layers may not need this (TensorFlow Lambda layer workaround) |
| **Multi-Model Bundle** | `autoencoder` + `diffraction_to_obj` in single zip | Single model checkpoint | 🔴 **CRITICAL** — spec requires dual-model support |
| **Manifest** | `manifest.dill` with model names + version | ❌ Not present | 🟡 **MODERATE** — enables multi-model validation |
| **Standalone Load** | No external dependencies (zip + TensorFlow) | Requires MLflow + Lightning runtime | 🔴 **CRITICAL** — spec §4.6 requires standalone archives |
| **Load Entry Point** | `load_inference_bundle(model_dir)` → standard API | No equivalent; requires MLflow run ID or checkpoint path | 🔴 **CRITICAL** — spec §4.5 lifecycle contract |

---

## 9. Side Effects & Critical Dependencies

### params.cfg Mutations
1. **On save** (`model_manager.py:74`): Snapshot captured via `params.cfg.copy()`
2. **On load** (`model_manager.py:119`): `params.cfg.update(loaded_params)` — **global state mutation**
3. **Implication:** Any code path that loads a model **MUST** expect `params.cfg` to be overwritten

### TensorFlow-Specific Machinery
1. **Custom objects registry** (`model_manager.py:136-167`): Imports ~25 custom layers/functions
2. **Unsafe deserialization** (`model_manager.py:173`): Required for Lambda layers in Keras 3
3. **Blank model reconstruction** (`model_manager.py:176`): Uses `create_model_with_gridsize(gridsize, N)` — requires `gridsize` & `N` from params

### PyTorch Replacement Requirements
1. **Must** capture `params.cfg` snapshot (or equivalent dataclass) at save time
2. **Must** restore `params.cfg` (via `update_legacy_dict`) at load time before inference
3. **Must** support dual-model bundle (`autoencoder` + `diffraction_to_obj`) in single archive
4. **Must** produce `wts.h5.zip`-compatible archive (or adapter layer for ptychodus)
5. **Should** avoid MLflow dependency for core persistence (optional logging OK, required loading NOT OK)

---

## 10. Open Questions & Confirmation Plan

### Q1: Do PyTorch models need custom_objects serialization?
- **Context:** TensorFlow requires this for Lambda layers; PyTorch may not (standard nn.Module architecture)
- **Confirmation:** Check if `ptycho_torch/model.py` uses any dynamic layer construction requiring serialization beyond state_dict
- **Impact:** If not needed, reduces Phase D3.B complexity

### Q2: Can MLflow be made optional for persistence?
- **Context:** Current PyTorch train.py tightly couples MLflow autolog to persistence
- **Confirmation:** Test whether Lightning checkpoint alone is sufficient for inference; review Phase D1 decision log
- **Impact:** Determines whether Phase D3.B needs MLflow→zip adapter or direct Lightning→zip path

### Q3: How to handle intensity_scale in PyTorch?
- **Context:** TensorFlow computes this dynamically during training; unclear if PyTorch equivalent exists
- **Confirmation:** Search `ptycho_torch/` for `intensity_scale` or equivalent normalization factor
- **Impact:** May need Phase D3.B hook to capture this value during training

### Q4: Should PyTorch reuse ModelManager or write TorchModelManager?
- **Context:** Phase D1 decision log recommends orchestration shims (Option B)
- **Confirmation:** Review Phase D1.C decision doc (already produced)
- **Impact:** Architecture choice for Phase D3.B implementation

---

## 11. Gaps & Unknowns

1. **Dynamic trace not captured:** No minimal training run executed (evidence-only loop per input.md)
2. **MLflow autolog internals:** Exact artifact paths/formats require inspection of actual MLflow output directory
3. **PyTorch checkpoint structure:** `.ckpt` internal layout not inspected (requires torch.load or Lightning docs)
4. **Custom object necessity:** Assumption that PyTorch doesn't need `custom_objects.dill` — requires validation

**Next Steps for Confirmation:**
1. Execute minimal PyTorch training run with MLflow → inspect `mlruns/` directory structure
2. Load PyTorch `.ckpt` via `torch.load(...)` → document state_dict keys
3. Review Phase D1.C decision doc → confirm TorchModelManager vs ModelManager reuse
4. Search `ptycho_torch/` for intensity_scale computation

---

## 12. Summary

**TensorFlow Persistence Contract (Complete):**
- Entry: `run_cdi_example` → `train_cdi_model` → training loop → `model_manager.save` → `ModelManager.save_multiple_models`
- Archive: `wts.h5.zip` containing dual-model bundle (`autoencoder` + `diffraction_to_obj`) with params snapshots and custom objects
- Load: `load_inference_bundle` → `ModelManager.load_multiple_models` → architecture-aware model reconstruction + params.cfg restoration
- Side effects: **CONFIG-001 critical** — `params.cfg` overwritten on load

**PyTorch Delta (Gaps):**
- ❌ No `wts.h5.zip` equivalent
- ❌ No `params.cfg` snapshot/restoration
- ❌ No dual-model bundle support
- ❌ MLflow-dependent persistence (violates spec §4.6 standalone requirement)
- ❌ No `load_inference_bundle` equivalent

**Phase D3.B Requirements (Derived):**
1. Implement Lightning→zip adapter capturing params snapshot
2. Support dual-model bundle in single archive
3. Provide `load_inference_bundle_torch` matching TensorFlow API
4. Ensure `update_legacy_dict` call on load (CONFIG-001 compliance)
5. Optional: MLflow logging can coexist but must not be required for loading
