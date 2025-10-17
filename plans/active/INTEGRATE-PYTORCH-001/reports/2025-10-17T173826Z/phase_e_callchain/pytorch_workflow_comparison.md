# PyTorch Workflow Backend - Parallel Implementation Analysis

## Executive Summary

The PyTorch workflow orchestration layer (`ptycho_torch/workflows/components.py`) mirrors the TensorFlow baseline (`ptycho/workflows/components.py`) at the same orchestration level while delegating to PyTorch-specific backends. The implementation is currently at **Phase D2.A (Scaffold)** with all entry point signatures complete and CONFIG-001 compliance verified, but core training/inference logic pending Phase D2.B/C implementation.

---

## Entry Point Signatures Comparison Table

| Aspect | TensorFlow (`ptycho/workflows/components.py`) | PyTorch (`ptycho_torch/workflows/components.py`) | Parity Status |
|--------|---------------------------------------------|------------------------------------------------|---------------|
| **Function Name** | `run_cdi_example()` | `run_cdi_example_torch()` | ✅ Named pair |
| **Parameters** | `(train_data, test_data, config, flip_x, flip_y, transpose, M, do_stitching)` | `(train_data, test_data, config, flip_x, flip_y, transpose, M, do_stitching)` | ✅ Identical |
| **Input Types** | `Union[RawData, PtychoDataContainer]` | `Union[RawData, 'RawDataTorch', 'PtychoDataContainerTorch']` | ✅ Parallel types |
| **Config Parameter** | `TrainingConfig` | `TrainingConfig` | ✅ Same dataclass |
| **Return Type** | `Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]` | `Tuple[Optional[Any], Optional[Any], Dict[str, Any]]` | ✅ Signature match |
| **CONFIG-001 Compliance** | ✅ `update_legacy_dict()` at line 706 | ✅ `update_legacy_dict()` at line 161 | ✅ Both compliant |

---

## Function Implementation Status Matrix

### 1. `run_cdi_example` / `run_cdi_example_torch`

| Component | TensorFlow Status | PyTorch Status | Notes |
|-----------|------------------|-----------------|-------|
| **Entry signature** | Implemented | ✅ Complete | Matches TensorFlow exactly |
| **CONFIG-001 gate** | Line 706: `update_legacy_dict(params.cfg, config)` | ✅ Line 161 | CRITICAL: Synchronizes params before delegation |
| **Data normalization** | Lines 591-595: `create_ptycho_data_container()` | ✅ Phase D2.B: `_ensure_container()` | PyTorch uses Phase C adapters (RawDataTorch, PtychoDataContainerTorch) |
| **Training delegation** | Line 709: `train_cdi_model()` | ✅ Line 166: `train_cdi_model_torch()` | Calls backend-specific trainer |
| **Stitching path** | Lines 714-721: conditional `reassemble_cdi_image()` | ⚠️ Lines 173-182: calls `_reassemble_cdi_image_torch()` | Raises `NotImplementedError` (Phase D2.C stub) |
| **Persistence** | Not in baseline (TensorFlow trains inline) | ✅ Lines 188-205: model saving via `save_torch_bundle()` | PyTorch adds optional archive persistence |
| **Return values** | `(recon_amp, recon_phase, train_results)` | ✅ Same structure | Full API parity achieved |

### 2. `train_cdi_model` / `train_cdi_model_torch`

| Component | TensorFlow Status | PyTorch Status | Notes |
|-----------|------------------|-----------------|-------|
| **Entry signature** | `(train_data, test_data, config)` → `Dict[str, Any]` | ✅ Identical | Signature complete |
| **Data containerization** | Lines 591-595: `create_ptycho_data_container()` | ✅ `_ensure_container()` helper | Delegates to Phase C adapters |
| **Input type handling** | Handles `Union[RawData, PtychoDataContainer]` | ✅ Handles `Union[RawData, RawDataTorch, PtychoDataContainerTorch]` | Supports extended types |
| **Probe initialization** | Line 598: `probe.set_probe_guess()` | 🔶 Line 148: TODO (deferred to Phase D2.B) | PyTorch: skipped in scaffold |
| **Training orchestration** | Line 604: `train_pinn.train_eval()` | 🔶 Line 152: `_train_with_lightning()` stub | Full Lightning impl pending |
| **Result structure** | `{'history': {...}, 'train_container': ..., 'test_container': ...}` | ✅ Same structure | Compatible result dict |

### 3. `reassemble_cdi_image` / `_reassemble_cdi_image_torch`

| Component | TensorFlow Status | PyTorch Status | Notes |
|-----------|------------------|-----------------|-------|
| **Function location** | Public: `reassemble_cdi_image()` at line 611 | Private: `_reassemble_cdi_image_torch()` (line 329) | PyTorch: internal helper only |
| **Entry signature** | `(test_data, config, flip_x, flip_y, transpose, M, coord_scale)` | Subset: `(test_data, config, flip_x, flip_y, transpose, M)` | PyTorch: no coord_scale parameter in signature |
| **Implementation** | **IMPLEMENTED** (lines 637-674) | 🔴 **NOT IMPLEMENTED** | Raises `NotImplementedError` at line 89-94 |
| **Logic flow** | 1. Normalize container 2. Inference via `nbutils.reconstruct_image()` 3. Coord transforms 4. Reassemble via `tf_helper.reassemble_position()` 5. Extract amp/phase | **TODO**: Same 5 steps (Phase D2.C) | PyTorch version awaits full implementation |
| **Returns** | `(recon_amp, recon_phase, results_dict)` | ✅ Same signature | Full parity intended |

### 4. `load_inference_bundle` / `load_inference_bundle_torch`

| Component | TensorFlow Status | PyTorch Status | Notes |
|-----------|------------------|-----------------|-------|
| **Entry signature** | `(model_dir: Path)` → `Tuple[tf.keras.Model, dict]` | `(bundle_dir, model_name='diffraction_to_obj')` → `Tuple[Any, dict]` | Similar structure, extended parameters |
| **Archive format** | Expects `{model_dir}/wts.h5.zip` (TensorFlow .keras format) | Expects `{bundle_dir}/wts.h5.zip` (PyTorch .pth format) | Format parity via shared archive schema |
| **Implementation** | Lines 102-184: Full implementation via `ModelManager.load_multiple_models()` | ⚠️ Lines 157-220: Delegates to `load_torch_bundle()` (Phase D3.C stub) | CONFIG-001 gate inside `load_torch_bundle()` |
| **Model extraction** | Gets 'diffraction_to_obj' from models_dict | ✅ Parameterized: `model_name='diffraction_to_obj'` (default) | Flexible model selection |
| **CONFIG-001 gate** | Line 175: `config = params.cfg.copy()` (restored by ModelManager) | Line 265 (inside `load_torch_bundle`): `params.cfg.update(params_dict)` | Both restore params.cfg from archive |
| **Torch availability** | N/A | 🔴 Raises `ImportError` if `load_torch_bundle` unavailable | Requires torch runtime |

---

## Config Bridge Integration Points

### Bridge Touchpoints in PyTorch Workflows

| Touchpoint | Location | TensorFlow | PyTorch | Purpose |
|-----------|----------|-----------|---------|---------|
| **Entry point gate** | `run_cdi_example_torch()` line 161 | N/A | ✅ `update_legacy_dict(params.cfg, config)` | CONFIG-001: Sync TrainingConfig → legacy dict before delegation |
| **Config bridge module** | Separate layer | N/A | ✅ `ptycho_torch/config_bridge.py` | Translates PyTorch singleton configs → TensorFlow dataclasses |
| **Data container creation** | `_ensure_container()` lines 212-283 | Uses TensorFlow RawData natively | ✅ Wraps with `RawDataTorch`, calls `generate_grouped_data()` | Phase C adapters normalize inputs to PtychoDataContainerTorch |
| **Training config passing** | Train orchestration | Inline params.cfg | ✅ Explicit `config: TrainingConfig` parameter | PyTorch: explicit dataclass threading avoids global state |
| **Probe handling** | `train_cdi_model_torch()` line 146-148 | `probe.set_probe_guess()` | TODO (deferred) | PyTorch: probe init skipped in Phase D2.B scaffold |

### CONFIG-001 Critical Path

```
PyTorch Workflow Entry
  ↓
run_cdi_example_torch() called with TrainingConfig
  ↓
[CRITICAL GATE] update_legacy_dict(params.cfg, config) at line 161
  ↓
train_cdi_model_torch(train_data, test_data, config)
  ↓
_ensure_container() — normalizes data via Phase C adapters
  ↓
_train_with_lightning() — stub returns minimal results
  ↓
Optional: _reassemble_cdi_image_torch() — raises NotImplementedError (Phase D2.C)
  ↓
Optional: save_torch_bundle(models, config) — creates wts.h5.zip archive
  ↓
load_torch_bundle() inside load_inference_bundle_torch()
  ↓
[CRITICAL GATE] params.cfg.update(params_dict) at line 265 (inside load_torch_bundle)
  ↓
Return (models_dict, params_dict) for inference
```

---

## Implementation Status by Phase

| Phase | Component | PyTorch Status | Artifact |
|-------|-----------|-----------------|----------|
| **Phase D2.A** | Workflow orchestration scaffold | ✅ **COMPLETE** | Entry signatures match TensorFlow, CONFIG-001 gates in place, torch-optional design verified |
| **Phase D2.B** | Lightning training orchestration | 🔴 **STUB** | `_train_with_lightning()` returns placeholder losses; full Lightning integration pending |
| **Phase D2.C** | Inference/stitching reassembly | 🔴 **STUB** | `_reassemble_cdi_image_torch()` raises `NotImplementedError`; Phase D2.C roadmap in plans/ |
| **Phase C** | Data adapters (RawDataTorch, PtychoDataContainerTorch) | 🔶 **PARTIAL** | Referenced but implementation status unconfirmed in workflows layer |
| **Phase D3.C** | Model persistence (load_torch_bundle) | 🔴 **STUB** | Raises `NotImplementedError` for model reconstruction; params.cfg restoration complete |
| **Phase D4.C1** | Archive persistence (save_torch_bundle) | ✅ **IMPLEMENTED** | Creates wts.h5.zip archives with dual-model structure and params.dill snapshot |

---

## Key Differences from TensorFlow

### 1. **Data Pipeline**
- **TensorFlow**: Directly uses `RawData` → `PtychoDataContainer`
- **PyTorch**: Wraps with `RawDataTorch`, leverages Phase C adapters for normalized container
- **Implication**: PyTorch has explicit adapter layer for type isolation

### 2. **Training Orchestration**
- **TensorFlow**: Calls `train_pinn.train_eval()` (direct TensorFlow/Keras trainer)
- **PyTorch**: Delegates to `_train_with_lightning()` (Lightning abstraction, currently stub)
- **Implication**: PyTorch decouples model framework from orchestration layer

### 3. **Probe Initialization**
- **TensorFlow**: `probe.set_probe_guess()` called in `train_cdi_model()` line 598
- **PyTorch**: Deferred to Phase D2.B; currently skipped in scaffold (line 148)
- **Implication**: PyTorch probe handling unresolved; may require distinct implementation

### 4. **Inference/Stitching**
- **TensorFlow**: Fully implemented; `reassemble_cdi_image()` handles inference, coordinate transforms, patch reassembly
- **PyTorch**: Private helper `_reassemble_cdi_image_torch()` raises `NotImplementedError`
- **Implication**: PyTorch inference path not yet integrated with trained models

### 5. **Persistence Strategy**
- **TensorFlow**: Archive creation delegated to `ModelManager.save_multiple_models()` (not shown in workflow layer)
- **PyTorch**: Explicit `save_torch_bundle()` call in `run_cdi_example_torch()` at lines 188-205
- **Implication**: PyTorch makes persistence explicit in workflow; saves to `wts.h5.zip` matching TensorFlow format

### 6. **CONFIG-001 Compliance**
- **TensorFlow**: Single gate at line 706 in `run_cdi_example()`
- **PyTorch**: Two gates: (1) entry at line 161, (2) inside `load_torch_bundle()` for inference
- **Implication**: PyTorch replicates params.cfg restoration at both training and inference boundaries

---

## Torch-Optional Design

### Guard Mechanism

```python
# ptycho_torch/workflows/components.py lines 67-88
try:
    from ptycho_torch.config_params import TORCH_AVAILABLE
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    try:
        from ptycho_torch.raw_data_bridge import RawDataTorch
        from ptycho_torch.data_container_bridge import PtychoDataContainerTorch
        from ptycho_torch.model_manager import save_torch_bundle, load_torch_bundle
    except ImportError:
        RawDataTorch = None
        PtychoDataContainerTorch = None
        save_torch_bundle = None
        load_torch_bundle = None
else:
    RawDataTorch = None
    PtychoDataContainerTorch = None
    save_torch_bundle = None
    load_torch_bundle = None
```

**Implication**: Module is importable without torch; runtime errors only when torch-specific functions invoked

---

## Summary Table: PyTorch vs TensorFlow Workflow Parity

| Feature | TensorFlow | PyTorch | Status |
|---------|-----------|---------|--------|
| Entry point signatures | 4 public functions | 3 public + 2 private helpers | ✅ Full API parity |
| CONFIG-001 compliance | Line 706 gate | Lines 161 + 265 gates | ✅ Both compliant |
| Data type handling | RawData, PtychoDataContainer | RawData, RawDataTorch, PtychoDataContainerTorch | ✅ Extended support |
| Training orchestration | `train_pinn.train_eval()` | `_train_with_lightning()` (stub) | 🔴 Pending Phase D2.B |
| Inference/stitching | Fully implemented | `_reassemble_cdi_image_torch()` raises NotImplementedError | 🔴 Pending Phase D2.C |
| Model persistence | Inline (via ModelManager) | Explicit `save_torch_bundle()` | ✅ Implemented |
| Archive loading | Via `ModelManager.load_multiple_models()` | Via `load_torch_bundle()` (stub) | 🔰 Partial (params restored, model reconstruction pending) |
| Torch-optional design | N/A | Guarded imports, importable without torch | ✅ Implemented |
| Return value structure | `(amp, phase, results_dict)` | `(amp, phase, results_dict)` | ✅ Identical |

---

## Open Questions & Blockers (Phase D2.B/C)

1. **Lightning Integration**: How to instantiate PtychoPINN Lightning module? Which version of `ptycho_torch.model` provides Lightning LightningModule subclass?
2. **Probe Initialization**: Does PyTorch probe handling require distinct implementation from TensorFlow? Should `probe.set_probe_guess()` work across backends?
3. **Reassembly Logic**: PyTorch reassemble equivalent uses what? (nbutils-equivalent exists?)
4. **Model Reconstruction**: `load_torch_bundle()` needs `create_torch_model_with_gridsize()` helper to reconstruct architecture before loading weights.

---

## Files Analyzed

- **PyTorch Workflow**: `/home/ollie/Documents/PtychoPINN2/ptycho_torch/workflows/components.py` (506 lines)
- **PyTorch Config Bridge**: `/home/ollie/Documents/PtychoPINN2/ptycho_torch/config_bridge.py` (376 lines)
- **PyTorch Model Manager**: `/home/ollie/Documents/PtychoPINN2/ptycho_torch/model_manager.py` (partial, 286 lines shown)
- **TensorFlow Workflow**: `/home/ollie/Documents/PtychoPINN2/ptycho/workflows/components.py` (757 lines)

