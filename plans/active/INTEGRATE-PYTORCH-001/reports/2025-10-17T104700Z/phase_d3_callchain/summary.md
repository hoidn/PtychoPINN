# Phase D3 Persistence Callchain — Executive Summary

**Initiative:** INTEGRATE-PYTORCH-001 | **Phase:** D3.A Evidence Collection | **Date:** 2025-10-17

---

## Analysis Question

**"How does TensorFlow produce `wts.h5.zip` bundles and what must PyTorch replicate for spec compliance?"**

---

## One-Page Answer

### The TensorFlow Persistence Contract

TensorFlow training produces a **dual-model zip archive** (`wts.h5.zip`) containing:
1. **Two models** (`autoencoder` + `diffraction_to_obj`) — spec §4.6 requires both
2. **Params snapshot** (`params.dill` per model) — captures full `params.cfg` state at training time
3. **Custom objects** (`custom_objects.dill`) — serializes ~25 Lambda layers/custom TensorFlow classes
4. **Manifest** (`manifest.dill`) — lists available models + version for validation

**Critical flow:** `run_cdi_example` → `train_cdi_model` → `model_manager.save` → `ModelManager.save_multiple_models` → zip creation

**CONFIG-001 enforcement:** On load, `params.cfg.update(loaded_params)` restores training-time configuration before model reconstruction. **Without this, gridsize/N mismatch causes shape errors.**

### PyTorch Current State (Gaps)

**What exists:**
- Lightning `.ckpt` checkpoint (model weights + optimizer state)
- MLflow autolog (hyperparams + metrics + maybe artifacts)

**What's missing (🔴 CRITICAL):**
1. **No `wts.h5.zip` equivalent** — violates spec §4.6 archive requirement
2. **No `params.cfg` snapshot** — CONFIG-001 violation on load (shape mismatch risk)
3. **No dual-model bundle** — current save is single-model only
4. **No `load_inference_bundle_torch` API** — violates spec §4.5 lifecycle contract
5. **MLflow-dependent persistence** — spec requires standalone archives

### First Implementation Step (Phase D3.B)

**Minimal viable persistence shim:**

```python
# ptycho_torch/model_manager.py (new module)
def save_torch_bundle(
    models_dict: Dict[str, nn.Module],  # {'autoencoder': ..., 'diffraction_to_obj': ...}
    base_path: str,
    config: TrainingConfig  # replaces params.cfg snapshot
) -> None:
    """Save PyTorch models to wts.h5.zip-compatible archive."""
    zip_path = f"{base_path}.zip"
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save manifest
        manifest = {'models': list(models_dict.keys()), 'version': '2.0-pytorch'}
        dill.dump(manifest, open(f"{temp_dir}/manifest.dill", 'wb'))

        # Save each model
        for name, model in models_dict.items():
            model_dir = os.path.join(temp_dir, name)
            os.makedirs(model_dir)

            # PyTorch state_dict (replaces model.keras)
            torch.save(model.state_dict(), f"{model_dir}/model.pth")

            # Params snapshot (CONFIG-001 critical)
            params_dict = dataclass_to_legacy_dict(config)  # Phase B bridge
            params_dict['_version'] = '2.0-pytorch'
            dill.dump(params_dict, open(f"{model_dir}/params.dill", 'wb'))

        # Zip archive
        shutil.make_archive(base_path, 'zip', temp_dir)
```

**Key design choices:**
- **Reuse `params.dill` schema** — enables cross-backend compatibility
- **Omit `custom_objects.dill`** — PyTorch nn.Module doesn't need Lambda layer workaround (validate in D3.B)
- **Dual-model support** — matches TensorFlow contract exactly
- **Version tag** — '2.0-pytorch' enables format detection

### Load Path (Phase D3.C)

```python
def load_torch_bundle(base_path: str) -> Tuple[nn.Module, dict]:
    """Load PyTorch model bundle with params restoration (CONFIG-001 compliant)."""
    zip_path = f"{base_path}.zip"
    with tempfile.TemporaryDirectory() as temp_dir:
        shutil.unpack_archive(zip_path, temp_dir, 'zip')

        # Load manifest
        manifest = dill.load(open(f"{temp_dir}/manifest.dill", 'rb'))

        # Load diffraction_to_obj (inference model)
        model_dir = os.path.join(temp_dir, 'diffraction_to_obj')

        # Restore params.cfg (CONFIG-001 gate)
        params_dict = dill.load(open(f"{model_dir}/params.dill", 'rb'))
        ptycho.params.cfg.update(params_dict)  # Critical side effect

        # Reconstruct model architecture
        gridsize = params_dict['gridsize']
        N = params_dict['N']
        model = create_torch_model_with_gridsize(gridsize, N)  # Phase D2.B helper

        # Load weights
        state_dict = torch.load(f"{model_dir}/model.pth")
        model.load_state_dict(state_dict)

        return model, params_dict
```

---

## Critical Findings

### 1. CONFIG-001 is Non-Negotiable
**Evidence:** `ptycho/model_manager.py:119` — TensorFlow **always** calls `params.cfg.update(loaded_params)` before model reconstruction.

**Implication:** PyTorch **MUST** do the same via `update_legacy_dict(params.cfg, restored_config)` or risk shape mismatch bugs.

**Test:** Phase D3.C must include a failing test demonstrating `gridsize` mismatch when params restoration is skipped.

### 2. Dual-Model Bundle is Spec-Required
**Evidence:** `specs/ptychodus_api_spec.md:192-202` — Ptychodus expects both `autoencoder` and `diffraction_to_obj` in the archive.

**Implication:** Even though inference only uses `diffraction_to_obj`, **both must be saved** to maintain contract parity.

**Implementation:** `save_torch_bundle` must accept `models_dict` with both keys populated.

### 3. MLflow Cannot Be a Load Dependency
**Evidence:** Spec §4.6 requires "standalone archives"; current PyTorch train.py ties persistence to MLflow autolog.

**Implication:** Phase D3.B must decouple archive creation from MLflow. MLflow logging can remain optional.

**Design:** Lightning checkpoint callback → save hook → call `save_torch_bundle` directly; MLflow autolog separately logs run metadata.

### 4. Custom Objects May Be PyTorch-Optional
**Evidence:** TensorFlow needs `custom_objects.dill` for Lambda layers (~25 serialized functions); PyTorch uses standard `nn.Module` subclasses.

**Implication:** PyTorch may skip `custom_objects.dill` **IF** all layers are standard. Needs validation in Phase D3.B.

**Validation:** Check if `ptycho_torch/model.py` uses any dynamic layer construction beyond `__init__`.

---

## Next Actions (Immediate)

### For Phase D3.B (Archive Writer Implementation)
1. **Create `ptycho_torch/model_manager.py`** with `save_torch_bundle` function
2. **Add Lightning save hook** at end of training to call `save_torch_bundle`
3. **Capture both models:** Modify training loop to provide `autoencoder` + `diffraction_to_obj` dict
4. **Test archive structure:** Write unit test validating zip contents match TensorFlow schema

### For Phase D3.C (Loader Implementation)
1. **Create `load_torch_bundle`** function matching TensorFlow `load_inference_bundle` API
2. **Add `load_inference_bundle_torch` wrapper** in `ptycho_torch/workflows/components.py` (Phase D2 module)
3. **Test params restoration:** Write failing test showing shape mismatch without `params.cfg.update`
4. **Test cross-backend load:** Validate PyTorch can load TensorFlow archives (if weights format allows)

### Open Questions (Block D3.B if Unresolved)
- **Q1:** Does `ptycho_torch/model.py` need `custom_objects.dill`? → **Action:** Code search + inspection
- **Q2:** Where is intensity_scale computed in PyTorch? → **Action:** Grep `ptycho_torch/` for normalization factors
- **Q3:** Should we reuse `ModelManager` or create `TorchModelManager`? → **Action:** Review Phase D1.C decision (already done: Option B = separate shims)

---

## Risks & Mitigation

| Risk | Severity | Mitigation |
|------|----------|-----------|
| **Params snapshot incomplete** | 🔴 CRITICAL | Use `dataclass_to_legacy_dict` from Phase B config bridge (tested) |
| **Archive format mismatch** | 🔴 CRITICAL | Follow exact TensorFlow schema (manifest.dill + per-model subdirs) |
| **Missing intensity_scale** | 🟡 MODERATE | Search PyTorch code; may need training loop hook to capture |
| **Cross-backend incompatibility** | 🟡 MODERATE | Document version tag ('2.0-pytorch'); TensorFlow loader should skip unknown versions |
| **MLflow coupling** | 🟡 MODERATE | Decouple via direct Lightning callback; test without MLflow |

---

## Evidence Quality Assessment

**Strengths:**
- ✅ Complete static callgraph from entry to archive creation
- ✅ Exact file:line anchors for all critical paths
- ✅ Cross-referenced spec requirements with implementation
- ✅ Identified all CONFIG-001 touchpoints

**Limitations:**
- ⚠️ No dynamic trace captured (evidence-only loop per input.md)
- ⚠️ MLflow artifact structure not inspected (requires actual training run)
- ⚠️ PyTorch checkpoint internals not verified (requires `torch.load` inspection)

**Recommendation:** Phase D3.B can proceed with static analysis; dynamic validation deferred to D3.C test phase.

---

## Summary One-Liner

**TensorFlow persistence = dual-model zip + params snapshot + custom objects; PyTorch must replicate all except custom objects (TBD), with CONFIG-001 params restoration as non-negotiable gate.**
