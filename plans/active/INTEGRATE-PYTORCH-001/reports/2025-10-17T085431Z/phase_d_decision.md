# Phase D1.C: Orchestration Surface Decision

**Goal**: Compare API wrapper vs low-level shim approaches for PyTorch workflow integration.

**Context**: Reconstruct lifecycle (specs/ptychodus_api_spec.md §4), Phase D callchain analysis, Phase D asset inventory.

---

## 1. Decision Framework

### 1.1. Core Question

**How should PyTorch workflows expose functionality to satisfy the Ptychodus reconstructor contract?**

**Two Options**:
- **Option A**: Wrap or complete the experimental `ptycho_torch/api/*` layer
- **Option B**: Build new `ptycho_torch/workflows/components.py` mirroring TensorFlow

---

## 2. Option A: High-Level API Wrapper

### 2.1. Architecture

```
Ptychodus Reconstructor
    ↓ (calls)
ptycho_torch.api.trainer_api.train(...)  ← NEW: complete implementation
ptycho_torch.api.inference_api.infer(...) ← NEW: create module
    ↓ (delegates to)
ptycho_torch.train.main(...)              ← EXISTING: Lightning training
ptycho_torch.inference.load_and_predict(...) ← EXISTING: MLflow inference
    ↓ (uses)
PtychoDataModule → PtychoDataset          ← EXISTING: memory-mapped loading
```

**Config Bridge Insertion Point**: Inside API wrapper functions
```python
# ptycho_torch/api/trainer_api.py (NEW)
def train(train_npz: Path, test_npz: Path, config: TrainingConfig) -> Dict[str, Any]:
    """High-level training API for Ptychodus integration."""
    # Translate TensorFlow config → PyTorch singletons
    from ptycho_torch.config_bridge import to_training_config
    from ptycho.config.config import update_legacy_dict
    from ptycho import params

    update_legacy_dict(params.cfg, config)  # ← CONFIG BRIDGE

    # Delegate to existing Lightning training
    from ptycho_torch import train as torch_train
    run_ids = torch_train.main(str(train_npz), existing_config=(...))
    return {'model_path': ..., 'run_id': run_ids['training']}
```

### 2.2. Pros

1. **Minimal Code Changes**: Reuse existing `train.py` and `inference.py` with thin wrappers (~200 lines total)
2. **Separation of Concerns**: API layer isolates Ptychodus from Lightning/MLflow internals
3. **Backward Compatibility**: Existing PyTorch workflows (CLI, notebooks) continue working unchanged
4. **Incremental Adoption**: Can add Ptychodus support without refactoring core training loop

### 2.3. Cons

1. **Double Wrapping**: Ptychodus → API → train.py → Lightning (deep call stack)
2. **Config Translation Overhead**: Must convert TensorFlow dataclass → PyTorch singletons → back to dataclass for `params.cfg`
3. **MLflow Dependency Leakage**: API still depends on `train.py` MLflow coupling unless refactored
4. **Incomplete API Layer**: `ptycho_torch/api/base_api.py` is 995 lines of abstractions with no implementations - must complete or discard
5. **Test Surface Complexity**: Testing requires mocking API → train.py → Lightning chain

### 2.4. Implementation Effort

| Task | Effort |
|------|--------|
| Complete `ptycho_torch/api/trainer_api.py` | 1 day |
| Create `ptycho_torch/api/inference_api.py` | 1 day |
| Wire config_bridge into API wrappers | 0.5 days |
| Add MLflow optionality to `train.py` | 0.5 days |
| Test API layer (torch-optional) | 1 day |
| **Total** | **4 days** |

---

## 3. Option B: Low-Level Orchestration Shims

### 3.1. Architecture

```
Ptychodus Reconstructor
    ↓ (calls)
ptycho_torch.workflows.components.run_cdi_example_torch(...)  ← NEW MODULE
    ↓ (orchestrates)
ptycho_torch.workflows.components.train_cdi_model_torch(...)
    ↓ (builds)
RawDataTorch → PtychoDataContainerTorch   ← PHASE C adapters
Lightning Trainer + PtychoPINN_Lightning  ← DIRECT calls (no train.py)
TorchModelManager.save(...)               ← NEW: persistence shim
```

**Config Bridge Insertion Point**: At orchestration entry
```python
# ptycho_torch/workflows/components.py (NEW ~300 lines)
def run_cdi_example_torch(
    train_data: Union[RawDataTorch, PtychoDataContainerTorch],
    test_data: Optional[...],
    config: TrainingConfig,  # TensorFlow-style dataclass
    do_stitching: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """PyTorch equivalent of ptycho.workflows.components.run_cdi_example."""

    # Config bridge at workflow entry (mirrors TensorFlow pattern)
    from ptycho.config.config import update_legacy_dict
    from ptycho import params
    update_legacy_dict(params.cfg, config)  # ← CRITICAL

    # Train using Lightning (direct call, no train.py wrapper)
    train_results = train_cdi_model_torch(train_data, test_data, config)

    # Optional stitching
    if do_stitching and test_data is not None:
        recon_amp, recon_phase = reassemble_torch(...)
        return recon_amp, recon_phase, train_results

    return None, None, train_results
```

### 3.2. Pros

1. **Mirror TensorFlow Structure**: Identical API signatures between `ptycho.workflows.components` and `ptycho_torch.workflows.components`
2. **Shallow Call Stack**: Ptychodus → shim → Lightning (2 levels vs 4 in Option A)
3. **Direct Lightning Control**: Orchestration layer owns Trainer configuration, callbacks, and persistence
4. **Phase C Adapter Integration**: Natural fit for `RawDataTorch` → `PtychoDataContainerTorch` pipeline
5. **MLflow Optionality**: Shim layer can disable MLflow by default, enable via flag
6. **Test Surface Clarity**: Test shim directly, mock Lightning components

### 3.3. Cons

1. **Code Duplication**: Reimplements some `train.py` logic (Trainer setup, callbacks)
2. **Maintenance Burden**: Two training paths (CLI via `train.py` vs Ptychodus via shim)
3. **Breaking Change**: Existing PyTorch workflows must migrate to new orchestration OR we maintain both
4. **More New Code**: ~300 lines of orchestration + ~200 lines of persistence shim vs ~200 lines of API wrappers

### 3.4. Implementation Effort

| Task | Effort |
|------|--------|
| Create `ptycho_torch/workflows/components.py` | 1.5 days |
| Implement `train_cdi_model_torch` | 1 day |
| Implement `reassemble_torch` wrapper | 0.5 days |
| Create `TorchModelManager` (D3 overlap) | 1.5 days |
| Test orchestration layer (torch-optional) | 1 day |
| **Total** | **5.5 days** |

---

## 4. Decision Matrix

### 4.1. Evaluation Criteria

| Criterion | Weight | Option A (API) | Option B (Shim) | Notes |
|-----------|--------|---------------|----------------|-------|
| **Maintainability** | 25% | 6/10 | 8/10 | Shim has clearer separation; API adds wrapper layer |
| **TensorFlow Parity** | 25% | 5/10 | 9/10 | Shim mirrors TF structure exactly; API diverges |
| **Implementation Speed** | 15% | 8/10 | 6/10 | API reuses existing code; shim builds new |
| **MLflow Optionality** | 15% | 6/10 | 9/10 | API still coupled to `train.py` MLflow; shim decouples |
| **Test Surface** | 10% | 5/10 | 8/10 | API deep call stack harder to test; shim direct |
| **Backward Compatibility** | 10% | 9/10 | 7/10 | API doesn't touch existing flows; shim may diverge |

**Weighted Scores**:
- Option A: (6×0.25) + (5×0.25) + (8×0.15) + (6×0.15) + (5×0.10) + (9×0.10) = **6.45/10**
- Option B: (8×0.25) + (9×0.25) + (6×0.15) + (9×0.15) + (8×0.10) + (7×0.10) = **7.95/10**

---

## 5. MLflow Dependency Analysis

### 5.1. Current MLflow Coupling (Both Options)

**In `train.py`** (lines 169-210):
```python
mlflow.pytorch.autolog(checkpoint_monitor=val_loss_label)
mlflow.set_experiment(training_config.experiment_name)
with mlflow.start_run() as run:
    # ... training ...
    run_ids['training'] = run.info.run_id
```

**In `inference.py`** (lines 77-94):
```python
tracking_uri = f"file:{os.path.abspath(relative_mlflow_path)}"
mlflow.set_tracking_uri(tracking_uri)
model_uri = f"runs:/{run_id}/model"
loaded_model = mlflow.pytorch.load_model(model_uri)
```

### 5.2. Option A MLflow Strategy

**Approach**: Add `--disable_mlflow` flag to `train.py`, pass through API wrapper

**Pros**:
- Minimal change to existing code (5 lines)
- MLflow users keep current workflow

**Cons**:
- API still indirectly depends on MLflow via `train.py`
- Inference loading still requires MLflow (no checkpoint fallback)

**Implementation**:
```python
# ptycho_torch/train.py:165
if not training_config.disable_mlflow:
    mlflow.pytorch.autolog(...)

# ptycho_torch/api/trainer_api.py
def train(..., enable_mlflow=False):
    config_dict['disable_mlflow'] = not enable_mlflow
    run_ids = torch_train.main(..., existing_config=config_dict)
```

### 5.3. Option B MLflow Strategy

**Approach**: Shim layer defaults to MLflow disabled, enable via orchestration flag

**Pros**:
- Clean separation: Ptychodus never sees MLflow
- Shim can support both Lightning checkpoints AND MLflow URIs

**Cons**:
- Must implement checkpoint loading separately

**Implementation**:
```python
# ptycho_torch/workflows/components.py
def train_cdi_model_torch(..., enable_mlflow=False):
    if enable_mlflow:
        mlflow.pytorch.autolog(...)
        with mlflow.start_run() as run:
            trainer.fit(model, datamodule)
            return {'run_id': run.info.run_id, 'checkpoint': trainer.checkpoint_callback.best_model_path}
    else:
        trainer.fit(model, datamodule)
        return {'checkpoint': trainer.checkpoint_callback.best_model_path}
```

---

## 6. Persistence Strategy Comparison

### 6.1. Option A Persistence

**Format**: MLflow model registry (default) OR Lightning checkpoint (if MLflow disabled)

**TensorFlow Compatibility**: ❌ None - separate archive formats

**Implementation**:
```python
# ptycho_torch/api/trainer_api.py
if enable_mlflow:
    return {'model_uri': f'runs:/{run_id}/model'}
else:
    return {'checkpoint_path': trainer.checkpoint_callback.best_model_path}

# ptycho_torch/api/inference_api.py
def infer(model_path: Union[str, Path], ...):
    if str(model_path).startswith('runs:/'):
        model = mlflow.pytorch.load_model(model_path)
    else:
        model = PtychoPINN_Lightning.load_from_checkpoint(model_path)
```

**Pros**: Reuses Lightning checkpoint format
**Cons**: No TensorFlow archive compatibility, dual loading paths

### 6.2. Option B Persistence

**Format**: Custom `wts.h5.zip`-style archives containing Lightning checkpoint + `params.cfg` snapshot

**TensorFlow Compatibility**: ✅ Possible (see D3 design)

**Implementation**:
```python
# ptycho_torch/model_manager_torch.py (NEW)
class TorchModelManager:
    @staticmethod
    def save(model: PtychoPINN_Lightning, checkpoint_path: Path, out_prefix: Path):
        """Save PyTorch model in wts.h5.zip-compatible format."""
        with zipfile.ZipFile(f"{out_prefix}/wts.h5.zip", 'w') as zf:
            # Lightning checkpoint
            zf.write(checkpoint_path, 'torch_checkpoint.ckpt')
            # params.cfg snapshot (CONFIG-001 compliance)
            with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
                dill.dump(params.cfg.copy(), f)
                zf.write(f.name, 'params.dill')
            # Metadata
            manifest = {'backend': 'pytorch', 'lightning_version': L.__version__}
            with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
                dill.dump(manifest, f)
                zf.write(f.name, 'manifest_torch.dill')
```

**Pros**: Unified archive format, TensorFlow interop possible
**Cons**: More complex implementation (D3 scope)

---

## 7. Config Bridge Touchpoint Comparison

### 7.1. Option A: Bridge in API Wrapper

**Location**: `ptycho_torch/api/trainer_api.py`

**Flow**:
```
Ptychodus (TensorFlow TrainingConfig)
    ↓
API wrapper: update_legacy_dict(params.cfg, config)
    ↓
train.py: builds PyTorch singleton configs from JSON/dict
    ↓
Lightning Trainer
```

**Issue**: Config translated TWICE (TF → params.cfg at API, then params.cfg → PyTorch singletons in train.py)

### 7.2. Option B: Bridge at Orchestration Entry

**Location**: `ptycho_torch/workflows/components.py:run_cdi_example_torch`

**Flow**:
```
Ptychodus (TensorFlow TrainingConfig)
    ↓
Shim: update_legacy_dict(params.cfg, config)
    ↓
Shim: read params.cfg, build Lightning Trainer directly
    ↓
Lightning Trainer
```

**Advantage**: Single translation step, mirrors TensorFlow pattern exactly

---

## 8. Test Surface Implications

### 8.1. Option A Test Strategy

**Test Layers**:
1. Unit tests for API wrapper functions (mock `train.py`)
2. Integration tests for `train.py` (existing)
3. End-to-end Ptychodus integration tests

**Torch-Optional Handling**:
- API wrapper tests use `conftest.py` whitelist (like Phase B/C)
- Must mock Lightning components in API tests

**Complexity**: Medium-High (3 test layers)

### 8.2. Option B Test Strategy

**Test Layers**:
1. Unit tests for `ptycho_torch.workflows.components` functions
2. Integration tests for Lightning orchestration (mock model training)
3. End-to-end Ptychodus integration tests

**Torch-Optional Handling**:
- Orchestration tests use same whitelist pattern
- Direct Lightning mocking (fewer indirection layers)

**Complexity**: Medium (2 primary test layers)

---

## 9. Recommendation

### 9.1. Preferred Option: **B (Low-Level Orchestration Shims)**

**Rationale**:
1. **Higher Parity Score** (7.95 vs 6.45): Better mirrors TensorFlow structure, critical for maintainability
2. **Cleaner MLflow Decoupling**: Shim layer naturally isolates MLflow dependency
3. **Phase C Adapter Fit**: Direct integration with `RawDataTorch` → `PtychoDataContainerTorch` pipeline
4. **Test Simplicity**: Shallow call stack easier to test torch-optionally
5. **Persistence Flexibility**: D3 `TorchModelManager` can support TensorFlow archive compatibility

**Trade-off Accepted**:
- +1.5 days implementation time vs Option A
- Some code duplication with `train.py` (Trainer setup)

**Mitigation**:
- Keep `train.py` for CLI/notebook users (backward compatibility)
- Extract shared Lightning setup into `train_utils.py` helpers
- Document divergence in `docs/workflows/pytorch.md`

### 9.2. Decision Record

**Decision**: Build `ptycho_torch/workflows/components.py` orchestration layer (Option B)

**Owner**: Ralph (engineer loops D2.A-D2.C)

**Target**: Phase D2 implementation

**Reversibility**: Medium (can migrate to Option A later if needed, but persistence format may lock us in)

---

## 10. Implementation Roadmap (Phase D2 Guidance)

### 10.1. Phase D2.A: Scaffold Orchestration Module

**Deliverable**: `ptycho_torch/workflows/components.py` with entry points

**Tasks**:
1. Create module with torch-optional import guards
2. Define `run_cdi_example_torch` signature (mirror TensorFlow)
3. Add config bridge call at entry: `update_legacy_dict(params.cfg, config)`
4. Stub out helper functions (`train_cdi_model_torch`, `reassemble_torch`)

**Artifact**: `phase_d2_scaffold.md` documenting module structure

### 10.2. Phase D2.B: Implement Training Path

**Deliverable**: `train_cdi_model_torch` function with Lightning orchestration

**Tasks**:
1. Accept `TrainingConfig` + RawDataTorch/PtychoDataContainerTorch inputs
2. Build `PtychoDataModule` using Phase C adapters (replace `PtychoDataset`)
3. Configure Lightning Trainer (callbacks, epochs, devices)
4. Optional MLflow via `enable_mlflow` flag (default False)
5. Return checkpoint path + training metadata

**Artifact**: `phase_d2_training.md` with validation log

### 10.3. Phase D2.C: Implement Inference + Stitching Path

**Deliverable**: Inference orchestration + optional stitching

**Tasks**:
1. Load model from checkpoint OR MLflow URI
2. Run inference using Phase C `PtychoDataContainerTorch`
3. Optional stitching via `reconstruct_image_barycentric`
4. Match TensorFlow `save_outputs()` behavior (amplitude/phase PNGs)

**Artifact**: `phase_d2_inference.md` with parity checks

---

## 11. Open Questions for Phase D3

1. **Archive Format**: Should `TorchModelManager` produce TensorFlow-compatible `wts.h5.zip` OR separate `wts_torch.zip`?
2. **Cross-Loading**: Should TensorFlow `ModelManager` be able to load PyTorch archives (and vice versa)?
3. **params.cfg Versioning**: How to handle `params.cfg` schema evolution across TensorFlow/PyTorch boundaries?
4. **Custom Layer Equivalence**: How to document PyTorch `nn.Module` equivalents of TensorFlow Lambda layers?

**Decision Dependency**: D3 persistence design must align with D2 orchestration choice (Option B → unified archive format feasible)

---

**Summary**: Proceed with **Option B (Orchestration Shims)** for Phase D2 implementation, targeting TensorFlow parity and MLflow optionality.
