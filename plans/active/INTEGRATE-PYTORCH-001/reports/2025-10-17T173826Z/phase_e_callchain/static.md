# Callchain Analysis: Backend Selection and Invocation Flow

## Analysis Question

**How does Ptychodus select and invoke the TensorFlow backend, and what would a PyTorch backend selection mechanism require?**

Context: Phase E1.A evidence-only analysis mapping the current TensorFlow workflow invocation to inform PyTorch backend integration design per `specs/ptychodus_api_spec.md` §4.

---

## Candidate Entry Points

| Candidate | Relevance Signals | Confidence | Expected Code Region |
|-----------|------------------|------------|---------------------|
| `ptycho/workflows/components.py::run_cdi_example()` | Main training+inference orchestrator, referenced in spec §4.5 | **High** | Entry point for complete train→infer workflow |
| `ptycho/workflows/components.py::train_cdi_model()` | Training-only orchestrator, called by `run_cdi_example` | **High** | Training delegation layer |
| `ptycho/workflows/components.py::load_inference_bundle()` | Model loading for inference, referenced in spec §4.6 | **High** | Inference entry point |
| `ptycho/model_manager.py::save_multiple_models()` | Model persistence layer, creates `wts.h5.zip` | **Medium** | Persistence backend |
| `ptycho/model_manager.py::load_multiple_models()` | Archive restoration, loads TensorFlow models | **Medium** | Loading backend |

**Ranking**: All three workflow entry points selected (run_cdi_example, train_cdi_model, load_inference_bundle) as they form the primary API surface spec §4.1–4.6 requires.

**Selected Entrypoints**: `run_cdi_example()` (primary), `train_cdi_model()` (training path), `load_inference_bundle()` (inference path)

**Why**: These three functions define the complete TensorFlow backend contract per spec §4. Any PyTorch backend must provide parallel implementations with identical signatures to maintain API parity.

---

## Config Flow

### TensorFlow Backend Config Path

| Stage | Location | Keys/Actions | Anchors |
|-------|----------|--------------|---------|
| **Entry config** | `run_cdi_example(config: TrainingConfig)` | Receives modern dataclass with all params | `ptycho/workflows/components.py:676` |
| **CONFIG-001 gate** | `update_legacy_dict(params.cfg, config)` | Synchronizes `TrainingConfig` → legacy `params.cfg` dict | `ptycho/workflows/components.py:706` |
| **Legacy propagation** | `params.cfg` global dict | All downstream modules read via `params.get()` | `ptycho/params.py:1-100` |
| **Key fields** | `params.cfg` keys | `N`, `gridsize`, `model_type`, `nphotons`, `output_prefix`, etc. | Mapped per `ptycho/config/config.py:275-304` (KEY_MAPPINGS) |

**Critical Finding (CONFIG-001)**: The `update_legacy_dict()` call at line 706 is **mandatory** before any data loading or model construction. Skipping this causes shape mismatches and silent failures.

### PyTorch Backend Config Path (Parallel Implementation)

| Stage | Location | Keys/Actions | Anchors |
|-------|----------|--------------|---------|
| **Entry config** | `run_cdi_example_torch(config: TrainingConfig)` | Same modern dataclass interface | `ptycho_torch/workflows/components.py:127` |
| **CONFIG-001 gate** | `update_legacy_dict(params.cfg, config)` | **Identical gate placement** | `ptycho_torch/workflows/components.py:161` |
| **Config bridge** | `ptycho_torch/config_bridge.py` | Translates PyTorch singleton configs → TensorFlow dataclasses when needed | Phase B deliverable |
| **Inference gate** | `params.cfg.update(params_dict)` inside `load_torch_bundle()` | Restores saved params from archive | `ptycho_torch/model_manager.py:265` |

**Parity Status**: ✅ PyTorch implements CONFIG-001 gates at both training (line 161) and inference (line 265) boundaries, matching TensorFlow's contract.

---

## Core Pipeline Stages

### TensorFlow Workflow Pipeline

| Stage | Purpose | Anchors |
|-------|---------|---------|
| **1. Entry** | `run_cdi_example(train_data, test_data, config, ...)` | `ptycho/workflows/components.py:676` |
| **2. Config sync** | `update_legacy_dict(params.cfg, config)` (CONFIG-001 gate) | Line 706 |
| **3. Data normalization** | `create_ptycho_data_container(train_data)` → `PtychoDataContainer` | Lines 591-595 |
| **4. Training** | `train_cdi_model(train_container, test_container, config)` | Line 709 |
| **5. Stitching (optional)** | `reassemble_cdi_image(test_data, config, ...)` | Lines 714-721 (conditional on `do_stitching`) |
| **6. Return** | `(recon_amp, recon_phase, train_results)` tuple | Line 723 |

### PyTorch Workflow Pipeline (Parallel)

| Stage | Purpose | Anchors | Implementation Status |
|-------|---------|---------|----------------------|
| **1. Entry** | `run_cdi_example_torch(train_data, test_data, config, ...)` | `ptycho_torch/workflows/components.py:127` | ✅ Complete |
| **2. Config sync** | `update_legacy_dict(params.cfg, config)` | Line 161 | ✅ Complete |
| **3. Data normalization** | `_ensure_container(data, config)` → `PtychoDataContainerTorch` | Lines 212-283 | ✅ Complete (Phase C adapters) |
| **4. Training** | `train_cdi_model_torch(train_container, test_container, config)` | Line 166 | 🔴 Stub (Phase D2.B pending) |
| **5. Stitching (optional)** | `_reassemble_cdi_image_torch(test_data, config, ...)` | Lines 173-182 | 🔴 NotImplementedError (Phase D2.C pending) |
| **6. Persistence** | `save_torch_bundle(models, config.output_dir)` | Lines 188-205 | ✅ Complete (Phase D3.B) |
| **7. Return** | `(recon_amp, recon_phase, train_results)` tuple | Line 207 | ✅ Complete |

**Parity Gap Summary**:
- ✅ Entry signatures, config sync, data adapters, persistence → **Complete**
- 🔴 Training orchestration, inference/stitching → **Stubs** (Phase D2.B/C blockers)

---

## Training Path Deep Dive

### TensorFlow Training Delegation

| Stage | Purpose | Anchors |
|-------|---------|---------|
| **Entry** | `train_cdi_model(train_data, test_data, config)` | `ptycho/workflows/components.py:535` |
| **Data containerization** | `create_ptycho_data_container(train_data)` | Lines 591-595 |
| **Probe initialization** | `probe.set_probe_guess(train_container.probeGuess)` | Line 598 |
| **Model construction** | `create_model_with_gridsize(...)` (implicit via train_eval) | `ptycho/model.py:220-560` |
| **Training loop** | `train_pinn.train_eval(train_container, test_container, ...)` | Line 604 |
| **Result packaging** | `{'history': {...}, 'train_container': ..., 'test_container': ...}` | Lines 606-610 |

### PyTorch Training Delegation (Parallel)

| Stage | Purpose | Anchors | Status |
|-------|---------|---------|--------|
| **Entry** | `train_cdi_model_torch(train_data, test_data, config)` | `ptycho_torch/workflows/components.py:326` | ✅ Signature complete |
| **Data containerization** | `_ensure_container(train_data, config)` | Lines 335-340 | ✅ Uses Phase C adapters |
| **Probe initialization** | TODO (architectural decision pending) | Line 148 (commented) | 🔴 Deferred to Phase D2.B |
| **Lightning setup** | `_train_with_lightning(train_container, test_container, config)` | Line 152 | 🔴 Stub (returns placeholder losses) |
| **Result packaging** | Same dict structure as TensorFlow | Lines 342-346 | ✅ Compatible |

**Critical Blocker**: `_train_with_lightning()` stub at `ptycho_torch/workflows/components.py:283-323` must be replaced with full Lightning Trainer integration per Phase D2.B requirements.

---

## Inference Path Deep Dive

### TensorFlow Inference/Stitching

| Stage | Purpose | Anchors |
|-------|---------|---------|
| **Entry** | `reassemble_cdi_image(test_data, config, flip_x, flip_y, transpose, M, coord_scale)` | `ptycho/workflows/components.py:611` |
| **Container normalization** | `create_ptycho_data_container(test_data)` | Lines 637-640 |
| **Model inference** | `nbutils.reconstruct_image(test_container, coord_scale)` | Line 648 |
| **Coordinate transforms** | Apply flip_x/flip_y/transpose/M transformations | Lines 649-665 |
| **Patch reassembly** | `tf_helper.reassemble_position(patches, global_offsets, ...)` | Line 667 |
| **Phase extraction** | `np.angle(recon_complex)` → recon_phase | Line 673 |

### PyTorch Inference/Stitching (Parallel)

| Stage | Purpose | Anchors | Status |
|-------|---------|---------|--------|
| **Entry** | `_reassemble_cdi_image_torch(test_data, config, flip_x, flip_y, transpose, M)` | `ptycho_torch/workflows/components.py:329` | ✅ Signature defined |
| **Full implementation** | All 5 steps (containerization → inference → transforms → reassembly → extraction) | Lines 89-94 | 🔴 Raises NotImplementedError |

**Critical Blocker**: Phase D2.C must implement the full inference pipeline including:
1. PyTorch model inference equivalent to `nbutils.reconstruct_image()`
2. Reassembly logic matching `tf_helper.reassemble_position()` behavior

---

## Persistence Layer

### TensorFlow Model Persistence

| Operation | Entry Point | Archive Format | Anchors |
|-----------|-------------|----------------|---------|
| **Save** | `ModelManager.save_multiple_models(models_dict, base_path)` | Creates `wts.h5.zip` with dual-model structure | `ptycho/model_manager.py:210-290` |
| **Archive layout** | Root: `manifest.dill`, Subdirs: `{model_name}/` with `model.keras`, `params.dill`, `custom_objects.dill`, `model.h5` | TensorFlow .keras format | Lines 235-260 |
| **CONFIG-001 snapshot** | `params.dill` contains full `params.cfg` state at training time | Ensures inference params match training | Line 245 |
| **Load** | `ModelManager.load_multiple_models(base_path)` | Restores models and params from archive | Lines 90-209 |
| **Params restoration** | `params.cfg.update(loaded_params)` from `params.dill` | Critical for shape consistency | Line 119 |

### PyTorch Model Persistence (Parallel)

| Operation | Entry Point | Archive Format | Anchors | Status |
|-----------|-------------|----------------|---------|--------|
| **Save** | `save_torch_bundle(models_dict, base_path, config)` | Creates `wts.h5.zip` with dual-model structure | `ptycho_torch/model_manager.py:95-194` | ✅ Complete (Phase D3.B) |
| **Archive layout** | Root: `manifest.dill` (version='2.0-pytorch'), Subdirs: `{model_name}/` with `model.pth`, `params.dill` | PyTorch state_dict format | Lines 140-165 | ✅ Complete |
| **CONFIG-001 snapshot** | `params.dill` contains `dataclass_to_legacy_dict(config)` | Same spec compliance as TensorFlow | Line 156 | ✅ Complete |
| **Load** | `load_torch_bundle(base_path, model_name)` | Restores params and model (stub) | Lines 197-286 | 🔰 Partial (params ✅, model reconstruction 🔴) |
| **Params restoration** | `params.cfg.update(params_dict)` from `params.dill` | CONFIG-001 compliant | Line 265 | ✅ Complete |

**Parity Status**:
- ✅ Archive format, params snapshot/restoration → **Complete**
- 🔴 Model reconstruction (requires `create_torch_model_with_gridsize()` helper) → **Phase D3.C blocker**

---

## Callgraph Edge List

### TensorFlow Backend Call Graph

| From | To | Purpose | Anchors |
|------|----|---------|---------|
| External caller | `run_cdi_example()` | Orchestrate train+infer workflow | `ptycho/workflows/components.py:676` → `train_cdi_model:709` |
| `run_cdi_example()` | `update_legacy_dict()` | CONFIG-001 gate | Line 706 → `ptycho/config/config.py:288` |
| `run_cdi_example()` | `train_cdi_model()` | Training delegation | Line 709 → `train_cdi_model:535` |
| `train_cdi_model()` | `create_ptycho_data_container()` | Data normalization | Line 591 → `ptycho/loader.py:93` |
| `train_cdi_model()` | `probe.set_probe_guess()` | Probe initialization | Line 598 → `ptycho/probe.py:60` |
| `train_cdi_model()` | `train_pinn.train_eval()` | Core training loop | Line 604 → `ptycho/train_pinn.py:162` |
| `run_cdi_example()` | `reassemble_cdi_image()` | Inference+stitching (conditional) | Line 714 → `reassemble_cdi_image:611` |
| `reassemble_cdi_image()` | `nbutils.reconstruct_image()` | Model inference | Line 648 → `ptycho/nbutils.py:580` |
| `reassemble_cdi_image()` | `tf_helper.reassemble_position()` | Patch reassembly | Line 667 → `ptycho/tf_helper.py:1050` |
| External caller | `load_inference_bundle()` | Load saved model | → `ptycho/workflows/components.py:94` |
| `load_inference_bundle()` | `ModelManager.load_multiple_models()` | Archive restoration | Line 133 → `ptycho/model_manager.py:90` |
| `ModelManager.load_multiple_models()` | `params.cfg.update()` | CONFIG-001 params restoration | Line 119 (implicit) |

### PyTorch Backend Call Graph (Parallel)

| From | To | Purpose | Anchors | Status |
|------|----|---------|---------|---------|
| External caller | `run_cdi_example_torch()` | Orchestrate train+infer workflow | `ptycho_torch/workflows/components.py:127` → `train_cdi_model_torch:166` | ✅ |
| `run_cdi_example_torch()` | `update_legacy_dict()` | CONFIG-001 gate | Line 161 → `ptycho/config/config.py:288` | ✅ |
| `run_cdi_example_torch()` | `train_cdi_model_torch()` | Training delegation | Line 166 → `train_cdi_model_torch:326` | ✅ |
| `train_cdi_model_torch()` | `_ensure_container()` | Data normalization via Phase C adapters | Line 335 → `_ensure_container:212` | ✅ |
| `_ensure_container()` | `RawDataTorch()` | Wrap RawData with PyTorch bridge | Line 243 → `ptycho_torch/raw_data_bridge.py:80` | ✅ |
| `_ensure_container()` | `PtychoDataContainerTorch()` | Torch-optional container | Line 267 → `ptycho_torch/data_container_bridge.py:70` | ✅ |
| `train_cdi_model_torch()` | `_train_with_lightning()` | Lightning training (stub) | Line 152 → `_train_with_lightning:283` | 🔴 Stub |
| `run_cdi_example_torch()` | `_reassemble_cdi_image_torch()` | Inference+stitching (conditional) | Line 173 → `_reassemble_cdi_image_torch:329` | 🔴 NotImplementedError |
| `run_cdi_example_torch()` | `save_torch_bundle()` | Model persistence | Line 197 → `ptycho_torch/model_manager.py:95` | ✅ |
| External caller | `load_inference_bundle_torch()` | Load saved model | → `ptycho_torch/workflows/components.py:223` | ✅ Signature |
| `load_inference_bundle_torch()` | `load_torch_bundle()` | Archive restoration (stub) | Line 254 → `ptycho_torch/model_manager.py:197` | 🔰 Partial |
| `load_torch_bundle()` | `params.cfg.update()` | CONFIG-001 params restoration | Line 265 | ✅ |

---

## Data/Units & Constants

### Key Configuration Fields

| Field | Units/Type | Purpose | Legacy Key | Anchors |
|-------|-----------|---------|------------|---------|
| `N` | pixels (int) | Diffraction pattern size | `N` | `ptycho/config/config.py:72` |
| `gridsize` | dimensionless (int) | Group cardinality (gridsize²) | `gridsize` | Line 73 |
| `model_type` | enum ('pinn', 'supervised') | Physics vs pure supervised | `model_type` | Line 75 |
| `nphotons` | photons (float) | Expected photon count for Poisson model | `nphotons` | Line 95 |
| `output_dir` | pathlib.Path | Save directory for artifacts | `output_prefix` | Line 132 (mapped via KEY_MAPPINGS) |
| `train_data_file` | pathlib.Path | Training NPZ path | `train_data_file_path` | Line 88 |

**Critical**: All Path fields auto-converted to strings by `update_legacy_dict()` per `ptycho/config/config.py:304-309`.

---

## Device/dtype Handling

### TensorFlow Backend

- **Device placement**: Implicit via TensorFlow device placement (CPU/GPU auto-selected)
- **Dtype conventions**:
  - Diffraction: `float32`
  - Coordinates: `float64`
  - Complex ground truth: `complex64` (critical per DATA-001 finding)
  - Offsets: `float64`
- **Anchors**: `ptycho/loader.py:178-186` (dtype conversions in PtychoDataContainer)

### PyTorch Backend

- **Device placement**: Torch-optional with `TORCH_AVAILABLE` flag; falls back to NumPy when unavailable
- **Dtype conventions**: Same as TensorFlow (enforced by Phase C adapters)
  - `PtychoDataContainerTorch` validates complex64 for Y patches (`ptycho_torch/data_container_bridge.py:183-189`)
- **Anchors**: `ptycho_torch/data_container_bridge.py:70-280`

**Parity Status**: ✅ PyTorch replicates TensorFlow dtype requirements via Phase C adapters.

---

## Gaps/Unknowns & Confirmation Plan

### Known Gaps (Phase D2.B/C Blockers)

| Gap ID | Description | Blocking | Confirmation Plan |
|--------|-------------|----------|------------------|
| **D2.B-TRAIN** | Lightning training stub returns placeholder losses | Training workflows | Implement full Lightning Trainer integration; see Phase D2.B roadmap |
| **D2.C-INFER** | Inference/stitching raises NotImplementedError | Inference workflows | Implement PyTorch equivalents for `nbutils.reconstruct_image()` and `tf_helper.reassemble_position()` |
| **D3.C-LOAD** | Model reconstruction stub in `load_torch_bundle()` | Model loading | Create `create_torch_model_with_gridsize()` helper matching TensorFlow signature |
| **D2.B-PROBE** | Probe initialization deferred (architectural decision pending) | Training setup | Resolve whether PyTorch probe handling requires distinct implementation |

### Open Questions for Phase E1.C

1. **Backend Selection Mechanism**: How should Ptychodus **select** between TensorFlow and PyTorch backends?
   - Option A: Config flag (`backend='pytorch'` in TrainingConfig/InferenceConfig)?
   - Option B: Separate reconstructor classes in `PtychoPINNReconstructorLibrary` (spec §4.1)?
   - Option C: Environment variable (`PTYCHOPINN_BACKEND=pytorch`)?

2. **Import Guards**: Should PyTorch workflows be torch-optional at **all** layers, or only at the adapter boundaries?

3. **Fallback Behavior**: If PyTorch backend selected but torch unavailable, should system:
   - Raise immediate error?
   - Fall back to TensorFlow with warning?
   - Fail late (only when torch-specific functions invoked)?

4. **TEST-PYTORCH-001 Coordination**: What test fixtures/selectors does TEST-PYTORCH-001 require from Phase E integration tests?

---

## Summary

### One-Page Answer

**How does Ptychodus select and invoke the TensorFlow backend?**

Currently, there is **no explicit backend selection**. The system defaults to TensorFlow by importing `ptycho.workflows.components` functions (`run_cdi_example`, `train_cdi_model`, `load_inference_bundle`).

**Entry Flow**:
1. External caller (e.g., Ptychodus reconstructor) invokes `run_cdi_example(train_data, test_data, config, ...)`
2. **CONFIG-001 GATE** at line 706: `update_legacy_dict(params.cfg, config)` synchronizes modern dataclass → legacy dict
3. Training delegation: `train_cdi_model()` → `train_pinn.train_eval()` (TensorFlow/Keras)
4. Optional stitching: `reassemble_cdi_image()` → `nbutils.reconstruct_image()` → `tf_helper.reassemble_position()`
5. Persistence: `ModelManager.save_multiple_models()` creates `wts.h5.zip` archive

**PyTorch Parallel**:

The PyTorch backend provides **API-identical entry points** in `ptycho_torch.workflows.components`:
- ✅ `run_cdi_example_torch()`, `train_cdi_model_torch()`, `load_inference_bundle_torch()` → signatures complete
- ✅ CONFIG-001 gates at lines 161 (training) + 265 (inference) → params sync complete
- ✅ Data adapters (`RawDataTorch`, `PtychoDataContainerTorch`) → Phase C complete
- ✅ Persistence (`save_torch_bundle`, `load_torch_bundle` params restoration) → Phase D3.B complete
- 🔴 Training orchestration stub → Phase D2.B pending
- 🔴 Inference/stitching stub → Phase D2.C pending

**First Implementation Step for Phase E1.C**:

Design a **backend selection handshake** allowing Ptychodus to dispatch to either:
```python
# TensorFlow path (existing)
from ptycho.workflows.components import run_cdi_example

# PyTorch path (new)
from ptycho_torch.workflows.components import run_cdi_example_torch as run_cdi_example
```

**Most Likely Implementation Strategy** (per Phase E1.B red tests):
- Add `backend='tensorflow'` parameter to `TrainingConfig`/`InferenceConfig` (default: 'tensorflow')
- Ptychodus reconstructor reads config flag and imports appropriate workflow module
- Guard PyTorch imports with try/except; raise actionable error if torch unavailable when `backend='pytorch'`

**Next Steps** (Phase E1.B):
1. Author torch-optional failing pytest cases asserting backend selection behavior
2. Capture red logs under `phase_e_red_backend_selection.log`
3. Document implementation design in `phase_e_backend_design.md`

---

## Environment Metadata

**File**: `env/trace_env.json`

```json
{
  "commit_sha": "360afa81",
  "branch": "feature/torchapi",
  "analysis_type": "static",
  "trace_timestamp": "2025-10-17T17:38:26Z",
  "python_version": "3.x",
  "platform": "linux",
  "time_budget_minutes": 30,
  "scope_filter": "ptycho, ptycho_torch",
  "tools_used": ["static code analysis", "file:line anchoring"],
  "dynamic_trace": false
}
```

---

**End of Callchain Analysis**
