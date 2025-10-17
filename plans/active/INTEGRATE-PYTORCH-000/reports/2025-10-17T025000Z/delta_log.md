# Delta Log: PyTorch Integration Plan Gaps

**Generated:** 2025-10-17T02:50:00Z
**Basis:** Comparison of `module_inventory.md` against `plans/ptychodus_pytorch_integration_plan.md`
**Scope:** Planning-impacting divergences that require plan document updates

---

## Critical Deltas (Block Phase B Without Resolution)

### 🚨 Delta-1: Configuration Schema Mismatch

**Evidence:** `ptycho_torch/config_params.py:1-50` vs `specs/ptychodus_api_spec.md:220-273`

**Gap:**
- PyTorch `ModelConfig` uses **different field names and types** than TensorFlow spec:
  - `grid_size: Tuple[int, int]` ≠ `gridsize: int`
  - `mode: 'Supervised' | 'Unsupervised'` ≠ `model_type: 'pinn' | 'supervised'`
  - Missing spec-mandated fields: `gaussian_smoothing_sigma`, `probe_scale`, `pad_object`
- PyTorch backend **does not populate `params.cfg`** (no `update_legacy_dict()` equivalent)
- No `KEY_MAPPINGS` translation layer

**Spec Sections Affected:**
- `specs/ptychodus_api_spec.md:11-86` (Configuration system architecture)
- `specs/ptychodus_api_spec.md:220-273` (Field reference tables)

**Plan Sections Requiring Update:**
- `plans/ptychodus_pytorch_integration_plan.md` — Section 2 ("Configuration & Legacy Bridge Alignment")
- Integration assumptions based on TensorFlow config bridge are **invalid** for PyTorch

**Proposed Resolution:**
1. **Option A (Harmonize):** Refactor `ptycho_torch/config_params.py` to match TensorFlow schema exactly
2. **Option B (Document):** Update spec to acknowledge dual schemas + provide translation layer in `ptychodus` integration code
3. **Option C (Shim):** Create `ptycho_torch/config/compat.py` bridge that emulates TensorFlow API surface

**Action Owner:** Phase B.B1 drafting; requires architectural decision before plan revision

---

### 🚨 Delta-2: New High-Level API Layer

**Evidence:** `ptycho_torch/api/base_api.py` (994 lines, not in legacy plan)

**Gap:**
- Introduces `ConfigManager`, `PtychoModel`, `Trainer`, `InferenceEngine` classes wrapping lower-level modules
- Provides MLflow-centric persistence (`save_mlflow()`, `load_from_mlflow()`)
- **Abstracts Lightning orchestration** away from direct model instantiation
- **Not referenced** in `plans/ptychodus_pytorch_integration_plan.md` Phase structure

**Spec Sections Affected:**
- `specs/ptychodus_api_spec.md:129-212` (Reconstructor contract assumes direct module calls)

**Plan Sections Requiring Update:**
- **Phase 0 (Planning)** — Add decision: target `api/` layer or bypass it?
- **Phase 1 (Config Parity)** — If using `api/`, delegate to `ConfigManager` instead of direct dataclass instantiation
- **Phase 2 (Data Pipeline)** — `PtychoDataLoader` encapsulates Lightning DataModule
- **Phase 4 (Persistence)** — `PtychoModel.save_mlflow()` replaces manual archive creation

**Proposed Resolution:**
- **Immediate:** Document `api/` layer in plan overview (Phase A.A2 task)
- **Phase B.B1:** Decide integration strategy (high-level API vs low-level modules) and update plan accordingly

**Action Owner:** Phase A.A2 annotation; Phase B.B1 architectural decision

---

## High-Priority Deltas (Require Plan Updates)

### ⚠️ Delta-3: Data Generation Package

**Evidence:** `ptycho_torch/datagen/` package (3 modules, ~600 lines combined)

**Gap:**
- Provides `from_simulation()`, `simulate_multiple_experiments()` for synthetic data generation
- **Not covered** in legacy plan's data pipeline section
- Duplicates some functionality expected from `ptycho.diffsim` (TensorFlow physics simulation)

**Spec Sections Affected:**
- `specs/data_contracts.md:1-74` (NPZ format requirements)

**Plan Sections Requiring Update:**
- **Phase 2 (Data Pipeline)** — Note `datagen/` package as synthetic data source; cross-reference with TensorFlow `ptycho.diffsim` equivalence

**Proposed Resolution:**
- Add `datagen/` package overview to plan Section 2
- Flag for parity testing: verify `from_simulation()` produces spec-compliant NPZ

**Action Owner:** Phase B.B1 plan revision

---

### ⚠️ Delta-4: Reassembly Module Suite

**Evidence:** `ptycho_torch/reassembly_alpha.py`, `reassembly_beta.py`, `reassembly.py`

**Gap:**
- Implements **vectorized barycentric accumulator** for patch stitching (alternative to TensorFlow's `tf_helper.reassemble_position`)
- Includes **multi-GPU inference** with DataParallel support
- Performance profiling infrastructure (inference time vs assembly time tracking)
- **Not documented** in legacy plan's inference section

**Spec Sections Affected:**
- `specs/ptychodus_api_spec.md:176-178` (Stitching contract references TensorFlow helpers)

**Plan Sections Requiring Update:**
- **Phase 3 (Inference Workflow)** — Document reassembly strategy divergence
- **Phase E (Testing)** — Add parity test: compare PyTorch barycentric output vs TensorFlow `reassemble_position` on synthetic data

**Proposed Resolution:**
- Update plan to note alternative reassembly implementation
- Add to test plan: numerical parity check for stitching results

**Action Owner:** Phase B.B1 plan revision; Phase E test plan expansion

---

### ⚠️ Delta-5: Lightning + MLflow Orchestration

**Evidence:** `ptycho_torch/train.py:23-226` (Lightning `Trainer`, MLflow autologging)

**Gap:**
- Training uses **PyTorch Lightning conventions** (callbacks, DataModule, DDP strategy)
- TensorFlow backend uses `ptycho.workflows.components.run_cdi_example()` direct orchestration
- **Multi-stage training** logic embedded in `train.py` (stage_1/2/3_epochs with physics weight scheduling)
- Legacy plan assumes TensorFlow-style orchestration

**Spec Sections Affected:**
- `specs/ptychodus_api_spec.md:185-191` (Training workflow contract)

**Plan Sections Requiring Update:**
- **Phase 1 (Config Parity)** — Note Lightning-specific `TrainingConfig` fields (`stage_1/2/3_epochs`, `scheduler`, etc.)
- **Phase 2 (Data Pipeline)** — `PtychoDataModule` replaces manual data grouping calls

**Proposed Resolution:**
- Update plan to document Lightning orchestration divergence
- Clarify whether `ptychodus` integration will invoke Lightning trainer or lower-level model API

**Action Owner:** Phase B.B1 architectural decision + plan revision

---

## Low-Priority Deltas (Informational)

### ℹ️ Delta-6: Notebooks Addition

**Evidence:** `ptycho_torch/notebooks/` (6 notebooks)

**Gap:**
- Notebooks added for analysis, FRC calculation, probe similarity, supervised dataset creation
- Useful for reference but **not integration-critical**

**Proposed Resolution:**
- Note in plan as "Reference materials" section; omit from integration scope

---

### ℹ️ Delta-7: Memory-Mapped Dataloader

**Evidence:** `ptycho_torch/dset_loader_pt_mmap.py`

**Gap:**
- PyTorch uses **memory-mapped datasets** for efficient large-scale data loading
- TensorFlow backend uses in-memory `RawData` objects
- Performance characteristics differ

**Proposed Resolution:**
- Note in plan as implementation detail; verify that memory-mapped loading satisfies `specs/data_contracts.md` NPZ requirements

---

## Legacy Plan Section Coverage Analysis

| Legacy Plan Section | Coverage Status | Notes |
|-------------------|----------------|-------|
| **Phase 0: Planning** | ❌ Missing | No mention of `api/` layer decision |
| **Phase 1: Config Parity** | ⚠️ Outdated | Assumes TensorFlow config schema |
| **Phase 2: Data Pipeline** | ⚠️ Partial | Missing `datagen/` package, memory-mapped loader |
| **Phase 3: Inference** | ⚠️ Outdated | Missing reassembly module documentation |
| **Phase 4: Persistence** | ⚠️ Partial | Missing MLflow-centric API layer |
| **Phase 5: Testing** | ✅ Adequate | Test plan structure still valid |

---

## Recommended Phase B.B1 Actions

1. **Architectural Decisions (Block Plan Refresh):**
   - [ ] Decision: Use `api/` layer or bypass for `ptychodus` integration?
   - [ ] Decision: Harmonize config schemas or document dual-schema approach?
   - [ ] Decision: Lightning orchestration or direct model API for training integration?

2. **Plan Document Updates:**
   - [ ] Add `api/` layer overview to Phase 0
   - [ ] Revise Phase 1 with PyTorch config schema details
   - [ ] Update Phase 2 with `datagen/` and memory-mapped dataloader
   - [ ] Document reassembly divergence in Phase 3
   - [ ] Add MLflow persistence notes to Phase 4

3. **Spec Alignment:**
   - [ ] Flag `specs/ptychodus_api_spec.md` for potential dual-schema amendment
   - [ ] Cross-reference config field tables (lines 220-273) against `ptycho_torch/config_params.py`

4. **Test Plan Expansion:**
   - [ ] Add config schema translation parity test
   - [ ] Add reassembly output parity test (PyTorch barycentric vs TensorFlow `reassemble_position`)
   - [ ] Add MLflow persistence round-trip test

---

## File Pointers for Phase A.A3 (Spec Touchpoints)

- `specs/ptychodus_api_spec.md:11-29` — ModelConfig field requirements (conflicts with `ptycho_torch/config_params.py:36-58`)
- `specs/ptychodus_api_spec.md:220-273` — Field reference tables (schema mismatch)
- `specs/data_contracts.md:1-74` — NPZ format (verify `datagen/` compliance)
- `ptycho_torch/api/base_api.py:1-994` — New API layer (not in spec)
- `ptycho_torch/config_params.py:1-150` — PyTorch config schema (diverges from spec)

---

## Next Steps (Phase A.A3)

1. Review this delta_log.md with architectural stakeholders
2. Make architectural decisions (API layer, config schema, orchestration)
3. Proceed to Phase B.B1 plan redline drafting with decisions documented
