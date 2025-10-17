# Phase A Summary — PyTorch Backend Integration Parity Baseline

**Initiative:** INTEGRATE-PYTORCH-001
**Phase:** A — Refresh Parity Baseline
**Date:** 2025-10-17
**Artifacts:** `parity_map.md`, `glossary_and_ownership.md` (generated)

---

## Objectives Completed

✅ **Task A1:** Inventoried current TensorFlow workflow touchpoints used by Ptychodus
✅ **Task A2:** Audited PyTorch equivalents and identified gaps
✅ **Task A3:** Defined initiative glossary + owner map

---

## Key Findings

### 1. Integration Maturity Assessment

**TensorFlow Backend:** ✅ Fully integrated with Ptychodus
- Complete dataclass-driven configuration
- RawData + PtychoDataContainer pipeline
- ModelManager persistence with `.h5.zip` archives
- Workflow orchestration via `run_cdi_example()`
- Ptychodus reconstructor fully wired

**PyTorch Backend:** ⚠️ Standalone implementation, NOT Ptychodus-compatible
- Singleton configuration (no dataclass ingestion)
- Memory-mapped dataset (incompatible with RawData API)
- Lightning training (no orchestration layer)
- MLflow checkpoints (incompatible with ModelManager)
- **No Ptychodus integration hooks**

### 2. Critical Path Blockers

**7 gaps identified**, organized by blocking impact:

| Gap | Blocks | Phase | Priority |
|:----|:-------|:------|:---------|
| Configuration dataclass bridge | Everything | B | **CRITICAL** |
| RawDataTorch compatibility | Training + Inference | C | **CRITICAL** |
| PtychoDataContainerTorch | Model execution | C | **CRITICAL** |
| Workflow orchestration | Ptychodus train() | D | HIGH |
| Model persistence shim | Ptychodus open_model() | D | HIGH |
| Inference script | Ptychodus reconstruct() | D | HIGH |
| Dual-backend wiring | Production use | E | MEDIUM |

**All gaps are serial dependencies.** Cannot proceed to Phase C until Phase B completes.

### 3. Configuration Bridge Deep Dive

**Why this is the #1 blocker:**

1. **Global state contract:** Legacy modules (`ptycho/raw_data.py`, `ptycho/loader.py`, `ptycho/model.py`) read `ptycho.params.cfg` at runtime
2. **Dataclass-driven API:** Ptychodus instantiates `ModelConfig`, `TrainingConfig`, `InferenceConfig` dataclasses per spec
3. **Bridge function missing:** PyTorch has NO equivalent to `update_legacy_dict(params.cfg, config)`
4. **Cascade failure:** Without bridge, every downstream module receives incorrect/missing configuration

**Example failure scenario:**
```python
# Ptychodus call
config = InferenceConfig(model=ModelConfig(N=64, gridsize=2), ...)
# PyTorch backend: NO BRIDGE CALL
ptycho_torch.workflows.run_inference(config)  # How does PyTorch ingest this?

# Downstream failure
from ptycho.raw_data import RawData
data.generate_grouped_data(...)  # Reads params.cfg['gridsize'] → UNSET → crash
```

**Solution direction (Phase B):**
- Create `ptycho_torch/config_bridge.py` with `ingest_training_config(config: TrainingConfig)`
- Call `update_legacy_dict(ptycho.params.cfg, config)` to populate global state
- Optionally populate PyTorch singletons for backward compatibility
- Unit test: dataclass → `params.cfg` → downstream module reads correct values

---

## Risks & Mitigations

### Risk 1: Lightning + MLflow Dependencies

**Problem:** Ptychodus GUI cannot assume Lightning/MLflow are available; training may run on constrained systems

**Mitigation:**
- Make Lightning/MLflow optional imports in `ptycho_torch/train.py`
- Provide fallback: pure PyTorch training loop without Lightning when unavailable
- Document environment requirements in `docs/workflows/pytorch.md`

### Risk 2: Memory-Mapped Tensor Compatibility

**Problem:** PyTorch uses `tensordict.MemoryMappedTensor`; TensorFlow uses NumPy memmap. Cache formats differ.

**Mitigation:**
- `RawDataTorch` wrapper MUST delegate neighbor grouping to existing `ptycho/raw_data.py`
- Reuse `.groups_cache.npz` files for group discovery
- Only memory-map diffraction stacks; keep coordinates/metadata in shared format

### Risk 3: Persistence Format Divergence

**Problem:** Lightning `.ckpt` ≠ TensorFlow `.h5.zip`. Ptychodus expects `MODEL_FILE_NAME = 'wts.h5.zip'`.

**Mitigation Strategy (two options):**

**Option A: Unified Archive Format (recommended)**
- Create `ptycho_torch/persistence.py` wrapping Lightning checkpoint + `params.cfg` bundle in `.h5.zip`-compatible archive
- Ptychodus sees consistent file extension across backends
- Pro: Minimal Ptychodus changes
- Con: Additional serialization layer

**Option B: Dual File Filter**
- Update Ptychodus `MODEL_FILE_NAME` to accept `.h5.zip` OR `.ckpt`
- Backend selection based on file extension
- Pro: Simpler PyTorch implementation
- Con: Ptychodus UI complexity

**Recommendation:** Option A for Phase D

### Risk 4: Test Fixture Availability

**Problem:** TEST-PYTORCH-001 and INTEGRATE-PYTORCH-001 need shared minimal datasets for testing

**Mitigation:**
- Coordinate with TEST-PYTORCH-001 to build fixture under `tests/fixtures/pytorch_integration/`
- Fixture requirements: ~10 diffraction patterns, single probe, known ground truth
- Store as canonical NPZ format consumable by both backends

### Risk 5: Incomplete Spec Coverage

**Problem:** `specs/ptychodus_api_spec.md` documents TensorFlow behavior but lacks PyTorch-specific guidance

**Mitigation:**
- Phase D/E loops MUST update spec with PyTorch-specific sections
- Document backend selection mechanism
- Clarify persistence format options
- Update config field tables (§5) if PyTorch adds settings

---

## Unresolved Questions

### Q1: Should PyTorch backend support incremental model updates?

**Context:** TensorFlow backend allows loading base model and fine-tuning. Lightning checkpoints include optimizer state.

**Decision Point:** Phase D (persistence design)
**Options:**
1. Support full checkpoint (weights + optimizer) for resumable training
2. Export inference-only weights (lighter, simpler)
3. Both, with Ptychodus UI selecting mode

**Recommendation:** Defer to Phase D; start with inference-only for MVP

### Q2: How should backend selection be exposed in Ptychodus?

**Context:** `PtychoPINNReconstructorLibrary` currently returns one reconstructor. Need dual-backend support.

**Decision Point:** Phase E (integration)
**Options:**
1. Environment variable (`PTYCHOPINN_BACKEND=pytorch|tensorflow`)
2. Settings UI dropdown
3. Auto-detect based on available dependencies

**Recommendation:** Option 1 for Phase E MVP; Option 2 for user-facing release

### Q3: Can we share tests between backends?

**Context:** Both backends should satisfy same acceptance criteria

**Decision Point:** Phase E (test expansion)
**Approach:**
- Parametrize integration tests with `@pytest.mark.parametrize("backend", ["tensorflow", "pytorch"])`
- Shared fixtures for data preparation
- Backend-specific mocks for persistence layer

**Recommendation:** Coordinate with TEST-PYTORCH-001 owner

---

## Metrics & Evidence

### Artifact Inventory

| Artifact | Purpose | Location |
|:---------|:--------|:---------|
| `parity_map.md` | Exhaustive TF↔PyTorch touchpoint catalog | `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T020000Z/` |
| `glossary_and_ownership.md` | Terminology + task ownership map | `plans/active/INTEGRATE-PYTORCH-001/` |
| This summary | Phase A findings + risks + next steps | `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T020000Z/` |

### Source Citation Count

- **Spec references:** 12 (`specs/ptychodus_api_spec.md` sections §4.1-§5.3)
- **TensorFlow file:line pointers:** 23
- **PyTorch file:line pointers:** 15
- **Gap analysis entries:** 7 (categorized by criticality)

### Phase A Completeness

✅ **100% of Phase A tasks completed**
- A1: TensorFlow touchpoints inventoried (23 integration points)
- A2: PyTorch equivalents audited (15 modules reviewed)
- A3: Glossary + ownership defined (9 terms, 13 tasks mapped)

---

## Recommended Next Steps

### Immediate (Phase B — Configuration Bridge)

**Goal:** Enable Ptychodus to configure PyTorch backend using dataclasses

**Entry Point:** `plans/active/INTEGRATE-PYTORCH-001/implementation.md` Phase B tasks

**Suggested Loop Sequence:**

1. **Loop 1 (Design):** Author `reports/<timestamp>/config_bridge_design.md`
   - Map every field from `TrainingConfig`/`InferenceConfig` to singleton keys
   - Document `KEY_MAPPINGS` equivalent for dotted legacy keys
   - Specify `update_legacy_dict` call order

2. **Loop 2 (TDD):** Add `tests/torch/test_config_bridge.py`
   - Failing test: `test_dataclass_to_singleton_parity()`
   - Assert `ptycho.params.cfg['gridsize']` == dataclass value after bridge call

3. **Loop 3 (Implement):** Create `ptycho_torch/config_bridge.py`
   - `ingest_training_config(config: TrainingConfig) -> None`
   - Populate singletons + call `update_legacy_dict`
   - Pass test from Loop 2

4. **Loop 4 (Validate):** Run parity check
   - Compare `ptycho.params.cfg` dumps from TF and PyTorch workflows
   - Store diff in `reports/<timestamp>/cfg_diff.txt`
   - Investigate any mismatches

### Short-Term (Phase C — Data Pipeline)

**Prerequisite:** Phase B complete
**Goal:** PyTorch workflows can consume `RawData` and produce model-ready tensors

**Suggested First Loop:**
- Design `RawDataTorch` wrapper specification
- Document API: `RawDataTorch.from_raw_data(tf_rawdata)` delegation pattern
- Clarify caching strategy (reuse `.groups_cache.npz` vs new format)

### Medium-Term (Phase D — Workflows + Persistence)

**Prerequisite:** Phases B + C complete
**Goal:** Ptychodus can train and save PyTorch models

**Critical Decisions:**
- Persistence format (unified archive vs dual filter)
- MLflow/Lightning dependency handling
- Return value contract for `run_cdi_example_torch()`

### Long-Term (Phase E — Integration + Testing)

**Prerequisite:** Phases B-D complete
**Goal:** Ptychodus integration tests pass with PyTorch backend

**Success Criteria:**
- `pytest tests/torch/test_integration_workflow.py` passes (to be created)
- Dual-backend parametrized tests run in CI
- Documentation updated (`docs/workflows/pytorch.md`, spec updates)

---

## Phase A Exit Criteria

✅ **All criteria satisfied:**
- Consolidated parity document saved (`parity_map.md`)
- Reviewed against spec sections (`specs/ptychodus_api_spec.md §4-5`)
- Glossary and ownership map complete
- Risks and mitigations documented
- Next steps clearly defined

**Phase A Status:** ✅ **COMPLETE**

**Proceed to Phase B:** Configuration & Legacy Bridge Alignment

---

## Appendix: File Inventory

### Source Files Read (38 total)

**Specifications:**
- `specs/ptychodus_api_spec.md`
- `specs/data_contracts.md`

**TensorFlow Implementation:**
- `ptycho/config/config.py`
- `ptycho/workflows/components.py`
- `ptycho/loader.py`
- `ptycho/raw_data.py`
- `ptycho/model_manager.py`
- `ptycho/model.py`
- `ptycho/train_pinn.py`
- `ptycho/tf_helper.py`

**PyTorch Implementation:**
- `ptycho_torch/config_params.py`
- `ptycho_torch/train.py`
- `ptycho_torch/dset_loader_pt_mmap.py`
- `ptycho_torch/model.py`
- `ptycho_torch/helper.py`
- `ptycho_torch/patch_generator.py`

**Documentation:**
- `docs/architecture.md`
- `docs/workflows/pytorch.md`
- `docs/DEVELOPER_GUIDE.md`
- `docs/findings.md`
- `plans/active/INTEGRATE-PYTORCH-001/implementation.md`
- `plans/pytorch_integration_test_plan.md`

### Commands Executed

None (documentation-only loop per `input.md` constraints)

### Test Commands (for future loops)

```bash
# Phase B validation
pytest tests/torch/test_config_bridge.py -v

# Phase C validation
pytest tests/torch/test_data_container.py -k cache -v

# Phase D validation
pytest tests/torch/test_integration_workflow.py -v

# Full regression (Phase E)
pytest tests/torch/ -v
```

---

**Loop Output Complete.** Proceeding to update `docs/fix_plan.md` Attempts History.
