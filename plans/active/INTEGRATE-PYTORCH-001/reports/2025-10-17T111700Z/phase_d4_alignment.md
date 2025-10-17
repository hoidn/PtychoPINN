# Phase D4 Alignment Narrative — TEST-PYTORCH-001 Activation Strategy & Ownership

**Initiative:** INTEGRATE-PYTORCH-001 Phase D4.A1
**Date:** 2025-10-17
**Status:** Planning Document
**Purpose:** Define activation strategy for TEST-PYTORCH-001 and document ownership boundaries

---

## Executive Summary

This document captures the handoff strategy between INTEGRATE-PYTORCH-001 (adapter implementation) and TEST-PYTORCH-001 (integration harness), clarifies responsibility ownership, and defines activation criteria for spinning TEST-PYTORCH-001 into active execution.

**Key Decision:** TEST-PYTORCH-001 activation is gated on INTEGRATE-PYTORCH-001 Phase D4.B readiness — specifically, torch-optional regression tests must exist and demonstrate baseline failures before the integration harness can be meaningfully authored.

**Ownership Principle:** INTEGRATE-PYTORCH-001 owns adapter-level parity (config bridge, data pipeline, persistence) while TEST-PYTORCH-001 owns end-to-end workflow validation (train → save → load → infer subprocess orchestration).

---

## Context: Current State (2025-10-17)

### INTEGRATE-PYTORCH-001 Progress

**Completed Phases:**
- **Phase B** (Configuration Bridge): `ptycho_torch/config_bridge.py` operational with 43/43 parity tests passing (`tests/torch/test_config_bridge.py`)
  - Covers 38 spec-required fields (ModelConfig: 11, TrainingConfig: 18, InferenceConfig: 9)
  - Torch-optional execution confirmed (no PyTorch runtime required)
  - CONFIG-001 compliance validated via baseline comparison test
  - Artifacts: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T050930Z/` (parity green plan)

- **Phase C** (Data Pipeline): `RawDataTorch`, `PtychoDataContainerTorch`, `MemmapDatasetBridge` adapters functional
  - 188/188 tests passing including full data contract validation
  - Delegation to TensorFlow `RawData.generate_grouped_data()` ensures zero grouping logic duplication
  - Cache-free deterministic generation via seed parameter
  - Artifacts: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T084500Z/` (memmap bridge)

- **Phase D1-D3** (Workflow Orchestration & Persistence): Orchestration scaffold + persistence shims operational
  - `ptycho_torch/workflows/components.py`: `run_cdi_example_torch`, `train_cdi_model_torch`, `load_inference_bundle_torch` entry points
  - `ptycho_torch/model_manager.py`: `save_torch_bundle`, `load_torch_bundle` persistence layer
  - CONFIG-001 guard at all entry points (automatic `update_legacy_dict` invocation)
  - Artifacts: D2 scaffold (`reports/2025-10-17T091450Z/`), D3 persistence (`reports/2025-10-17T110500Z/`)

**Outstanding Work (Phase D4):**
- **D4.A** (Current): Alignment narrative + selector map (this document)
- **D4.B**: Author failing regression tests for persistence + orchestration
- **D4.C**: Turn regression tests green + prepare TEST-PYTORCH-001 handoff package

### TEST-PYTORCH-001 Status

**Current State:** `pending` (ledger entry exists, no active implementation)

**Charter:** `plans/pytorch_integration_test_plan.md`
- Goal: Create subprocess-based integration test mirroring `tests/test_integration_workflow.py` (TensorFlow baseline)
- Scope: train → save → load → infer cycle with real NPZ data
- Environment: CPU-only, MLflow-disabled, <2min runtime

**Blockers:**
1. Requires functional PyTorch persistence layer (D3 ✅ but not validated via failing tests)
2. Requires torch-optional regression baseline (D4.B pending)
3. Needs fixture strategy aligned with INTEGRATE-PYTORCH-001 adapter contracts

---

## Ownership & Responsibility Matrix

| Responsibility Domain | Owner | Scope | Exit Criteria |
|-----------------------|-------|-------|---------------|
| **Configuration Bridge** | INTEGRATE-PYTORCH-001 | Translation PyTorch configs → TensorFlow dataclasses → `params.cfg`; KEY_MAPPINGS; override validation | 43/43 parity tests passing; baseline comparison test green |
| **Data Pipeline Adapters** | INTEGRATE-PYTORCH-001 | `RawDataTorch`, `PtychoDataContainerTorch`, `MemmapDatasetBridge`; delegation to TF grouping | 188/188 tests passing; DATA-001 compliance verified |
| **Persistence Shims** | INTEGRATE-PYTORCH-001 | `save_torch_bundle` / `load_torch_bundle`; wts.h5.zip compatibility; dual-model archives | D3 tests passing; CONFIG-001 params restoration validated |
| **Orchestration Wrappers** | INTEGRATE-PYTORCH-001 | `run_cdi_example_torch`, `train_cdi_model_torch`; Lightning delegation stubs; CONFIG-001 guards | D2 tests passing; entry point smoke tests green |
| **Regression Test Harness** | INTEGRATE-PYTORCH-001 (D4.B) | Torch-optional adapter-level tests (`tests/torch/test_model_manager.py`, `test_workflows_components.py`); baseline failure capture | D4.B red logs captured; failing tests documented |
| **Integration Test Suite** | TEST-PYTORCH-001 | End-to-end subprocess orchestration (`tests/test_pytorch_integration.py`); fixture management; CI/CD integration | TensorFlow parity achieved; <2min CPU runtime |
| **Fixture Strategy** | TEST-PYTORCH-001 | Minimal NPZ + probe datasets under `tests/fixtures/pytorch_integration/`; deterministic seeds | Fixtures consumable by both TF and PyTorch backends |
| **MLflow Toggles** | TEST-PYTORCH-001 | Environment overrides (`MLFLOW_TRACKING_URI=memory`); CLI flag implementation (`--disable_mlflow`) | No external service dependencies in CI |

**Critical Handoff Point:** INTEGRATE-PYTORCH-001 D4.C delivers torch-optional regression tests (passing) → TEST-PYTORCH-001 consumes as baseline and extends to subprocess-level integration.

---

## Activation Strategy for TEST-PYTORCH-001

### Phase 1: Dependency Resolution (INTEGRATE-PYTORCH-001 D4.B → D4.C)

**Prerequisite Work:**
1. INTEGRATE-PYTORCH-001 D4.B completes torch-optional regression tests:
   - Persistence round-trip tests (`test_model_manager.py::TestLoadTorchBundle::test_archive_roundtrip`)
   - Orchestration smoke tests (`test_workflows_components.py::TestWorkflowsComponentsRun`)
   - Manifest validation tests (dual-model bundle structure)
   - CONFIG-001 restoration tests (params.cfg snapshot roundtrip)

2. INTEGRATE-PYTORCH-001 D4.C turns regression tests green:
   - Implements missing glue (model reconstruction in `load_torch_bundle`)
   - Resolves any cross-backend compatibility issues
   - Captures green logs + artifact trees

**Output Artifacts:**
- `plans/active/INTEGRATE-PYTORCH-001/reports/<ts>/phase_d4_handoff.md` (D4.C3)
- Pytest selectors for regression baseline (D4.A3, this document's companion)
- Green test logs demonstrating adapter-level functionality

**Decision Gate:** Once D4.C complete, TEST-PYTORCH-001 activation is unblocked.

---

### Phase 2: TEST-PYTORCH-001 Activation (Post D4.C)

**Trigger Event:** INTEGRATE-PYTORCH-001 Phase D4 marked complete in `docs/fix_plan.md`

**Activation Steps:**
1. **Create Active Plan:**
   - Copy `plans/pytorch_integration_test_plan.md` → `plans/active/TEST-PYTORCH-001/implementation.md`
   - Add phased checklist (Phases A-E per TDD methodology)
   - Update with references to INTEGRATE-PYTORCH-001 artifacts (D4 handoff package)

2. **Align Fixture Requirements:**
   - Review INTEGRATE-PYTORCH-001 adapter contracts (NPZ schema, probe format, memmap expectations)
   - Build minimal fixture dataset (recommendation: subsample 10-20 patterns from existing fly dataset)
   - Validate fixtures consumable by both `ptycho_torch.MemmapDatasetBridge` and TensorFlow `RawData.from_file`

3. **Define Test Scope (Initial MVP):**
   - **Train Phase:** Invoke `python -m ptycho_torch.train` with:
     - `--ptycho_dir <tmpdir>/ptycho`
     - `--probe_dir <tmpdir>/probes`
     - `--max_epochs 1` (fast runtime)
     - `--device cpu` (CI compatibility)
     - `--disable_mlflow` (no external services)
   - **Save Phase:** Verify checkpoint + `wts.h5.zip` archive existence
   - **Load Phase:** Call `load_torch_bundle(archive_path)` in subprocess
   - **Infer Phase:** Run `trainer.predict` or manual forward pass on single batch
   - **Validation:** Assert finite output tensor, expected shapes, no stderr errors

4. **Extend Regression Coverage:**
   - Incorporate INTEGRATE-PYTORCH-001 D4 regression selectors as baseline
   - Add subprocess orchestration layer on top
   - Test cross-process persistence (serialize in Train subprocess, deserialize in Infer subprocess)

---

### Phase 3: Coordination & Handoff (D4.C → TEST-PYTORCH-001)

**Handoff Deliverables (from INTEGRATE-PYTORCH-001 D4.C3):**
1. **Selector Map:** Authoritative pytest commands for adapter-level regression tests (companion document to this narrative)
2. **Environment Matrix:** Required env vars (`CUDA_VISIBLE_DEVICES`, `MLFLOW_TRACKING_URI`), optional PyTorch toggle
3. **Artifact Expectations:** Archive structure, log formats, params snapshot schemas
4. **Known Limitations:** Documented gaps (e.g., model reconstruction stubs, Lightning trainer stubs)

**TEST-PYTORCH-001 Inputs:**
1. Green regression tests from D4.C (establishes adapter-level baseline)
2. Selector map with runtime expectations (<60s per selector on CPU)
3. Fixture requirements (NPZ schema, probe format, memmap layout)
4. CONFIG-001 enforcement patterns (from INTEGRATE-PYTORCH-001 evidence)

**Shared Responsibilities:**
- **Fixture Maintenance:** TEST-PYTORCH-001 owns fixtures but must coordinate schema changes with INTEGRATE-PYTORCH-001
- **Torch-Optional Logic:** Both initiatives share `tests/conftest.py` whitelist; coordinate additions via `docs/fix_plan.md`
- **Spec Updates:** If integration testing reveals spec gaps, raise issues in both initiative plans

---

## Dependency Gating: D4.B Readiness Criteria

TEST-PYTORCH-001 **CANNOT** meaningfully activate until INTEGRATE-PYTORCH-001 D4.B produces failing regression tests. Rationale:

1. **Test-First Discipline:** Integration tests should build on validated adapter-level tests (TDD methodology per `docs/DEVELOPER_GUIDE.md`)
2. **Baseline Establishment:** Failing tests document expected behavior before implementation stabilizes
3. **Scope Clarity:** Red tests define integration harness requirements (fixture shapes, env vars, assertion patterns)

**D4.B Readiness Checklist:**
- [ ] `tests/torch/test_model_manager.py::TestLoadTorchBundle::test_archive_roundtrip` exists and fails (or passes if D4.C already complete)
- [ ] `tests/torch/test_workflows_components.py::TestWorkflowsComponentsRun::test_run_cdi_example_invokes_training` exists and passes
- [ ] Persistence red logs captured (`phase_d4_red_persistence.log`)
- [ ] Orchestration red logs captured (`phase_d4_red_workflows.log`)
- [ ] D4.B3 summary artifact documents prerequisites for green phase

**Once D4.B Complete:** TEST-PYTORCH-001 can begin fixture authoring and integration test scaffolding in parallel with INTEGRATE-PYTORCH-001 D4.C (turning tests green).

**Once D4.C Complete:** TEST-PYTORCH-001 is fully unblocked for subprocess-level integration test authoring.

---

## Open Questions & Decision Log

### Q1: MLflow Dependency Handling (from Stakeholder Brief Q7)

**Question:** Are Lightning and MLflow mandatory dependencies for TEST-PYTORCH-001, or should the integration test support optional fallback paths?

**Status:** Deferred to TEST-PYTORCH-001 Phase A (fixture design)

**Recommendation:** TEST-PYTORCH-001 should test with MLflow **disabled** (`MLFLOW_TRACKING_URI=memory`) to avoid external service dependencies in CI. Lightning is mandatory (core orchestration abstraction).

**Rationale:**
- INTEGRATE-PYTORCH-001 D2 stub implementation already demonstrates Lightning delegation
- MLflow autologging can be disabled via env var or CLI flag (TEST-PYTORCH-001 to implement `--disable_mlflow` if missing)
- CI environments should not require network access for test execution

**Impact:** TEST-PYTORCH-001 fixtures should not assume MLflow artifact structure; rely on `wts.h5.zip` archives from `save_torch_bundle` instead.

---

### Q2: Fixture Ownership & Maintenance

**Question:** Who owns the fixture datasets under `tests/fixtures/pytorch_integration/`, and how are schema updates coordinated?

**Decision:** TEST-PYTORCH-001 owns fixture files but must coordinate schema changes with INTEGRATE-PYTORCH-001 when data contracts evolve.

**Process:**
1. TEST-PYTORCH-001 authors initial fixtures (minimal NPZ + probe, subsample of fly dataset)
2. Fixtures must conform to `specs/data_contracts.md` (canonical NPZ schema)
3. If INTEGRATE-PYTORCH-001 adapter changes require fixture updates (e.g., new required keys), raise issue in TEST-PYTORCH-001 plan and coordinate schema migration
4. Both backends (TensorFlow and PyTorch) should validate fixtures for cross-backend compatibility

**Validation Check:** Fixtures should be loadable by:
- `ptycho.raw_data.RawData.from_file(fixture_path)`
- `ptycho_torch.MemmapDatasetBridge(fixture_path, config)`

---

### Q3: Regression Test Scope for D4.B

**Question:** Which specific torch-optional regression tests must exist before TEST-PYTORCH-001 activation?

**Decision:** Minimum viable regression coverage (D4.B):
1. **Persistence Tests** (`tests/torch/test_model_manager.py`):
   - `test_archive_roundtrip`: Save dual-model bundle → load → verify params.cfg restoration
   - `test_manifest_validation`: Verify manifest.dill structure + version tag
   - `test_params_snapshot`: Validate CONFIG-001 fields in params.dill
   - `test_malformed_archive_error_handling`: Graceful degradation for corrupted archives

2. **Orchestration Tests** (`tests/torch/test_workflows_components.py`):
   - `test_run_cdi_example_invokes_training`: Verify train delegation + optional stitching
   - `test_ensure_container_normalization`: Validate input type handling (RawData, RawDataTorch, PtychoDataContainerTorch)
   - `test_config_bridge_invocation`: Confirm `update_legacy_dict` called at entry points

**Rationale:** These tests establish adapter-level baseline. TEST-PYTORCH-001 extends by adding subprocess orchestration layer.

---

### Q4: TEST-PYTORCH-001 Runtime Budget

**Question:** What is the target runtime for the integration test suite?

**Decision:** <2 minutes total on CPU-only environment (per `plans/pytorch_integration_test_plan.md` acceptance criteria)

**Breakdown:**
- Fixture setup: <10s (copy files to tempdir)
- Train subprocess: <60s (1 epoch, batch_size=4, CPU)
- Save + load: <10s (archive I/O)
- Infer subprocess: <30s (single batch forward pass)
- Cleanup: <10s (tempdir removal)

**If runtime exceeds budget:** Reduce fixture size (fewer patterns) or add ROI parameter documentation for selective test execution.

---

## Next Actions & Timeline

### Immediate (D4.A completion - Current Loop)

1. ✅ This alignment narrative authored
2. [ ] Companion selector map authored (`phase_d4_selector_map.md`)
3. [ ] Update `docs/fix_plan.md` Attempts History with artifact paths
4. [ ] Mark D4.A1 and D4.A3 complete in `phase_d4_regression.md`

### Near-Term (D4.B - Next Loop)

1. Author failing regression tests per Q3 scope
2. Capture red logs under `reports/<ts>/phase_d4_red_*.log`
3. Document prerequisites for green phase in `phase_d4_red_summary.md`

### Mid-Term (D4.C - Following Loops)

1. Implement missing glue (model reconstruction, orchestration wiring)
2. Turn regression tests green
3. Produce handoff package (`phase_d4_handoff.md`)
4. Mark INTEGRATE-PYTORCH-001 Phase D4 complete

### Long-Term (TEST-PYTORCH-001 Activation - Post D4)

1. Copy test plan to `plans/active/TEST-PYTORCH-001/`
2. Build fixtures conforming to data contract
3. Implement subprocess integration test
4. Validate TensorFlow parity

---

## References & Artifact Paths

**Planning Documents:**
- Phase D4 checklist: `plans/active/INTEGRATE-PYTORCH-001/phase_d4_regression.md`
- Phase D workflow: `plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md`
- TEST-PYTORCH-001 charter: `plans/pytorch_integration_test_plan.md`

**Evidence from Prior Phases:**
- Config bridge parity: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T050930Z/parity_green_plan.md`
- Data pipeline summary: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T084500Z/cache_semantics.md`
- Persistence callchain: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T104700Z/phase_d3_callchain/`
- D2 scaffold: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T091450Z/phase_d2_scaffold.md`

**Stakeholder Context:**
- Stakeholder brief: `plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T031500Z/stakeholder_brief.md`
- Open questions Q1, Q4, Q7 (MLflow dependencies, config schema, ownership)

**Specification Contracts:**
- Configuration: `specs/ptychodus_api_spec.md` §2-3, §5
- Data format: `specs/data_contracts.md` §1-2
- Reconstructor lifecycle: `specs/ptychodus_api_spec.md` §4

**Knowledge Base:**
- CONFIG-001 finding: `docs/findings.md:9` (params.cfg initialization order)
- TDD methodology: `docs/DEVELOPER_GUIDE.md` §3.3
- Torch-optional patterns: `tests/conftest.py:38-46` (whitelist exemption)

---

**Document Status:** Ready for ledger integration
**Next Artifact:** `phase_d4_selector_map.md` (D4.A3)
