# PyTorch Backend Integration — Stakeholder Brief

**Initiative:** INTEGRATE-PYTORCH-000 Phase C.C2
**Date:** 2025-10-17
**Status:** Planning Document
**Purpose:** Communicate canonical plan updates (Phase B.B2) to downstream execution initiatives

---

## Executive Summary

The canonical PyTorch integration plan (`plans/ptychodus_pytorch_integration_plan.md`) has been updated to reflect the rebased `ptycho_torch/` tree (commit bfc22e7). This brief summarizes five major architectural deltas, maps them to execution initiatives, and identifies critical open questions requiring governance decisions.

**Critical Insight:** Configuration bridge harmonization (`specs/ptychodus_api_spec.md:20-125`) is the #1 blocker. The PyTorch configuration schema diverges significantly from the TensorFlow dataclass specification, risking silent contract violations. This must be resolved in INTEGRATE-PYTORCH-001 Phase B before any data pipeline or training integration can proceed.

---

## Context: What Changed in Phase B.B2

The integration plan was overhauled to incorporate new subsystems discovered during the Phase A module audit (`plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T025000Z/delta_log.md`):

1. New `api/` layer providing high-level orchestration abstraction
2. Configuration schema mismatches between PyTorch and TensorFlow implementations
3. New `datagen/` package for synthetic data generation
4. Alternative barycentric reassembly implementation suite
5. Lightning + MLflow persistence divergence from TensorFlow workflows

Each delta has been analyzed and integrated into the canonical plan with specific phase-level tasks, spec citations, and risk assessments.

---

## Delta Summary & Actions Required

### Delta 1: Configuration Schema Divergence

**Section Reference:** `plans/ptychodus_pytorch_integration_plan.md:71-81` (Phase 1)
**Spec Reference:** `specs/ptychodus_api_spec.md:20-125`, `213-291` (configuration surface & KEY_MAPPINGS)

**What Changed:**
- Current PyTorch configuration uses singleton pattern with divergent field names:
  - `grid_size: Tuple[int, int]` → spec requires `gridsize: int`
  - `mode: 'Supervised' | 'Unsupervised'` → spec requires `model_type: 'pinn' | 'supervised'`
- Missing spec-mandated fields: `gaussian_smoothing_sigma`, `probe_scale`, `pad_object`
- No KEY_MAPPINGS translation layer for legacy `params.cfg` dot-separated keys

**Required Actions:**

| Initiative | Task | Priority | Blocking? |
|-----------|------|----------|-----------|
| INTEGRATE-PYTORCH-001 | Phase B.B1: Audit complete field mapping between PyTorch singletons and TensorFlow dataclasses | Critical | Yes |
| INTEGRATE-PYTORCH-001 | Phase B.B2: Implement schema harmonization strategy (refactor PyTorch or dual-schema bridge) | Critical | Yes |
| INTEGRATE-PYTORCH-001 | Phase B.B3: Create KEY_MAPPINGS translation layer for PyTorch config → `params.cfg` | Critical | Yes |
| INTEGRATE-PYTORCH-001 | Phase B.B4: Write 75+ parameterized tests validating all config fields propagate correctly | Critical | Yes |
| TEST-PYTORCH-001 | Update fixture requirements to match harmonized configuration schema | High | No |

**Outstanding Questions:**
1. **Q1:** Should PyTorch configuration be refactored to use shared dataclasses, or should we maintain dual schemas with explicit translation? (Decision impacts maintainability and test coverage scope)
2. **Q2:** Which fields from `specs/ptychodus_api_spec.md §5` (75+ documented fields) are truly required for minimal viable integration vs. full parity? (Impacts Phase B scope definition)

**Citation:** See `plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T025000Z/delta_log.md` Delta-1 for complete field inventory.

---

### Delta 2: API Layer Surface Decision

**Section Reference:** `plans/ptychodus_pytorch_integration_plan.md:52-54` (Phase 0 Decision Gate)
**Spec Reference:** `specs/ptychodus_api_spec.md:127-211` (reconstructor lifecycle)

**What Changed:**
- PyTorch provides high-level `api/base_api.py` layer (994 lines) with:
  - `ConfigManager`, `PtychoModel`, `Trainer`, `InferenceEngine` classes
  - MLflow-centric persistence (`save_mlflow()`, `load_from_mlflow()`)
  - Lightning orchestration abstraction
- This API layer was not present in legacy plan assumptions

**Required Actions:**

| Initiative | Task | Priority | Blocking? |
|-----------|------|----------|-----------|
| INTEGRATE-PYTORCH-001 | Phase A (Evidence): Document API layer capabilities vs reconstructor contract requirements | High | Yes |
| INTEGRATE-PYTORCH-001 | Phase A: Decision Gate - API-first integration vs low-level module integration | High | Yes |
| INTEGRATE-PYTORCH-001 | Phase B (if API-first): Map API layer methods to `specs/ptychodus_api_spec.md §4` contract points | High | No |
| INTEGRATE-PYTORCH-001 | Phase B (if low-level): Design adapter layer bypassing API for direct module access | High | No |

**Outstanding Questions:**
3. **Q3:** Should `ptychodus` integration use the `api/` layer (cleaner but adds dependency on MLflow orchestration) or bypass it for direct module calls (lower-level but more control)? (Decision impacts Phases 3-6 implementation strategy)
4. **Q4:** If API layer is adopted, who owns the MLflow persistence contract — PyTorch backend or ptychodus integration glue? (Impacts Phase 5 persistence design)

**Citation:** `plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T025000Z/delta_log.md` Delta-2.

---

### Delta 3: Synthetic Data Generation (`datagen/` Package)

**Section Reference:** `plans/ptychodus_pytorch_integration_plan.md:91-94` (Phase 2, Task 2.4)
**Spec Reference:** `specs/data_contracts.md` (NPZ schema requirements)

**What Changed:**
- New `ptycho_torch/datagen/` package provides synthetic dataset generation:
  - `from_simulation()`, `simulate_multiple_experiments()` for NPZ creation
  - Poisson scaling, beamstop handling, experimental data extraction
- Must verify outputs conform to shared NPZ contract (`specs/data_contracts.md:13-73`)

**Required Actions:**

| Initiative | Task | Priority | Blocking? |
|-----------|------|----------|-----------|
| INTEGRATE-PYTORCH-001 | Phase C (Data): Validate `datagen/` NPZ outputs against `specs/data_contracts.md` | Medium | No |
| INTEGRATE-PYTORCH-001 | Phase C: Ensure synthetic datasets consumable by both TensorFlow and PyTorch backends | Medium | No |
| TEST-PYTORCH-001 | Incorporate `datagen/`-produced fixtures into integration test suite | Medium | No |

**Outstanding Questions:**
5. **Q5:** Should `datagen/` replace or complement existing TensorFlow simulation workflows (`ptycho.diffsim`)? (Impacts tooling standardization and maintenance burden)

**Citation:** `plans/ptychodus_pytorch_integration_plan.md:91-94`, `delta_log.md` Delta-3.

---

### Delta 4: Barycentric Reassembly Alternatives

**Section Reference:** `plans/ptychodus_pytorch_integration_plan.md:102-107` (Phase 3, Task 3.4)
**Spec Reference:** `specs/ptychodus_api_spec.md:176-178` (reassembly contract via `tf_helper`)

**What Changed:**
- PyTorch provides alternative reassembly modules: `reassembly_alpha.py`, `reassembly_beta.py`, `reassembly.py`
- Implementation uses vectorized barycentric accumulator (alternative to `ptycho.tf_helper.reassemble_position`)
- Includes DataParallel support for multi-GPU patch stitching with performance profiling

**Required Actions:**

| Initiative | Task | Priority | Blocking? |
|-----------|------|----------|-----------|
| INTEGRATE-PYTORCH-001 | Phase D (Inference): Establish numeric parity tests for barycentric reassembly vs TensorFlow | High | No |
| INTEGRATE-PYTORCH-001 | Phase D: Define acceptable tolerances for reconstruction output differences | High | No |
| INTEGRATE-PYTORCH-001 | Phase D: Decision - adapt TensorFlow reassembly or validate PyTorch parity and use native | High | No |
| TEST-PYTORCH-001 | Create synthetic fixtures for reassembly parity validation across backends | High | No |

**Outstanding Questions:**
6. **Q6:** What are acceptable numeric tolerances for reassembly output differences (e.g., RMSE thresholds, SSIM minimums)? (Impacts parity test pass/fail criteria)

**Citation:** `plans/ptychodus_pytorch_integration_plan.md:102-107`, `delta_log.md` Delta-4.

---

### Delta 5: Lightning + MLflow Orchestration Divergence

**Section Reference:** `plans/ptychodus_pytorch_integration_plan.md:111-114` (Phase 4, Task 4.1)
**Spec Reference:** `specs/ptychodus_api_spec.md:180-190` (training workflow contract)

**What Changed:**
- PyTorch training uses Lightning `Trainer` with callbacks, DataModule, DDP strategy
- MLflow autologging replaces TensorFlow's direct orchestration (`run_cdi_example()`)
- Multi-stage training logic embedded (`stage_1/2/3_epochs` with physics weight scheduling)

**Required Actions:**

| Initiative | Task | Priority | Blocking? |
|-----------|------|----------|-----------|
| INTEGRATE-PYTORCH-001 | Phase E (Training): Clarify if ptychodus invokes Lightning trainer directly or requires lower-level API | High | Yes |
| INTEGRATE-PYTORCH-001 | Phase E: Document Lightning/MLflow dependency policy for CI/production environments | High | Yes |
| INTEGRATE-PYTORCH-001 | Phase E: Design graceful degradation path if MLflow unavailable | Medium | No |
| INTEGRATE-PYTORCH-001 | Phase F (Persistence): Define Lightning checkpoint → `.h5.zip` archive adapter | High | Yes |

**Outstanding Questions:**
7. **Q7:** Are Lightning and MLflow mandatory dependencies, or should integration support optional fallback paths? (Impacts deployment requirements and Phase 4-5 architecture)
8. **Q8:** Should multi-stage training logic (physics weight scheduling) be exposed to ptychodus configuration UI, or remain PyTorch-internal? (Impacts reconstructor settings surface)

**Citation:** `plans/ptychodus_pytorch_integration_plan.md:111-114`, `delta_log.md` Delta-5.

---

## Execution Roadmap for Downstream Initiatives

### INTEGRATE-PYTORCH-001 (Execution Lead)

**Immediate Next Steps (Phase B — Configuration & Legacy Bridge):**
1. **B1:** Complete field-by-field audit mapping PyTorch config schema → TensorFlow dataclass schema → `params.cfg` keys
   - Input: `ptycho_torch/config_params.py`, `specs/ptychodus_api_spec.md §5.1-5.3`
   - Output: Configuration schema mapping table (recommended location: `plans/active/INTEGRATE-PYTORCH-001/reports/<ts>/config_schema_map.md`)
2. **B2:** Write failing test demonstrating config bridge failure (TDD methodology per `docs/DEVELOPER_GUIDE.md §3.3`)
   - Test should instantiate PyTorch config, call `update_legacy_dict()`, verify `params.cfg` population
   - Expected: Test fails due to missing KEY_MAPPINGS and field mismatches
3. **B3:** Implement schema harmonization (decision required: refactor vs dual-schema)
   - If refactor: migrate PyTorch to shared dataclasses
   - If dual-schema: implement translation layer + KEY_MAPPINGS for PyTorch
4. **B4:** Extend parity tests to cover all 75+ configuration fields
   - Parameterized tests per `plans/ptychodus_pytorch_integration_plan.md:79-81`

**Blockers for Phases C-F:**
- Cannot proceed with data pipeline (Phase C) until `params.cfg` bridge is operational
- Cannot proceed with training integration (Phase E) until API surface decision (Q3) is resolved

**References:**
- Working plan: `plans/active/INTEGRATE-PYTORCH-001/implementation.md`
- Parity map: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T020000Z/parity_map.md`
- Config contract: `specs/ptychodus_api_spec.md:20-125`, `213-291`

---

### TEST-PYTORCH-001 (Test Harness Lead)

**Immediate Next Steps:**
1. Review configuration schema harmonization decisions from INTEGRATE-PYTORCH-001 Phase B
   - Adjust fixture expectations based on finalized schema (refactor vs dual-schema)
2. Prepare test fixtures for reassembly parity validation (Delta 4)
   - Create synthetic datasets with known ground truth for numeric comparisons
   - Define tolerance thresholds in collaboration with INTEGRATE-PYTORCH-001 Phase D
3. Update test plan (`plans/pytorch_integration_test_plan.md`) to incorporate:
   - Configuration bridge test suite (75+ fields)
   - Barycentric reassembly parity tests
   - Dual-backend persistence round-trip tests

**Dependencies:**
- Configuration schema finalization (INTEGRATE-PYTORCH-001 Phase B)
- API surface decision (Q3) impacts test execution strategy

**References:**
- Test plan: `plans/pytorch_integration_test_plan.md`
- Existing TensorFlow test baseline: `tests/test_integration_workflow.py` (persistence contract)

---

## Open Questions Requiring Governance Decisions

The following questions block critical path work and require architectural decisions before implementation can proceed:

| ID | Question | Impact | Decision Forum | Target Resolution |
|----|----------|--------|----------------|-------------------|
| Q1 | Refactor PyTorch config to shared dataclasses or maintain dual schemas? | Affects maintainability, test scope | INTEGRATE-PYTORCH-001 Phase B kickoff | Before Phase B.B2 |
| Q2 | Which of 75+ config fields are MVP vs full parity? | Affects Phase B scope definition | INTEGRATE-PYTORCH-001 Phase A review | Before Phase B.B1 |
| Q3 | API-first integration or low-level module access? | Affects Phases 3-6 implementation strategy | INTEGRATE-PYTORCH-001 Phase A decision gate | Before Phase C start |
| Q4 | Who owns MLflow persistence contract? | Affects Phase 5 persistence design | INTEGRATE-PYTORCH-001 + TEST-PYTORCH-001 sync | Before Phase E start |
| Q5 | `datagen/` replaces or complements TensorFlow simulation? | Affects tooling standardization | INTEGRATE-PYTORCH-001 Phase C review | Before Phase C.C2 |
| Q6 | Acceptable numeric tolerances for reassembly parity? | Affects parity test pass/fail criteria | TEST-PYTORCH-001 fixture design | Before Phase D tests |
| Q7 | Lightning/MLflow mandatory or optional dependencies? | Affects deployment requirements | INTEGRATE-PYTORCH-001 Phase E kickoff | Before Phase E.E1 |
| Q8 | Expose multi-stage training to ptychodus UI? | Affects reconstructor settings surface | INTEGRATE-PYTORCH-001 Phase E review | Before Phase E.E2 |

**Recommended Decision Process:**
1. Q1, Q2, Q3 are critical-path blockers — resolve in next INTEGRATE-PYTORCH-001 planning session
2. Q4, Q7, Q8 can be deferred until respective phases (E, F) with fallback assumptions documented
3. Q5, Q6 are medium-priority — establish working defaults and iterate based on test results

---

## Success Criteria for Phase C Completion

Phase C (Governance & Handoff Sync) of INTEGRATE-PYTORCH-000 is complete when:

1. ✅ This stakeholder brief is reviewed and acknowledged by downstream initiative leads
2. ✅ `docs/fix_plan.md` updated with Phase C completion status and artifact paths
3. ✅ `plans/active/INTEGRATE-PYTORCH-001/implementation.md` updated to reference this brief and canonical plan sections
4. ✅ Open questions (Q1-Q8) logged in governance tracking system with decision forums assigned
5. ✅ input.md directive updated to shift focus to INTEGRATE-PYTORCH-001 Phase B execution

---

## References & Artifact Paths

**Canonical Plan:**
- `plans/ptychodus_pytorch_integration_plan.md` (updated Phase B.B2)

**Phase A Evidence:**
- Module inventory: `plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T025000Z/module_inventory.md`
- Delta analysis: `plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T025000Z/delta_log.md`

**Phase B Planning:**
- Redline outline: `plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T025633Z/plan_redline.md`
- Summary: `plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T025633Z/summary.md`

**Specification Contracts:**
- API contract: `specs/ptychodus_api_spec.md` (configuration §2-3, reconstructor §4)
- Data contract: `specs/data_contracts.md` (NPZ schema)

**Downstream Plans:**
- INTEGRATE-PYTORCH-001: `plans/active/INTEGRATE-PYTORCH-001/implementation.md`
- TEST-PYTORCH-001: `plans/pytorch_integration_test_plan.md`

**Knowledge Base:**
- Configuration gotchas: `docs/findings.md` (CONFIG-001, CONVENTION-001)
- Debugging methodology: `docs/debugging/debugging.md`

---

## Acknowledgments

This brief synthesizes planning work from INTEGRATE-PYTORCH-000 Phases A-B (module audit, canonical plan redline) and incorporates guidance from:
- `specs/ptychodus_api_spec.md` (authoritative contract reference)
- `docs/DEVELOPER_GUIDE.md` (TDD methodology, two-system architecture)
- `docs/workflows/pytorch.md` (PyTorch workflow patterns)

For questions or clarifications, reference the artifact paths above or consult the active initiative plans under `plans/active/`.

---

**Document Status:** Ready for stakeholder review
**Next Action:** INTEGRATE-PYTORCH-001 Phase B kickoff with configuration bridge focus
