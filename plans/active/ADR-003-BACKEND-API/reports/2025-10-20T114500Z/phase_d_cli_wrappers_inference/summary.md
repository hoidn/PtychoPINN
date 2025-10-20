# Phase D.C C1 Completion Summary — Inference CLI Thin Wrapper Blueprint

**Date:** 2025-10-20
**Phase:** D.C — Inference CLI Thin Wrapper Refactoring
**Checklist Row:** C1 (inference blueprint)

---

## Deliverables

### Primary Artifact

- **`inference_refactor.md`** (51 KB): Comprehensive blueprint for refactoring `ptycho_torch/inference.py` into a thin CLI wrapper mirroring the training CLI refactor (Phase D.B).

**Key Sections:**
1. **Executive Summary**: Design principles and backward compatibility strategy
2. **Current State Analysis**: Call graph, identified problems (duplicate device mapping, manual validation, inline inference orchestration)
3. **Target Architecture**: Module structure, proposed thin wrapper flow, helper reuse strategy
4. **Component Specifications**: Detailed usage of `cli/shared.py` helpers (resolve_accelerator, build_execution_config_from_args, validate_paths)
5. **RawData Ownership Decision**: Option A (CLI retains loading) chosen for Phase D consistency; migration path documented for Phase E
6. **Inference Orchestration Refactor**: Recommends extracting `_run_inference_and_reconstruct()` helper (Option 2) to satisfy thin wrapper principle
7. **Open Questions & Design Choices**: Addressed 3 design questions (--quiet-only flag, helper placement, bundle loading scope)
8. **Test Strategy**: RED coverage plan for Phase C2 with 5 new unit tests and 3 inference-mode tests for shared helpers
9. **Implementation Sequence**: 9-step plan for Phase C3 GREEN implementation
10. **Success Criteria**: 7-point checklist tracking blueprint → RED → GREEN → docs progression

---

## Key Decisions Documented

### D1: Helper Reuse Strategy
**Decision:** Reuse `ptycho_torch/cli/shared.py` helpers from Phase D.B training refactor.
**Rationale:** Consistency, no duplication, minimal test churn.

### D2: RawData Ownership
**Decision:** CLI retains RawData loading (Option A) for Phase D.C.
**Rationale:** Matches training CLI pattern, explicit CONFIG-001 ordering, minimal test impact. Migration to workflow delegation planned for Phase E.

### D3: Inference Orchestration
**Decision:** Extract inline logic (lines 563-641) to `_run_inference_and_reconstruct()` helper function (Option 2).
**Rationale:** Thin wrapper delegates to testable helper; avoids premature workflow component migration; Phase E can adopt full reassembly workflow when required.

### D4: Accelerator Flag Handling
**Decision:** Reuse `resolve_accelerator()` from shared.py; emit DeprecationWarning for `--device` usage.
**Rationale:** Same deprecation strategy as training CLI; no new logic needed.

### D5: Quiet Mode Mapping
**Decision:** Inference CLI accepts `--quiet` only (not `--disable_mlflow`).
**Rationale:** MLflow tracking never implemented for inference CLI; cleaner interface; no legacy users relying on flag.

### D6: Bundle Loading Scope
**Decision:** Keep `load_inference_bundle_torch()` call in CLI thin wrapper.
**Rationale:** Factory-validated function already handles CONFIG-001; no benefit from helper indirection.

---

## Alignment with Phase D.B Training Blueprint

This blueprint explicitly mirrors the training CLI refactor to ensure consistency:

| Aspect | Training CLI (Phase D.B) | Inference CLI (Phase D.C) |
|--------|-------------------------|--------------------------|
| Helper module | Introduced `cli/shared.py` | Reuses `cli/shared.py` |
| Device deprecation | `resolve_accelerator()` | Same helper, no changes |
| Execution config | `build_execution_config_from_args(mode='training')` | `build_execution_config_from_args(mode='inference')` |
| Path validation | `validate_paths(train_file, test_file, output_dir)` | `validate_paths(None, test_file, output_dir)` |
| RawData ownership | CLI retains loading (Option A) | CLI retains loading (Option A) |
| Orchestration logic | Delegates to `run_cdi_example_torch()` | Extracts to `_run_inference_and_reconstruct()` helper |
| Legacy interface | Preserved `main()` unchanged | Preserved `load_and_predict()` unchanged |

---

## Test Coverage Plan (Phase C2)

### New RED Tests Required

**File:** `tests/torch/test_cli_inference_torch.py`
- `test_cli_delegates_to_helper_for_data_loading`
- `test_cli_delegates_to_inference_helper`
- `test_cli_calls_save_individual_reconstructions`
- `test_cli_validates_test_file_existence_before_factory`
- `test_quiet_flag_suppresses_progress_output`

**File:** `tests/torch/test_cli_shared.py` (if not already covered from Phase D.B)
- `test_build_execution_config_inference_mode_defaults`
- `test_build_execution_config_inference_mode_custom_batch_size`
- `test_build_execution_config_inference_mode_no_deterministic_warning`

**Expected RED Behavior:** Tests fail with `AttributeError: no attribute '_run_inference_and_reconstruct'` or `AssertionError` for incorrect delegation flow.

---

## Risks & Open Questions

### Risks

1. **Test Churn Risk (LOW):** Helper extraction may require updating existing integration test mocks. Mitigation: Keep RawData loading in CLI to preserve mock structure.
2. **Scope Creep Risk (MEDIUM):** Temptation to migrate to full `_reassemble_cdi_image_torch()` workflow component prematurely. Mitigation: Blueprint explicitly recommends helper extraction (Option 2) for Phase D.C, deferring workflow migration to Phase E.
3. **Legacy Interface Preservation (LOW):** MLflow-based inference path must remain unchanged. Mitigation: Blueprint documents clear separation via interface detection logic.

### Open Questions

1. **Q1:** Should `_run_inference_and_reconstruct()` be public or private? **Decision:** Private (`_` prefix) to signal internal helper, not part of public API.
2. **Q2:** Should inference tests verify execution config propagation to factory? **Decision:** Already covered by Phase C4.D1 GREEN tests; no new tests needed.
3. **Q3:** Should blueprint document migration path for full reassembly workflow? **Decision:** Yes, noted in "Phase E Migration Path" for RawData ownership and inference orchestration sections.

---

## Next Steps for Ralph (Phase D.C C2)

1. **RED Test Authoring:** Write 5-8 new tests in `tests/torch/test_cli_inference_torch.py` capturing expected helper delegation behavior.
2. **RED Log Capture:** Run `pytest tests/torch/test_cli_inference_torch.py -vv > pytest_cli_inference_thin_red.log 2>&1` and archive log.
3. **Shared Helper Inference Tests:** Add 3 inference-mode tests to `tests/torch/test_cli_shared.py` (if not already GREEN from Phase D.B).
4. **Plan Update:** Update `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md` row C1 state to `[x]` with artifact links.
5. **Fix Plan Attempt:** Log Attempt entry in `docs/fix_plan.md` referencing this artifact hub (`plans/active/ADR-003-BACKEND-API/reports/2025-10-20T114500Z/phase_d_cli_wrappers_inference/`).

---

## Artifact Metadata

**Location:** `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T114500Z/phase_d_cli_wrappers_inference/`
**Files Created:**
- `inference_refactor.md` (51 KB, complete specification)
- `summary.md` (this file, 6 KB, recap + decisions)

**References:**
- Training blueprint: `../../../2025-10-20T105408Z/phase_d_cli_wrappers_training/training_refactor.md`
- Plan: `../../../2025-10-20T130900Z/phase_d_cli_wrappers/plan.md`
- Implementation plan: `plans/active/ADR-003-BACKEND-API/implementation.md` (Phase D checklist)
- Spec: `specs/ptychodus_api_spec.md` §4.6, §4.8, §7

**Traceability:** This summary and blueprint fulfill Phase D.C C1 exit criteria (blueprint authored). Next loop will execute C2 (RED tests).
