# Phase D.B Training CLI Thin Wrapper Summary

**Initiative:** ADR-003-BACKEND-API — Standardize PyTorch backend API
**Phase:** D.B (Tasks B1-B2) — Training CLI Blueprint + RED Test Coverage
**Date:** 2025-10-20
**Loop:** Ralph (Engineer)
**Status:** ✅ COMPLETE (B1✅, B2✅ — implementation deferred to B3)

---

## Executive Summary

Phase D.B1-B2 successfully delivered:

1. ✅ **Complete Training CLI Refactor Blueprint** (`training_refactor.md`)
   - Documented helper/module layout (`ptycho_torch/cli/shared.py`)
   - Specified delegation flow (CLI → helpers → factory → workflow)
   - Resolved RawData ownership (Option A: CLI retains loading for Phase D)
   - Defined accelerator warning strategy (deprecation + performance warnings)
   - Clarified `--disable_mlflow` handling (maps to `enable_progress_bar` via `--quiet` alias)

2. ✅ **RED Test Coverage Established** (`tests/torch/test_cli_shared.py`)
   - 20 new unit tests for helper functions (all RED as expected)
   - Tests for `resolve_accelerator()`: 5 tests
   - Tests for `build_execution_config_from_args()`: 9 tests
   - Tests for `validate_paths()`: 6 tests
   - All tests fail with `ModuleNotFoundError: No module named 'ptycho_torch.cli'` (expected RED behavior)

3. ✅ **Baseline Tests GREEN** (`tests/torch/test_cli_train_torch.py`)
   - 7/7 existing Phase C4 tests PASSED (execution config integration)
   - No regressions introduced by new test file

**Test Results Summary:**
- **Total Tests:** 27
- **PASSED:** 7 (baseline Phase C4 tests)
- **FAILED:** 20 (new RED tests for helpers — expected)
- **Runtime:** 4.97s

---

## Artifacts Generated

| Artifact | Location | Purpose | Status |
|----------|----------|---------|--------|
| **Blueprint** | `training_refactor.md` | Architectural specification for thin wrapper refactor | ✅ Complete |
| **RED Tests** | `tests/torch/test_cli_shared.py` | Unit tests for helper functions (pre-implementation) | ✅ Complete |
| **RED Log** | `pytest_cli_train_thin_red.log` | Captured pytest output showing expected failures | ✅ Complete |
| **Summary** | `summary.md` (this file) | Loop closeout notes | ✅ Complete |

**Artifact Hub:** `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T105408Z/phase_d_cli_wrappers_training/`

---

## Blueprint Highlights (training_refactor.md)

### New Module Structure

```
ptycho_torch/
├─ cli/
│  ├─ __init__.py (NEW)
│  └─ shared.py (NEW)
│     ├─ resolve_accelerator(accelerator, device) -> str
│     ├─ build_execution_config_from_args(args, mode) -> PyTorchExecutionConfig
│     └─ validate_paths(train_file, test_file, output_dir) -> None
└─ train.py (REFACTORED in Phase B3)
   └─ cli_main() (simplified to ~80 lines from current 377)
```

### Key Design Decisions Captured

1. **RawData Loading Ownership (D6):** CLI retains `RawData.from_file()` calls for Phase D (minimal test churn). Phase E will migrate to workflow-internal loading.

2. **Device Mapping Strategy (D1, D5):** Extract duplicate 12-line `--device` → `--accelerator` resolution to `resolve_accelerator()` in `cli/shared.py`. Both training and inference CLIs will reuse this helper.

3. **Validation Responsibility (D4):** Move `num_workers >= 0`, `learning_rate > 0` checks from CLI to `PyTorchExecutionConfig.__post_init__()` (dataclass is authoritative).

4. **Progress Bar Control (D3):** `--disable_mlflow` and `--quiet` both map to `enable_progress_bar` field. Add `--quiet` as preferred alias, deprecate `--disable_mlflow` in Phase E.

5. **Warning Emission (D1, Performance):**
   - `DeprecationWarning` for `--device` usage (via `resolve_accelerator()`)
   - `UserWarning` for `deterministic=True` + `num_workers > 0` (via `build_execution_config_from_args()`)

6. **Error Handling Pattern:** Helpers raise exceptions (`FileNotFoundError`, `ValueError`), CLI wrapper catches and formats for user-friendly terminal output.

---

## RED Test Coverage Analysis

### Test File Structure: `tests/torch/test_cli_shared.py`

#### Class: `TestResolveAccelerator` (5 tests)

| Test | Assertion | Expected RED Failure | Status |
|------|-----------|---------------------|--------|
| `test_default_no_device` | `resolve_accelerator('cpu', None) → 'cpu'` | `ModuleNotFoundError` | ✅ RED |
| `test_legacy_device_cpu` | `resolve_accelerator('auto', 'cpu') → 'cpu' + DeprecationWarning` | `ModuleNotFoundError` | ✅ RED |
| `test_legacy_device_cuda_maps_to_gpu` | `resolve_accelerator('auto', 'cuda') → 'gpu'` | `ModuleNotFoundError` | ✅ RED |
| `test_conflict_accelerator_wins` | `resolve_accelerator('cpu', 'cuda') → 'cpu' + warning` | `ModuleNotFoundError` | ✅ RED |
| `test_all_accelerator_values_passthrough` | All valid values unchanged | `ModuleNotFoundError` | ✅ RED |

#### Class: `TestBuildExecutionConfig` (9 tests)

| Test | Assertion | Expected RED Failure | Status |
|------|-----------|---------------------|--------|
| `test_training_mode_defaults` | Default PyTorchExecutionConfig fields | `ModuleNotFoundError` | ✅ RED |
| `test_training_mode_custom_values` | Custom execution config values | `ModuleNotFoundError` | ✅ RED |
| `test_inference_mode` | Inference-specific fields | `ModuleNotFoundError` | ✅ RED |
| `test_emits_deterministic_warning` | UserWarning for deterministic+workers | `ModuleNotFoundError` | ✅ RED |
| `test_handles_quiet_flag` | `--quiet` → `enable_progress_bar=False` | `ModuleNotFoundError` | ✅ RED |
| `test_handles_disable_mlflow_flag` | `--disable_mlflow` → `enable_progress_bar=False` | `ModuleNotFoundError` | ✅ RED |
| `test_quiet_or_disable_mlflow_both_true` | Logical OR behavior | `ModuleNotFoundError` | ✅ RED |
| `test_invalid_mode_raises_value_error` | ValueError for invalid mode | `ModuleNotFoundError` | ✅ RED |
| `test_resolves_accelerator_from_device_flag` | Internal call to `resolve_accelerator()` | `ModuleNotFoundError` | ✅ RED |

#### Class: `TestValidatePaths` (6 tests)

| Test | Assertion | Expected RED Failure | Status |
|------|-----------|---------------------|--------|
| `test_creates_output_dir` | `mkdir -p` behavior | `ModuleNotFoundError` | ✅ RED |
| `test_raises_if_train_file_missing` | `FileNotFoundError` for missing train file | `ModuleNotFoundError` | ✅ RED |
| `test_raises_if_test_file_missing` | `FileNotFoundError` for missing test file | `ModuleNotFoundError` | ✅ RED |
| `test_accepts_none_test_file` | Optional `test_file` parameter | `ModuleNotFoundError` | ✅ RED |
| `test_works_with_pathlib_path_objects` | `Path` object compatibility | `ModuleNotFoundError` | ✅ RED |
| `test_accepts_none_train_file_for_inference_mode` | Optional `train_file` for inference | `ModuleNotFoundError` | ✅ RED |

---

## Baseline Test Results (No Regressions)

### `tests/torch/test_cli_train_torch.py::TestExecutionConfigCLI` (7 tests — all PASSED)

| Test | Description | Status |
|------|-------------|--------|
| `test_accelerator_flag_roundtrip` | CLI → factory execution_config.accelerator | ✅ PASSED |
| `test_deterministic_flag_roundtrip` | CLI → factory execution_config.deterministic | ✅ PASSED |
| `test_no_deterministic_flag_roundtrip` | `--no-deterministic` flag | ✅ PASSED |
| `test_num_workers_flag_roundtrip` | CLI → factory execution_config.num_workers | ✅ PASSED |
| `test_learning_rate_flag_roundtrip` | CLI → factory execution_config.learning_rate | ✅ PASSED |
| `test_multiple_execution_config_flags` | Combined flags integration | ✅ PASSED |
| `test_bundle_persistence` | Dual-model wts.h5.zip bundle creation | ✅ PASSED |

**Warnings Observed (Expected):**
1. `UserWarning` for `num_workers > 0` with deterministic mode (expected behavior from `ptycho_torch/train.py:569`)
2. `UserWarning` for missing probeGuess in dummy test fixture (expected — test creates empty NPZ)
3. `UserWarning` for params.cfg already populated (expected — factory defensive check)

---

## Open Questions for Phase B3

### Q1: Should `PyTorchExecutionConfig.__post_init__()` validation be added in Phase B3?

**Context:** Blueprint specifies moving CLI validation logic to dataclass.

**Recommendation:** Yes — implement validation in `ptycho/config/config.py` during Phase B3 helper implementation. This ensures helpers are self-contained and CLI validation is removed atomically.

---

### Q2: Should `--quiet` flag be added to training CLI in Phase B3?

**Context:** Blueprint proposes `--quiet` as alias for `--disable_mlflow`.

**Recommendation:** Yes — add to argparse in Phase B3 refactor. Update `build_execution_config_from_args()` to handle both flags via logical OR.

---

### Q3: Should we update docs before or after Phase B3 implementation?

**Context:** `docs/workflows/pytorch.md` CLI examples need deprecation notices.

**Recommendation:** After Phase B3 GREEN tests. Docs should reflect implemented behavior, not planned behavior.

---

## Next Steps (Phase B3 — Deferred to Next Loop)

**Task Sequence:**

1. **Create `ptycho_torch/cli/__init__.py`** (empty file for Python package)

2. **Implement `ptycho_torch/cli/shared.py`** with three helper functions:
   - `resolve_accelerator(accelerator, device)`
   - `build_execution_config_from_args(args, mode)`
   - `validate_paths(train_file, test_file, output_dir)`

3. **Add validation to `PyTorchExecutionConfig.__post_init__()`** in `ptycho/config/config.py`:
   - `num_workers >= 0`
   - `learning_rate > 0`
   - `inference_batch_size > 0` (if provided)
   - `accelerator in {'auto', 'cpu', 'gpu', 'cuda', 'tpu', 'mps'}`

4. **Refactor `ptycho_torch/train.py` `cli_main()`**:
   - Add `--quiet` flag to argparse
   - Update `--device` help text to mark deprecated
   - Replace device mapping logic with `resolve_accelerator()` call
   - Replace path validation with `validate_paths()` call
   - Replace execution config construction with `build_execution_config_from_args()` call
   - Remove CLI-level validation (lines 558-564)
   - Keep RawData loading in CLI (Option A)

5. **Run GREEN tests:**
   - `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_shared.py -vv` (20 tests should PASS)
   - `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py -vv` (7 tests should PASS)
   - Capture `pytest_cli_train_thin_green.log`

6. **Update `docs/workflows/pytorch.md`** §12 (CLI usage):
   - Add deprecation notice for `--device`
   - Recommend `--quiet` over `--disable_mlflow`
   - Update CLI examples to use `--accelerator`

7. **Update plan checklists:**
   - Mark `phase_d_cli_wrappers/plan.md` B3→`[x]`
   - Mark `implementation.md` Phase D.D1→`[x]`
   - Log attempt in `docs/fix_plan.md`

---

## Loop Decisions & Deviations

### Deviation: RawData Loading Not Removed

**Planned (Blueprint):** Remove RawData loading from CLI, delegate to workflow.

**Actual (Phase D):** Retained RawData loading in CLI per Option A decision.

**Rationale:**
- Minimal test churn (existing mocks unchanged)
- Explicit CONFIG-001 ordering visible in CLI flow
- Phase E can migrate to workflow-internal loading after signature stabilizes

**Impact:** No impact on Phase D goals (thin wrapper still achieved). Defers workflow signature change to Phase E.

---

### Deviation: No argparse extraction to `training_args.py`

**Planned (Blueprint Option):** Optionally extract argparse logic to separate module.

**Actual (Phase D):** Keep argparse inline in `cli_main()`.

**Rationale:**
- Argparse definition is already declarative and readable
- Extracting to separate module adds indirection without clear benefit
- Inline argparse makes CLI entry point self-contained

**Impact:** `cli_main()` remains ~80-100 lines instead of ~60 lines. Acceptable tradeoff for maintainability.

---

## Conclusion

**Phase D.B1-B2 Success Criteria:**

✅ **Blueprint Authored:** Complete specification in `training_refactor.md` with helper functions, call flows, design decisions, and test strategy.

✅ **RED Tests Written:** 20 new unit tests in `tests/torch/test_cli_shared.py` fail with expected `ModuleNotFoundError`.

✅ **RED Log Captured:** `pytest_cli_train_thin_red.log` shows 20 FAILED, 7 PASSED (baseline).

✅ **No Regressions:** All 7 Phase C4 baseline tests remain GREEN.

✅ **Artifacts Organized:** All artifacts stored in timestamped directory per repository conventions.

**Phase B3 Readiness:**

🟢 **Ready to Implement:** Blueprint provides complete specification for helper implementation.

🟢 **Test Coverage Complete:** 20 RED tests will turn GREEN when helpers are implemented.

🟢 **Baseline Stable:** No test churn risk; existing tests unaffected.

**Blockers:** None.

**Follow-up:** Proceed to Phase B3 implementation in next loop (implement helpers + refactor CLI + GREEN tests).

---

**Loop Artifacts Summary:**
- Blueprint: `training_refactor.md` (21 KB)
- RED Tests: `tests/torch/test_cli_shared.py` (18 KB, 568 lines)
- RED Log: `pytest_cli_train_thin_red.log` (23 KB)
- Summary: `summary.md` (this file, 10 KB)

**Total Artifact Size:** ~72 KB (well within repository limits)

**Loop Runtime:** ~30 minutes (blueprint authoring + test writing + execution)

**Next Loop Estimate:** ~60-90 minutes (helper implementation + CLI refactor + GREEN verification)

---

**Phase D.B1-B2 COMPLETE — Ready for B3 Implementation**
