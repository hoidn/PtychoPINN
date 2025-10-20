# Phase D.C C3 Inference CLI Thin-Wrapper Test Fix (Follow-Up)

**Date:** 2025-10-20
**Initiative:** ADR-003-BACKEND-API
**Phase:** D.C (CLI Thin-Wrapper Delegation)
**Checklist Item:** C3
**Mode:** TDD (Test Fix + GREEN Evidence)

---

## Objective

Fix failing thin-wrapper delegation tests in `tests/torch/test_cli_inference_torch.py` by:
1. Updating `test_cli_delegates_to_validate_paths()` to inspect `mock_validate_paths.call_args.kwargs` (keyword invocation pattern)
2. Seeding all mocked bundle loaders with `{'diffraction_to_obj': MagicMock()}` to satisfy helper execution path

---

## Changes Implemented

### Test File Updates (`tests/torch/test_cli_inference_torch.py`)

**1. Fixed `test_cli_delegates_to_validate_paths()` (lines 237-281):**
- **Changed:** Assertion logic from positional args (`call_args[0][0]`) to keyword args (`call_args.kwargs`)
- **Rationale:** The `validate_paths()` helper uses keyword arguments per shared helper contract (`ptycho_torch/cli/shared.py`)
- **Before:**
  ```python
  call_args = mock_validate_paths.call_args
  assert call_args[0][0] is None  # Positional
  ```
- **After:**
  ```python
  call_kwargs = mock_validate_paths.call_args.kwargs
  assert call_kwargs.get('train_file') is None  # Keyword
  ```

**2. Updated Bundle Loader Mocks (2 tests):**
- **test_cli_delegates_to_validate_paths()** (line 257)
- **test_cli_delegates_to_helper_for_data_loading()** (line 304)
- **Changed:** `MagicMock(return_value=({}, {}))` → `MagicMock(return_value=({'diffraction_to_obj': MagicMock()}, {}))`
- **Rationale:** The CLI workflow expects the bundle loader to return a dictionary containing at least the `diffraction_to_obj` model key. Without this, the helper path fails to execute fully, causing downstream assertions to fail.

**3. Verified Existing Tests:**
- Tests at lines 329 (`test_cli_delegates_to_inference_helper`), 380 (`test_cli_calls_save_individual_reconstructions`), and 433 (`test_quiet_flag_suppresses_progress_output`) already had correct `{'diffraction_to_obj': MagicMock()}` setup
- No changes needed for these tests

---

## Test Results

### CLI Inference Tests (Selector 1)
**Command:**
```bash
CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_inference_torch.py -vv
```

**Result:** ✅ **9 passed in 4.59s**

**Log:** `pytest_cli_inference_green.log` (29 KB)

**Test Breakdown:**
- `TestInferenceCLI` (4 tests): All execution config flag roundtrip tests PASSED
- `TestInferenceCLIThinWrapper` (5 tests): All delegation contract tests PASSED
  - `test_cli_delegates_to_validate_paths`: ✅ PASSED (keyword args assertion fixed)
  - `test_cli_delegates_to_helper_for_data_loading`: ✅ PASSED (bundle loader seeded)
  - `test_cli_delegates_to_inference_helper`: ✅ PASSED
  - `test_cli_calls_save_individual_reconstructions`: ✅ PASSED
  - `test_quiet_flag_suppresses_progress_output`: ✅ PASSED

### Integration Test (Selector 2)
**Command:**
```bash
CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv
```

**Result:** ✅ **1 passed in 16.75s**

**Log:** `pytest_cli_integration_green.log` (18 KB)

**Test:** `test_run_pytorch_train_save_load_infer` validates the complete train → save → load → infer cycle with PyTorch backend

---

## Artifacts Generated

All artifacts stored under:
```
plans/active/ADR-003-BACKEND-API/reports/2025-10-20T122425Z/phase_d_cli_wrappers_inference_followup/
```

| Artifact | Size | Description |
|----------|------|-------------|
| `summary.md` | 4.2 KB | This document |
| `pytest_cli_inference_green.log` | 2.9 KB | Full pytest output for CLI inference tests (9 passed) |
| `pytest_cli_integration_green.log` | 1.8 KB | Full pytest output for integration test (1 passed) |
| `train_debug.log` | 29.5 KB | Training debug log (relocated from repo root) |

---

## Phase D.C C3 Exit Criteria

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Update `test_cli_delegates_to_validate_paths` to inspect kwargs | ✅ | Line 278-281 in test file |
| Seed bundle loader mocks with `diffraction_to_obj` | ✅ | Lines 257, 304 in test file |
| CLI inference test selector GREEN | ✅ | 9/9 passed in 4.59s |
| Integration test selector GREEN | ✅ | 1/1 passed in 16.75s |
| Artifacts stored correctly | ✅ | All logs under timestamped directory |
| train_debug.log relocated from repo root | ✅ | Moved to artifact directory |

---

## Key Observations

1. **Keyword Args Pattern:** The shared helper contract (`validate_paths()`) uses keyword arguments, not positional. Tests must inspect `.call_args.kwargs` to avoid IndexError on positional tuple unpacking.

2. **Bundle Loader Contract:** The inference CLI workflow requires the bundle loader to return a dictionary with at least the `diffraction_to_obj` key. Mocking with an empty dictionary `{}` causes the workflow to fail before reaching delegation assertions.

3. **Test Hygiene:** Three of five thin-wrapper tests already had correct bundle loader setup, confirming the pattern was established during earlier Phase C4 work.

4. **Integration Test Alignment:** The integration test continues to pass, demonstrating that CLI changes maintain end-to-end workflow integrity.

---

## Next Steps (Phase D.C C4)

Per `input.md` guidance:
1. Update `phase_d_cli_wrappers/plan.md` to mark C3 row as `[x]`
2. Append Attempt #50 entry to `docs/fix_plan.md` with artifact references
3. Proceed to Phase D.C C4 (docs + hygiene cleanup)

---

## References

- **Plan:** `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md` §C3
- **Implementation:** `ptycho_torch/inference.py:365-546` (CLI main with helper delegation)
- **Shared Helpers:** `ptycho_torch/cli/shared.py:1-150` (`validate_paths`, `resolve_accelerator`, etc.)
- **Factory Design:** `.../2025-10-19T232336Z/phase_b_factories/factory_design.md`
- **Spec:** `specs/ptychodus_api_spec.md` §4.8 (CONFIG-001 ordering)
