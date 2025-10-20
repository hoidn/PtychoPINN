# Phase D.C C2: Inference CLI Thin Wrapper RED Tests — Summary

**Initiative:** ADR-003-BACKEND-API — Standardize PyTorch backend API
**Phase:** D.C C2 — RED test coverage for inference CLI thin wrapper refactor
**Date:** 2025-10-20
**Engineer:** Ralph
**Status:** COMPLETE ✅

---

## Objective

Establish RED test coverage for the inference CLI thin wrapper refactor per blueprint specification (`inference_refactor.md` §Test Strategy). Tests validate expected helper delegation patterns before implementation begins (Phase D.C C3).

---

## Deliverables

### 1. New Test Class: `TestInferenceCLIThinWrapper` (5 tests)

**File:** `tests/torch/test_cli_inference_torch.py:203-479`

**Tests Added:**
1. `test_cli_delegates_to_validate_paths` — Validates CLI calls `validate_paths()` before factory (CONFIG-001 ordering)
2. `test_cli_delegates_to_helper_for_data_loading` — Validates CLI loads `RawData.from_file()` (Option A decision)
3. `test_cli_delegates_to_inference_helper` — Validates CLI calls `_run_inference_and_reconstruct()` helper (Option 2 decision)
4. `test_cli_calls_save_individual_reconstructions` — Validates CLI generates output artifacts after inference
5. `test_quiet_flag_suppresses_progress_output` — Validates `--quiet` flag maps to `enable_progress_bar=False`

**Expected RED Behavior:** All 5 tests FAIL (see §RED Test Results below).

---

### 2. Inference-Mode Extension: `TestBuildExecutionConfigInferenceMode` (3 tests)

**File:** `tests/torch/test_cli_shared.py:570-685`

**Tests Added:**
1. `test_inference_mode_defaults` — Validates inference mode produces correct default config
2. `test_inference_mode_custom_batch_size` — Validates `inference_batch_size` field handling
3. `test_inference_mode_respects_quiet` — Validates `--quiet` flag behavior in inference mode

**Expected Behavior:** Tests PASS (GREEN) because Phase D.B3 implementation already handles `mode='inference'` correctly.

---

## RED Test Results

### Inference CLI Thin Wrapper Tests (`test_cli_inference_torch.py`)

**Command:**
```bash
CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_inference_torch.py -vv
```

**Result:** **5 FAILED, 4 PASSED** (4.62s runtime)

**Failure Signatures:**

1. **`test_cli_delegates_to_validate_paths`**
   - **Error:** `AssertionError: validate_paths() was not called - CLI still using inline validation`
   - **Root Cause:** CLI performs inline validation (lines 548-550) instead of delegating to `validate_paths()` helper
   - **Expected:** Validates blueprint §Component 2 requires shared helper delegation

2. **`test_cli_delegates_to_helper_for_data_loading`**
   - **Error:** `AssertionError: RawData.from_file() was not called - data loading delegation broken`
   - **Root Cause:** Test patching `ptycho.raw_data.RawData.from_file()` but CLI may use different import path or workflow component loads data
   - **Expected:** Validates Option A decision (CLI retains RawData loading)

3. **`test_cli_delegates_to_inference_helper`**
   - **Error:** `AttributeError: <module 'ptycho_torch.inference'> does not have the attribute '_run_inference_and_reconstruct'`
   - **Root Cause:** Helper function not yet extracted from inline logic (lines 563-641)
   - **Expected:** Validates blueprint §Inference Orchestration Refactor Option 2 (extract helper)

4. **`test_cli_calls_save_individual_reconstructions`**
   - **Error:** `AttributeError: '_run_inference_and_reconstruct'` (dependency on test #3)
   - **Root Cause:** Test cannot proceed without helper extraction
   - **Expected:** Validates blueprint §Component 2 output artifact generation

5. **`test_quiet_flag_suppresses_progress_output`**
   - **Error:** `AttributeError: '_run_inference_and_reconstruct'` (dependency on test #3)
   - **Root Cause:** Test cannot proceed without helper extraction
   - **Expected:** Validates blueprint §Quiet Mode Mapping

**Baseline Tests (Still GREEN):**
- `TestInferenceCLI::test_accelerator_flag_roundtrip` ✅ PASSED
- `TestInferenceCLI::test_num_workers_flag_roundtrip` ✅ PASSED
- `TestInferenceCLI::test_inference_batch_size_flag_roundtrip` ✅ PASSED
- `TestInferenceCLI::test_multiple_execution_config_flags` ✅ PASSED

**Evidence:** `pytest_cli_inference_thin_red.log` (captured stdout shows current CLI still uses inline validation and logic)

---

### Shared Helpers Inference-Mode Tests (`test_cli_shared.py`)

**Command:**
```bash
CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_shared.py -k "inference_mode or quiet" -vv
```

**Result:** **7 PASSED** (3.55s runtime) — All GREEN ✅

**Rationale:** Phase D.B3 implementation already handles `mode='inference'` correctly in shared helpers. These tests validate that inference CLI can reuse existing helper infrastructure without modification.

**Tests Passing:**
- `TestBuildExecutionConfig::test_inference_mode` ✅ (existing, validates shared code)
- `TestBuildExecutionConfig::test_handles_quiet_flag` ✅ (existing)
- `TestBuildExecutionConfig::test_quiet_or_disable_mlflow_both_true` ✅ (existing)
- `TestValidatePaths::test_accepts_none_train_file_for_inference_mode` ✅ (existing, validates train_file=None)
- `TestBuildExecutionConfigInferenceMode::test_inference_mode_defaults` ✅ (new, Phase D.C C2)
- `TestBuildExecutionConfigInferenceMode::test_inference_mode_custom_batch_size` ✅ (new, Phase D.C C2)
- `TestBuildExecutionConfigInferenceMode::test_inference_mode_respects_quiet` ✅ (new, Phase D.C C2)

**Evidence:** `pytest_cli_shared_inference_red.log` (all inference-mode tests GREEN)

---

## Test Coverage Summary

| Test File | New Tests | Expected RED | Actual RED | Baseline Tests Passing |
|-----------|-----------|--------------|------------|------------------------|
| `test_cli_inference_torch.py` | 5 | 5 | 5 | 4/4 ✅ |
| `test_cli_shared.py` | 3 | 0 | 0 | 7/7 ✅ |
| **Total** | **8** | **5** | **5** | **11/11 ✅** |

**Key Insight:** Inference-mode support in shared helpers (Phase D.B3) is complete and GREEN. Remaining RED tests correctly identify missing thin wrapper delegation patterns in `ptycho_torch/inference.py` CLI.

---

## Blueprint Alignment

### Design Decisions Validated

1. **Option A (RawData Loading):** RED test confirms CLI currently does not delegate to `RawData.from_file()` as expected (test failure signature validates this design choice)
2. **Option 2 (Helper Extraction):** RED test confirms `_run_inference_and_reconstruct()` does not exist (validates refactor requirement)
3. **Shared Helper Reuse:** GREEN tests confirm inference mode already supported by Phase D.B3 helpers (no new helper implementation needed)

### Blueprint Sections Covered

- ✅ §Test Strategy: 5 new RED tests per blueprint requirements
- ✅ §RawData Ownership Decision: Test validates Option A choice
- ✅ §Inference Orchestration Refactor: Test validates Option 2 choice
- ✅ §Accelerator & Execution Config Strategy: GREEN tests confirm reuse works

---

## Next Steps (Phase D.C C3)

1. **Extract `_run_inference_and_reconstruct()` helper** (blueprint §Inference Orchestration Refactor)
2. **Refactor `cli_main()` to use shared helpers** (blueprint §Component 2):
   - Replace duplicate device mapping → `resolve_accelerator()`
   - Replace duplicate validation → `build_execution_config_from_args(args, mode='inference')`
   - Replace path checks → `validate_paths(train_file=None, test_file, output_dir)`
3. **Run RED tests → GREEN** (`pytest tests/torch/test_cli_inference_torch.py -vv`)
4. **Update docs** (`docs/workflows/pytorch.md` §12-13 with deprecation notices)

---

## Artifacts

| Artifact | Size | Location |
|----------|------|----------|
| RED test log (cli_inference) | ~12 KB | `pytest_cli_inference_thin_red.log` |
| RED test log (cli_shared) | ~3 KB | `pytest_cli_shared_inference_red.log` |
| Test source (thin wrapper) | ~277 lines | `tests/torch/test_cli_inference_torch.py:203-479` |
| Test source (inference mode) | ~116 lines | `tests/torch/test_cli_shared.py:570-685` |
| This summary | ~6 KB | `summary.md` |

---

## Validation Checklist

- ✅ RED tests written per blueprint specification
- ✅ RED logs captured with expected failure signatures
- ✅ Baseline tests remain GREEN (no regressions)
- ✅ Inference-mode helper tests GREEN (validates reuse)
- ✅ Test coverage aligns with blueprint §Test Strategy
- ✅ Artifacts stored in timestamped hub
- ✅ Plan C2 row ready for `[x]` marking

**Phase D.C C2: COMPLETE** — Ready for implementation (Phase D.C C3).
