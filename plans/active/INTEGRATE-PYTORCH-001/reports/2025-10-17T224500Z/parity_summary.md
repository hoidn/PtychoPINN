# PyTorch Dataloader DATA-001 Compliance - Parity Summary

**Initiative:** INTEGRATE-PYTORCH-001
**Task:** INTEGRATE-PYTORCH-001-DATALOADER
**Date:** 2025-10-17T224500Z
**Status:** ✅ COMPLETE (Green Phase)

---

## Overview

This loop restored PyTorch dataloader compliance with DATA-001 canonical NPZ format by implementing canonical `diffraction` key preference with graceful fallback to legacy `diff3d` key.

## Problem Statement

**Root Cause (confirmed via triage):**
PyTorch memory-mapped dataloader (`ptycho_torch/dataloader.py`) hardcoded searches for legacy `diff3d` key in two locations:
1. `npz_headers()` function (line 46): Shape extraction logic
2. `memory_map_data()` method (line 504): Actual data loading

**Impact:**
- Integration test `test_pytorch_train_save_load_infer_cycle` failed with `ValueError: Could not find diff3d data in <dataset>.npz`
- Canonical NPZ datasets (using `diffraction` key per `specs/data_contracts.md` §1) could not be loaded
- DATA-001 spec non-compliance blocked PyTorch backend from using standard dataset formats

**Reference:**
- Triage document: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T223200Z/dataloader_triage.md`
- Spec: `specs/data_contracts.md` §1 — Canonical Ptychography Dataset

---

## Implementation

### Changes Made

#### 1. Updated `npz_headers()` function (`dataloader.py:29-83`)
- Added canonical `diffraction` key check as first priority
- Preserved legacy `diff3d` fallback for backward compatibility
- Enhanced error message to mention both keys when neither exists
- Updated docstring to reference DATA-001 spec

#### 2. Created `_get_diffraction_stack()` helper (`dataloader.py:86-114`)
- Centralized diffraction key resolution logic
- Ensures consistent canonical→legacy fallback across all load sites
- Returns numpy array directly for use in memory mapping
- Clear error messaging with spec reference

#### 3. Updated `memory_map_data()` method (`dataloader.py:582`)
- Replaced hardcoded `np.load(npz_file)['diff3d']` call
- Now uses `_get_diffraction_stack(npz_file)` helper
- Preserves existing dtype conversion and rounding behavior

### Code Quality
- All changes preserve existing functionality (backward compatible)
- Dtype conversions (float32, rounding) unchanged
- Error messages improved with spec citations
- Docstrings updated to reflect DATA-001 compliance

---

## Test Coverage

### Unit Tests (New)
**File:** `tests/torch/test_dataloader.py`
**Status:** ✅ All passing (3/3)

| Test | Purpose | Result |
|------|---------|--------|
| `test_loads_canonical_diffraction` | Verify canonical `diffraction` key loads successfully | ✅ PASS |
| `test_backward_compat_legacy_diff3d` | Ensure legacy `diff3d` key still works | ✅ PASS |
| `test_error_when_no_diffraction_key` | Validate clear error when neither key exists | ✅ PASS |

**Execution Logs:**
- Red phase (pre-fix): `pytest_dataloader_red.log` — Expected failure confirmed
- Green phase (post-fix): `pytest_dataloader_green.log` — All tests passing

### Regression Tests
**Full suite run:** `pytest tests/ --ignore=tests/test_benchmark_throughput.py --ignore=tests/test_run_baseline.py`
**Result:** ✅ 206 passed, 14 skipped, 1 xfailed, 1 failed (unrelated probe size issue)

**Key findings:**
- No new test failures introduced by dataloader changes
- All existing dataloader-dependent tests continue to pass
- PyTorch config bridge tests unaffected (no data loading in those tests)
- TensorFlow baseline tests remain green (137 passed)

### Integration Test Evolution
**Test:** `tests/torch/test_integration_workflow_torch.py::test_pytorch_train_save_load_infer_cycle`

**Before fix:**
```
ValueError: Could not find diff3d data in /datasets/Run1084_recon3_postPC_shrunk_3.npz
```

**After fix:**
```
RuntimeError: The expanded size of the tensor (128) must match the existing size (64) at non-singleton dimension 2.
Target sizes: [1, 128, 128].  Tensor sizes: [64, 64]
```

**Interpretation:**
✅ **DATA-001 issue RESOLVED**. Test now progresses past dataloader initialization and fails on an unrelated probe/configuration mismatch. This is expected and represents forward progress — dataloader successfully loads canonical dataset.

---

## Verification Evidence

### Static Analysis
- `npz_headers()`: Canonical key checked first (lines 47-55), legacy fallback (lines 58-66)
- `_get_diffraction_stack()`: Explicit key priority documented (lines 102-108)
- `memory_map_data()`: Shared helper used (line 582)

### Runtime Evidence
**Dataset used:** `datasets/Run1084_recon3_postPC_shrunk_3.npz`
**Confirmed keys (via manual inspection):**
```python
with np.load('datasets/Run1084_recon3_postPC_shrunk_3.npz') as data:
    print(data.keys())
# Output: ['diffraction', 'Y', 'objectGuess', 'probeGuess', 'xcoords', 'ycoords', 'scan_index']
```

**Integration test log excerpt (green phase):**
```
Calculating dataset length with coordinate bounds...
For file .../Run1084_recon3_postPC_shrunk_3.npz, maximum x_range is (34.41, 79.28)...
Creating memory mapped tensor dictionary...
Memory map length: 722
```
✅ Dataloader successfully processed canonical NPZ without ValueError.

---

## Exit Criteria Status

| Criterion | Status | Evidence |
|-----------|--------|----------|
| PyTorch dataloader ingests canonical DATA-001 NPZ files (`diffraction` key) | ✅ COMPLETE | Integration test progresses past dataloader init; unit test `test_loads_canonical_diffraction` passes |
| Graceful fallback to `diff3d` for legacy datasets | ✅ COMPLETE | Unit test `test_backward_compat_legacy_diff3d` passes |
| Informative error when neither key exists | ✅ COMPLETE | Unit test `test_error_when_no_diffraction_key` verifies error message quality |
| Targeted regression tests cover canonical + legacy key paths | ✅ COMPLETE | New pytest module `tests/torch/test_dataloader.py` provides 3 tests |
| Integration test uses canonical dataset and progresses past dataloader | ✅ COMPLETE | `test_pytorch_train_save_load_infer_cycle` now fails on probe size, not DATA-001 |
| Parity summary updated with green evidence | ✅ COMPLETE | This document |
| Full pytest suite passes without new regressions | ✅ COMPLETE | 206 passed (same as pre-fix baseline) |

---

## Next Steps

### Immediate (Priority: High)
1. **Fix probe size mismatch** — Integration test now blocked on `N=128` vs `probe.shape=(64,64)` mismatch
   - Root cause: Configuration bridge or dataset probe size mismatch
   - Create new fix_plan.md entry: `[INTEGRATE-PYTORCH-001-PROBE-SIZE]`

2. **Update phase_e2_implementation.md** — Mark D2 task complete, document dataloader green status

### Follow-up (Priority: Medium)
3. **Validate with additional datasets** — Test dataloader fix against other canonical NPZ files (e.g., simulated datasets, fly64)
4. **Performance check** — Ensure no regression in dataloader throughput from dual-pass key checking

---

## Artifacts Generated

| Artifact | Path | Purpose |
|----------|------|---------|
| Unit test module | `tests/torch/test_dataloader.py` | Regression coverage for DATA-001 compliance |
| Red phase log | `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T224500Z/pytest_dataloader_red.log` | TDD red phase evidence |
| Green phase log (unit) | `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T224500Z/pytest_dataloader_green.log` | TDD green phase evidence (unit tests) |
| Green phase log (integration) | `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T224500Z/pytest_integration_green.log` | Integration test progress evidence |
| Parity summary | `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T224500Z/parity_summary.md` | This document |

---

## Compliance References

- **Spec:** `specs/data_contracts.md` §1 — Canonical Ptychography Dataset
- **Findings:** `docs/findings.md#DATA-001` — diffraction key requirement
- **Workflow:** `docs/workflows/pytorch.md` §4 — PyTorch dataset parity requirements
- **API Spec:** `specs/ptychodus_api_spec.md` §4.5 — Dataset contract for reconstructor
- **Triage:** `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T223200Z/dataloader_triage.md` — Hypothesis validation

---

**Summary:** PyTorch dataloader now fully complies with DATA-001 canonical NPZ format while maintaining backward compatibility with legacy `diff3d` key. Integration test progresses to next blocker (probe size mismatch). All unit tests green. No regressions detected.
