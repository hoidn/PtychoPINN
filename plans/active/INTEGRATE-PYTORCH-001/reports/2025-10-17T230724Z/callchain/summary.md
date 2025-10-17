# INTEGRATE-PYTORCH-001-DATALOADER-INDEXING Root Cause & Resolution Summary

**Date:** 2025-10-17
**Loop:** Ralph evidence + implementation
**Status:** ✅ RESOLVED

## Problem Statement

Integration test `tests/torch/test_integration_workflow_torch.py::test_pytorch_train_save_load_infer_cycle` failed with:

```
IndexError: index 367 is out of bounds for dimension 0 with size 64
```

Occurred at `ptycho_torch/dataloader.py:617` when trying to index into diffraction stack.

## Root Cause Analysis

### The Bug
Dataset `datasets/Run1084_recon3_postPC_shrunk_3.npz` uses **legacy (H, W, N) format**: `(64, 64, 1087)` instead of DATA-001 compliant `(N, H, W)` format: `(1087, 64, 64)`.

### The Failure Mechanism
1. `npz_headers()` read shape `(64, 64, 1087)` from NPZ header
2. Memory maps allocated with H=64, W=64 (treating dim[0] as N=64)
3. `_get_diffraction_stack()` loaded the array as-is: `(64, 64, 1087)`
4. `group_coords()` generated `nn_indices` containing global coordinate indices (0-1086)
5. Line 617 tried: `diff_stack[nn_indices[j:local_to]]` with indices up to 367
6. **IndexError**: `diff_stack` dimension 0 only had size 64, not 1087

### Evidence
- Debugger agent verified NPZ file structure: `data['diffraction'].shape = (64, 64, 1087)`
- TensorFlow backend has commented-out transpose logic at `ptycho/loader.py:398`
- No prior tests exercised this specific dataset with PyTorch backend

## Solution Implemented

### Auto-Transpose with Format Detection

**Modified Functions:**

1. **`_get_diffraction_stack()` (lines 86-129)**
   - Added heuristic: if `shape[2] > max(shape[0], shape[1])`, transpose `(H,W,N) → (N,H,W)`
   - Emits warning when legacy format detected
   - Works with both `diffraction` and `diff3d` keys

2. **`npz_headers()` (lines 75-81)**
   - Applied **same heuristic** to shape read from NPZ header
   - **CRITICAL**: Memory map allocation uses this shape, so both functions MUST agree

### Test Coverage

Added `TestDataloaderFormatAutoTranspose` class with 6 tests:
- Legacy (64,64,100) → (100,64,64) transpose
- Canonical (100,64,64) preserved
- Edge case (65,64,64) preserved (ambiguous)
- `diff3d` key backward compat
- Real Run1084 dimensions (64,64,1087) → (1087,64,64)
- `npz_headers()` shape transpose verification

**All unit tests GREEN** ✅

## Verification

### Integration Test Status
- **Before fix**: IndexError at line 617 during memory map population
- **After fix**: Training completes successfully, emits warning:
  ```
  ⚠ Legacy format (64, 64, 1087) detected in .../Run1084_recon3_postPC_shrunk_3.npz,
  transposing to DATA-001 compliant (N, H, W)
  ```
- **New failure point**: Inference checkpoint loading (different issue, outside scope)

### Test Suite Regression Check
```bash
pytest tests/ --ignore=tests/test_benchmark_throughput.py --ignore=tests/test_run_baseline.py
```

**Results:**
- 217 passed ✅
- 14 skipped (expected)
- 1 xfailed (expected)
- 1 failed (integration test, inference phase - separate issue)

**Conclusion:** No regressions introduced. All previously passing tests remain green.

## Artifacts
- Unit tests: `tests/torch/test_dataloader.py` (lines 181-369)
- Implementation: `ptycho_torch/dataloader.py` (lines 75-81, 118-127)
- Failure log (original): `pytest_integration_fail.log`
- Fixed run log: `pytest_integration_fixed.log`

## Next Steps
Integration test now blocks on:
```
TypeError: PtychoPINN_Lightning.__init__() missing 4 required positional arguments
```

This is a **separate issue** tracked under inference workflow integration (likely Phase D2 stub completion). The dataloader indexing bug is **resolved**.

## Recommendations
1. Document FORMAT-001 finding in `docs/findings.md`
2. Consider preprocessing canonical datasets to avoid runtime transpose
3. Add DATA-001 compliance check to dataset validation scripts
4. Update `specs/data_contracts.md` with explicit (N,H,W) ordering requirement
