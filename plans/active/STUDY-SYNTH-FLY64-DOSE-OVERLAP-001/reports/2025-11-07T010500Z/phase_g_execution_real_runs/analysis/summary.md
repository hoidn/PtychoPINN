# Phase G Comparison Path Fix — Summary

**Timestamp:** 2025-11-07T01:05:00Z
**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
**Loop:** 2025-11-07T010500Z
**Focus:** Phase G comparison builder dose-specific path correction

---

## Acceptance Criteria

**AT-G2.1:** Phase G comparison builder uses dose-specific Phase E paths
- ✅ Test fixture updated to reflect training.py:184,226 structure
- ✅ New test validates dose/view-specific checkpoint paths
- ✅ Comparison builder updated to use `dose_{dose}/{view}/gs2/wts.h5.zip` structure
- ✅ Baseline checkpoint path uses `dose_{dose}/baseline/gs1/wts.h5.zip` structure
- ✅ All existing tests pass with new path resolution

---

## Problem Statement

**Prior Bug (comparison.py:95-96):**
```python
pinn_checkpoint = phase_e_root / 'pinn' / 'checkpoint.h5'
baseline_checkpoint = phase_e_root / 'baseline' / 'checkpoint.h5'
```

**Issue:** Hardcoded flat structure did not match actual Phase E training outputs.

**Actual Phase E Structure (per training.py:184,226):**
```
phase_e_root/
  dose_1000/
    baseline/gs1/wts.h5.zip      # gridsize=1 baseline
    dense/gs2/wts.h5.zip          # gridsize=2 dense overlap
    sparse/gs2/wts.h5.zip         # gridsize=2 sparse overlap
```

---

## Implementation Summary

### Code Changes

**File:** `studies/fly64_dose_overlap/comparison.py:94-99`

**Before:**
```python
# Phase E checkpoints
pinn_checkpoint = phase_e_root / 'pinn' / 'checkpoint.h5'
baseline_checkpoint = phase_e_root / 'baseline' / 'checkpoint.h5'
```

**After:**
```python
# Phase E checkpoints (dose-specific structure per training.py:184,226)
# Baseline: dose_{dose}/baseline/gs1/wts.h5.zip
# View (dense/sparse): dose_{dose}/{view}/gs2/wts.h5.zip
dose_suffix = f'dose_{dose}'
pinn_checkpoint = phase_e_root / dose_suffix / view / 'gs2' / 'wts.h5.zip'
baseline_checkpoint = phase_e_root / dose_suffix / 'baseline' / 'gs1' / 'wts.h5.zip'
```

**Rationale:** Aligns with Phase E training module's dose-specific artifact directory structure, ensuring comparison jobs can locate baseline and PINN checkpoints correctly.

**File:** `studies/fly64_dose_overlap/comparison.py:45-47`

**Docstring Update:**
```python
phase_e_root : Path
    Root directory containing Phase E checkpoints (dose-specific structure:
    dose_{dose}/baseline/gs1/wts.h5.zip and dose_{dose}/{view}/gs2/wts.h5.zip)
```

---

## Test Evidence

### Test Addition

**File:** `tests/study/test_dose_overlap_comparison.py:260-323`

**New Test:** `test_build_comparison_jobs_uses_dose_specific_phase_e_paths`

**Validates:**
1. PINN checkpoint path: `phase_e_root/dose_1000/dense/gs2/wts.h5.zip`
2. Baseline checkpoint path: `phase_e_root/dose_1000/baseline/gs1/wts.h5.zip`
3. Paths include dose-specific directory (`dose_1000`)
4. View-specific structure for PINN (dense/gs2)
5. All paths exist and are accessible

**Test Fixture Updated:** `fake_phase_artifacts` (lines 41-53) now creates dose-specific Phase E structure matching training.py output.

### RED/GREEN Evidence

**RED State (Expected Failure):**
```
FileNotFoundError: PINN checkpoint not found: /tmp/.../phase_e/pinn/checkpoint.h5
```
Log: `plans/active/.../red/pytest_phase_e_paths_red.log`

**GREEN State (Pass):**
```
test_build_comparison_jobs_uses_dose_specific_phase_e_paths PASSED [100%]
```
Log: `plans/active/.../green/pytest_phase_e_paths_green.log`

**Full Suite:**
```
tests/study/test_dose_overlap_comparison.py::test_build_comparison_jobs_creates_all_conditions PASSED
tests/study/test_dose_overlap_comparison.py::test_execute_comparison_jobs_invokes_compare_models PASSED
tests/study/test_dose_overlap_comparison.py::test_execute_comparison_jobs_records_summary PASSED
tests/study/test_dose_overlap_comparison.py::test_build_comparison_jobs_uses_dose_specific_phase_e_paths PASSED

4 passed in 0.85s
```
Log: `plans/active/.../green/pytest_comparison_full.log`

### Comprehensive Test Suite

**Results:**
- **398 passed** (includes our new test + all existing tests)
- **1 failed** (pre-existing: `test_interop_h5_reader`)
- **17 skipped**
- **Time:** 415.98s (0:06:55)

**Delta from baseline:** +1 test (new path validation test)

Log: `plans/active/.../green/pytest_full.log`

---

## Artifacts

All under: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T010500Z/phase_g_execution_real_runs/`

- `red/pytest_phase_e_paths_red.log` — RED test (expected failure)
- `green/pytest_phase_e_paths_green.log` — GREEN test (pass after fix)
- `green/pytest_comparison_full.log` — Full comparison module tests
- `green/pytest_full.log` — Comprehensive test suite
- `analysis/summary.md` — This file

---

## SPEC/ADR Alignment

**SPEC Reference:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:184-189`
> Phase E produces training bundles under `dose_{dose}/{baseline|view}/gs{1|2}` with `wts.h5.zip` archives.

**Code Reference:** `studies/fly64_dose_overlap/training.py:184,226`
> ```python
> artifact_dir=artifact_root / dose_suffix / "baseline" / "gs1"
> artifact_dir=artifact_root / dose_suffix / view / "gs2"
> ```

**Alignment:** Comparison builder now follows Phase E's actual artifact layout, eliminating hardcoded flat paths that never existed.

---

## Next Steps

1. **Phase C/D/E/F Evidence Regeneration:** With path fix in place, regenerate dose_1000 assets under the new hub per input.md Do Now steps 3-7.
2. **Phase G Execution:** Run dense/train and dense/test comparisons with corrected paths (input.md steps 8-9).
3. **Findings Update:** No new finding required—this was a straightforward path alignment bug.

---

## Notes

- **No ARCH/SPEC conflict:** Both documents consistently describe dose-specific Phase E structure; comparison.py was simply out of sync.
- **Test hygiene:** Existing tests passed because fixture mocked flat structure; new test enforces real structure alignment.
- **Type safety:** Paths use `pathlib.Path` throughout per TYPE-PATH-001.
