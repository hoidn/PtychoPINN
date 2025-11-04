# Phase G2 Execution Runs Summary

**Date:** 2025-11-05
**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
**Phase:** G2 - Deterministic comparison execution
**Artifacts Root:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T190500Z/phase_g_execution_runs/`

---

## Objectives

1. Extend `execute_comparison_jobs()` to record `n_success`/`n_failed` summary counts in execution manifests
2. Update CLI to display these summary counts
3. Execute dose=1000 dense train comparison to capture real-run evidence
4. Document blocking conditions for sparse comparisons

---

## Implementation Summary

### Code Changes

**File:** `studies/fly64_dose_overlap/comparison.py`

1. **`execute_comparison_jobs()` function (lines 239-252):**
   - Added calculation of `n_success` (jobs with returncode==0) and `n_failed` (all other jobs)
   - Updated manifest dict to include `n_success` and `n_failed` fields
   - These fields are now persisted in the execution manifest JSON

2. **`main()` CLI function (lines 363-381):**
   - Updated manifest merge to include `n_success` and `n_failed` from execution results
   - Updated CLI summary output to use `execution_manifest['n_success']` and `execution_manifest['n_failed']`
   - Updated return code logic to use `execution_manifest['n_failed']`

### Test Coverage

**File:** `tests/study/test_dose_overlap_comparison.py`

Added `test_execute_comparison_jobs_records_summary` (lines 193-251):
- Tests that `n_success` and `n_failed` fields are present in manifest
- Tests that counts are calculated correctly (1 success, 1 failure in mock scenario)
- Tests that `n_success + n_failed == n_executed` invariant holds
- **TDD Cadence:** RED test run captured in `red/pytest_phase_g_executor_red.log` showing expected failure
- **GREEN test run** captured in `green/pytest_phase_g_executor_green.log` showing PASSED status

---

## Test Results

### Targeted Test (Phase G2 Executor)

**RED test:**
- File: `red/pytest_phase_g_executor_red.log`
- Result: FAILED with `AssertionError: Manifest missing 'n_success' field`
- This confirmed the test correctly identified missing functionality

**GREEN test:**
- File: `green/pytest_phase_g_executor_green.log`
- Result: PASSED (1 passed in 0.83s)
- Implementation correctly provides `n_success` and `n_failed` fields

### Full Comparison Test Suite

**File:** `green/pytest_phase_g_suite_green.log`
- All 3 comparison tests PASSED:
  - `test_build_comparison_jobs_creates_all_conditions`
  - `test_execute_comparison_jobs_invokes_compare_models`
  - `test_execute_comparison_jobs_records_summary` (new)

### Test Collection

**File:** `collect/pytest_phase_g_collect.log`
- Selector `tests/study/test_dose_overlap_comparison.py -k comparison` collects 3 tests
- All tests have clear docstrings with exit criteria
- New test documents AT-G2.1 acceptance criteria

### Comprehensive Test Gate

- **Full suite:** 394 passed, 17 skipped, 1 failed (pre-existing unrelated failure in `test_interop_h5_reader` due to missing ptychodus module)
- **Study tests:** 46/46 passed
- **No regressions** introduced by Phase G2 changes

---

## CLI Execution Results

### dose=1000 dense train

**Command:**
```bash
python -m studies.fly64_dose_overlap.comparison \
  --phase-c-root tmp/phase_c_f2_cli \
  --phase-e-root tmp/phase_e_training_gs2 \
  --phase-f-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/phase_f_cli_test \
  --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T190500Z/phase_g_execution_runs/cli/dose1000_dense_train \
  --dose 1000 --view dense --split train
```

**Result:**
- Return code: Non-zero (1 failure)
- Manifest: `cli/dose1000_dense_train/comparison_manifest.json`
- Log: `cli/dose1000_dense_train.log`

**Execution Summary (from CLI output):**
```
Total jobs: 1
Executed: 1
Success: 0
Failed: 1
```

**Failure Cause:**
- Missing `wts.h5.zip` in Phase E checkpoint directory (`tmp/phase_e_training_gs2/pinn/`)
- Error: `FileNotFoundError: Model archive not found at: tmp/phase_e_training_gs2/pinn/wts.h5.zip`
- This is expected - Phase E checkpoints are stub files (0 bytes) from scaffolding, not real trained models

**Verification:**
- Executor correctly captured the failure with returncode=1
- Manifest includes `n_success: 0`, `n_failed: 1`
- Log file contains full stdout/stderr from failed comparison script

---

## Blocking Conditions

### Phase E Training Missing

**Issue:** Phase E checkpoint artifacts (`wts.h5.zip`) do not exist
- Current state: `checkpoint.h5` files are 0-byte stubs
- Impact: All Phase G comparisons will fail until real Phase E training completes
- Required action: Execute Phase E training jobs to generate real PINN and baseline checkpoints

### Phase C Sparse View Missing

**Issue:** Sparse view datasets not generated for dose=1000
- Missing: `tmp/phase_c_f2_cli/dose_1000/sparse/sparse_{train,test}.npz`
- Current state: Only dense view (`patched_{train,test}.npz`) exists
- Impact: Cannot execute sparse comparisons as planned in `input.md`
- Required action: Generate Phase C sparse view datasets per Phase B overlap specifications

### Phase F Sparse Manifests Missing

**Issue:** Phase F pty-chi manifests only exist for dense_train
- Exists: `phase_f_cli_test/dose_1000_dense_train/manifest.json`
- Missing: Manifests for dense_test, sparse_train, sparse_test
- Impact: Cannot execute any comparisons except dense_train
- Required action: Execute Phase F pty-chi reconstruction for all view/split combinations

---

## Metrics File Inventory

No metrics CSV files were generated due to blocking Phase E condition (comparison scripts failed before producing outputs).

**Expected artifacts when Phase E unblocked:**
```
cli/dose1000_dense_train/dose_1000/dense/train/
  ├── comparison.log (exists, captured failure)
  ├── metrics.csv (will exist after Phase E training)
  ├── correlation_heatmap.png (will exist after Phase E training)
  └── ms_ssim_heatmap.png (will exist after Phase E training)
```

---

## Outstanding Gaps

1. **Phase E Training:** Blocking all real comparisons
   - Priority: HIGH
   - Action: Execute Phase E training jobs with real datasets to generate `wts.h5.zip` artifacts

2. **Phase C Sparse Generation:** Blocking sparse comparisons
   - Priority: MEDIUM
   - Action: Run Phase B overlap filtering for dose=1000 sparse views

3. **Phase F Remaining Splits:** Blocking dense_test and all sparse comparisons
   - Priority: MEDIUM (depends on Phase C sparse completion for sparse views)
   - Action: Execute Phase F pty-chi reconstruction for dense_test, sparse_train, sparse_test

4. **Documentation Updates:** Selector registry needs Phase G2 test
   - Priority: LOW
   - Action: Update `docs/TESTING_GUIDE.md` §Phase G and `docs/development/TEST_SUITE_INDEX.md`

---

## Exit Criteria Status

### Phase G2.1 Deliverables (from implementation plan)

- [x] **Execution manifest includes summary fields:** `n_success` and `n_failed` fields implemented and tested
- [x] **CLI displays summary counts:** Updated `main()` to emit success/failure counts
- [x] **RED→GREEN test evidence:** Captured in `red/` and `green/` subdirectories
- [x] **Real-run logs captured:** `cli/dose1000_dense_train.log` demonstrates end-to-end orchestration
- [ ] **Successful comparison metrics:** BLOCKED by Phase E (expected per inventory analysis)

**Overall Status:** Phase G2 implementation COMPLETE; execution evidence captured; comparisons blocked by Phase E training as expected.

---

## Next Actions

1. **Prioritize Phase E Training:** This is the critical path blocker for all Phase G comparisons
2. **After Phase E completes:** Re-run dose=1000 dense_train comparison to capture success case with real metrics
3. **Generate Phase C sparse views:** Unblocks sparse comparison workflows
4. **Execute remaining Phase F reconstructions:** Enables full dose=1000 comparison matrix
5. **Update documentation:** Once all comparisons run successfully, update selector registry and test index

---

## Artifact Manifest

```
plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T190500Z/phase_g_execution_runs/
├── red/
│   └── pytest_phase_g_executor_red.log          (RED test showing expected failure)
├── green/
│   ├── pytest_phase_g_executor_green.log       (GREEN test PASSED)
│   └── pytest_phase_g_suite_green.log          (Full comparison suite PASSED)
├── collect/
│   └── pytest_phase_g_collect.log              (Test collection proof)
├── cli/
│   ├── dose1000_dense_train.log                (CLI execution log with failure capture)
│   └── dose1000_dense_train/
│       ├── comparison_manifest.json            (Manifest with n_success/n_failed)
│       ├── comparison_summary.txt              (Job enumeration summary)
│       └── dose_1000/dense/train/
│           └── comparison.log                  (Subprocess stderr/stdout capture)
├── analysis/
│   └── summary.md                              (This file)
└── docs/
    └── (reserved for selector registry updates)
```

---

## References

- **Fix Plan:** `docs/fix_plan.md` STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 section
- **Implementation Plan:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md`
- **Phase G Plan:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T140500Z/phase_g_comparison_plan/plan/plan.md`
- **Inventory:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T162500Z/phase_g_inventory/analysis/inventory.md`
- **Test Strategy:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md`
