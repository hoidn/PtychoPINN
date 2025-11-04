# Phase E Training E1 Summary

**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
**Phase:** E (Train PtychoPINN)
**Task:** E1 - Define training job matrix and builder (TDD)
**Date:** 2025-11-04
**Artifact Hub:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T060200Z/phase_e_training_e1/`

## Objective
Implement the training job matrix enumeration logic for Phase E using TDD (RED→GREEN), enabling systematic training across all dose/view/gridsize combinations.

## Deliverables

### 1. Test Infrastructure Design (Task E1a)
- **Updated:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md` §Phase E
- **Content:** Test selectors, coverage plan, execution proof requirements, and findings alignment for Phase E
- **Status:** ✅ Complete

### 2. TrainingJob Dataclass & Builder (Task E1b)
- **Module:** `studies/fly64_dose_overlap/training.py`
- **Components:**
  - `TrainingJob` dataclass with fields: dose, view, gridsize, train_data_path, test_data_path, artifact_dir, log_path
  - `build_training_jobs()` function enumerating 9 jobs per dose (3 doses × 3 variants)
  - Validation logic for view-gridsize invariants and dataset path existence
- **Status:** ✅ Complete
- **Lines:** 170 total (52 for dataclass, 118 for builder and docs)

### 3. RED→GREEN Test Cycle (Task E1c)
- **Test Module:** `tests/study/test_dose_overlap_training.py`
- **Test:** `test_build_training_jobs_matrix`
- **RED Phase:**
  - Log: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T060200Z/phase_e_training_e1/red/pytest_red.log`
  - Error: `ModuleNotFoundError: No module named 'studies.fly64_dose_overlap.training'`
  - Timestamp: 2025-11-04 (initial)
- **GREEN Phase:**
  - Log: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T060200Z/phase_e_training_e1/green/pytest_green.log`
  - Result: `1 passed in 1.47s`
  - Timestamp: 2025-11-04 (post-implementation)
- **Status:** ✅ Complete

### 4. Full Test Suite Validation (Hard Gate)
- **Log:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T060200Z/phase_e_training_e1/green/pytest_full_suite.log`
- **Result:** 377 passed, 17 skipped, 1 failed (pre-existing), 104 warnings in 245.73s
- **New Test Status:** `tests/study/test_dose_overlap_training.py::test_build_training_jobs_matrix PASSED [13%]`
- **Pre-existing Failure:** `tests/io/test_ptychodus_interop_h5.py::test_interop_h5_reader` (ModuleNotFoundError: ptychodus not installed)
- **Note:** The pre-existing failure is unrelated to Phase E changes and does not block this deliverable.
- **Status:** ✅ Complete (new test passed, no regressions introduced)

### 5. Test Documentation Updates (Task E1d)
- **Updated Files:**
  - `docs/TESTING_GUIDE.md` §Study Tests - added `test_dose_overlap_training.py` entry with selector
  - `docs/development/TEST_SUITE_INDEX.md` - added row for Phase E training job matrix test
- **Collection Log:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T060200Z/phase_e_training_e1/collect/pytest_collect.log`
- **Collected Tests:** 1 test (test_build_training_jobs_matrix)
- **Status:** ✅ Complete

## Test Coverage

### Test Assertions
The `test_build_training_jobs_matrix` test validates:
- Total job count: 9 jobs (3 doses × 3 variants)
- Job distribution: 3 jobs per dose (baseline, dense, sparse)
- View-gridsize invariants:
  - baseline → gridsize=1
  - dense/sparse → gridsize=2
- Dataset path existence (train and test NPZs)
- Artifact path structure (deterministic from dose/view/gridsize)
- Log path derivation (nested under artifact_dir)
- Path conventions:
  - Phase C baseline: `dose_{dose}/patched_{split}.npz`
  - Phase D overlap: `dose_{dose}/{view}_{split}.npz`

### Dependency Injection
Test uses `tmp_path` fixtures to fabricate:
- Phase C dataset tree (3 doses × 2 splits = 6 NPZs)
- Phase D dataset tree (3 doses × 2 views × 2 splits = 12 NPZs)
- Minimal DATA-001 compliant arrays (10 positions for Phase C, 5 for Phase D)

## Findings Applied
- **CONFIG-001:** `build_training_jobs()` remains pure (no `params.cfg` mutation). Legacy bridge via `update_legacy_dict()` deferred to execution helper (task E3).
- **DATA-001:** Dataset paths validated for existence; actual NPZ contract enforcement occurs during training via loader.
- **OVERSAMPLING-001:** Gridsize=2 jobs assume neighbor_count=7 from Phase D outputs.

## Exit Criteria Met
- [x] Test infrastructure design documented in test_strategy.md before implementation
- [x] RED test captured with expected failure (ModuleNotFoundError)
- [x] TrainingJob dataclass and build_training_jobs() implemented
- [x] GREEN test shows PASSED status with execution proof
- [x] Full test suite passed (377 passed, no new failures)
- [x] Test selectors documented in TESTING_GUIDE.md and TEST_SUITE_INDEX.md
- [x] Collection proof captured (1 test collected)

## Next Steps (E2)
Implement `run_training_job()` helper with:
- CONFIG-001 bridge via `update_legacy_dict(params.cfg, config)`
- CLI dry-run support for job execution preview
- Log capture to job-specific artifact directories

## References
- Implementation plan: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:133-144`
- Test strategy: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:84-115`
- Phase E plan: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T053500Z/phase_e_training_plan/plan.md:1-31`
- DATA-001 spec: `specs/data_contracts.md:190-260`
- CONFIG-001 guidance: `docs/DEVELOPER_GUIDE.md:68-104`
