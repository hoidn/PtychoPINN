# Test Suite Baseline Summary — 2025-10-16T230539Z

## Executive Summary
**Status:** ✅ **GREEN BASELINE** — All tests either passing or intentionally skipped.

- **Total Tests:** 165 collected (153 passed + 12 skipped)
- **Passed:** 153 (100% of runnable tests)
- **Failed:** 0
- **Skipped:** 12 (intentional, documented reasons)
- **Execution Time:** 201.42s (3m 21s)
- **Exit Code:** 0 (success)

## Key Findings

### 1. Test Suite Health: EXCELLENT
The pytest suite is in a **healthy, green state** with zero failures. This is a significant positive finding and indicates:
- Core functionality is stable
- Recent development has maintained test discipline
- CI/CD pipeline should be green
- No blocking issues for development work

### 2. Skipped Tests Analysis

#### A. PyTorch-Related (2 tests) — EXPECTED
- `tests/test_pytorch_tf_wrapper.py::21` - "PyTorch not available"
- `tests/torch/test_tf_helper.py::20` - "PyTorch not available"
- **Reason:** PyTorch installation issue (symbol error: `ncclCommWindowRegister`)
- **Classification:** Environmental, non-blocking for TensorFlow workflows
- **Recommendation:** Track under separate PyTorch integration initiative (see `docs/fix_plan.md` [TEST-PYTORCH-001])

#### B. Data Dependency (2 tests) — EXPECTED
- `tests/test_generic_loader.py::46` - "Data loading failed"
- `tests/test_integration_baseline_gs2.py::27` - "Test data not found at datasets/fly/fly001_transposed.npz"
- **Reason:** Missing optional test dataset
- **Classification:** Data availability, not code defect
- **Recommendation:** Document dataset acquisition in `docs/FLY64_DATASET_GUIDE.md` or mark as optional/CI-only

#### C. API Deprecation (1 test) — DOCUMENTED
- `tests/test_misc.py:45` - "Deprecated: generate_simulated_data API changed from (obj,probe,nimages) to (config,obj,probe) and memoization disabled"
- **Reason:** Intentional API evolution, test marked deprecated
- **Classification:** Expected, properly documented deprecation
- **Recommendation:** Remove test in future cleanup cycle or update to new API

#### D. TensorFlow Addons Migration (7 tests) — EXPECTED
- `tests/test_tf_helper.py::199` - "TensorFlow Addons removed in TF 2.19 migration"
- `tests/test_tf_helper_edge_aware.py` (6 tests) - "tensorflow_addons not available"
- **Reason:** TF 2.19 migration removed TF Addons dependency
- **Classification:** Known architectural change, documented migration
- **Recommendation:** Tests are properly skipped, alternative implementation in place

### 3. Test Coverage by Category

#### Passing Test Modules (100% pass rate)
- **Image Processing** (20 tests) - Registration, cropping, alignment ✅
- **Baseline Models** (1 test) - Model architecture ✅
- **Benchmarking** (13 tests) - Performance profiling ✅
- **CLI** (7 tests) - Argument parsing, logging config ✅
- **Coordinate Grouping** (15 tests) - Efficient sampling algorithm ✅
- **Integration** (1 test) - Full train→save→load→infer workflow ✅
- **Logging** (9 tests) - Advanced logging system ✅
- **Metadata** (6 tests) - nphotons metadata integration ✅
- **Oversampling** (5 tests) - Automatic oversampling triggers ✅
- **Projective Warp** (12 tests) - XLA-compatible transformation ✅
- **Raw Data** (9 tests) - Data grouping and sampling ✅
- **Baselines** (2 tests) - Baseline data preparation ✅
- **Scaling** (7 tests) - Physics scaling regression prevention ✅
- **Sequential Sampling** (10 tests) - Deterministic sampling mode ✅
- **Subsampling** (11 tests) - Independent sampling control ✅
- **TF Helper** (14 tests) - TensorFlow utilities ✅
- **Workflow Components** (7 tests) - Model loading robustness ✅
- **Tools** (2 tests) - Dataset update utilities ✅

## Failure Ledger
**No failures detected.** ✅

## Recommendations

### Immediate Actions (Ralph)
1. **Record this green baseline in docs/fix_plan.md** — Update [TEST-SUITE-TRIAGE] status to "done" with link to this report.
2. **No remediation needed** — Skip Phase B/C (failure classification/sequencing) since baseline is clean.
3. **Proceed with PyTorch work** — Green baseline clears path for [TEST-PYTORCH-001] and [INTEGRATE-PYTORCH-001] initiatives.

### Future Maintenance
1. **PyTorch Environment Fix:** Resolve `ncclCommWindowRegister` symbol error to enable PyTorch tests.
2. **Dataset Provisioning:** Document or provision `datasets/fly/fly001_transposed.npz` for completeness.
3. **Deprecation Cleanup:** Remove or modernize `test_misc.py::test_memoize_simulated_data` in next cleanup cycle.
4. **TF Addons Tests:** Consider removing TF Addons-related tests entirely if functionality confirmed replaced.

## Artifacts
- Full pytest log: `pytest.log`
- Environment metadata: `env.md`
- Package snapshot: `requirements.txt`

## Validation
- [x] Phase A Task A1: Environment documented in `env.md`
- [x] Phase A Task A2: Pytest sweep executed and logged
- [x] Phase A Task A3: Failure manifest extracted (N/A - zero failures)
- [x] Exit code captured: 0 (success)
- [x] Artifact paths recorded in this summary

## Next Steps
Per plan Phase B/C: **NOT REQUIRED** — No failures to classify or sequence.

Supervisor should:
1. Mark [TEST-SUITE-TRIAGE] as complete in `docs/fix_plan.md`
2. Update `input.md` to direct Ralph to next initiative (PyTorch integration)
3. Archive this report for future baseline comparison

---
**Generated:** 2025-10-16T23:05:39Z
**Command:** `pytest tests/ -vv`
**Branch:** `feature/torchapi`
**Commit:** 521fe85
