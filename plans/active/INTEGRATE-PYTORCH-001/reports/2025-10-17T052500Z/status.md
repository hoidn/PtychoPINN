# Phase B.B5 Loop Status — Config Bridge Parity Test Harness Refactor

**Timestamp:** 2025-10-17T052500Z
**Initiative:** INTEGRATE-PYTORCH-001
**Focus:** Phase B.B5.B0 (Convert parity tests to pytest style)
**Attempt:** #19

## Summary
Successfully refactored `TestConfigBridgeParity` from unittest.TestCase to pytest-style tests, unblocking parametrized test execution. All 34 parity tests and 1 MVP test now execute and pass without torch runtime dependency.

## Changes Made

### 1. Test Harness Refactor (`tests/torch/test_config_bridge.py`)
- Removed `unittest.TestCase` inheritance from `TestConfigBridgeParity` class (line 163)
- Created new pytest fixture `params_cfg_snapshot` (lines 151-160) to replace setUp/tearDown
- Converted all 13 test methods to use pytest fixture injection
- Replaced `self.assertEqual()` with plain `assert` statements
- Replaced `self.assertRaises()` with `pytest.raises()` context manager
- Replaced `self.assertIn()` with plain `assert ... in ...` checks
- Replaced `self.assertNotEqual()` with plain `assert ... !=` checks

### 2. Test Fixes (nphotons validation compliance)
- Added `nphotons=1e9` override to all TrainingConfig test calls to satisfy adapter's nphotons divergence validation (implemented in Attempt #17)
- Updated `test_train_data_file_required_error` to include nphotons override so validation reaches the train_data_file check
- Tests affected:
  - `test_training_config_direct_fields` (line 237)
  - `test_training_config_transform_fields` (line 300)
  - `test_training_config_override_fields` (line 361)
  - `test_train_data_file_required_error` (line 523)

## Test Results

### MVP Test (Unchanged Baseline)
```
pytest tests/torch/test_config_bridge.py::TestConfigBridgeMVP::test_mvp_config_bridge_populates_params_cfg -vv
1 passed in 2.95s
```
**Status:** ✅ PASSED — No regressions from harness refactor

### Targeted Parity Tests
```
pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity::test_model_config_direct_fields -vv
4 passed in 2.79s (N-direct, n_filters_scale-direct, object_big-direct, probe_big-direct)
```

```
pytest tests/torch/test_config_bridge.py -k "parity and (probe_mask or nphotons)" -vv
1 passed, 34 deselected in 2.86s (nphotons-divergence)
```
**Note:** No probe_mask-specific tests exist yet; field is covered in adapter implementation (Attempt #17).

### Full Parity Suite
```
pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity -v
34 passed in 2.72s
```
**Status:** ✅ ALL GREEN
- 4 direct field tests
- 1 training direct field test
- 6 model transform field tests
- 3 training transform field tests
- 4 model override field tests
- 7 training override field tests
- 2 inference override field tests
- 2 default divergence detection tests
- 3 gridsize/mode/activation error handling tests
- 2 required field error tests

### Full Regression Suite
```
pytest tests/ --ignore=tests/test_benchmark_throughput.py --ignore=tests/test_run_baseline.py -v
172 passed, 12 skipped in 214.10s (0:03:34)
```
**Status:** ✅ NO NEW FAILURES
- Pre-existing collection errors in benchmark/baseline tests (unrelated)
- Pre-existing skips (PyTorch unavailable, missing test data, deprecated APIs)

## Artifacts
- `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T052500Z/pytest_mvp.log` — MVP test baseline
- `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T052500Z/pytest_parity_direct.log` — Direct fields selector
- `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T052500Z/pytest_parity.log` — probe_mask/nphotons selector
- `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T052500Z/pytest_full.log` — Full regression suite
- `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T052500Z/status.md` — This summary

## Key Decisions
1. **Fixture Design:** Single fixture `params_cfg_snapshot` covers all parity tests; simpler than per-test fixtures given uniform teardown requirements.
2. **nphotons Override Policy:** Required in all TrainingConfig tests due to adapter validation (nphotons divergence detection is P0 blocker per parity plan); documented in field_matrix.md.
3. **Preserved Test Coverage:** No test cases removed; all 34 parameterized cases from red-phase design (Attempt #13) remain intact and now execute.

## Outstanding Gaps (Not Addressed in This Loop)
1. **Probe_mask parity tests:** Adapter implements probe_mask translation (Attempt #17), but no dedicated parameterized test cases exist. Consider adding explicit probe_mask test cases in Phase B.B5.B2 (deferred per parity plan Phase B notes).
2. **Baseline params.cfg comparison (Phase D.D1):** Deferred to future loop per parity_green_plan.md.
3. **n_subsample semantics (Phase C.C1):** Deferred; no tests currently cover this field.
4. **Override matrix documentation (Phase D.D2):** Deferred.

## Next Steps (Per Parity Green Plan)
1. Mark Phase B.B5.B0 complete in `plans/active/INTEGRATE-PYTORCH-001/implementation.md` ✅
2. Update `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T050930Z/parity_green_plan.md` to reflect B0 completion
3. Update docs/fix_plan.md Attempt #19 with this status summary
4. Consider Phase B.B5.B2/B4 parity assertion extensions for probe_mask (optional per plan guidance)
5. Proceed to Phase C or Phase D per initiative priorities

## Phase B.B5 Status
- [x] Phase A — Torch-optional harness (Attempt #15)
- [x] Phase B.B5.B0 — Refactor parity harness to pytest style (Attempt #19, this loop)
- [x] Phase B.B5.B1 — Implement probe_mask conversion (Attempt #17)
- [ ] Phase B.B5.B2 — Extend tests for probe_mask (optional; deferred)
- [x] Phase B.B5.B3 — Enforce nphotons override (Attempt #17)
- [ ] Phase B.B5.B4 — Tighten default divergence test (optional; covered by existing test_default_divergence_detection)
