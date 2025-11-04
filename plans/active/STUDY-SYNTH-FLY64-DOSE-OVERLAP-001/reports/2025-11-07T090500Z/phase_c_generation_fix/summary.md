# Phase C Generation Fix — Summary

**Loop Timestamp:** 2025-11-07T090500Z
**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
**Mode:** TDD
**Focus:** Fix Phase C dataset generation by wiring `TrainingConfig.n_images`

---

## Problem Statement

Phase C dataset generation failed with `TypeError: object of type 'float' has no len()` at `ptycho/raw_data.py:227`. The root cause was that `studies/fly64_dose_overlap/generation.py::build_simulation_plan` constructed a `TrainingConfig` with only `n_groups` set, leaving `n_images` as `None`. The legacy simulator `_generate_simulated_data_legacy_params` relies on `n_images` to size coordinate arrays correctly.

**SPEC Reference:**
From `specs/data_contracts.md:207` (canonical format requirements), datasets must contain proper coordinate arrays matching the number of scan positions.

**ADR/ARCH Reference:**
From `docs/architecture.md` CONFIG-001 bridge: `TrainingConfig` dataclass fields must be populated before being consumed by legacy modules that depend on global state.

---

## Search Summary

**Files examined:**
- `studies/fly64_dose_overlap/generation.py:50-100` — `build_simulation_plan` function
- `tests/study/test_dose_overlap_generation.py:60-180` — existing test coverage
- `ptycho/config/config.py:108-140` — `TrainingConfig` dataclass definition
- `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T070500Z/phase_g_execution_real_runs/cli/phase_c_generation.log` — blocker evidence

**What exists:**
- `TrainingConfig` has both `n_groups` (primary) and `n_images` (deprecated but required for backward compatibility) fields
- Test `test_generate_dataset_config_construction` validates config construction but lacked `n_images` assertion

**What was missing:**
- `build_simulation_plan` at line 85-91 did not set `n_images`
- Test coverage for `n_images` field

---

## Changes

### 1. Test Enhancement (RED step)
**File:** `tests/study/test_dose_overlap_generation.py:180-181`

Added assertion to verify `n_images` field is populated:
```python
# Verify n_images is set (required for legacy simulator coordinate arrays)
assert captured_config.n_images == 100  # must match base dataset length
```

### 2. Implementation Fix (GREEN step)
**File:** `studies/fly64_dose_overlap/generation.py:89`

Added `n_images` field to `TrainingConfig` construction:
```python
training_config = TrainingConfig(
    model=model_config,
    train_data_file=str(base_npz_path),
    n_groups=n_images,  # gs=1 initially (one group per scan position)
    n_images=int(n_images),  # Required for legacy simulator coordinate array sizing
    nphotons=int(dose),
    # Use defaults for other params; they'll be overridden in phase E training
)
```

**Rationale:** Cast to `int` to avoid numpy scalar or float type issues.

---

## Test Results

### RED (Expected Failure)
**Selector:** `pytest tests/study/test_dose_overlap_generation.py::test_generate_dataset_config_construction -vv`
**Result:** FAILED as expected with `AssertionError: assert None == 100`
**Log:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T090500Z/phase_c_generation_fix/RED_test.log`

### GREEN (Fix Validated)
**Selector:** `pytest tests/study/test_dose_overlap_generation.py::test_generate_dataset_config_construction -vv`
**Result:** PASSED (1 passed in 3.90s)
**Log:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T090500Z/phase_c_generation_fix/GREEN_test.log`

### Module Suite
**Selector:** `pytest tests/study/test_dose_overlap_generation.py -vv`
**Result:** 5 passed in 3.85s
**Log:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T090500Z/phase_c_generation_fix/module_test.log`

### Comprehensive Suite
**Selector:** `pytest -v tests/`
**Result:** 402 passed, 1 pre-existing fail (test_interop_h5_reader), 17 skipped in 250.25s
**Log:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T090500Z/phase_c_generation_fix/full_suite.log`

### Orchestrator Dry-Run
**Command:** `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub ... --collect-only`
**Result:** SUCCESS — all 8 commands (Phase C→G) planned correctly
**Log:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T090500Z/phase_c_generation_fix/orchestrator_collect.log`

---

## Metrics

- **Test Count:** 5 tests in module (all GREEN), 402 tests in full suite
- **Code Changes:** 2 files touched (1 test, 1 implementation)
- **Lines Added:** 3 lines (2 in test, 1 in implementation)
- **Regression Prevention:** New assertion ensures future changes don't break `n_images` wiring

---

## Artifacts

All artifacts archived under: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T090500Z/phase_c_generation_fix/`

- `RED_test.log` — Expected failure before fix
- `GREEN_test.log` — Passing test after fix
- `module_test.log` — Full module test results
- `full_suite.log` — Comprehensive test suite results
- `orchestrator_collect.log` — Dry-run validation of Phase C→G pipeline
- `summary.md` — This file

---

## Findings Applied

- **POLICY-001** — PyTorch dependency installed but Phase C uses TensorFlow pipeline
- **CONFIG-001** — Maintained legacy bridge expectations when constructing configs
- **DATA-001** — Dataset generation will now produce DATA-001 compliant artifacts
- **TYPE-PATH-001** — Preserved Path normalization across CLI/script usage

---

## Next Actions

1. ✅ Phase C fix complete and tested
2. ⏭️ Execute full Phase C→G pipeline with real runs to capture dense comparison evidence
3. ⏭️ Archive metrics (MS-SSIM, MAE) from Phase G comparison outputs
4. ⏭️ Update fix_plan.md to mark this item done and add new Phase G execution task

---

## Exit Criteria Met

- ✅ RED test confirmed expected failure
- ✅ GREEN test validated fix
- ✅ Full test module passed (5/5)
- ✅ Comprehensive test suite passed (402 passed, no new failures)
- ✅ Orchestrator dry-run validated command sequencing
- ✅ No pitfalls violated (stable modules untouched, TYPE-PATH-001 preserved, CONFIG-001 maintained)
- ✅ Test remains pytest-native (no unittest mix)
- ✅ `n_images` cast to `int` (no numpy scalar issues)
