# Phase E6 Dense/Baseline Evidence — Results

**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
**Date:** 2025-11-06
**Loop ID:** 2025-11-06T130500Z
**Task:** Phase E6 — Normalize CLI stdout bundle logging and add stdout relative-path assertions

---

## Problem Statement

**Quoted SPEC lines implemented:**
From `input.md:10`:
> Implement: tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path — **assert stdout bundle lines use artifact-relative paths and SHA digests match manifest entries**; adjust studies/fly64_dose_overlap/training.py::main to emit normalized paths while preserving CONFIG-001 guardrails.

**Relevant ADR/ARCH sections:**
- `specs/ptychodus_api_spec.md:239` — wts.h5.zip persistence + SHA contract
- `docs/TESTING_GUIDE.md:101` — Training CLI selectors for Phase E evidence
- `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:268` — Phase E6 evidence/test coverage

---

## Search Summary

**What exists:**
- `studies/fly64_dose_overlap/training.py:728-746` — CLI stdout emission for bundle_path and SHA256 (lines 733, 735)
- `studies/fly64_dose_overlap/training.py:743-758` — Manifest normalization converting absolute→relative paths
- `tests/study/test_dose_overlap_training.py:1450-1681` — Test `test_training_cli_records_bundle_path` validates manifest but not stdout paths

**What was missing:**
- Test assertion validating stdout bundle paths are artifact-relative (not absolute workstation paths)
- CLI stdout normalization before print (was emitting absolute paths directly from `result['bundle_path']`)

**File pointers:**
- Test: `tests/study/test_dose_overlap_training.py:1450-1681`
- Implementation: `studies/fly64_dose_overlap/training.py:728-746`

---

## Changes Made

### 1. Test Enhancement (RED phase)
**File:** `tests/study/test_dose_overlap_training.py:1645-1652`

Added stdout path validation assertions:
```python
# NEW Phase E6 Do Now: Validate bundle path is artifact-relative (not absolute)
# Extract path portion from line (after "]: ")
bundle_path_stdout = line.split(']: ')[1]
assert not Path(bundle_path_stdout).is_absolute(), \
    f"Stdout bundle path must be artifact-relative, got absolute path: {bundle_path_stdout}"
# Expect simple filename "wts.h5.zip" relative to artifact_dir
assert bundle_path_stdout == "wts.h5.zip", \
    f"Stdout bundle path should be 'wts.h5.zip', got: {bundle_path_stdout}"
```

**RED Evidence:** `red/pytest_training_cli_relative_red_v2.log`
```
AssertionError: Stdout bundle path must be artifact-relative, got absolute path: /tmp/pytest-of-ollie/pytest-226/test_training_cli_records_bund0/artifacts/dose_1000/baseline/gs1/wts.h5.zip
```

---

### 2. CLI Stdout Normalization (GREEN phase)
**File:** `studies/fly64_dose_overlap/training.py:728-746`

Updated stdout emission to normalize bundle_path before printing:
```python
# Phase E6: Emit bundle_path and bundle_sha256 to stdout for CLI log capture
# IMPORTANT: Normalize bundle_path to artifact-relative before stdout emission
# to avoid workstation-specific absolute paths in logged output
if not args.dry_run and result.get('bundle_path'):
    # Convert absolute bundle_path to artifact-relative path for portable logging
    bundle_path_abs = Path(result['bundle_path'])
    try:
        bundle_path_rel = bundle_path_abs.relative_to(job.artifact_dir)
        bundle_path_display = str(bundle_path_rel)
    except ValueError:
        # Defensive: if bundle_path not under artifact_dir, use as-is
        bundle_path_display = result['bundle_path']

    print(f"    → Bundle [{job.view}/dose={job.dose:.0e}]: {bundle_path_display}")
    if result.get('bundle_sha256'):
        print(f"    → SHA256 [{job.view}/dose={job.dose:.0e}]: {result['bundle_sha256']}")
```

**GREEN Evidence:** `green/pytest_training_cli_relative_green.log`
All assertions pass: stdout now emits `wts.h5.zip` (relative) instead of absolute paths.

---

## Test Results

### Targeted Tests
✅ **RED Test (with new assertion):** `red/pytest_training_cli_relative_red_v2.log`
- **Result:** FAILED (expected)
- **Error:** Stdout emitted absolute path `/tmp/.../wts.h5.zip` instead of relative `wts.h5.zip`

✅ **GREEN Test (after CLI fix):** `green/pytest_training_cli_relative_green.log`
- **Result:** PASSED in 3.72s
- **Validation:** Stdout now emits `wts.h5.zip` (artifact-relative)

✅ **Bundle SHA Test:** `green/pytest_bundle_sha_green.log`
- **Result:** PASSED in 3.77s
- **Validation:** SHA256 computation and matching confirmed

✅ **Training CLI Suite:** `green/pytest_training_cli_suite_green.log`
- **Result:** 4 PASSED in 3.62s
- **Tests:**
  - test_training_cli_filters_jobs
  - test_training_cli_manifest_and_bridging
  - test_training_cli_invokes_real_runner
  - test_training_cli_records_bundle_path

✅ **Collection Proof:** `collect/pytest_training_cli_collect.log`
- **Result:** 4/10 tests collected (6 deselected)
- **Validates:** Active selector `-k training_cli` maps to correct tests

---

### Full Test Suite
✅ **Comprehensive Regression:** `green/pytest_full_suite.log`
- **Result:** **397 passed, 1 failed (pre-existing), 17 skipped** in 249.29s
- **Failed (pre-existing):** `tests/io/test_ptychodus_interop_h5.py::test_interop_h5_reader` (ModuleNotFoundError unrelated to changes)
- **All Phase E training tests:** PASSED
- **No regressions introduced**

---

## Observations

### What Worked
1. **TDD RED→GREEN cycle validated stdout format issue:**
   - Adding stdout assertion to test correctly exposed absolute-path emission bug
   - Implementation fix (lines 736-742) properly normalizes paths before stdout

2. **Defensive error handling preserved:**
   - `try/except ValueError` at lines 737-742 handles edge case if bundle not under artifact_dir
   - Existing manifest normalization (lines 748-758) remains unchanged and working

3. **Test strategy alignment:**
   - Selectors (`-k training_cli`) collect exactly 4 tests as documented
   - Full suite passes with no regressions (397 passed)

### Configuration Compliance
- **CONFIG-001:** No changes to legacy bridge ordering; `update_legacy_dict` remains in `run_training_job`
- **POLICY-001:** PyTorch dependency assumption intact (no torch-optional branches added)
- **DATA-001:** No changes to NPZ contract validation

---

## Next Steps

**Immediate (same initiative):**
- ✅ Phase E6 complete — stdout normalization GREEN
- [ ] **Next:** Sparse view Phase E real run for dose=1000 (once dense/baseline evidence is validated)
- [ ] **Follow-up:** Phase G comparison analysis after Phase F ptychi-baseline outputs ready

**Deferred:**
- Real-run CLI executions for dense/baseline (deterministic mode) pending user request
- Archive script integration for Phase E outputs (template exists at `plans/.../bin/archive_phase_e_outputs.py`)

---

## Findings Applied

- **CONFIG-001:** Legacy bridge ordering maintained (docs/findings.md:10)
- **POLICY-001:** PyTorch runtime required; tests assume torch>=2.2 (docs/findings.md:8)
- **DATA-001:** NPZ contract validation unchanged (docs/findings.md:14)

---

## Commit Summary

**Files Modified:**
1. `tests/study/test_dose_overlap_training.py` (+7 lines: stdout path assertions at 1645-1652)
2. `studies/fly64_dose_overlap/training.py` (+11 lines: stdout normalization at 732-746)

**Tests Evidence:**
- RED: `red/pytest_training_cli_relative_red_v2.log`
- GREEN: `green/pytest_training_cli_relative_green.log`, `green/pytest_bundle_sha_green.log`, `green/pytest_training_cli_suite_green.log`
- COLLECT: `collect/pytest_training_cli_collect.log`
- FULL: `green/pytest_full_suite.log` (397 passed, 1 pre-existing fail)

**Acceptance Criteria Met:**
- ✅ Test asserts stdout bundle lines use artifact-relative paths
- ✅ CLI emits normalized paths (e.g., `wts.h5.zip` not `/abs/path/wts.h5.zip`)
- ✅ SHA256 digests match manifest entries (via existing test)
- ✅ CONFIG-001 guardrails preserved (no changes to bridge ordering)
- ✅ Full test suite passes (no regressions)
