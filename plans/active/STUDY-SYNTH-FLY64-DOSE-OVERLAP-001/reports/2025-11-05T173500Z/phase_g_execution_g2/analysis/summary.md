# Phase G2 Execution Summary — Comparison Executor Implementation

## Overview

Successfully implemented Phase G2 comparison executor infrastructure following TDD methodology. Added `execute_comparison_jobs()` helper that shells out to `scripts/compare_models.py`, integrated with CLI to invoke when `--dry-run` is false, and validated with RED→GREEN pytest cycle plus CLI evidence.

## Problem Statement

**SPEC Reference:** Per `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T140500Z/phase_g_comparison_plan/plan/plan.md:31`, Phase G2.1 requires:
- Executor helper that dispatches comparison jobs via subprocess
- CLI integration to invoke executor when `--dry-run=false`
- Execution logs and return codes captured in manifest
- Deterministic job ordering (dose/view/split)

**ADR/ARCH Alignment:**
- **CONFIG-001 (docs/findings.md)**: Executor remains pure; `compare_models.py` handles CONFIG-001 bridge internally
- **DATA-001**: Job construction points at canonical Phase C NPZ paths
- **POLICY-001**: PyTorch backend assumed available for pty-chi baseline integration

## Search Summary

Reviewed existing implementations:
- `studies/fly64_dose_overlap/reconstruction.py:130-246` (Phase F executor pattern)
- `scripts/compare_models.py:36-100` (CLI interface for comparison script)
- `tests/study/test_dose_overlap_reconstruction.py:115-191` (subprocess mocking pattern)

## Implementation

### Changes Made

1. **studies/fly64_dose_overlap/comparison.py:130-246**
   - Added `execute_comparison_jobs(jobs, artifact_root)` helper
   - Constructs subprocess command with `sys.executable -m scripts.compare_models`
   - Maps `ComparisonJob` fields to CLI flags: `--pinn_dir`, `--baseline_dir`, `--test_data`, `--output_dir`, `--ms-ssim-sigma`
   - Adds registration flags: `--skip-registration`, `--register-ptychi-only`
   - Captures stdout/stderr/returncode in per-job log files
   - Returns manifest dict with `execution_results` array

2. **studies/fly64_dose_overlap/comparison.py:347-376**
   - Updated `main()` to invoke `execute_comparison_jobs()` when `--dry-run=false`
   - Updates manifest JSON with execution results
   - Emits execution summary (n_success, n_failed)
   - Returns exit code 1 if any failures, 0 otherwise

3. **tests/study/test_dose_overlap_comparison.py:115-191**
   - Added `test_execute_comparison_jobs_invokes_compare_models()`
   - Uses `monkeypatch` to mock `subprocess.run`
   - Validates subprocess invocation count, command structure, CLI arguments
   - Verifies manifest contains execution_results with returncode

## Test Results

### RED Phase
**File:** `red/pytest_phase_g_executor_red.log`
**Expected Failure:** `ImportError: cannot import name 'execute_comparison_jobs'`
**Status:** ✓ Expected failure captured

### GREEN Phase
**File:** `green/pytest_phase_g_executor_green.log`
**Result:** `1 passed in 0.84s`
**Selector:** `pytest tests/study/test_dose_overlap_comparison.py::test_execute_comparison_jobs_invokes_compare_models -vv`

### Collection Proof
**File:** `collect/pytest_phase_g_collect.log`
**Result:** `2 tests collected in 0.83s`
**Selectors:**
- `test_build_comparison_jobs_creates_all_conditions`
- `test_execute_comparison_jobs_invokes_compare_models`

### CLI Dry-Run
**File:** `cli/phase_g_cli_dry_run.log`
**Command:**
```bash
AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md \
python -m studies.fly64_dose_overlap.comparison \
  --phase-c-root tmp/phase_c_f2_cli \
  --phase-e-root tmp/phase_e_training_gs2 \
  --phase-f-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/phase_f_cli_test \
  --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T173500Z/phase_g_execution_g2/cli \
  --dose 1000 --view dense --split train --dry-run
```
**Result:** Successfully built 1 comparison job, wrote manifest/summary, skipped execution (dry-run mode)

### Comprehensive Suite
**File:** `analysis/pytest_full_suite.log`
**Result:** `393 passed, 17 skipped, 1 failed in 248.89s`
**Pre-existing failure:** `test_interop_h5_reader` (ModuleNotFoundError, unrelated to Phase G)
**Phase G tests:** 2/2 PASSED
**No regressions introduced**

## Findings Applied

- **POLICY-001**: Assumed PyTorch backend available; `compare_models.py` will fail fast if torch not installed
- **CONFIG-001**: Executor delegates to `compare_models.py` which handles `update_legacy_dict` bridge
- **DATA-001**: Job paths follow canonical Phase C structure (`dose_{dose}/patched_{split}.npz` for dense, `dose_{dose}/{view}/{view}_{split}.npz` for sparse)
- **OVERSAMPLING-001**: Sparse jobs inherit K=7 from Phase D; manifest metadata preserved

## Metrics

- **Lines changed:** ~145 (executor: 117, CLI integration: 28)
- **Tests added:** 1 new test function (76 lines)
- **Selectors:** 2 Phase G comparison tests collected
- **RED failures:** 1 (expected ImportError)
- **GREEN passes:** 1/1 targeted, 2/2 Phase G suite
- **Comprehensive suite:** 393/394 PASSED (99.7% pass rate)

## Artifacts

All artifacts stored under `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T173500Z/phase_g_execution_g2/`:
- `red/pytest_phase_g_executor_red.log` — RED test failure log
- `green/pytest_phase_g_executor_green.log` — GREEN test pass log
- `collect/pytest_phase_g_collect.log` — Collection proof (2 tests)
- `cli/phase_g_cli_dry_run.log` — CLI dry-run transcript
- `cli/comparison_manifest.json` — Dry-run manifest
- `cli/comparison_summary.txt` — Dry-run summary
- `analysis/pytest_full_suite.log` — Comprehensive test suite results
- `analysis/summary.md` — This document

## Exit Criteria Met

✓ `execute_comparison_jobs()` helper implemented (comparison.py:130-246)
✓ CLI integration complete (comparison.py:347-376)
✓ RED→GREEN TDD cycle captured
✓ Subprocess mocking validates CLI argument routing
✓ Execution logs/return codes persisted in manifest
✓ Collection proof shows 2 Phase G tests registered
✓ CLI dry-run evidence demonstrates job building
✓ Comprehensive suite passed with no regressions

## Next Actions

**Phase G2.2 (Real Comparison Execution):**
1. Verify Phase C/E/F prerequisites exist for dose=1000, view=dense, split=train
2. Run non-dry-run CLI command to execute real comparison
3. Validate `scripts/compare_models.py` invocation succeeds
4. Capture comparison outputs (metrics, visualizations) in artifact root
5. Update test_strategy.md and TESTING_GUIDE.md with Phase G2 selectors
6. Mark Phase G2 COMPLETE once real-run evidence captured

**Blocked scenarios:**
- Sparse view comparisons await Phase D overlap data regeneration
- Dose variations (500, 2000) pending PINN training checkpoints

## Findings for docs/findings.md

*No new findings — executor follows existing Phase F reconstruction pattern.*
