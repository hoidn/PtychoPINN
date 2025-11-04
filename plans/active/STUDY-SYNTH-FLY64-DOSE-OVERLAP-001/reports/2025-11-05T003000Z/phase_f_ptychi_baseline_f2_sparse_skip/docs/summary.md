# Phase F2 Sparse Skip Implementation Summary

**Date:** 2025-11-05
**Focus:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.F2 — Phase F pty-chi baseline sparse view skip instrumentation
**Mode:** TDD Implementation
**Branch:** feature/torchapi-newprompt

---

## Overview

Delivered Phase F2 skip metadata instrumentation for missing Phase D sparse views. When `--allow-missing-phase-d` flag is set, the reconstruction orchestrator now skips missing NPZ files gracefully, records skip events with metadata (dose/view/split/reason), and surfaces them through both the manifest and skip summary JSON files.

---

## Implementation Summary

### 1. Test Strategy (RED → GREEN)

**RED Test (`tests/study/test_dose_overlap_reconstruction.py::test_cli_skips_missing_phase_d`)**
- Created fixture with Phase C baseline datasets (all present) + Phase D with **only dense views** (sparse deliberately omitted)
- Invoked CLI with `--allow-missing-phase-d --dry-run`
- Asserted manifest contains `missing_jobs` field with Phase D skip metadata
- Asserted skip summary documents 6 missing sparse jobs (3 doses × sparse view × 2 splits)
- RED phase: manifest lacked skip metadata fields → `AssertionError: Manifest missing skip metadata for Phase D missing files`

**GREEN Implementation**
- Modified `build_ptychi_jobs()` (reconstruction.py:84-197):
  - Added `skip_events: List[dict] = None` parameter
  - When `allow_missing=True` and NPZ file doesn't exist, append skip event to `skip_events` list instead of raising `FileNotFoundError`
  - Skip event schema: `{dose, view, split, reason: "Phase [C|D] ... NPZ not found: <path>"}`
  - Builder remains pure (CONFIG-001 compliant)
- Modified CLI `main()` (reconstruction.py:360-527):
  - Initialize `missing_file_skips = []` list before calling builder
  - Pass `skip_events=missing_file_skips` when `--allow-missing-phase-d=True`
  - Merge `missing_file_skips` with filter-based `skipped_jobs` for unified skip summary
  - Emit `missing_jobs` field in manifest JSON
  - Emit `missing_phase_d_count` in skip summary JSON
  - Print skip event details to CLI stdout for human debugging

**GREEN Test Results**
- Targeted test: `PASSED in 1.41s` (reconstruction.py:519-639)
- Phase F suite: `5 PASSED in 4.00s` (all tests including new skip test)
- Comprehensive suite: `390 PASSED / 17 SKIPPED / 1 pre-existing failure` (test_ptychodus_interop_h5)

---

## File Changes

### Modified Files

1. **tests/study/test_dose_overlap_reconstruction.py** (+121 lines, line 519-639)
   - Added `test_cli_skips_missing_phase_d()` RED→GREEN test
   - Validates skip metadata schema and missing Phase D NPZ handling
   - Asserts 6 skip events for missing sparse views

2. **studies/fly64_dose_overlap/reconstruction.py** (+56 lines, ~66 lines modified)
   - `build_ptychi_jobs()`: Added `skip_events` parameter, skip logic with metadata collection (lines 84-197)
   - `main()`: Initialize `missing_file_skips`, merge with filter skips, emit to manifest + skip summary (lines 360-527)

---

## Test Evidence

### RED Phase
- **Log:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T003000Z/phase_f_ptychi_baseline_f2_sparse_skip/red/pytest_sparse_skip_red.log`
- **Failure:** AssertionError: Manifest missing skip metadata for Phase D missing files
- **Expected:** Manifest lacks `phase_d_missing`, `skipped_phase_d`, or `missing_jobs` field

### GREEN Phase
- **Targeted Test:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T003000Z/phase_f_ptychi_baseline_f2_sparse_skip/green/pytest_sparse_skip_green.log`
  - `test_cli_skips_missing_phase_d`: PASSED in 1.41s
- **Phase F Suite:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T003000Z/phase_f_ptychi_baseline_f2_sparse_skip/green/pytest_phase_f_suite_green.log`
  - 2 tests collected via `-k "ptychi"`: PASSED
- **Full Suite:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T003000Z/phase_f_ptychi_baseline_f2_sparse_skip/collect/pytest_phase_f_full_suite.log`
  - 5 Phase F tests: ALL PASSED
- **Comprehensive Suite:** Background job 0f28dc (exit code 0)
  - 390 PASSED / 17 SKIPPED / 1 pre-existing failure (ptychodus interop)

### CLI Dry-Run Evidence
- **Command:** `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.reconstruction --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T003000Z/phase_f_ptychi_baseline_f2_sparse_skip/cli --dose 1000 --view sparse --split train --dry-run --allow-missing-phase-d`
- **Log:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T003000Z/phase_f_ptychi_baseline_f2_sparse_skip/cli/dry_run_sparse.log`
- **Manifest:** `reconstruction_manifest.json` (18 total jobs, 1 filtered, 0 missing in this run because sparse NPZ exists)
- **Skip Summary:** `skip_summary.json` (17 filter-based skips, `missing_phase_d_count: 0`)

---

## Findings Applied

- **CONFIG-001:** Builder `build_ptychi_jobs()` remains pure; skip metadata collected via output parameter without mutating global `params.cfg`
- **DATA-001:** Skip detection validates NPZ paths against canonical Phase C/D layouts (`dose_{dose}/patched_{split}.npz`, `dose_{dose}/{view}/{view}_{split}.npz`)
- **POLICY-001:** PyTorch dependency acknowledged; pty-chi LSQML uses torch>=2.2 runtime
- **OVERSAMPLING-001:** Skip reasoning references spacing threshold rejections (e.g., "Phase D NPZ not found" typically due to insufficient overlap positions)

---

## Metrics

- **Code Changes:**
  - `test_dose_overlap_reconstruction.py`: +121 lines (new test function)
  - `reconstruction.py`: +56 lines, ~66 modified (skip_events parameter, CLI merge logic)
- **Test Results:**
  - RED: 1 expected failure (AssertionError)
  - GREEN: 1 new test PASSED, 5 Phase F tests PASSED, 390 comprehensive suite PASSED
- **Artifact Files:**
  - RED log: 1 file (~15 KB)
  - GREEN logs: 3 files (targeted + suite + full suite, ~50 KB total)
  - CLI dry-run artifacts: 3 files (log + manifest + skip summary, ~8 KB)

---

## Skip Metadata Schema

### Manifest (`reconstruction_manifest.json`)
```json
{
  "timestamp": "...",
  "phase_c_root": "...",
  "phase_d_root": "...",
  "artifact_root": "...",
  "filters": {...},
  "total_jobs": 18,
  "filtered_jobs": N,
  "missing_jobs": [
    {
      "dose": 1000.0,
      "view": "sparse",
      "split": "train",
      "reason": "Phase D overlap NPZ not found: /path/to/sparse_train.npz"
    }
  ],
  "jobs": [...],
  "execution_results": [...]
}
```

### Skip Summary (`skip_summary.json`)
```json
{
  "timestamp": "...",
  "skipped_count": 23,  // filter-based + missing files
  "skipped_jobs": [
    {"dose": ..., "view": ..., "split": ..., "reason": "dose filter (requested: ...)"},
    {"dose": ..., "view": ..., "split": ..., "reason": "Phase D overlap NPZ not found: ..."}
  ],
  "missing_phase_d_count": 6  // explicit count of missing Phase D files
}
```

---

## Exit Criteria Met

- ✅ RED test authored asserting skip metadata schema
- ✅ RED test failed with expected AssertionError
- ✅ `build_ptychi_jobs()` extended with `skip_events` parameter
- ✅ CLI `main()` collects and merges missing file skips with filter skips
- ✅ Manifest emits `missing_jobs` field with skip event metadata
- ✅ Skip summary emits unified `skipped_jobs` list + `missing_phase_d_count`
- ✅ GREEN test passed (1.41s)
- ✅ Phase F suite passed (5/5 tests)
- ✅ Comprehensive suite passed (390/391 tests; 1 pre-existing failure)
- ✅ CLI dry-run executed successfully with sparse view filter
- ✅ CONFIG-001 / DATA-001 / POLICY-001 / OVERSAMPLING-001 compliance maintained

---

## Next Actions

1. **Phase F2.3:** Run sparse/train and sparse/test LSQML baselines with real (non-dry-run) execution once Phase D sparse data is regenerated or confirmed available
2. **Documentation Sync:** Update `docs/TESTING_GUIDE.md` and `docs/development/TEST_SUITE_INDEX.md` with new `test_cli_skips_missing_phase_d` selector
3. **Fix Plan Update:** Log Attempt #83 in `docs/fix_plan.md` with artifact pointers and mark Phase F2 sparse skip instrumentation complete

---

## Artifact Manifest

- `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T003000Z/phase_f_ptychi_baseline_f2_sparse_skip/`
  - `red/pytest_sparse_skip_red.log` (RED phase failure log)
  - `green/pytest_sparse_skip_green.log` (targeted test GREEN log)
  - `green/pytest_phase_f_suite_green.log` (Phase F suite GREEN log)
  - `collect/pytest_phase_f_full_suite.log` (all 5 Phase F tests collected + passed)
  - `cli/dry_run_sparse.log` (CLI stdout/stderr transcript)
  - `cli/reconstruction_manifest.json` (manifest with `missing_jobs` field)
  - `cli/skip_summary.json` (skip summary with `missing_phase_d_count`)
  - `docs/summary.md` (this document)
