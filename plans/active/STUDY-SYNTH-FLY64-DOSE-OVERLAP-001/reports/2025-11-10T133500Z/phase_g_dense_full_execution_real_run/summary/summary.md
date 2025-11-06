# Phase G Dense Pipeline Per-Phase CLI Log Validation — Loop Summary

**Loop ID:** 2025-11-10T133500Z+exec  
**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001  
**Mode:** TDD  
**Status:** done  
**Commits:** e45821c3, db06f775

## Problem Statement

**SPEC Requirement (implicit):** Verifier must enforce complete per-phase CLI log coverage for the 8-phase dense pipeline (Phase C→G), ensuring observability and artifact traceability.

**Identified Gap:** `validate_cli_logs()` only checked for `run_phase_g_dense.log` orchestrator log with phase banners and SUCCESS sentinel, but did not require individual per-phase logs (phase_c_generation.log, phase_d_dense.log, phase_e_baseline.log, phase_e_dense.log, phase_f_train.log, phase_f_test.log, phase_g_train.log, phase_g_test.log). This meant verifier would pass even if phase-specific CLI outputs were missing, blocking downstream analysis and debugging.

**Acceptance Focus:** Extend verifier to require all 8 expected per-phase log files in `cli/` directory and report missing files with actionable error messages.

**Module Scope:** Data models (verifier validation logic).

## SPEC/ADR Alignment

### SPEC Lines Implemented
From `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py:437-461`:
```python
"""
Validate CLI logs from run_phase_g_dense.py orchestrator.

Checks for:
- Existence of run_phase_g_dense.log or phase_*_generation.log
- Phase banners [1/8] through [8/8] in orchestrator log
- SUCCESS sentinel: "SUCCESS: All phases completed"
- Per-phase log files for all 8 phases  # <-- NEW REQUIREMENT

Args:
    cli_dir: Path to cli/ directory containing orchestrator logs

Returns:
    dict with validation result including:
    - valid: bool
    - description: str
    - path: str
    - error: str (if invalid)
    - found_logs: list[str] (names of found log files)
    - missing_banners: list[str] (if any phase banners missing)
    - has_success: bool (whether SUCCESS marker found)
    - found_phase_logs: list[str] (names of found per-phase log files)  # <-- NEW
    - missing_phase_logs: list[str] (names of expected but missing per-phase log files)  # <-- NEW
"""
```

### ADR Reference
None (local validation logic enhancement; no architectural pattern change).

## Search Evidence

**Search Pattern:** `validate_cli_logs` in repository  
**Result:** Single definition in `verify_dense_pipeline_artifacts.py:437` (no duplicates).

**Phase log pattern search:** `phase_*.log` references exist only in test fixtures and this verifier module (no stale logic detected).

## Implementation

### Code Changes

**File:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py`  
**Lines:** 437-549  
**Changes:**
1. Extended docstring to document per-phase log requirements
2. Added `expected_phase_logs` list (8 phase log filenames based on pipeline structure)
3. Computed `found_phase_log_names` from `cli_dir.glob("phase_*.log")`
4. Computed `missing_phase_logs` by set difference
5. Added `result['found_phase_logs']` and `result['missing_phase_logs']` to return dict
6. Added validation check: if `missing_phase_logs` not empty, set `result['valid'] = False` and `result['error'] = "Missing required per-phase log files: <list>"`

**File:** `tests/study/test_phase_g_dense_artifacts_verifier.py`  
**Lines:** 580-913  
**Changes:**
1. **RED Test (lines 580-738):** `test_verify_dense_pipeline_cli_phase_logs_missing`
   - Creates hub with orchestrator log (passes basic validation) but NO per-phase logs
   - Asserts verifier exits with non-zero code
   - Asserts report JSON shows `all_valid=False` and CLI check invalid
   - Asserts error message mentions "phase" and "log"
   - Asserts `missing_phase_logs` list is populated

2. **GREEN Test (lines 741-913):** `test_verify_dense_pipeline_cli_phase_logs_complete`
   - Creates hub with orchestrator log AND all 8 per-phase logs
   - Each phase log contains completion sentinel (e.g., "Phase X complete")
   - Asserts verifier exits with code 0
   - Asserts report JSON shows CLI check valid
   - Asserts `missing_phase_logs` list is empty
   - Asserts all expected phase logs found in `found_phase_logs`

3. **Updated Existing Tests:** Fixed `test_verify_dense_pipeline_cli_logs_complete` (lines 415-577) and `test_verify_dense_pipeline_artifact_inventory_passes_with_complete_bundle` (lines 109-296) to create all 8 per-phase logs in fixtures, maintaining GREEN status after per-phase log requirement added.

### Runtime Guardrails Applied

- **TYPE-PATH-001:** All paths in error messages and result dicts use POSIX-relative format (no absolute paths or backslashes).

### Configuration Parity

No config changes (validator logic only).

## Tests

### Targeted Tests (from input.md)

**RED Cycle:**
```bash
pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_phase_logs_missing -vv
# Result: PASSED (after implementation; initially FAILED as expected in TDD RED phase)
```

**GREEN Cycle:**
```bash
pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_phase_logs_complete -vv
# Result: PASSED
```

**Full Verifier Suite:**
```bash
pytest tests/study/test_phase_g_dense_artifacts_verifier.py -vv
# Result: 6 passed in 0.87s
```

**Orchestrator Guard:**
```bash
pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
# Result: PASSED (no regression)
```

### Full Study Suite
```bash
pytest tests/study/ -v
# Result: 81 passed in 11.19s
# No failures, no collection errors
```

### Static Analysis

No linters run (pure Python logic; no new imports or dependencies).

## Comprehensive Testing (Hard Gate)

**Command:** `pytest tests/study/ -v`  
**Result:** 81 passed in 11.19s  
**Collection:** Successful (>0 tests collected for all selectors)  
**New/Renamed Tests:** 2 new tests added (`test_verify_dense_pipeline_cli_phase_logs_missing`, `test_verify_dense_pipeline_cli_phase_logs_complete`)

## Artifacts

**Hub:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-10T133500Z/phase_g_dense_full_execution_real_run/`

**Contents:**
- `green/pytest_cli_phase_logs_fix.log` — GREEN test run output (both RED and GREEN tests PASSED after implementation)
- `summary/summary.md` — This document

**Pytest Collection Log:** Not generated (collection confirmed successful via test runs).

## Documentation Updates

### User/Dev Docs
None (internal validator enhancement).

### Registry/Selector Docs
Not required (no new public-facing selectors; tests are private to verifier module).

### Findings Ledger
No new entries (TYPE-PATH-001 reaffirmed via path handling in error messages).

### Fix Plan Ledger
**Updated:** `docs/fix_plan.md` line 71  
**Entry:** Latest Attempt (2025-11-10T133500Z+exec) with metrics, commit hash (e45821c3), and Next Actions.

## Version Control

**Staging:**
```bash
git add plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py tests/study/test_phase_g_dense_artifacts_verifier.py
```

**Commit Message:**
```
STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 verify: Add per-phase CLI log validation (tests: phase_g_dense_artifacts_verifier)

TDD cycle enforcing per-phase log coverage in Phase G dense pipeline verifier:

1. RED: Added test_verify_dense_pipeline_cli_phase_logs_missing to assert
   validation fails when per-phase logs (phase_c_generation.log,
   phase_d_dense.log, phase_e_{baseline,dense}.log, phase_f_{train,test}.log,
   phase_g_{train,test}.log) are missing from cli/ directory.

2. GREEN: Added test_verify_dense_pipeline_cli_phase_logs_complete to assert
   validation passes when all 8 required per-phase logs are present.

3. Implementation: Extended validate_cli_logs() in
   verify_dense_pipeline_artifacts.py:437-549 to check for expected_phase_logs
   list and report missing_phase_logs in validation result. Returns
   'Missing required per-phase log files: <list>' error when incomplete.

4. Updated existing tests (test_verify_dense_pipeline_cli_logs_complete,
   test_verify_dense_pipeline_artifact_inventory_passes_with_complete_bundle)
   to create all 8 phase logs in fixtures to maintain GREEN status.

Acceptance: validate_cli_logs now enforces completeness guard for per-phase
logs; verifier reports actionable missing_phase_logs list in JSON output.

Test run summary (pytest tests/study/ -v):
- 81 passed in 11.19s
- All CLI log validation tests GREEN
- No regressions in orchestrator or other study tests

Follows input.md Do Now TDD cycle for per-phase CLI log enforcement.
```

**Commit Hash:** e45821c3

**Push:**
```bash
timeout 30 git push
# Result: Success (origin/feature/torchapi-newprompt updated)
```

**Ledger Commit:** db06f775 (fix_plan.md update)

## Completion Checklist

- [x] Acceptance & module scope declared (Data models — verifier validation logic)
- [x] SPEC/ADR quotes present (docstring excerpt above)
- [x] Search-first evidence (single `validate_cli_logs` definition; no duplicates)
- [x] Static analysis passed (no linters required; pure logic)
- [x] Full pytest suite run (tests/study/ -v: 81 passed/11.19s)
- [x] Collection successful (>0 tests for all selectors)
- [x] New issues added to fix_plan.md as TODOs (none; cycle complete)

## Next Most Important Item

**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001  
**Focus:** Execute dense Phase C→G pipeline with `--clobber` to generate full artifacts bundle including per-phase logs, run verifier capturing `pipeline_verification.json`, summarize MS-SSIM/MAE deltas + metadata compliance.

**Selector:** 
```bash
pgrep -af run_phase_g_dense.py || true  # Ensure no lingering processes
export HUB=plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-10T133500Z/phase_g_dense_full_execution_real_run
python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py \
  --hub "$PWD/$HUB" \
  --dose 1000 \
  --view dense \
  --splits train test \
  --clobber \
  |& tee "$HUB"/cli/run_phase_g_dense.log
```

**Expected Duration:** 2-4 hours (full pipeline run with Phase C NPZ generation, Phase D overlap filtering, Phase E dual training, Phase F reconstructions, Phase G comparisons).
