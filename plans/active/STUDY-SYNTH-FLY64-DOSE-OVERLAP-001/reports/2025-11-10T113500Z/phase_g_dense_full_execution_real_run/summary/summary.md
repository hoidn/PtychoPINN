# Phase G Dense Pipeline TDD Cycle — CLI Log Validation

**Date:** 2025-11-06  
**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001  
**Focus:** Add CLI log validation via TDD and prepare for full pipeline execution  
**Branch:** feature/torchapi-newprompt  
**Commit:** 3f3c2e59

## Summary

Successfully completed TDD cycle to add CLI log validation to `verify_dense_pipeline_artifacts.py`. The new `validate_cli_logs()` function enforces orchestrator log completeness by checking for:

1. **Orchestrator log existence**: `run_phase_g_dense.log` is now REQUIRED
2. **Phase banners**: All 8 phase markers `[1/8]` through `[8/8]` must be present
3. **SUCCESS sentinel**: `"SUCCESS: All phases completed"` must appear in the log

## Implementation

### TDD Cycle (RED → GREEN)

**RED Test** (`test_verify_dense_pipeline_cli_logs_missing`):
- Created hub fixture with incomplete CLI logs (missing orchestrator log)
- Verified verifier correctly detects missing log and exits with status 1
- Captured failure in `red/pytest_cli_logs_fail.log`

**GREEN Test** (`test_verify_dense_pipeline_cli_logs_complete`):
- Created hub fixture with complete orchestrator log including all phase banners and SUCCESS marker
- Verified verifier passes CLI validation check
- Captured success in `green/pytest_cli_logs_fix.log`

### Code Changes

**File:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py`
- Added `validate_cli_logs(cli_dir: Path) -> dict[str, Any]` function (lines 437-520)
- Integrated into main() validation flow as check #9 (line 619-622)
- Replaced simple phase_g_*.log existence check with comprehensive orchestrator validation

**File:** `tests/study/test_phase_g_dense_artifacts_verifier.py`
- Added `test_verify_dense_pipeline_cli_logs_missing()` (lines 262-375)
- Added `test_verify_dense_pipeline_cli_logs_complete()` (lines 378-540)
- Updated `test_verify_dense_pipeline_artifact_inventory_passes_with_complete_bundle()` to include orchestrator log fixture (lines 205-243)

## Test Results

### Targeted Tests
```bash
pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_logs_missing -vv
# Status: PASSED (verifier correctly detects missing log)

pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_logs_complete -vv
# Status: PASSED (verifier validates complete logs)

pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
# Status: PASSED (orchestrator integration test)
```

### Full Test Suite
```bash
pytest tests/study/ -v
# Result: 79 passed in 11.11s
# No regressions introduced

pytest -v tests/
# Result: 431 passed, 17 skipped, 2 failed (pre-existing failures)
# New CLI validation does not break existing tests
```

## Compliance

- **POLICY-001**: PyTorch dependency remains installed (torch>=2.2)
- **CONFIG-001**: Validation helper preserves legacy bridge compatibility
- **DATA-001**: Phase C NPZ contract validation remains intact
- **TYPE-PATH-001**: POSIX-relative path enforcement maintained in artifact_inventory.txt
- **STUDY-001**: CLI validation prepares for MS-SSIM/MAE delta capture in full pipeline run

## Artifacts

All artifacts saved to: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-10T113500Z/phase_g_dense_full_execution_real_run/`

- `red/pytest_cli_logs_fail.log`: RED test failure evidence
- `green/pytest_cli_logs_fix.log`: GREEN test success evidence
- `green/pytest_orchestrator_dense_exec_cli_guard.log`: Orchestrator integration test evidence
- `summary/summary.md`: This file

## Next Steps

1. Execute full Phase G dense pipeline run with: `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-10T113500Z/phase_g_dense_full_execution_real_run" --dose 1000 --view dense --splits train test --clobber`
2. Run verifier after pipeline completes to validate all artifacts including new CLI checks
3. Capture MS-SSIM/MAE deltas vs Baseline/PtyChi for documentation
4. Update docs/fix_plan.md Attempts History and docs/findings.md with durable lessons

## Notes

- The 093500Z pipeline run was incomplete (only Phase C finished), so a fresh 113500Z run is planned
- CLI validation is now a hard gate: pipeline runs without complete orchestrator logs will fail verification
- Tests are device-neutral (no GPU-specific strings in validation logic)
- Orchestrator log path is normalized per TYPE-PATH-001 before validation
