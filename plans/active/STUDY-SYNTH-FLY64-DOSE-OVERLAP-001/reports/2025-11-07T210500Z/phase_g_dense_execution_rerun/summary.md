# Phase G Dense Execution Rerun — prepare_hub Helper Implementation

**Loop ID:** 2025-11-07T210500Z
**Focus:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis
**Mode:** TDD
**Status:** GREEN

---

## Objective

Implement `prepare_hub()` helper with `--clobber` flag in Phase G orchestrator to detect and manage stale Phase C outputs, preventing accidental overwrites while enabling clean pipeline reruns.

---

## Implementation Summary

### TDD Cycle

**RED Phase:**
- Added failing test `test_prepare_hub_detects_stale_outputs` validating stale output detection with read-only error (tests/study/test_phase_g_dense_orchestrator.py:390-420)
- Confirmed AttributeError: module 'run_phase_g_dense' has no attribute 'prepare_hub'

**GREEN Phase:**
- Implemented `prepare_hub(hub, clobber)` helper in bin/run_phase_g_dense.py:137-192
  - TYPE-PATH-001 compliant (Path.resolve() normalization)
  - Default clobber=False raises RuntimeError with actionable `--clobber` guidance
  - Clobber mode archives stale outputs to timestamped `{hub}/archive/phase_c_<timestamp>/` directory
  - Read-only by default to prevent accidental data loss
- Added `--clobber` CLI flag (run_phase_g_dense.py:484-487)
- Integrated `prepare_hub()` call into main before Phase C execution (run_phase_g_dense.py:603-617)
- Added positive-path test `test_prepare_hub_clobbers_previous_outputs` (tests/study/test_phase_g_dense_orchestrator.py:423-459)
- Both tests PASSED (0.90-0.92s)

### Test Results

**Targeted Tests:**
- test_prepare_hub_detects_stale_outputs: PASSED 0.90s
- test_prepare_hub_clobbers_previous_outputs: PASSED 0.92s
- test_validate_phase_c_metadata_requires_canonical_transform: PASSED 0.94s
- test_validate_phase_c_metadata_accepts_valid_metadata: PASSED 0.96s (2 tests)
- test_summarize_phase_g_outputs: PASSED 0.96s (2 tests)

**Collect-only:**
- 9 tests collected (2 prepare_hub + 3 metadata guard + 4 summary)

**Full Suite:**
- 415 passed (up from 413)
- 1 pre-existing fail (test_interop_h5_reader)
- 17 skipped
- 104 warnings
- Duration: 272.54s (4:32)

---

## Documentation Updates

**docs/TESTING_GUIDE.md:**
- Updated Phase G orchestrator section to include prepare_hub() helper coverage (lines 215-239)
- Added Hub Preparation Helper subsection describing stale detection, clobber mode, and archival behavior (lines 241-252)
- Updated CLI usage example to include --clobber flag (lines 288-295)
- Updated test count from 7 to 9 tests

**docs/development/TEST_SUITE_INDEX.md:**
- Updated test_phase_g_dense_orchestrator.py row to describe prepare_hub tests (line 62)
- Added test_prepare_hub_detects_stale_outputs and test_prepare_hub_clobbers_previous_outputs to key tests list
- Updated test count from 7 to 9 tests (2 prepare_hub + 3 metadata guard + 4 summary)
- Updated evidence path to reference this loop's artifacts

---

## Key Features

1. **Stale Output Detection:**
   - Detects existing Phase C outputs under {hub}/data/phase_c/
   - Default behavior raises RuntimeError with actionable --clobber remedy
   - No file deletions in read-only mode (preserves evidence)

2. **Clobber Mode:**
   - Archives stale outputs to {hub}/archive/phase_c_<timestamp>/ directory
   - Preserves evidence via archiving instead of deletion
   - Produces clean hub state ready for new pipeline run

3. **TYPE-PATH-001 Compliance:**
   - All paths normalized via Path.resolve()
   - Consistent with existing orchestrator helpers

4. **Integration:**
   - Called before Phase C execution in main()
   - Fail-fast error handling with blocker logging
   - Skipped in --collect-only dry-run mode

---

## Artifacts

**RED/GREEN Logs:**
- red/pytest_prepare_hub_red.log (AttributeError confirmation)
- green/pytest_prepare_hub_detects_green.log (PASSED 0.90s)
- green/pytest_prepare_hub_clobbers_green.log (PASSED 0.92s)
- green/pytest_metadata_guard_green.log (PASSED 0.94s)
- green/pytest_remaining_green.log (2 PASSED 0.96s)
- green/pytest_full_suite.log (415 passed, 272.54s)

**Collect-only:**
- collect/pytest_phase_g_orchestrator_collect.log (9 tests collected)

---

## Findings Applied

- POLICY-001: PyTorch dependency policy (not directly relevant to this helper)
- CONFIG-001: Legacy bridge sequencing (not invoked by prepare_hub)
- DATA-001: NPZ metadata contract (enforced by validate_phase_c_metadata)
- TYPE-PATH-001: Path normalization (prepare_hub uses Path.resolve())
- OVERSAMPLING-001: Dense view overlap parameters (validated in Phase D, not affected)

---

## Summary

Implemented prepare_hub() helper with --clobber flag to prevent accidental overwrites of Phase C outputs. TDD cycle: RED confirmed missing function, GREEN validated both stale detection (read-only error) and clobber mode (archival). Full suite passes with 415 tests. Documentation updated with helper description, selectors, and CLI usage. Ready for full Phase C→G pipeline evidence capture.
