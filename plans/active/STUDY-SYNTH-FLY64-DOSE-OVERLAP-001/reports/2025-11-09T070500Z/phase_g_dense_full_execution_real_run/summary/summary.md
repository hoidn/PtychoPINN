# Phase G Dense Execution — Digest Paths in Success Banner

**Timestamp:** 2025-11-09T070500Z
**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
**Mode:** TDD
**Status:** GREEN

## Objective

Surface `metrics_digest.md` and `cli/metrics_digest_cli.log` paths in the orchestrator success banner to improve artifact discoverability.

## Implementation Summary

### TDD Cycle: RED → GREEN

**RED Phase (Step 4 from input.md):**
- Enhanced `test_run_phase_g_dense_exec_runs_analyze_digest` (tests/study/test_phase_g_dense_orchestrator.py:856-1014) with stdout capture via `io.StringIO` + `contextlib.redirect_stdout`
- Added four new assertions requiring success banner to include:
  1. `"Metrics digest (Markdown):"` line
  2. `"Metrics digest log:"` line
  3. `"metrics_digest.md"` path mention
  4. `"metrics_digest_cli.log"` path mention
- Test FAILED as expected: success banner lacked the new digest path lines
- RED evidence: `red/pytest_digest_exec.log` (FAILED at assertion line 1005)

**GREEN Phase (Step 6 from input.md):**
- Updated `run_phase_g_dense.py::main` success banner (lines 830-831) to print two additional lines:
  ```python
  print(f"Metrics digest (Markdown): {metrics_digest_md}")
  print(f"Metrics digest log: {analyze_digest_log}")
  ```
- Path objects automatically convert to strings via f-string (TYPE-PATH-001 compliant)
- Test PASSED: all four stdout assertions satisfied
- GREEN evidence: `green/pytest_digest_exec.log` (PASSED in 2.73s)

**Regression Guards:**
- `test_run_phase_g_dense_collect_only_generates_commands`: PASSED (3.63s-3.89s, both RED/GREEN)
- `test_analyze_dense_metrics_success_digest`: PASSED (5.19s)

**Full Test Suite:**
- 426 passed / 1 pre-existing fail (test_interop_h5_reader) / 17 skipped
- No new failures introduced
- Full suite log: `green/pytest_full.log` (426 passed in 253s)

## Files Changed

### Production Code
- `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:830-831` — Added two banner lines

### Test Code
- `tests/study/test_phase_g_dense_orchestrator.py:951-959` — Added stdout capture mechanism
- `tests/study/test_phase_g_dense_orchestrator.py:1004-1014` — Added four digest path assertions

## Acceptance Criteria

✅ Test FAILED in RED phase with actionable message showing missing banner lines
✅ Implementation adds exactly two print statements with digest paths
✅ Test PASSED in GREEN phase
✅ Regression tests (collect-only, analyze success) remain GREEN
✅ Full test suite passes (426/1/17)
✅ Path formatting follows TYPE-PATH-001 (automatic string conversion)

## Artifacts

```
plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T070500Z/phase_g_dense_full_execution_real_run/
├── red/
│   ├── pytest_digest_exec.log          # RED phase: FAILED as expected
│   └── pytest_collect_only.log         # Regression guard: PASSED
├── green/
│   ├── pytest_digest_exec.log          # GREEN phase: PASSED
│   ├── pytest_collect_only.log         # Regression guard: PASSED
│   ├── pytest_analyze_success.log      # Additional regression: PASSED
│   └── pytest_full.log                 # Full suite: 426 passed
└── summary/
    └── summary.md                      # This file
```

## Findings Applied

- **TYPE-PATH-001:** Path objects converted to strings automatically via f-string interpolation; no explicit `str()` calls needed
- **CONFIG-001:** AUTHORITATIVE_CMDS_DOC exported before all pytest invocations
- **DATA-001:** No data contract changes in this loop (success banner only)

## Next Steps (Per input.md Step 10-14)

Deferred to follow-up loop (per Ralph nucleus principle):
- Execute `python .../run_phase_g_dense.py --hub <hub> --dose 1000 --view dense --splits train test --clobber`
- Capture CLI logs, verify digest paths in real execution stdout
- Extract MS-SSIM/MAE deltas from `analysis/metrics_digest.md`
- Update `docs/TESTING_GUIDE.md` §2.5 and `docs/development/TEST_SUITE_INDEX.md` if selector inventory changes

## Notes

- Nucleus complete: test-driven banner enhancement GREEN
- Dense Phase C→G pipeline execution (2-4 hour evidence run) deferred per Ralph principle
- Success banner now surfaces both Markdown digest file and its CLI log for improved observability
- No collect-only output changes; command order assertions unaffected
