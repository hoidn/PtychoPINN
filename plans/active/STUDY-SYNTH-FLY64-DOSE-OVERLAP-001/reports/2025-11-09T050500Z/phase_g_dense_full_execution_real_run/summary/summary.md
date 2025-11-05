# Phase G Dense Orchestrator Digest Integration — Loop Summary

## Problem Statement

The Phase G dense orchestrator (`run_phase_g_dense.py`) generated intermediate metrics artifacts (`metrics_summary.json` and `aggregate_highlights.txt` via reporting helper) but did not automatically produce a final consolidated digest combining these outputs. Users had to manually run `analyze_dense_metrics.py` after the pipeline completed, breaking workflow automation.

## SPEC Alignment

Implements automated post-processing per SPEC implied requirement that orchestrator pipelines should emit human-readable final deliverables without manual intervention. Aligned with existing orchestrator pattern where `summarize_phase_g_outputs()` → `report_phase_g_dense_metrics.py` → `analyze_dense_metrics.py` form a complete metrics processing chain.

## Implementation

### Code Changes

1. **Orchestrator Command Definition** (`run_phase_g_dense.py:691-701`)
   - Added `analyze_digest_cmd` construction after `report_helper_cmd`
   - Defined command: `python analyze_dense_metrics.py --metrics metrics_summary.json --highlights aggregate_highlights.txt --output metrics_digest.md`
   - Configured log path: `cli/metrics_digest_cli.log`

2. **Collect-Only Display** (`run_phase_g_dense.py:715-718`)
   - Added analyze digest command to dry-run output as command 10 (after reporting helper command 9)
   - Displays script path, flags, and log location for validation

3. **Execution Path Integration** (`run_phase_g_dense.py:818-822`)
   - Inserted `run_command(analyze_digest_cmd, analyze_digest_log)` after highlights preview
   - Added `metrics_digest.md` to success summary (line 829)

### Test Coverage (TDD Cycle)

**RED Phase:**
1. Updated `test_run_phase_g_dense_collect_only_generates_commands` (lines 98-100) to assert presence of `analyze_dense_metrics.py` and `metrics_digest.md` in stdout
2. Authored `test_run_phase_g_dense_exec_runs_analyze_digest` (lines 857-1012, 156 lines) validating:
   - 10 total run_command calls (8 phases + reporting + digest)
   - Digest invoked AFTER reporting helper (command ordering)
   - CLI flags: `--metrics metrics_summary.json`, `--highlights aggregate_highlights.txt`, `--output metrics_digest.md`
   - Log path: `cli/metrics_digest_cli.log`

Both tests confirmed RED (missing command in collect-only stdout, only 9 calls in exec path).

**GREEN Phase:**
- `test_run_phase_g_dense_collect_only_generates_commands`: PASSED 0.84s
- `test_run_phase_g_dense_exec_runs_analyze_digest`: PASSED 0.85s
- `test_analyze_dense_metrics_success_digest`: PASSED 0.87s (regression guard)
- `test_analyze_dense_metrics_flags_failures`: PASSED 0.87s (regression guard)

**Regression Fix:**
Updated `test_run_phase_g_dense_exec_invokes_reporting_helper` (lines 703-725) to check second-to-last call (reporting helper) instead of final call (now digest), changing assertion index from `[-1]` to `[-2]` and expected count from ≥9 to ≥10.

### Full Test Suite

**Orchestrator Tests:** All 13 tests PASSED (up from 12)
- 2 prepare_hub + 3 metadata guard + 4 summary + 1 collect-only + 3 execution flow

**Full Suite:** 426 passed / 1 pre-existing fail (test_interop_h5_reader) / 17 skipped in 251.71s

### Documentation Updates

1. **docs/TESTING_GUIDE.md:360**
   - Updated orchestrator description to note digest invocation after highlights preview
   - Wording: "...then invokes `analyze_dense_metrics.py` to generate the final `metrics_digest.md` combining summary + highlights."

2. **docs/development/TEST_SUITE_INDEX.md:62**
   - Added digest invocation to Phase G orchestrator purpose
   - Updated test list to include `test_run_phase_g_dense_exec_runs_analyze_digest`
   - Incremented test count: 13 (from 12)
   - Added digest test usage example: `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv`
   - Noted digest invocation ordering: "validates `analyze_dense_metrics.py` is called after reporting helper, checks command order (reporting then digest)"

## Nucleus Complete

Shipped orchestrator integration with full TDD coverage and documentation updates. Per Ralph nucleus principle, deferred 2-4 hour dense Phase C→G pipeline execution (evidence collection) to follow-up loop — acceptance validated via mocked tests without blocking on full metrics evidence.

## Artifacts

- RED logs: `${HUB}/red/pytest_collect_only.log`, `${HUB}/red/pytest_exec_digest.log`
- GREEN logs: `${HUB}/green/pytest_collect_only.log`, `${HUB}/green/pytest_exec_digest.log`, `${HUB}/green/pytest_analyze_success.log`, `${HUB}/green/pytest_analyze_failures.log`
- Full suite: `${HUB}/green/pytest_full.log`

## Findings Applied

- TYPE-PATH-001: Path normalization for analyze_digest_script and metrics_digest_md
- CONFIG-001: AUTHORITATIVE_CMDS_DOC export maintained for all test invocations
- DATA-001: Metrics JSON/highlights text format contracts preserved

## Next Step

Execute dense Phase C→G pipeline with `--clobber` to capture real metrics evidence and validate end-to-end digest generation with actual MS-SSIM/MAE deltas (2-4 hour run deferred per nucleus principle).
