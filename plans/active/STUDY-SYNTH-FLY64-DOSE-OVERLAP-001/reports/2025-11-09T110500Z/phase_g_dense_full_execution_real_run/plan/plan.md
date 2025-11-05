# Phase G Dense Real-Run Evidence — Supervisor Plan (2025-11-09T110500Z)

## Objective
Capture real Phase G dense metrics evidence after the delta banner landed by persisting the computed deltas to a structured artifact and running the full Phase C→G pipeline with `--clobber`.

## Scope for Ralph
1. **Persist delta summary**
   - Extend `run_phase_g_dense.py::main` to serialize the computed MS-SSIM/MAE deltas to `analysis/metrics_delta_summary.json` (and print the saved path in the success banner).
   - Ensure JSON schema includes both model comparisons (`ptycho_vs_baseline`, `ptycho_vs_ptychi`) with amplitude/phase entries for MS-SSIM and MAE; use `null` when data is unavailable.
2. **Test tightening**
   - Update `tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest` to assert the JSON file is created with expected content and that the success banner surfaces the path.
   - Keep existing delta stdout assertions intact (regression guard).
3. **Documentation touch-up**
   - Amend `docs/TESTING_GUIDE.md` Phase G section to note the new delta summary JSON artifact and how to inspect it during evidence runs.
4. **Evidence run**
   - Execute `run_phase_g_dense.py --clobber` against the dense hub, capture CLI log, and archive delta JSON + digest outputs under the 2025-11-09T110500Z hub.
   - Extract delta values into a short highlight note for summary.md and docs/fix_plan.md.

## Required Tests
- `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv`
- `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands -vv`
- `pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_success_digest -vv`

## Artifacts to Produce
- `$HUB/red/pytest_*.log` for failing RED (delta JSON assertions) if any.
- `$HUB/green/pytest_*.log` for GREEN targeted tests and collect-only guard.
- `$HUB/cli/run_phase_g_dense.log` capturing the full pipeline stdout/stderr (with delta block and banner path).
- `$HUB/analysis/metrics_summary.json`, `aggregate_report.md`, `aggregate_highlights.txt`, `metrics_digest.md`, and new `metrics_delta_summary.json`.
- `$HUB/analysis/metrics_delta_highlights.txt` summarizing key deltas (crafted post-run).
- `$HUB/analysis/artifact_inventory.txt` snapshot of key evidence.

## Exit Criteria
- Delta JSON persisted with numeric values matching stdout banner (3-decimal precision) and mentioned in success banner.
- All mapped selectors GREEN with logs archived under the new hub.
- Dense pipeline completes successfully (exit 0) with fresh metrics artifacts.
- `summary/summary.md` updated with MS-SSIM/MAE deltas (PtychoPINN vs Baseline/PtyChi) and artifact links.
- `docs/fix_plan.md` Attempts History updated with this execution loop and captured evidence pointers.
