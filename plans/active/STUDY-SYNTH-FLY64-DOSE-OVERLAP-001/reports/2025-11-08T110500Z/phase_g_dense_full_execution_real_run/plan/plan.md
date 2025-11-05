# Phase G Dense Full Execution — Real Evidence Run Plan (2025-11-08T110500Z)

**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001  
**Focus:** Phase G comparison & analysis (dense real evidence + automated report)  
**Action Type:** Planning (supervisor loop)  
**Target State:** ready_for_implementation

---

## Context

- Reporting helper integration (Markdown + highlights) and regression coverage are GREEN as of 2025-11-08T090500Z+exec. No real dense run has been executed since integrating highlights.
- We still need a production `--clobber` run for dose=1000, view=dense to capture actual MS-SSIM/MAE deltas and archive CLI/analysis artifacts under a fresh hub.
- Reviewing highlights in situ requires opening the Markdown file; we want the orchestrator to echo a concise preview so findings can be sanity-checked quickly in logs.

## Objectives

1. Extend `bin/run_phase_g_dense.py` so that, after calling the reporting helper, it reads the generated highlights text and prints the first few lines under an "Aggregate highlights" banner (while still writing the file to disk).
2. Add pytest coverage ensuring the new highlights preview is emitted in execution mode without breaking collect-only output ordering.
3. Execute the dense Phase C→G pipeline (`--clobber`) using the updated orchestrator and archive CLI transcripts, metrics JSON/Markdown, highlights text, and aggregated outputs under this loop's hub.
4. Summarize the measured MS-SSIM/MAE deltas in `summary/summary.md`, update `docs/fix_plan.md`, and record the Turn Summary per workflow guardrails.

## Deliverables

- Updated orchestrator success flow (post-report helper) that loads `analysis/aggregate_highlights.txt` and prints its contents to stdout with clear separators; failure to read must raise with actionable message.
- Tests:
  - `tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_invokes_reporting_helper` updated (or companion test) to assert the highlights preview appears in captured stdout during execution mode.
  - Collect-only regression test still validates command list order (no extra preview there).
- RED→GREEN pytest evidence (targeted selector + full suite) archived under `red/` & `green/`.
- Hub populated with CLI logs, metrics summary (JSON + Markdown), highlights text, aggregate report, and summary write-up.

## Task Breakdown

1. **TDD — Highlights preview coverage**
   - Arrange regression test to simulate successful execution with stubbed helper writing deterministic highlights text (e.g., "Phase MS-SSIM Δ vs Baseline: +0.123").
   - RED: assert preview lines are present post-run; expect failure before implementation.

2. **Implementation — Orchestrator preview emission**
   - After helper command returns, open the highlights file, validate non-empty content, and print with separators (respect TYPE-PATH-001 by using `Path`).
   - Ensure preview only occurs in execution mode (skip during collect-only) and does not swallow subprocess errors.

3. **Execution — Dense Phase C→G run**
   - `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`.
   - Run `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T110500Z/phase_g_dense_full_execution_real_run --dose 1000 --view dense --splits train test --clobber`.
   - Capture CLI logs and verify highlights preview appears in stdout and CLI log (`cli/aggregate_report_cli.log`).

4. **Documentation & Ledger**
   - Update `summary/summary.md` with MS-SSIM/MAE deltas, pipeline status, and guardrail notes (include Turn Summary block at top).
   - Update `docs/fix_plan.md` Attempts history with this run (findings: POLICY-001, CONFIG-001, DATA-001, TYPE-PATH-001, OVERSAMPLING-001).

## Guardrails & Findings Reinforced

- **POLICY-001:** PyTorch dependency remains mandatory; do not mask ImportErrors.
- **CONFIG-001:** Maintain CONFIG-001 bridge set-up before invoking legacy components (already handled by orchestrator).
- **DATA-001:** Investigate immediately if metadata validator fails after rerun.
- **TYPE-PATH-001:** Coerce filesystem paths to `Path` objects and avoid bare strings.
- **OVERSAMPLING-001:** Treat unexpected metric regressions (|Δ MS-SSIM| > 0.05, |Δ MAE| > 0.01) as blockers; defer ledger status if encountered.

## Exit Criteria

- Highlights preview feature covered by tests (RED→GREEN) with orchestrator output verified in execution logs.
- Dense Phase C→G pipeline run completes successfully with updated artifacts archived under this hub.
- summary.md updated with key deltas + Turn Summary; docs/fix_plan.md references this attempt and findings.
- Focus ready for follow-up (either closing study or branching to sparse view execution) once evidence reviewed.
