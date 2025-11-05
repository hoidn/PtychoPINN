# Phase G Dense Full Execution — Highlight Export & Evidence Run Plan (2025-11-08T090500Z)

**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001  
**Focus:** Phase G comparison & analysis (dense real evidence + automated report)  
**Action Type:** Planning (supervisor loop)  
**Target State:** ready_for_implementation

---

## Context

- Reporting helper integration (collect + exec) is GREEN, and regression coverage now asserts the helper executes during real runs.
- Dense Phase C→G pipeline (dose=1000, view=dense) has **not** been executed since the helper landed; no CLI logs, metrics summary, or aggregate report exist under a fresh hub.
- Existing helper outputs a Markdown aggregate report but lacks a compact highlights export for quick ledger updates.

## Objectives

1. Enhance `bin/report_phase_g_dense_metrics.py` to optionally emit a concise highlights text file (top-line deltas) without altering existing CLI semantics.
2. Update `bin/run_phase_g_dense.py` to request the highlights export so real runs leave `analysis/aggregate_highlights.txt` alongside the Markdown report.
3. Execute the dense Phase C→G pipeline (`--clobber`) and archive CLI transcripts, metrics JSON/Markdown, highlights, and aggregate report under the new hub.
4. Summarize MS-SSIM/MAE deltas in `summary/summary.md`, update `docs/fix_plan.md`, and capture Turn Summary (with artifacts path) per workflow guardrails.

## Deliverables

- Updated helper script supporting `--highlights <path>` argument that writes a short textual summary (phase & amplitude MS-SSIM/MAE deltas vs Baseline + PtyChi).
- Updated orchestrator command wiring passing the highlights flag; success message notes both Markdown and highlights outputs.
- RED→GREEN pytest evidence:
  - `tests/study/test_phase_g_dense_metrics_report.py` extended to cover highlights export.
  - `tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands` and `::test_run_phase_g_dense_exec_invokes_reporting_helper` updated for the new argument/path.
- Hub `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T090500Z/phase_g_dense_full_execution_real_run/` populated with:
  - `cli/*.log` transcripts (Phase C→G, aggregate helper).
  - `analysis/metrics_summary.json`, `analysis/metrics_summary.md`, `analysis/aggregate_report.md`, `analysis/aggregate_highlights.txt`.
  - `summary/summary.md` capturing Turn Summary + metric highlights.
- Ledger (`docs/fix_plan.md`) Attempt entry referencing artifacts and findings.

## Task Breakdown

1. **TDD — Highlights export support**
   - Update helper argparse to accept optional `--highlights`.
   - Implement helper function to derive top-line text (phase/amplitude MS-SSIM & MAE deltas).
   - RED: extend existing helper tests to expect failure when highlights path missing (if flagged) before implementation.
   - GREEN: ensure both helper tests cover Markdown + highlights outputs.

2. **TDD — Orchestrator wiring**
   - Adjust orchestrator collect-only output to display the new highlights command arguments.
   - Update exec path to pass highlights file path (`analysis/aggregate_highlights.txt`) to helper.
   - Extend orchestrator regression tests to assert highlights argument and log path.

3. **Execution — Dense Phase C→G run**
   - `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`.
   - Run `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T090500Z/phase_g_dense_full_execution_real_run --dose 1000 --view dense --splits train test --clobber`.
   - Verify CLI logs recorded; confirm helper writes Markdown + highlights with coherent values.
   - Archive pytest, collect-only, and CLI outputs under the hub (red/green/collect/cli/analysis).

4. **Documentation & Ledger**
   - Summarize metrics in `summary/summary.md` (including highlights excerpt).
   - Update `docs/fix_plan.md` Attempts history and link artifacts.
   - Append Turn Summary block to `summary/summary.md` and respond with same block.

## Guardrails & Findings Reinforced

- **POLICY-001:** PyTorch dependency remains required; do not bypass ImportErrors during helper execution.
- **CONFIG-001:** Orchestrator already handles legacy bridge; ensure no new code paths skip it.
- **DATA-001:** Respect metadata validation gate; investigate immediately if validator fails post-`--clobber`.
- **TYPE-PATH-001:** Normalize filesystem interactions via `Path`.
- **OVERSAMPLING-001:** Do not alter dense overlap parameters; treat unexpected metric regressions (>±0.05 MS-SSIM, ±0.01 MAE) as blockers.

## Exit Criteria

- Helper + orchestrator modifications merged with RED→GREEN pytest evidence archived.
- Dense pipeline run completes successfully in the new hub with metrics JSON/Markdown + highlights.
- Turn Summary + ledger updates reference artifact paths and findings.
- Ready to transition focus to closing Phase G evidence (or next initiative) after artifacts are reviewed.
