# Phase G Dense Full Execution — Evidence Run Plan (2025-11-08T070500Z)

**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001  
**Focus:** Phase G comparison & analysis (dense evidence + automated report)  
**Action Type:** Planning (supervisor loop)  
**Target State:** ready_for_implementation  

---

## Context

- Reporting helper integration is GREEN (2025-11-08T050500Z+exec). Collect-only mode shows the new command, and orchestrator executes it after `summarize_phase_g_outputs()`.
- Real Phase C→G pipeline run with the automated reporting helper remains outstanding; no hub contains fresh CLI logs, metrics summary, or Markdown report generated via the orchestrator.
- Guard selectors (`test_summarize_phase_g_outputs`, `test_report_phase_g_dense_metrics`) cover helper behavior, but there is no regression test ensuring the real execution path invokes the reporting helper.

## Objectives

1. Extend orchestrator tests to assert the reporting helper command executes in non-collect runs (stubbing heavy phases).
2. Execute the dense Phase C→G pipeline (`--dose 1000 --view dense --splits train test --clobber`) using the orchestrator, capturing full CLI logs and generated artifacts.
3. Validate resulting artifacts (`metrics_summary.json`, `metrics_summary.md`, `aggregate_report.md`, `aggregate_report_cli.log`) and summarize key deltas.
4. Update documentation/ledger with evidence paths and findings alignment.

## Deliverables

- New regression test `test_run_phase_g_dense_exec_invokes_reporting_helper` under `tests/study/test_phase_g_dense_orchestrator.py`.
- Hub `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T070500Z/phase_g_dense_full_execution_real_run/` populated with:
  - `cli/phase_*.log` transcripts for Phase C→G plus `aggregate_report_cli.log`.
  - `analysis/metrics_summary.json`, `analysis/metrics_summary.md`, `analysis/aggregate_report.md`, and optional highlights file.
  - `summary/summary.md` updated with Turn Summary + metric deltas.
- Ledger (`docs/fix_plan.md`) Attempt entry documenting the evidence run and artifacts.

## Task Breakdown

1. **TDD — Reporting helper execution test**
   - Add `test_run_phase_g_dense_exec_invokes_reporting_helper`:
     - Import orchestrator module via spec loader.
     - Stub `prepare_hub`, `validate_phase_c_metadata`, and `summarize_phase_g_outputs` to avoid heavy work.
     - Monkeypatch `run_command` to record invocations.
     - Run `main()` against a tmp hub (without `--collect-only`) and assert:
       - Exit code is 0.
       - Final recorded command targets `report_phase_g_dense_metrics.py` with `--output <aggregate_report.md>`.
       - Log path equals `cli/aggregate_report_cli.log`.
   - Capture RED log before code updates.

2. **Execution — Dense pipeline with automated report**
   - `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`.
   - Run `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T070500Z/phase_g_dense_full_execution_real_run --dose 1000 --view dense --splits train test --clobber`.
   - Ensure CLI logs populate under `cli/` and analysis outputs under `analysis/`.
   - Verify guard helpers succeed; record a brief highlights file (e.g., `analysis/aggregate_highlights.txt`) summarizing MS-SSIM/MAE deltas vs. Baseline/PtyChi.

3. **Validation & Documentation**
   - Inspect `aggregate_report.md` for sanity (no placeholder values, deltas signed with three decimals).
   - Update `summary/summary.md`, `docs/fix_plan.md`, and ledger artifacts with evidence references and findings alignment (POLICY-001, CONFIG-001, DATA-001, TYPE-PATH-001, OVERSAMPLING-001).
   - If selector inventory changes, refresh `docs/TESTING_GUIDE.md` and `docs/development/TEST_SUITE_INDEX.md` after GREEN.

## Guardrails & Findings Reinforced

- **POLICY-001:** PyTorch runtime remains mandatory; do not suppress ImportError if CLI surfaces torch issues.
- **CONFIG-001:** Always call `update_legacy_dict(params.cfg, config)` before legacy modules (already handled by orchestrator script).
- **DATA-001:** Phase C metadata guard must stay active; investigate immediately if it blocks the run.
- **TYPE-PATH-001:** Normalize all filesystem interactions with `Path`.
- **OVERSAMPLING-001:** Dense gridsize/oversampling settings must remain unchanged; report unexpected deltas (> ±0.05 MS-SSIM or ±0.01 MAE).

## Exit Criteria

- New regression test added and GREEN with orchestrator modifications applied.
- Dense Phase C→G CLI run completes without blockers; `analysis/metrics_summary.json|.md|aggregate_report.md` + logs present.
- Turn Summary + highlights recorded; ledger references artifact paths.
- No open TODOs for this hub; ready for closure pending engineering execution.
