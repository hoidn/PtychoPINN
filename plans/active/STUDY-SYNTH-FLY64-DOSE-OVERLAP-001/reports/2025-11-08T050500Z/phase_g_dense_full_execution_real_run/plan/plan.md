# Phase G Dense Full Execution — Real Evidence Plan (2025-11-08T050500Z)

**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001  
**Focus:** Phase G comparison & analysis (dense evidence)  
**Action Type:** Planning (supervisor loop)  
**Target State:** ready_for_implementation

---

## Context

- Aggregate metrics helper (`summarize_phase_g_outputs`) and reporting CLI (`report_phase_g_dense_metrics.py`) are GREEN as of 2025-11-08T030500Z+exec.
- No end-to-end dense Phase C→G run has executed with the new aggregate/delta tooling; `analysis/metrics_summary.json` and Markdown report are absent in the current hub.
- Engineers still run the reporting helper manually after `summarize_phase_g_outputs`, increasing drift risk for future runs.

## Objectives

1. Integrate the reporting helper into the orchestrator so dense runs automatically emit a Markdown delta report (ensuring AUTHORITATIVE_CMDS_DOC compliance and deterministic artifact placement).
2. Execute the dense Phase C→G pipeline (dose=1000, view=dense, splits=train/test) on a clean hub with `--clobber`, capturing CLI logs and verifying guard checks.
3. Archive reporting helper output (`aggregate_report.md` + CLI log) under the hub and summarize key deltas for the ledger.
4. Update documentation/ledger with evidence paths and guard + delta findings.

## Deliverables

- Updated `plans/active/.../bin/run_phase_g_dense.py` calling the reporting helper via `run_command` with a dedicated CLI log and Markdown output.
- Extended pytest coverage in `tests/study/test_phase_g_dense_orchestrator.py` asserting the collect-only command inventory includes the new reporting helper invocation.
- Hub `plans/active/.../reports/2025-11-08T050500Z/phase_g_dense_full_execution_real_run/` populated with:
  - `cli/phase_*` transcripts for Phase C→G.
  - `analysis/metrics_summary.json`, `analysis/metrics_summary.md`, `analysis/aggregate_report.md`, `analysis/aggregate_report_cli.log`.
  - Guard logs for metadata validation and summarization.
- `summary/summary.md` capturing Turn Summary + highlight deltas (PtychoPINN vs Baseline/PtyChi) with pointers to logs/reports.
- Ledger/doc updates referencing findings (POLICY-001, CONFIG-001, DATA-001, TYPE-PATH-001, OVERSAMPLING-001).

## Task Breakdown

1. **TDD — Orchestrator command inventory**
   - Update `test_run_phase_g_dense_collect_only_generates_commands` to expect a “Phase G: Aggregate Metrics Report” command pointing to `report_phase_g_dense_metrics.py`.
   - Capture RED evidence (test fails because orchestrator has not yet appended the command).

2. **Implementation — Reporting helper integration**
   - Modify `run_phase_g_dense.py::main` to append the reporting helper command to the command list (respecting TYPE-PATH-001) and, after `summarize_phase_g_outputs`, invoke it via `run_command` writing to `analysis/aggregate_report_cli.log`.
   - Ensure collect-only mode lists this command with the correct metrics/output arguments.
   - Generate Markdown output at `analysis/aggregate_report.md`.

3. **Execution — Dense pipeline & reporting**
   - `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`.
   - Run `python plans/.../bin/run_phase_g_dense.py --hub <HUB> --dose 1000 --view dense --splits train test --clobber`.
   - On success, confirm `analysis/metrics_summary.json`, `analysis/metrics_summary.md`, `analysis/aggregate_report.md`, and `analysis/aggregate_report_cli.log` exist with fresh content.

4. **Validation & Documentation**
   - Record guard outputs (`validate_phase_c_metadata`, `summarize_phase_g_outputs`) and capture new aggregate deltas in `analysis/aggregate_highlights.txt` or equivalent.
   - Update `summary/summary.md`, `docs/fix_plan.md`, `docs/TESTING_GUIDE.md`, and `docs/development/TEST_SUITE_INDEX.md` with evidence references.

## Findings & Guardrails Reinforced

- **POLICY-001:** PyTorch dependency remains mandatory; CLI commands must not suppress ImportError from pty-chi.
- **CONFIG-001:** Orchestrator continues to bridge params.cfg before legacy consumers; reporting helper runs read-only.
- **DATA-001:** Phase C metadata guard remains in post-run validation.
- **TYPE-PATH-001:** All new paths (helper script, metrics, logs) normalized via `Path`.
- **OVERSAMPLING-001:** Dense overlap parameters unchanged; report should flag unexpected deltas (>±0.05 MS-SSIM, etc.).

## Exit Criteria

- Collect-only selector reports include the new reporting command (pytest GREEN evidence).
- Dense run completes without blockers; all CLI logs present under the hub.
- Reporting helper Markdown and CLI log generated automatically by orchestrator.
- Findings documented in ledger with artifact paths; no pending TODOs for this hub.
