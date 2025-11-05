## Phase G Dense Full Execution — Real Evidence Run Plan (2025-11-08T130500Z)

**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001  
**Focus:** Phase G comparison & analysis (dense real evidence + automated report)  
**Action Type:** Planning (supervisor loop)  
**Target State:** ready_for_implementation

---

### Context
- Highlights preview support landed in the previous loop (2025-11-08T110500Z+exec) with regression coverage and doc updates, but no full Phase C→G run has produced real metrics since the preview path was introduced.
- The orchestrator script `bin/run_phase_g_dense.py` is stable after the Phase C `n_images` fix; earlier blocker (`TypeError: object of type 'float' has no len()`) is resolved.
- We need a production `--clobber` run for dose=1000, view=dense to capture actual MS-SSIM/MAE deltas and verify the new stdout preview in a real log.

### Objectives
1. Re-validate the highlights preview regression test (`test_run_phase_g_dense_exec_prints_highlights_preview`) before the real run to ensure harness behavior has not regressed.
2. Execute `bin/run_phase_g_dense.py` end-to-end with `--clobber --dose 1000 --view dense --splits train test`, exporting `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`.
3. Archive CLI transcripts, metrics JSON/Markdown, highlights text, aggregate report, and stdout preview evidence under `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T130500Z/phase_g_dense_full_execution_real_run/`.
4. Update `summary/summary.md` with measured MS-SSIM/MAE deltas, CLI observations, and the turn summary block; ensure docs/fix_plan.md logs this execution attempt.

### Deliverables
- `green/pytest_highlights_preview.log` capturing a fresh GREEN run for the highlights preview selector.
- CLI logs for all eight pipeline commands under `cli/` plus `cli/aggregate_report_cli.log` showing the stdout preview.
- Updated analysis artifacts: `analysis/metrics_summary.json`, `analysis/metrics_summary.md`, `analysis/aggregate_report.md`, `analysis/aggregate_highlights.txt`.
- Updated `summary/summary.md` (this hub) with key deltas, preview transcript snippet, and any anomalies.
- Ledger update in `docs/fix_plan.md` documenting this loop.

### Guardrails & Findings
- **POLICY-001 / CONFIG-001 / DATA-001 / TYPE-PATH-001 / OVERSAMPLING-001** remain in force; abort run if any guard is violated.
- If pipeline fails again, capture the error signature, log paths, and mark the attempt as blocked in docs/fix_plan.md + galph_memory.md.
- Respect storage hygiene: no artifacts outside this hub; delete transient scratch data under `tmp/` after run.

### Exit Criteria
- Highlights preview selector passes (Green) with log archived.
- Phase C→G pipeline completes successfully (all commands exit 0) and stdout preview appears in CLI log.
- summary.md updated with metrics + findings references; docs/fix_plan.md latest attempt documents the real run.
- If blocked, blocker rationale + remediation steps captured in summary.md and docs/fix_plan.md with artifacts pointing to relevant logs.
