Summary: Execute the dense Phase C→G pipeline to capture real MS-SSIM/MAE deltas and verify the highlights preview behaves in a production run.
Mode: Perf
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Mapped tests:
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T130500Z/phase_g_dense_full_execution_real_run/

Do Now:
- Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
- Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T130500Z/phase_g_dense_full_execution_real_run/summary/summary.md::Phase G dense evidence write-up — record real run deltas, preview transcript, and findings references.
- Validate: pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview -vv
- Execute: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T130500Z/phase_g_dense_full_execution_real_run --dose 1000 --view dense --splits train test --clobber
- Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T130500Z/phase_g_dense_full_execution_real_run/

How-To Map:
1. `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`
2. `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T130500Z/phase_g_dense_full_execution_real_run/green/pytest_highlights_preview_green.log`
3. `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T130500Z/phase_g_dense_full_execution_real_run --dose 1000 --view dense --splits train test --clobber | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T130500Z/phase_g_dense_full_execution_real_run/cli/run_phase_g_dense_cli.log`
4. After completion, verify `{hub}/analysis/metrics_summary.json`, `metrics_summary.md`, `aggregate_report.md`, and `aggregate_highlights.txt` exist and stdout preview text appears in `cli/aggregate_report_cli.log`.
5. Summarize measured MS-SSIM/MAE deltas and preview snippet in `summary/summary.md` (replace placeholder, keep turn summary at top).
6. Update `docs/fix_plan.md` Latest Attempt entry with execution results, findings applied, and artifact links.

Pitfalls To Avoid:
- Do not skip exporting `AUTHORITATIVE_CMDS_DOC`; orchestrator asserts it.
- Keep all artifacts within this hub; delete any temporary files created elsewhere.
- Treat validator or CLI failures as hard blockers; capture logs and stop instead of retrying blindly.
- Monitor runtime (2–4 hr expected); if Stage fails, capture precise command + exit code in summary and ledger.
- Do not modify stable core modules (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`).
- Ensure highlights preview remains execution-only; collect-only output should stay unchanged.
- Record actual MS-SSIM/MAE deltas even if surprising—do not downplay anomalies.

If Blocked:
- Archive failing pytest/pipeline logs under `red/` within this hub, add blocker note to `summary/summary.md`, and update docs/fix_plan.md with blocker status.
- Capture the exact error signature and command path; include pointers to CLI log files.
- Ping supervisor by noting the block in galph_memory with `state=switch_focus` if remediation requires upstream fixes.

Findings Applied (Mandatory):
- POLICY-001 — PyTorch dependency is mandatory; abort if comparison phase hits ImportError.
- CONFIG-001 — Legacy bridge must run before any legacy phases; confirm orchestrator logs reflect it.
- DATA-001 — Phase C metadata validator failures are stop-the-line issues.
- TYPE-PATH-001 — Continue using `Path` objects for all IO (highlight preview reads).
- OVERSAMPLING-001 — Large MS-SSIM/MAE deviations may signal overlap misconfiguration; investigate immediately.

Pointers:
- docs/fix_plan.md:35 — Latest highlights preview implementation context.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T130500Z/phase_g_dense_full_execution_real_run/plan/plan.md:1 — Current execution plan objectives.
- docs/TESTING_GUIDE.md:268 — Phase G orchestrator + reporting helper workflow.
- docs/development/TEST_SUITE_INDEX.md:62 — Test inventory for Phase G orchestrator selectors.

Next Up (optional):
- If dense run succeeds early, stage a follow-up plan for the sparse-view pipeline execution using the same highlights workflow.

Mapped Tests Guardrail:
- Ensure the mapped selector above collects exactly one test; if it fails to collect, stop and diagnose before proceeding.

Hard Gate:
- Do not mark the loop complete unless the dense pipeline exits 0 and the highlights preview log is captured; otherwise label the attempt blocked with artifacts noted.
