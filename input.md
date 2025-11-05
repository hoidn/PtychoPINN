Summary: Run the dense Phase C→G pipeline with the automated reporting helper and capture verifiable evidence.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Mapped tests:
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_invokes_reporting_helper -vv
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T070500Z/phase_g_dense_full_execution_real_run/

Do Now:
- Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
- Implement: tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_invokes_reporting_helper — add stubbed execution test that records run_command calls and asserts the reporting helper invocation (script path + aggregate_report.md output + cli/aggregate_report_cli.log).
- Validate: pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_invokes_reporting_helper -vv
- Execute: python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T070500Z/phase_g_dense_full_execution_real_run --dose 1000 --view dense --splits train test --clobber
- Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T070500Z/phase_g_dense_full_execution_real_run/

How-To Map:
1. `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`
2. RED proof: `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_invokes_reporting_helper -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T070500Z/phase_g_dense_full_execution_real_run/red/pytest_reporting_helper_exec_red.log`
3. Add the new test following plan details (stub prepare_hub/validate_phase_c_metadata/summarize_phase_g_outputs, monkeypatch run_command, assert final call targets report helper script, aggregate_report.md, and cli/aggregate_report_cli.log).
4. GREEN proof: `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_invokes_reporting_helper -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T070500Z/phase_g_dense_full_execution_real_run/green/pytest_reporting_helper_exec_green.log`
5. Regression guard: `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T070500Z/phase_g_dense_full_execution_real_run/green/pytest_collect_only_green.log`
6. Selector inventory: `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T070500Z/phase_g_dense_full_execution_real_run/collect/pytest_phase_g_orchestrator_collect.log`
7. Dense run: `stdbuf -oL -eL python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T070500Z/phase_g_dense_full_execution_real_run --dose 1000 --view dense --splits train test --clobber | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T070500Z/phase_g_dense_full_execution_real_run/green/run_phase_g_dense_cli.log`
8. Sanity checks: `ls -R plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T070500Z/phase_g_dense_full_execution_real_run/analysis` and review `aggregate_report.md` + `aggregate_report_cli.log` for signed deltas; summarize key metrics in `summary/summary.md`.
9. Documentation: Update `docs/fix_plan.md`, `summary/summary.md`, and—after GREEN + collect-only—refresh `docs/TESTING_GUIDE.md` Phase G section and `docs/development/TEST_SUITE_INDEX.md` with the new selector evidence.

Pitfalls To Avoid:
- Do not skip exporting AUTHORITATIVE_CMDS_DOC; orchestrator guards expect it.
- Keep hub path exactly as provided; no extra staging directories to avoid broken relative paths.
- Ensure run_command stub in tests records both cmd and log_path; forgetting log assertions weakens coverage.
- Do not delete existing Phase G artifacts from prior hubs; work only inside 2025-11-08T070500Z.
- Avoid running dense pipeline without `--clobber`; stale outputs will short-circuit phases.
- Preserve Path usage (TYPE-PATH-001) when editing tests; never hardcode strings without Path().
- Capture pytest logs via tee; missing RED/GREEN evidence will block closure.
- Treat TypeError from metadata guard as blocker; document signature in ledger if encountered.

If Blocked:
- Capture failing command output into `red/` (pytest) or `analysis/blocker.log` (pipeline).
- Add blocking note to docs/fix_plan.md Attempts History with error summary.
- Update galph_memory.md dwell/state and mark next_action=switch_focus if dependency emerges.
- Halt further execution until supervisor guidance arrives.

Findings Applied (Mandatory):
- POLICY-001 — PyTorch runtime mandatory; do not silence torch import errors during CLI run.
- CONFIG-001 — Maintain update_legacy_dict bridge within orchestrator before legacy modules; verify no edits regress ordering.
- DATA-001 — Metadata guard must stay enabled; follow TROUBLESHOOTING.md if it fails.
- TYPE-PATH-001 — Normalize all paths with Path() in new test assertions.
- OVERSAMPLING-001 — Dense gridsize/overlap assumptions unchanged; report deviations in summary.

Pointers:
- `docs/findings.md:8` — POLICY-001 (PyTorch dependency policy).
- `docs/findings.md:10` — CONFIG-001 (params.cfg bridge rule).
- `docs/findings.md:14` — DATA-001 (NPZ contract expectations).
- `docs/findings.md:21` — TYPE-PATH-001 (path normalization requirement).
- `docs/development/TEST_SUITE_INDEX.md:60` — Phase G orchestrator selectors.
- `docs/TESTING_GUIDE.md:280` — Phase G reporting helper workflow.
- `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:95` — Dense execution checklist context.

Next Up (optional):
- If time remains, begin drafting parity comparison between aggregate_report.md and prior manual reports for ledger inclusion.

Doc Sync Plan (Conditional):
- After GREEN + collect-only, append new selector entry to `docs/development/TEST_SUITE_INDEX.md` and expand `docs/TESTING_GUIDE.md` Phase G section with automated report execution notes. Archive `pytest_phase_g_orchestrator_collect.log` under `collect/` as proof.

Mapped Tests Guardrail:
- Confirm `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_invokes_reporting_helper -vv` collects (log under `collect/pytest_phase_g_helper_exec_collect.log` if run separately). Do not mark complete until selector collects >0.

Hard Gate:
- If any mapped selector collects 0 or pipeline command exits non-zero, stop and mark blocked with artifacts; do not claim completion.
