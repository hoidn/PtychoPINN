Summary: Add highlights export to the Phase G reporting helper, update orchestrator wiring, and run the dense Phase C→G pipeline with full evidence capture.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Mapped tests:
  - pytest tests/study/test_phase_g_dense_metrics_report.py::test_report_phase_g_dense_metrics -vv
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands -vv
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_invokes_reporting_helper -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T090500Z/phase_g_dense_full_execution_real_run/

Do Now:
- Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
- Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/report_phase_g_dense_metrics.py::main — add an optional `--highlights` output that writes top-line MS-SSIM/MAE deltas and update `run_phase_g_dense.py::main` plus associated tests to consume the new argument.
- Validate: pytest tests/study/test_phase_g_dense_metrics_report.py::test_report_phase_g_dense_metrics -vv
- Execute: pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands -vv && pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_invokes_reporting_helper -vv
- Execute: stdbuf -oL -eL python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T090500Z/phase_g_dense_full_execution_real_run --dose 1000 --view dense --splits train test --clobber
- Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T090500Z/phase_g_dense_full_execution_real_run/

How-To Map:
1. `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`
2. RED helper proof: `pytest tests/study/test_phase_g_dense_metrics_report.py::test_report_phase_g_dense_metrics -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T090500Z/phase_g_dense_full_execution_real_run/red/pytest_report_helper_red.log`
3. Update `bin/report_phase_g_dense_metrics.py` to accept `--highlights`, write concise deltas to the specified path, and extend tests for highlights export (handle RED➔GREEN cycle).
4. GREEN helper proof: `pytest tests/study/test_phase_g_dense_metrics_report.py::test_report_phase_g_dense_metrics -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T090500Z/phase_g_dense_full_execution_real_run/green/pytest_report_helper_green.log`
5. Orchestrator RED/GREEN: `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T090500Z/phase_g_dense_full_execution_real_run/green/pytest_collect_only_green.log`
6. Regression exec path: `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_invokes_reporting_helper -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T090500Z/phase_g_dense_full_execution_real_run/green/pytest_reporting_helper_exec_green.log`
7. Selector inventory: `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T090500Z/phase_g_dense_full_execution_real_run/collect/pytest_phase_g_orchestrator_collect.log`
8. Dense run with highlights: `stdbuf -oL -eL python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T090500Z/phase_g_dense_full_execution_real_run --dose 1000 --view dense --splits train test --clobber | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T090500Z/phase_g_dense_full_execution_real_run/green/run_phase_g_dense_cli.log`
9. Sanity check outputs: ensure `analysis/metrics_summary.json`, `analysis/metrics_summary.md`, `analysis/aggregate_report.md`, and `analysis/aggregate_highlights.txt` exist; review highlights for plausible deltas (±0.05 MS-SSIM, ±0.01 MAE thresholds).
10. Documentation: update `summary/summary.md` with Turn Summary + metric excerpt, then refresh `docs/fix_plan.md`, `docs/TESTING_GUIDE.md` Phase G section, and `docs/development/TEST_SUITE_INDEX.md` after GREEN + collect-only.

Pitfalls To Avoid:
- Do not bypass AUTHORITATIVE_CMDS_DOC export; orchestrator guard aborts otherwise.
- Keep hub path exactly as provided; highlights command expects sibling `analysis/` directory.
- Maintain Path() normalization when editing tests (TYPE-PATH-001).
- Capture RED evidence before implementation; missing red logs will block sign-off.
- Ensure helper writes highlights only when flag supplied; avoid unconditional I/O.
- Do not run dense pipeline without `--clobber`; stale artifacts lead to skipped phases.
- Preserve JSON schema from `summarize_phase_g_outputs`; highlights should read existing keys, not mutate structure.
- Treat validator or CLI failures as blockers—log outputs under `red/` or `analysis/blocker.log` immediately.
- Do not edit core TensorFlow/PyTorch backends; limit changes to initiative scripts/tests.

If Blocked:
- Save failing pytest output to `red/` and CLI failures to `analysis/blocker.log`.
- Document error signature + mitigation attempt in `docs/fix_plan.md` Attempts history.
- Update `summary/summary.md` with blocker details and ping supervisor via galph_memory (state=switch_focus if needed).
- Halt further steps until supervisor guidance arrives; do not downgrade tests or skip evidence.

Findings Applied (Mandatory):
- POLICY-001 — PyTorch dependency is mandatory; surface any torch ImportError immediately.
- CONFIG-001 — Maintain legacy bridge order inside orchestrator; confirm helper changes do not skip it.
- DATA-001 — Respect metadata validator output; dense run must pass dataset checks.
- TYPE-PATH-001 — Normalize filesystem paths via `Path` in scripts/tests.
- OVERSAMPLING-001 — Dense overlap parameters unchanged; flag unexpected metric deviations.

Pointers:
- docs/findings.md:8 — POLICY-001 dependency guardrails.
- docs/findings.md:10 — CONFIG-001 legacy bridge rule.
- docs/findings.md:14 — DATA-001 NPZ contract requirements.
- docs/findings.md:21 — TYPE-PATH-001 path normalization note.
- docs/development/TEST_SUITE_INDEX.md:60 — Orchestrator/Phase G selectors.
- docs/TESTING_GUIDE.md:280 — Phase G helper workflow.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:189 — Phase G comparison checklist.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T090500Z/phase_g_dense_full_execution_real_run/plan/plan.md — Current loop plan details.

Next Up (optional):
- If highlights workflow completes quickly, stage Phase H doc updates summarizing dense vs. sparse comparisons.

Doc Sync Plan (Conditional):
- After GREEN + collect-only, archive `pytest_phase_g_orchestrator_collect.log` and update `docs/development/TEST_SUITE_INDEX.md` plus `docs/TESTING_GUIDE.md` with highlights export instructions.

Mapped Tests Guardrail:
- Confirm `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -vv` reports >0 tests; log under `collect/pytest_phase_g_orchestrator_collect.log`.

Hard Gate:
- If any mapped selector collects 0 or dense pipeline exits non-zero, stop immediately, record artifacts, and mark the attempt as blocked rather than done.
