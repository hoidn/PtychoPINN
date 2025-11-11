Executive Summary
- Repeated hub scaffolding without execution: the last 4 hubs (2025-11-10T193500Z, 2025-11-11T001033Z, 2025-11-11T003351Z, 2025-11-11T005802Z) contain empty red/green/cli/collect directories, indicating planning without tests or runs. Evidence: see summary notes and empty dirs under each hub.
- Excessive CLI-tweak cycles: multiple recent hubs focused on CLI log pattern fixes and guards rather than producing Phase D–G artifacts and SSIM/MS-SSIM deliverables. Evidence: presence of pytest_cli_* logs vs. missing pipeline outputs.
- Documentation drift: TESTING_GUIDE claims uniform 3-decimal precision for all deltas, while the implementation moved to MS-SSIM (3) and MAE (6). This caused rework and confusion. Evidence: docs/TESTING_GUIDE.md lines ~333–348 vs ledger notes.
- Contract slippage: Do Now often valid, but execution stalls; stall-autonomy not enforced (new hubs created before finishing runs on prior hubs). This encourages evidence-only loops.
- Missing minimal driver: no tiny, testable SSIM grid driver to publish results quickly; work instead routes through the dense orchestrator and heavy end-to-end runs.
- MVP: add a minimal `ssim_grid.py` under initiative bin + a smoke test; enforce preview-phase-only guard; run one real pipeline to closure with verifier evidence; publish one-page grid summary.

Flow Analysis (last ~20 summaries)
- Presence of pytest evidence under hubs:
  • 2025-11-09T210500Z: red+green pytest logs present (inventory failure/fix, full suite)
  • 2025-11-10T093500Z/113500Z/133500Z/153500Z/173500Z: red+green pytest logs present (CLI guard/logs, highlights checks)
  • 2025-11-10T193500Z, 2025-11-11T001033Z, 2025-11-11T003351Z, 2025-11-11T005802Z: no pytest evidence (empty {red,green,cli,collect})
- Rough classification (last 12 hubs): 8 Code+Tests, 4 Docs-only/prep. The most recent 4 hubs are Docs-only, aligning with the stall perception.
- Evidence vs outcome: recent turn summaries emphasize prep/validation planning over producing Phase D–G artifacts; several hubs lack CLI pipeline logs entirely.

Hypotheses (ranked)
1) Process guardrails allow stalls (High likelihood, High impact, Small effort, Medium risk)
   - Evidence: Multiple consecutive hubs with empty red/green/cli (no execution), while input.md lists valid Do Now steps including run + verify.
   - Disambiguation: Enforce stall-autonomy — if previous hub has empty execution dirs, the next loop must implement+run or switch focus; forbid creating a new hub in Docs mode.

2) Documentation drift on precision and preview artifacts (High likelihood, Medium impact, Small effort, Low risk)
   - Evidence: docs/TESTING_GUIDE.md still states 3-decimal formatting for all deltas and omits preview artifact details; ledger indicates MS-SSIM (3) vs MAE (6) and the preview file requirements.
   - Disambiguation: Update TESTING_GUIDE and run a single selector validating preview precision and absence of "amplitude" lines; confirm green.

3) Over-emphasis on CLI/log grooming (Medium-High likelihood, Medium impact, Small effort, Low risk)
   - Evidence: numerous pytest_cli_* logs and CLI-guard tests; fewer artifacts confirming full Phase D–G outputs per hub.
   - Disambiguation: Add a release-critical KPI: a hub is “counted” only if it contains metrics_delta_summary.json + highlights + preview + digest; track % counted over last 10 hubs.

4) Missing minimal, testable SSIM grid driver (Medium likelihood, High impact, Medium effort, Low risk)
   - Evidence: Work funnels through the dense orchestrator; no tiny script + smoke test to extract/publish grid deltas across doses/overlaps independent of a full run.
   - Disambiguation: Implement `bin/ssim_grid.py` to read existing JSON/CSV outputs and write a small digest; add `tests/study/test_ssim_grid.py::test_smoke_ssim_grid` using fixtures.

5) Measurement ambiguity on preview-phase-only rule (Medium likelihood, Medium impact, Small effort, Low risk)
   - Evidence: Validator previously accepted preview with amplitude content; hardening is planned but not executed.
   - Disambiguation: Add explicit RED test injecting amplitude text into preview; make it GREEN, then run pipeline once and verify.

Minimum Viable Path (2 loops)
- Loop 1 (Implementation nucleus + validation)
  • Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/ssim_grid.py::main — read `{hub}/analysis/metrics_delta_summary.json` per dose×overlap, emit `analysis/ssim_grid_summary.md` with MS-SSIM (3 decimals) and MAE (6 decimals); fail if preview.txt contains any "amplitude" lines.
  • Implement: tests/study/test_ssim_grid.py::test_smoke_ssim_grid — create a tmp hub with minimal metrics_delta_summary.json and preview.txt; assert parser outputs expected table and enforces preview-phase-only rule.
  • Selector: pytest tests/study/test_ssim_grid.py::test_smoke_ssim_grid -q
  • Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-11T213000Z/retrospective/

- Loop 2 (Single real run + publish)
  • Run: `bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber`
  • Verify: run `verify_dense_pipeline_artifacts.py` + highlights check; require JSON + highlights.txt + preview.txt + digest + ssim_grid_summary.md in hub.
  • Publish: one-page SSIM grid summary with acceptance thresholds recorded in summary.md and linked in fix_plan.

Process Guardrails
- Stall-autonomy rule: if the previous hub has empty {red, green, cli}, the next loop must be code+tests implementing the smallest viable change or switch focus. Creating a new hub in Docs mode is disallowed.
- Do Now contract strictness: reject Do Now lacking a single Implement nucleus + a validating pytest selector; forbid multiple new hubs within 24h without a counted run.
- Counted hub definition: Only hubs containing metrics_delta_summary.json, metrics_delta_highlights.txt, preview.txt, aggregate_highlights.txt, and metrics_digest.md count toward progress metrics.
- Evidence floor: at least one green pytest log per hub and a validator report JSON; otherwise mark the loop as evidence-only (and block creating a new hub).

Pointers (evidence)
- input.md:1
- docs/fix_plan.md:25
- docs/fix_plan.md:27
- docs/fix_plan.md:28
- docs/TESTING_GUIDE.md:333
- docs/TESTING_GUIDE.md:345
- tests/study/test_phase_g_dense_artifacts_verifier.py:1
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-10T153500Z/phase_g_dense_full_execution_real_run/green/pytest_full_suite.log
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-10T153500Z/phase_g_dense_full_execution_real_run/red/pytest_cli_phase_logs_pattern_fail.log
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-10T193500Z/phase_g_dense_full_execution_real_run/summary/summary.md
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-11T005802Z/summary/summary.md

Next Loop Do Now (ready for Ralph)
- Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/ssim_grid.py::main — minimal SSIM grid emitter over existing hub(s).
- Implement: tests/study/test_ssim_grid.py::test_smoke_ssim_grid — RED→GREEN for preview-phase-only and precision formatting (MS-SSIM 3, MAE 6).
- Validate: pytest tests/study/test_ssim_grid.py::test_smoke_ssim_grid -q
- Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-11T213000Z/retrospective/

