Summary: Extend the dense highlights checker to enforce the new ssim_grid summary contract, then run the full Phase C→G pipeline to capture real MS-SSIM/MAE evidence.
Mode: Perf
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_check_dense_highlights_match.py::test_summary_mismatch_fails -vv; pytest tests/study/test_check_dense_highlights_match.py::test_summary_matches_json -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/

Do Now (hard validity contract)
- Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py::main — parse analysis/ssim_grid_summary.md, enforce preview metadata (`phase-only: ✓`), and assert the table’s MS-SSIM ±0.000 / MAE ±0.000000 values match metrics_delta_summary.json plus the highlights/preview text (PREVIEW-PHASE-001, STUDY-001).
- Implement: tests/study/test_check_dense_highlights_match.py::{test_summary_mismatch_fails,test_summary_matches_json} — add RED/GREEN fixtures for the upgraded checker, tee RED output to $HUB/red/pytest_highlights_checker.log and GREEN output to $HUB/green/pytest_highlights_checker.log, and keep the collect-only log under $HUB/collect/.
- Execute: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log to produce the counted Phase C→G bundle with ssim_grid artifacts.
- Validate: pytest tests/study/test_check_dense_highlights_match.py::test_summary_mismatch_fails -vv && pytest tests/study/test_check_dense_highlights_match.py::test_summary_matches_json -vv (collect log first via pytest --collect-only tests/study/test_check_dense_highlights_match.py -vv | tee "$HUB"/collect/pytest_collect_highlights.log).
- Validate: python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py --hub "$HUB" --report "$HUB"/analysis/verification_report.json --dose 1000 --view dense | tee "$HUB"/analysis/verify_dense_stdout.log && python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py --hub "$HUB" | tee "$HUB"/analysis/check_dense_highlights.log (rerun ssim_grid.py --hub "$HUB" if summary drift is detected).
- Document: Update docs/development/TEST_SUITE_INDEX.md (Phase G row) with the new highlights checker selectors, refresh `$HUB/summary/summary.md` with MS-SSIM/MAE deltas + preview verdict + pytest/log pointers, and log outcomes + evidence paths back to docs/fix_plan.md and galph_memory.md before exit.
- Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/ (analysis/verification_report.json, analysis/check_dense_highlights.log, analysis/ssim_grid_summary.md, cli/run_phase_g_dense_stdout.log, collect/red/green pytest logs, summary/summary.md).

How-To Map
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md; export HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier
2. Update plans/active/.../bin/check_dense_highlights_match.py — read analysis/ssim_grid_summary.md, parse the Markdown table (MS-SSIM ±0.000, MAE ±0.000000), confirm preview metadata (`preview source: ... (phase-only: ✓)`), and compare values against metrics_delta_summary.json + metrics_delta_highlights{,_preview}.txt; fail with actionable errors when missing/mismatched, and print summary results with hub-relative paths.
3. Create tests/study/test_check_dense_highlights_match.py with fixtures that synthesize a minimal hub (analysis with metrics_delta_summary.json/highlights/preview/ssim_grid_summary.md). RED test tampers with the summary table/value; GREEN test keeps everything aligned. Use tmp_path for hub scaffolding per TEST-CLI-001.
4. pytest --collect-only tests/study/test_check_dense_highlights_match.py -vv | tee "$HUB"/collect/pytest_collect_highlights.log (ensure selector collects >0 tests).
5. pytest tests/study/test_check_dense_highlights_match.py -vv |& tee /tmp/pytest_highlights.log; copy the first failing run to "$HUB"/red/pytest_highlights_checker.log (if failure occurs) and the passing run to "$HUB"/green/pytest_highlights_checker.log.
6. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log (expect SUCCESS banner + SSIM grid paths; stop and archive blocker logs if any phase fails).
7. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py --hub "$HUB" --report "$HUB"/analysis/verification_report.json --dose 1000 --view dense | tee "$HUB"/analysis/verify_dense_stdout.log.
8. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py --hub "$HUB" | tee "$HUB"/analysis/check_dense_highlights.log; if it fails because ssim_grid_summary.md is missing/out-of-sync, rerun python plans/active/.../bin/ssim_grid.py --hub "$HUB" and rerun the checker.
9. Update docs/development/TEST_SUITE_INDEX.md Phase G row with the new highlights checker selectors + evidence path; refresh `$HUB`/summary/summary.md with MS-SSIM/MAE deltas, preview verdict, checker outcome, CLI log locations, and doc/test references; append attempts to docs/fix_plan.md + galph_memory.md.

Pitfalls To Avoid
- Do not mint a new reports hub; reuse the 2025-11-12T010500Z directory until a counted dense run is archived (INITIATIVE_WORKFLOW_GUIDE).
- Missing AUTHORITATIVE_CMDS_DOC export will resurrect CONFIG-001 regressions before the pipeline even launches.
- The summary parser must handle both MS-SSIM (±0.000) and MAE (±0.000000); truncating precision violates STUDY-001 and will break tests.
- Keep all inventory/log references hub-relative (TYPE-PATH-001); absolute paths in success banners or docs are blockers.
- Always capture RED pytest output before fixing the tests (TEST-CLI-001); copy logs into $HUB/red/ even if the failure is trivial.
- Do not touch protected physics files (ptycho/model.py, ptycho/diffsim.py, ptycho/tf_helper.py); keep work inside plans/active scripts/tests/docs only.
- If run_phase_g_dense.py stops mid-phase, archive cli/*.log + analysis/blocker.log immediately and stop instead of re-running over partial outputs.
- SSIM grid helper returns non-zero when preview text still mentions "amplitude"; keep the offending preview file intact for triage if that occurs.

If Blocked
- If pytest selectors refuse to collect, save the collect-only output under $HUB/collect/pytest_collect_highlights.log, fix the import/name, and rerun before touching the pipeline; log the issue in summary.md + docs/fix_plan.md.
- If the dense run fails (timeout, OOM, missing dependency), archive cli/*.log plus analysis/blocker.log, summarize the failure + elapsed runtime in summary.md, mark the Attempt as blocked in docs/fix_plan.md, and await supervisor guidance.
- If disk space prevents writing analysis artifacts, capture `df -h` output in summary.md, stop further execution, and document the constraint in docs/fix_plan.md + galph_memory.md.

Findings Applied (Mandatory)
- POLICY-001 — Dense pipeline depends on PyTorch (Phase F baselines); confirm torch>=2.2 available before launch.
- CONFIG-001 — Always export AUTHORITATIVE_CMDS_DOC so legacy params.cfg bridges run before Phase C helpers.
- DATA-001 — `verify_dense_pipeline_artifacts.py` + the upgraded highlights checker enforce NPZ/JSON/delta contracts and must pass before declaring success.
- TYPE-PATH-001 — Inventory, success banners, docs, and summary entries must reference hub-relative POSIX paths only.
- STUDY-001 — Report MS-SSIM/MAE deltas with ±0.000/±0.000000 precision across JSON/highlights/preview/ssim_grid summary.
- TEST-CLI-001 — Preserve RED/GREEN pytest logs and orchestrator CLI logs within the hub for reproducibility.
- PREVIEW-PHASE-001 — Ensure the preview and SSIM grid summary stay phase-only; fail fast if "amplitude" resurfaces.
- PHASEC-METADATA-001 — Keep verify_dense pipeline metadata checks enabled so the dense run can trust Phase C inputs.

Pointers
- docs/fix_plan.md:4 — Active focus metadata + guardrails for this initiative.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/plan/plan.md:1 — Current objectives + execution sketch for the dense run.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py:1 — Highlights checker to extend with SSIM grid parsing.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:980 — CLI pipeline wiring that already invokes ssim_grid.py.
- docs/TESTING_GUIDE.md:331 — Phase G delta persistence requirements (JSON/highlights/preview + precision rules).
- docs/development/TEST_SUITE_INDEX.md:62 — Phase G test registry row to update with the new highlights checker selectors.

Next Up (optional)
1. After the dense view is green, rerun the sparse view through the same helper/verifier chain for comparison parity.
2. Automate SSIM grid summary aggregation across multiple hubs once at least two runs (dense + sparse) are archived.

Doc Sync Plan (Conditional)
- After the new tests pass, append the highlights checker selectors + evidence hub to docs/development/TEST_SUITE_INDEX.md (Phase G). If user-facing instructions change, add a short note under docs/TESTING_GUIDE.md §Phase G Delta Metrics Persistence referencing the SSIM grid table cross-check.

Mapped Tests Guardrail
- `pytest --collect-only tests/study/test_check_dense_highlights_match.py -vv` must collect (>0). Store the log under $HUB/collect/pytest_collect_highlights.log before running the dense pipeline command.

Hard Gate
- Do not call this loop done until `$HUB` contains populated `analysis/` (metrics_summary.json, metrics_delta_summary.json, metrics_delta_highlights{,_preview}.txt, ssim_grid_summary.md, verification_report.json, check_dense_highlights.log), `cli/` (run_phase_g_dense_stdout.log, run_phase_g_dense.log, phase_* logs, aggregate_report_cli.log, metrics_digest_cli.log, ssim_grid_cli.log), `$HUB/collect|red|green/` pytest evidence for the new checker, and `$HUB/summary/summary.md` describing MS-SSIM/MAE deltas, preview verdict, verifier/checker outcomes, and doc/test updates.
