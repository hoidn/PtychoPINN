Summary: Harden the dense verifier/tests for the new ssim_grid helper, then run one counted dense pipeline to capture MS-SSIM/MAE + doc evidence in the new hub.
Mode: Perf
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_logs_require_ssim_grid_log -vv; pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_requires_ssim_grid_summary -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/

Do Now (hard validity contract)
- Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py::validate_cli_logs — require `cli/ssim_grid_cli.log`, bubble helper metadata into the JSON report, and wire a new `validate_ssim_grid_summary()` check so PREVIEW-PHASE-001 failures stop the run if `analysis/ssim_grid_summary.md` is missing or phase-only formatting regresses.
- Implement: tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_logs_require_ssim_grid_log & ::test_verify_dense_pipeline_requires_ssim_grid_summary — add RED fixtures that omit the helper log/summary, GREEN fixtures that include them, and capture pytest logs under `{red,green}/` in the new hub.
- Execute: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber (tee stdout to cli/run_phase_g_dense_stdout.log) so Phase C→G, `ssim_grid.py`, and all helper logs land in this hub.
- Validate: pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_logs_require_ssim_grid_log -vv && pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_requires_ssim_grid_summary -vv (store GREEN logs under $HUB/green/ and keep RED evidence if initial runs fail).
- Validate: python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py --hub "$HUB" --report "$HUB"/analysis/verification_report.json --dose 1000 --view dense && python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py --hub "$HUB"; rerun plans/.../bin/ssim_grid.py --hub "$HUB" if orchestrator exited early.
- Document: Update docs/TESTING_GUIDE.md (§Phase G Delta Metrics Persistence) and docs/development/TEST_SUITE_INDEX.md (Phase G row) to describe ssim_grid helper, preview-only guard, ±0.000/±0.000000 precision, and the new pytest selectors; summarize MS-SSIM/MAE deltas + verifier outcome + doc diffs in $HUB/summary/summary.md and log CLI paths per TYPE-PATH-001.
- Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/

How-To Map
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md; export HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier
2. Edit plans/active/.../bin/verify_dense_pipeline_artifacts.py — add `ssim_grid_cli.log` to helper log list, emit `found_helper_logs` metadata, and insert a new `validate_ssim_grid_summary()` that checks `analysis/ssim_grid_summary.md` (non-empty, preview metadata) before running CLI validation.
3. Grow tests/study/test_phase_g_dense_artifacts_verifier.py with RED fixtures that omit the helper log/summary and GREEN fixtures that include them (plus collect-only evidence); capture logs under $HUB/red/ and $HUB/green/.
4. pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_logs_require_ssim_grid_log -vv | tee "$HUB"/green/pytest_verifier_cli_log.log after it passes; stash any failing log under $HUB/red/ first.
5. pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_requires_ssim_grid_summary -vv | tee "$HUB"/green/pytest_verifier_summary.log (again keep RED evidence if needed).
6. pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -k ssim_grid -vv | tee "$HUB"/collect/pytest_collect_verifier.log (Mapped Tests Guardrail).
7. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log (expect ssim_grid summary/log paths in the success banner).
8. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py --hub "$HUB" --report "$HUB"/analysis/verification_report.json --dose 1000 --view dense | tee "$HUB"/analysis/verify_dense_stdout.log
9. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py --hub "$HUB" | tee "$HUB"/analysis/check_dense_highlights.log; rerun ssim_grid.py --hub "$HUB" --output "$HUB"/analysis/ssim_grid_summary.md if the orchestrator aborted before generating it.
10. Update docs/TESTING_GUIDE.md (§Phase G) and docs/development/TEST_SUITE_INDEX.md (Phase G row) with helper + selector details; save git diff excerpts to $HUB/summary/.
11. Summarize MS-SSIM/MAE deltas, preview guard outcome, verifier status, doc/test updates, and log locations in $HUB/summary/summary.md; refresh docs/fix_plan.md + galph_memory.md before exit.

Pitfalls To Avoid
- Forgetting AUTHORITATIVE_CMDS_DOC leads to CONFIG-001 regressions before Phase C even starts.
- Skipping RED logs leaves TEST-CLI-001 unmet; keep failing pytest output under $HUB/red/ before fixes.
- Hub-relative paths only (TYPE-PATH-001); don’t hardcode /tmp or absolute workspace paths in inventories, docs, or success banners.
- Do not rerun Phase C in a different hub mid-loop—use the new $HUB with --clobber so artifacts stay collocated.
- Dense run is long; if it fails, archive cli/*.log + analysis/blocker.log immediately instead of wiping evidence.
- Preview guard must stay phase-only; if ssim_grid.py detects “amplitude”, keep the offending preview file in analysis/ for triage.
- Doc updates must reflect the exact pytest selectors you actually ran; stale instructions block future approvals.
- Do not touch `ptycho/model.py` / other protected physics modules; all changes stay in plans/active scripts + docs/tests.

If Blocked
- If run_phase_g_dense.py fails, capture `$HUB`/analysis/blocker.log plus the command stdout, note failure reason in summary.md, and stop before attempting verifier/tests.
- If pytest selectors refuse to collect (e.g., typoed test name), record the collect-only log under $HUB/collect/, fix the selector, and rerun before touching the pipeline.
- If disk space/runtime makes the dense run infeasible, log evidence (df, elapsed time) in summary.md, update docs/fix_plan.md Attempts History as blocked, and await supervisor guidance.

Findings Applied (Mandatory)
- POLICY-001 — Keep PyTorch available for the Phase F baseline steps invoked by the dense run.
- CONFIG-001 — Export AUTHORITATIVE_CMDS_DOC before invoking any orchestration helper that touches legacy configs.
- DATA-001 — Use verify_dense_pipeline_artifacts.py and highlights checker to prove NPZ/JSON bundles match the canonical contract.
- TYPE-PATH-001 — Only reference hub-relative paths in inventories, success banners, docs, and summary.md.
- STUDY-001 — Report MS-SSIM/MAE deltas with ± precision and phase emphasis in ssim_grid_summary.md + summary.md.
- TEST-CLI-001 — Preserve CLI + pytest red/green logs under the hub for reproducibility.
- PREVIEW-PHASE-001 — Enforce phase-only preview content via the new verifier guard + ssim_grid helper evidence.

Pointers
- docs/fix_plan.md:17 — Active focus status, guardrails, and new attempt metadata for this initiative.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:27 — Phase roadmap + execution guardrails for dense runs.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:1011 — ssim_grid helper wiring inside the orchestrator.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py:632 — CLI log validator that needs the new helper + summary checks.
- tests/study/test_phase_g_dense_artifacts_verifier.py:328 — Existing CLI log RED test scaffold to extend for the helper log.
- docs/TESTING_GUIDE.md:331 — Phase G Delta Metrics Persistence section that must mention ssim_grid + precision rules.
- docs/development/TEST_SUITE_INDEX.md:62 — Phase G orchestrator test row to refresh with the new selectors.

Next Up (optional)
1. After the dense hub is green, rerun the sparse view through the same verifier + helper workflow for comparison parity.
2. Automate SSIM grid aggregation across multiple hubs once at least two runs exist (dense + sparse).

Doc Sync Plan (Conditional)
- Update docs/TESTING_GUIDE.md (§Phase G Delta Metrics Persistence) and docs/development/TEST_SUITE_INDEX.md (Phase G row) after the code/tests pass; stash diffs under $HUB/summary/ and keep selectors aligned with the new guard.

Mapped Tests Guardrail
- `pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -k ssim_grid -vv` must collect (>0) before executing the dense run; keep the collect log under $HUB/collect/.

Hard Gate
- Do not call this loop done until `$HUB` contains `analysis/ssim_grid_summary.md`, `analysis/metrics_delta_summary.json`, `analysis/metrics_delta_highlights{,_preview}.txt`, `analysis/verification_report.json`, `$HUB/cli/run_phase_g_dense_stdout.log`, `$HUB/cli/ssim_grid_cli.log`, and summary.md describing MS-SSIM/MAE deltas + verifier outcome with pointers to doc/test diffs.
