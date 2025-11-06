Summary: Add CLI log validation via TDD and finish the dense Phase G pipeline run with full verifier + documentation under the 2025-11-10T113500Z hub.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_logs_missing -vv; pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_logs_complete -vv; pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-10T113500Z/phase_g_dense_full_execution_real_run/

Do Now (hard validity contract)
- Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py::validate_cli_logs — author RED tests in `tests/study/test_phase_g_dense_artifacts_verifier.py` covering missing/complete CLI log bundles, then add the helper (POSIX-relative enforcement, `[1/8]`…`[8/8]` banners, `SUCCESS: All phases completed`) and integrate it into `main()` before rerunning the tests to GREEN (capture RED→GREEN logs under `$HUB/red|green`).
- Validate: pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_logs_missing -vv; pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_logs_complete -vv; pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv (tee outputs to `$HUB/red/pytest_cli_logs_fail.log`, `$HUB/green/pytest_cli_logs_fix.log`, `$HUB/green/pytest_orchestrator_dense_exec_cli_guard.log`).
- Execute: After confirming the prior 093500Z pipeline has finished, export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md and HUB=plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-10T113500Z/phase_g_dense_full_execution_real_run, then run `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$PWD/$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense.log` to generate fresh Phase C→G artifacts in this hub.
- Verify: `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py --hub "$PWD/$HUB" --report "$HUB"/analysis/pipeline_verification.json |& tee "$HUB"/analysis/verifier_cli.log` once `[8/8]` is logged; ensure artifact_inventory.txt and new CLI checks pass.
- Document: Summarize MS-SSIM/MAE deltas, metadata compliance, and verifier status in `$HUB/summary/summary.md`; update docs/fix_plan.md Attempts History and add any new durable lessons to docs/findings.md.

How-To Map
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
2. export HUB=plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-10T113500Z/phase_g_dense_full_execution_real_run
3. mkdir -p "$HUB"/{analysis,cli,collect,green,red,summary}
4. pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_logs_missing -vv | tee "$HUB"/red/pytest_cli_logs_fail.log
5. pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_logs_complete -vv | tee "$HUB"/green/pytest_cli_logs_fix.log
6. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/green/pytest_orchestrator_dense_exec_cli_guard.log
7. pgrep -af run_phase_g_dense.py || true  # confirm previous pipeline finished before relaunch
8. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$PWD/$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense.log
9. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py --hub "$PWD/$HUB" --report "$HUB"/analysis/pipeline_verification.json |& tee "$HUB"/analysis/verifier_cli.log
10. pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv | tee "$HUB"/collect/pytest_collect_cli_logs.log

Pitfalls To Avoid
- Do not skip RED logs for the new CLI validation tests; capture them before implementing the helper.
- Keep CLI log parsing tolerant to ANSI-free text only; avoid GPU-specific strings so checks remain device-neutral.
- Treat hub paths as read-only outside run/verify commands; never edit `data/` artifacts manually.
- Ensure `validate_cli_logs` enforces POSIX-relative paths and reports missing sentinel lines with actionable context.
- Wait for current long-running pipeline to finish before starting the new hub run to prevent GPU contention.
- Do not delete or truncate the 093500Z hub; reference it for troubleshooting but keep evidence immutable.
- Record any verifier failures immediately in summary.md and halt instead of rerunning silently.

If Blocked
- If CLI log fixtures cannot reproduce a RED failure, document the attempt in `$HUB/summary/summary.md`, keep the failing log, and mark the loop blocked in docs/fix_plan.md.
- If the pipeline aborts mid-run, move the failing CLI log to `$HUB/red/`, capture the error signature in summary.md, and stop for escalation.
- If verifier reports missing artifacts, preserve the hub untouched, log the failure in summary.md, and update docs/fix_plan.md with blocker details.

Findings Applied (Mandatory)
- POLICY-001 — PyTorch dependency remains installed; tests rely on torch>=2.2.
- CONFIG-001 — Run/verify commands preserve the legacy bridge call chain; mention compliance in summary.
- DATA-001 — Phase C NPZ contract validated via verifier; confirm amplitude/complex64 adherence.
- TYPE-PATH-001 — New CLI validation must normalize paths and forbid absolute/backslash entries.
- OVERSAMPLING-001 — Dense overlap parameters (0.7) unchanged; confirm in metrics log notes.
- STUDY-001 — Capture MS-SSIM/MAE deltas vs Baseline/PtyChi in summary + ledger.
- PHASEC-METADATA-001 — Ensure metadata compliance block stays GREEN and reference results in documentation.

Pointers
- docs/fix_plan.md:4
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-10T113500Z/phase_g_dense_full_execution_real_run/plan/plan.md:1
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py:309
- tests/study/test_phase_g_dense_artifacts_verifier.py:1
- tests/study/test_phase_g_dense_orchestrator.py:977
- specs/data_contracts.md:1
- docs/findings.md:8

Next Up (optional)
- Draft sparse-view pipeline plan once dense hub artifacts are verified and archived.

Doc Sync Plan (Conditional)
- After GREEN, run `pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv | tee "$HUB"/collect/pytest_collect_cli_logs.log` and update docs/TESTING_GUIDE.md §2 plus docs/development/TEST_SUITE_INDEX.md with the new selectors referencing CLI log validation.

Mapped Tests Guardrail
- Confirm the three mapped selectors collect >0 tests; if collection fails, author/fix the missing test before proceeding with implementation.
