Summary: Enforce per-phase CLI log coverage via TDD, then execute the dense Phase C→G pipeline and verifier to capture MS-SSIM/MAE evidence under the 2025-11-10T133500Z hub.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_logs_missing -vv; pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_logs_complete -vv; pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_phase_logs_missing -vv; pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_phase_logs_complete -vv; pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-10T133500Z/phase_g_dense_full_execution_real_run/

Do Now (hard validity contract)
- Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py::validate_cli_logs — add RED fixtures for missing per-phase logs in `tests/study/test_phase_g_dense_artifacts_verifier.py`, extend the helper to require every expected `phase_*.log` (plus report helpers) and detect completion sentinels, then rerun the suite to GREEN capturing `$HUB/red/pytest_cli_phase_logs_fail.log` and `$HUB/green/pytest_cli_phase_logs_fix.log` alongside the existing orchestrator guards.
- Validate: pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_logs_missing -vv; pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_logs_complete -vv; pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_phase_logs_missing -vv; pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_phase_logs_complete -vv; pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv (tee outputs to `$HUB`/red or `$HUB`/green as mapped).
- Execute: Ensure no prior pipeline processes are running (`pgrep -af run_phase_g_dense.py || true`); export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md and HUB=plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-10T133500Z/phase_g_dense_full_execution_real_run; run `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$PWD/$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense.log`.
- Verify: After `[8/8]` appears, execute `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py --hub "$PWD/$HUB" --report "$HUB"/analysis/pipeline_verification.json |& tee "$HUB"/analysis/verifier_cli.log`; ensure `analysis/artifact_inventory.txt` and per-phase CLI checks pass.
- Document: Summarize MS-SSIM/MAE deltas, metadata compliance, and verifier status in `$HUB/summary/summary.md`; update docs/fix_plan.md Attempts History and add any durable lessons to docs/findings.md (linking HUB evidence).

How-To Map
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
2. export HUB=plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-10T133500Z/phase_g_dense_full_execution_real_run
3. mkdir -p "$HUB"/{analysis,cli,collect,green,red,summary}
4. pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_logs_missing -vv | tee "$HUB"/red/pytest_cli_logs_fail.log
5. pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_logs_complete -vv | tee "$HUB"/green/pytest_cli_logs_fix.log
6. pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_phase_logs_missing -vv | tee "$HUB"/red/pytest_cli_phase_logs_fail.log
7. pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_phase_logs_complete -vv | tee "$HUB"/green/pytest_cli_phase_logs_fix.log
8. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/green/pytest_orchestrator_dense_exec_cli_guard.log
9. pgrep -af run_phase_g_dense.py || true
10. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$PWD/$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense.log
11. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py --hub "$PWD/$HUB" --report "$HUB"/analysis/pipeline_verification.json |& tee "$HUB"/analysis/verifier_cli.log
12. pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv | tee "$HUB"/collect/pytest_collect_cli_logs.log

Pitfalls To Avoid
- Capture RED logs before implementing per-phase enforcement; do not overwrite them with GREEN.
- Keep CLI parsing device-neutral (plain text only) and forbid absolute/backslash paths per TYPE-PATH-001.
- Treat existing hubs as immutable; never edit `data/` artifacts manually outside orchestrator/verify commands.
- Ensure `validate_cli_logs` aggregates missing files and incomplete sentinels so the JSON report is actionable.
- Confirm no lingering `run_phase_g_dense.py` processes before launching the clobber run.
- Stop immediately if the pipeline aborts; log blocker context instead of relaunching silently.
- Maintain CONFIG-001 bridge by running provided CLI entry points (never import modules directly).

If Blocked
- If per-phase log fixtures cannot reproduce a RED failure, archive the attempt in `$HUB/summary/summary.md`, keep log artifacts, and mark the loop blocked in docs/fix_plan.md.
- If the pipeline halts mid-run, move the log to `$HUB/red/`, capture the error signature in summary.md, and pause for escalation.
- If the verifier flags missing artifacts or CLI logs, preserve the hub untouched, document the failure, and update docs/fix_plan.md with blocker details.

Findings Applied (Mandatory)
- POLICY-001 — Ensure torch>=2.2 remains available for verifier imports.
- CONFIG-001 — Run orchestrator/verify CLIs so the legacy bridge executes before dependent modules.
- DATA-001 — Reference verifier outputs to confirm amplitude/complex64 NPZ contract holds.
- TYPE-PATH-001 — Enforce POSIX-relative hub paths in logs and artifact inventory.
- OVERSAMPLING-001 — Dense overlap 0.7 must remain unchanged; cite if metrics regress.
- STUDY-001 — Capture MS-SSIM/MAE deltas vs Baseline/PtyChi in summary and ledger.
- PHASEC-METADATA-001 — Surface metadata compliance results in summary.md.

Pointers
- docs/fix_plan.md:4
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-10T133500Z/phase_g_dense_full_execution_real_run/plan/plan.md:1
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py:400
- tests/study/test_phase_g_dense_artifacts_verifier.py:1
- tests/study/test_phase_g_dense_orchestrator.py:977
- specs/data_contracts.md:1
- docs/findings.md:8

Next Up (optional)
- Draft sparse-view pipeline plan after dense evidence is GREEN and archived.

Doc Sync Plan (Conditional)
- After GREEN, run `pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv | tee "$HUB"/collect/pytest_collect_cli_logs.log`, then update docs/TESTING_GUIDE.md §2 and docs/development/TEST_SUITE_INDEX.md to list the new CLI log selectors.

Mapped Tests Guardrail
- Confirm all mapped selectors collect ≥1 test; author/fix missing nodes before implementation if collection fails.
