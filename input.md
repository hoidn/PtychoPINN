Summary: Patch the CLI verifier to match real log filename patterns and completion sentinels, then execute the dense Phase C→G pipeline and archive the verification evidence under the 2025-11-10T153500Z hub.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_phase_logs_wrong_pattern -vv; pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_phase_logs_incomplete -vv; pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_phase_logs_complete -vv; pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_logs_complete -vv; pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-10T153500Z/phase_g_dense_full_execution_real_run/

Do Now (hard validity contract)
- Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py::validate_cli_logs — add RED fixtures for filename pattern and sentinel enforcement in `tests/study/test_phase_g_dense_artifacts_verifier.py`, then update the helper to match dose/view-specific filenames (e.g., `phase_e_baseline_gs1_dose1000.log`, `phase_e_dense_gs2_dose1000.log`, `phase_f_dense_train.log`, etc.), require aggregate helper logs, and flag incomplete per-phase logs lacking completion sentinels; capture `$HUB/red/pytest_cli_phase_logs_pattern_fail.log` and `$HUB/red/pytest_cli_phase_logs_incomplete.log` before turning GREEN.
- Validate: pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_phase_logs_wrong_pattern -vv; pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_phase_logs_incomplete -vv; pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_phase_logs_complete -vv; pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_logs_complete -vv; pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv (tee outputs into `$HUB`/red or `$HUB`/green as mapped).
- Execute: Ensure no prior pipeline processes are active (`pgrep -af run_phase_g_dense.py || true`); export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md and HUB=plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-10T153500Z/phase_g_dense_full_execution_real_run; run `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$PWD/$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense.log`.
- Verify: After `[8/8]` plus `"SUCCESS: All phases completed"` emit, execute `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py --hub "$PWD/$HUB" --report "$HUB"/analysis/pipeline_verification.json |& tee "$HUB"/analysis/verifier_cli.log`; confirm `missing_phase_logs` and `incomplete_phase_logs` are empty and that `artifact_inventory.txt` is regenerated.
- Document: Summarize MS-SSIM/MAE deltas, metadata compliance, and the resolved filename-pattern gap in `$HUB/summary/summary.md`; update docs/fix_plan.md Attempts History and add a durable lesson to docs/findings.md if the pattern mismatch warrants one.

How-To Map
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
2. export HUB=plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-10T153500Z/phase_g_dense_full_execution_real_run
3. mkdir -p "$HUB"/{analysis,cli,collect,green,red,summary}
4. pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_phase_logs_wrong_pattern -vv | tee "$HUB"/red/pytest_cli_phase_logs_pattern_fail.log
5. pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_phase_logs_incomplete -vv | tee "$HUB"/red/pytest_cli_phase_logs_incomplete.log
6. pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_phase_logs_complete -vv | tee "$HUB"/green/pytest_cli_phase_logs_fix.log
7. pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_logs_complete -vv | tee "$HUB"/green/pytest_cli_logs_fix.log
8. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/green/pytest_orchestrator_dense_exec_cli_guard.log
9. pgrep -af run_phase_g_dense.py || true
10. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$PWD/$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense.log
11. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py --hub "$PWD/$HUB" --report "$HUB"/analysis/pipeline_verification.json |& tee "$HUB"/analysis/verifier_cli.log
12. pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv | tee "$HUB"/collect/pytest_collect_cli_logs.log

Pitfalls To Avoid
- Record RED evidence before modifying `validate_cli_logs`; never overwrite the failure logs when rerunning.
- Match the real filenames from `run_phase_g_dense.py` (`phase_e_baseline_gs1_dose{dose}.log`, etc.); hard-coding generic names will fail on execution.
- Treat the hub as append-only evidence—do not hand-edit `data/` outputs or analyzer files.
- Keep CLI parsing ASCII-only and enforce POSIX relative paths to satisfy TYPE-PATH-001.
- Abort immediately and document if the pipeline terminates early; log the exit code and error snippet in summary.md.
- Verify `aggregate_report_cli.log` and `metrics_digest_cli.log` are produced; missing helpers should cause the verifier to fail.
- Ensure CONFIG-001 bridge runs by invoking the CLI binaries rather than importing internals.
- Clean up hung orchestrator processes before starting the new run to avoid shared-resource conflicts.

If Blocked
- If pattern-aware tests cannot be made RED, capture the attempted fixture under `$HUB/summary/summary.md`, store logs, and mark the loop blocked in docs/fix_plan.md with rationale.
- If the dense pipeline fails mid-run, move the offending CLI log under `$HUB/red/`, summarize the failure mode (command, exit code, stack trace) in summary.md, and halt for follow-up.
- If the verifier still reports missing or incomplete logs after implementation, archive the JSON + CLI output, cross-reference the failing patterns in docs/fix_plan.md, and pause for design review.

Findings Applied (Mandatory)
- POLICY-001 — Torch dependency required so verifier/test imports succeed.
- CONFIG-001 — Running orchestrator/verify CLIs preserves the legacy bridge ordering.
- DATA-001 — Use verifier outputs to confirm amplitude/complex64 dataset compliance.
- TYPE-PATH-001 — Keep hub paths POSIX relative in inventories and error messages.
- OVERSAMPLING-001 — Dense overlap coefficient (0.7) should remain constant; note deviations.
- STUDY-001 — Summarize MS-SSIM/MAE deltas vs Baseline & PtyChi.
- PHASEC-METADATA-001 — Ensure metadata compliance details surface in summary/docs.
- TEST-CLI-001 — Maintain explicit RED/GREEN fixtures validating CLI instrumentation.

Pointers
- docs/fix_plan.md:4
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-10T153500Z/phase_g_dense_full_execution_real_run/plan/plan.md:1
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:736
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py:440
- tests/study/test_phase_g_dense_artifacts_verifier.py:580
- tests/study/test_phase_g_dense_orchestrator.py:977
- docs/findings.md:21
- specs/ptychodus_api_spec.md:1

Next Up (optional)
- After dense evidence lands, scope sparse-view pipeline verification using the same guard patterns.

Doc Sync Plan (Conditional)
- Once new selectors are GREEN, run `pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv | tee "$HUB"/collect/pytest_collect_cli_logs.log`, then refresh docs/TESTING_GUIDE.md §2 and docs/development/TEST_SUITE_INDEX.md with the additional CLI log checks.

Mapped Tests Guardrail
- Ensure each mapped selector collects ≥1 test; if `--collect-only` shows 0, add/repair tests before concluding the loop.
