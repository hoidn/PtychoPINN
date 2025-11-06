Summary: Tighten dense Phase G highlight guards via TDD, then execute the dense pipeline and archive verifier evidence under the new 2025-11-10T173500Z hub.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_missing_model -vv; pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_mismatched_value -vv; pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_complete -vv; pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_logs_complete -vv; pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv; pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-10T173500Z/phase_g_dense_full_execution_real_run/

Do Now (hard validity contract)
- Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py::validate_metrics_delta_highlights — add RED fixtures in `tests/study/test_phase_g_dense_artifacts_verifier.py` for missing-model and mismatched-value highlight scenarios (persist logs before code changes), then extend the helper to load `metrics_delta_summary.json`, enforce presence of preview text, and require properly formatted ± deltas for each model/metric pair; update the GREEN fixture to assert the new fields; capture RED logs under `$HUB/red/` before going GREEN.
- Validate: pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_missing_model -vv | tee "$HUB"/red/pytest_highlights_missing_model.log; pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_mismatched_value -vv | tee "$HUB"/red/pytest_highlights_mismatched_value.log; pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_complete -vv | tee "$HUB"/green/pytest_highlights_complete.log; pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_logs_complete -vv | tee "$HUB"/green/pytest_cli_logs_complete.log; pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/green/pytest_orchestrator_dense_exec.log.
- Execute: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md and HUB=plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-10T173500Z/phase_g_dense_full_execution_real_run; mkdir -p "$HUB"/{analysis,cli,collect,green,red,summary}; pgrep -af run_phase_g_dense.py || true; python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$PWD/$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense.log (wait for `[8/8]` + SUCCESS banner).
- Verify: python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py --hub "$PWD/$HUB" --report "$HUB"/analysis/pipeline_verification.json |& tee "$HUB"/analysis/verifier_cli.log; python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py --hub "$PWD/$HUB" |& tee "$HUB"/analysis/highlights_check.log; pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv | tee "$HUB"/collect/pytest_collect_cli_logs.log.
- Document: Summarize MS-SSIM/MAE deltas, metadata compliance, and highlight/preview guard results in `$HUB/summary/summary.md`; update docs/fix_plan.md with the 2025-11-10T153500Z+exec implementation attempt and this loop’s outcomes; add a docs/findings.md entry if highlight drift produces a reusable lesson.

How-To Map
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
2. export HUB=plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-10T173500Z/phase_g_dense_full_execution_real_run
3. mkdir -p "$HUB"/{analysis,cli,collect,green,red,summary}
4. pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_missing_model -vv | tee "$HUB"/red/pytest_highlights_missing_model.log
5. pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_mismatched_value -vv | tee "$HUB"/red/pytest_highlights_mismatched_value.log
6. pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_complete -vv | tee "$HUB"/green/pytest_highlights_complete.log
7. pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_logs_complete -vv | tee "$HUB"/green/pytest_cli_logs_complete.log
8. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/green/pytest_orchestrator_dense_exec.log
9. pgrep -af run_phase_g_dense.py || true
10. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$PWD/$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense.log
11. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py --hub "$PWD/$HUB" --report "$HUB"/analysis/pipeline_verification.json |& tee "$HUB"/analysis/verifier_cli.log
12. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py --hub "$PWD/$HUB" |& tee "$HUB"/analysis/highlights_check.log
13. pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv | tee "$HUB"/collect/pytest_collect_cli_logs.log

Pitfalls To Avoid
- Capture RED logs before touching the validator; do not overwrite them when iterating.
- Keep highlight fixtures realistic (include dose/view strings) so the updated validator exercises pattern checks.
- Treat the hub as append-only evidence—do not manually edit analyzer outputs.
- Wait for `[8/8]` + SUCCESS in the orchestrator log; abort and record failure if the pipeline stops early.
- Ensure helper logs (`aggregate_report_cli.log`, `metrics_digest_cli.log`) persist; missing helpers should cause the verifier to fail.
- Preserve POSIX-relative paths in artifact_inventory.txt to satisfy TYPE-PATH-001.
- Avoid parallel orchestrator runs; confirm `pgrep` clear before launching.
- If highlight mismatch surfaces real data drift, archive the raw JSON/TXT before fixing tests.

If Blocked
- If RED fixtures refuse to fail, stash the attempted test snippets in `$HUB/summary/summary.md`, keep the RED logs, and mark the loop blocked in docs/fix_plan.md with the hypothesis.
- If the dense pipeline terminates with an error, move failing CLI logs into `$HUB/red/`, record exit code + stack snippet in summary.md, and halt for follow-up.
- If verifier/highlights checks still report mismatches after implementation, archive their JSON/output under analysis/, note offending keys in docs/fix_plan.md, and pause for design review.

Findings Applied (Mandatory)
- POLICY-001 — Torch dependency required for verifier imports and orchestrator parity.
- CONFIG-001 — Running official CLIs keeps the legacy bridge ordering intact.
- DATA-001 — Use the verifier output to confirm Phase C NPZ contract remains compliant.
- TYPE-PATH-001 — Artifact inventory must stay POSIX-relative.
- OVERSAMPLING-001 — Dense overlap (0.7) invariant; document deviations in summary.
- STUDY-001 — Report MS-SSIM/MAE deltas vs Baseline & PtyChi.
- PHASEC-METADATA-001 — Surface metadata compliance status from metrics_summary.json.
- TEST-CLI-001 — Maintain explicit RED/GREEN fixtures for CLI/highlight validation steps.

Pointers
- docs/fix_plan.md:17
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-10T173500Z/phase_g_dense_full_execution_real_run/plan/plan.md:1
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py:309
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py:1
- tests/study/test_phase_g_dense_artifacts_verifier.py:600
- tests/study/test_phase_g_dense_orchestrator.py:977
- docs/findings.md:2
- docs/TESTING_GUIDE.md:1

Next Up (optional)
- Once dense evidence lands, adapt the highlight guard and pipeline execution for the sparse view hub.

Doc Sync Plan (Conditional)
- After GREEN, run `pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv | tee "$HUB"/collect/pytest_collect_cli_logs.log` and update docs/TESTING_GUIDE.md §2 plus docs/development/TEST_SUITE_INDEX.md with the new highlight guard selectors.
