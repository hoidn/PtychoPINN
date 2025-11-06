Summary: Harden dense Phase G highlight validation, then run the dense pipeline and archive verifier evidence in the new 2025-11-10T193500Z hub.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_missing_preview -vv; pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_preview_mismatch -vv; pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_delta_mismatch -vv; pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_complete -vv; pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_logs_complete -vv; pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv; pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-10T193500Z/phase_g_dense_full_execution_real_run/

Do Now (hard validity contract)
- Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py::validate_metrics_delta_highlights — add RED fixtures (`tests/study/test_phase_g_dense_artifacts_verifier.py`) for missing preview, preview mismatch, and highlight delta mismatch (persist RED logs before code changes); enhance the helper to load `analysis/metrics_delta_summary.json`, enforce both highlights + preview files, and emit structured metadata (`checked_models`, `missing_models`, `missing_metrics`, `missing_preview_values`, `mismatched_highlight_values`) with formatted ± deltas (MS-SSIM → 3 decimals, MAE → 6); update GREEN fixtures accordingly and ensure `main()` surfaces the enriched result. Also align `plans/active/.../bin/check_dense_highlights_match.py` with the same schema/formatting so CLI validation matches pytest expectations.
- Validate: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md and HUB=plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-10T193500Z/phase_g_dense_full_execution_real_run; mkdir -p "$HUB"/{analysis,cli,collect,green,red,summary}; pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_missing_preview -vv | tee "$HUB"/red/pytest_highlights_missing_preview.log; pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_preview_mismatch -vv | tee "$HUB"/red/pytest_highlights_preview_mismatch.log; pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_delta_mismatch -vv | tee "$HUB"/red/pytest_highlights_delta_mismatch.log; after implementation, rerun the three selectors plus `pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_complete -vv | tee "$HUB"/green/pytest_highlights_complete.log` and `pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_logs_complete -vv | tee "$HUB"/green/pytest_cli_logs_complete.log`; keep regression `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/green/pytest_orchestrator_dense_exec.log` GREEN.
- Execute: pgrep -af run_phase_g_dense.py || true; python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$PWD/$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense.log (wait for `[8/8]` and SUCCESS banner).
- Verify: python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py --hub "$PWD/$HUB" --report "$HUB"/analysis/pipeline_verification.json |& tee "$HUB"/analysis/verifier_cli.log; python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py --hub "$PWD/$HUB" |& tee "$HUB"/analysis/highlights_check.log; pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv | tee "$HUB"/collect/pytest_collect_highlights.log.
- Document: Summarize MS-SSIM/MAE deltas, highlight/preview guard results, CLI validation status, and metadata compliance in `$HUB/summary/summary.md`; update docs/fix_plan.md (Attempts History), append durable lessons to docs/findings.md if new guard rules surface, and cross-link artifacts.

How-To Map
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
2. export HUB=plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-10T193500Z/phase_g_dense_full_execution_real_run
3. mkdir -p "$HUB"/{analysis,cli,collect,green,red,summary}
4. pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_missing_preview -vv | tee "$HUB"/red/pytest_highlights_missing_preview.log
5. pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_preview_mismatch -vv | tee "$HUB"/red/pytest_highlights_preview_mismatch.log
6. pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_delta_mismatch -vv | tee "$HUB"/red/pytest_highlights_delta_mismatch.log
7. <implement code per above>
8. pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_missing_preview -vv | tee "$HUB"/green/pytest_highlights_missing_preview.log
9. pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_preview_mismatch -vv | tee "$HUB"/green/pytest_highlights_preview_mismatch.log
10. pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_delta_mismatch -vv | tee "$HUB"/green/pytest_highlights_delta_mismatch.log
11. pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_complete -vv | tee "$HUB"/green/pytest_highlights_complete.log
12. pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_logs_complete -vv | tee "$HUB"/green/pytest_cli_logs_complete.log
13. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/green/pytest_orchestrator_dense_exec.log
14. pgrep -af run_phase_g_dense.py || true
15. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$PWD/$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense.log
16. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py --hub "$PWD/$HUB" --report "$HUB"/analysis/pipeline_verification.json |& tee "$HUB"/analysis/verifier_cli.log
17. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py --hub "$PWD/$HUB" |& tee "$HUB"/analysis/highlights_check.log
18. pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv | tee "$HUB"/collect/pytest_collect_highlights.log
19. Update docs/fix_plan.md Attempts History + docs/findings.md (if new lesson)
20. Write summary to `$HUB/summary/summary.md`

Pitfalls To Avoid
- Capture RED pytest logs before touching implementation; do not overwrite them during refactors.
- Keep fixture data realistic (dose/view strings, preview file paths) so validation logic exercises the real schema.
- Ensure the validator tolerates missing amplitude deltas (None) without crashing but still reports gaps correctly.
- Reuse helper formatting rules (`±0.000` vs `±0.000000`) consistently between validator and CLI checker.
- Prevent absolute paths in `artifact_inventory.txt`; TYPE-PATH-001 requires POSIX-relative entries.
- Wait for `[8/8]` success before recording dense pipeline evidence; abort and log failure if orchestrator exits early.
- Do not launch a second orchestrator while one is running; confirm `pgrep` is empty first.
- Avoid editing core TensorFlow/PyTorch modules (`ptycho/model.py`, etc.); keep changes in initiative tooling/tests.
- Retain CLI helper logs (`aggregate_report_cli.log`, `metrics_digest_cli.log`) to keep verifier happy.

If Blocked
- If RED fixtures refuse to fail, archive the attempted fixture files + pytest output under `$HUB/red/` and log the hypothesis in docs/fix_plan.md, then halt.
- If preview file is absent after pipeline run, copy the CLI log snippet into `$HUB/summary/summary.md`, mark the loop blocked, and capture verifier JSON showing the failure.
- If the highlights checker and verifier disagree, archive both outputs under `$HUB/analysis/`, note the mismatch in docs/fix_plan.md, and pause for follow-up.

Findings Applied (Mandatory)
- POLICY-001 — Torch≥2.2 required for verifier modules.
- CONFIG-001 — Running official CLIs maintains legacy bridge ordering.
- DATA-001 — Phase C NPZ guard stays active via orchestrator before trusting deltas.
- TYPE-PATH-001 — Artifact inventory + JSON provenance remain POSIX-relative.
- OVERSAMPLING-001 — Dense overlap fixed at 0.7; document deviations.
- STUDY-001 — Summaries must compare MS-SSIM/MAE vs Baseline & PtyChi.
- PHASEC-METADATA-001 — Surface metadata compliance status in summary and verifier output.
- TEST-CLI-001 — Maintain RED/GREEN fixtures for CLI + highlight validation.

Pointers
- docs/fix_plan.md:17
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-10T193500Z/phase_g_dense_full_execution_real_run/plan/plan.md:1
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py:309
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py:1
- tests/study/test_phase_g_dense_artifacts_verifier.py:1321
- tests/study/test_phase_g_dense_orchestrator.py:977
- docs/findings.md:21
- docs/TESTING_GUIDE.md:1

Next Up (optional)
- Once dense evidence lands, port the highlight guard to the sparse view hub (`run_phase_g_sparse.py`).

Doc Sync Plan (Conditional)
- After GREEN, run `pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv | tee "$HUB"/collect/pytest_collect_highlights.log` and update docs/TESTING_GUIDE.md §2 plus docs/development/TEST_SUITE_INDEX.md with the new highlight selectors once the pipeline evidence is recorded.
