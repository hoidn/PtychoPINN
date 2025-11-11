Summary: Lock the highlight metadata guard via TDD, then run the dense Phase G pipeline into the new 2025-11-11T001033Z hub and archive verifier evidence.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_missing_preview -vv; pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_preview_mismatch -vv; pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_delta_mismatch -vv; pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_complete -vv; pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_logs_complete -vv; pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv; pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-11T001033Z/phase_g_dense_full_execution_real_run/

Do Now (hard validity contract)
- Implement: tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_missing_preview — extend this and the related preview/delta mismatch cases to assert the structured metadata returned by `validate_metrics_delta_highlights` (checked_models ordering, formatted `missing_preview_values`, `mismatched_highlight_values`, empty lists when GREEN). Update the GREEN fixture to verify those lists are empty and `line_count==4`. Then patch plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py::validate_metrics_delta_highlights so it pre-populates `checked_models`, `missing_models`, `missing_metrics`, `missing_preview_values`, and `mismatched_highlight_values` before any early return, and document the ±0.000 / ±0.000000 precision rules inline.
- Validate: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md and HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-11T001033Z/phase_g_dense_full_execution_real_run; mkdir -p "$HUB"/{analysis,cli,collect,green,red,summary}; run the three RED selectors in order (missing preview, preview mismatch, delta mismatch) piping logs to "$HUB"/red/*.log; after code changes rerun those selectors plus `pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_complete -vv` and `pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_logs_complete -vv`, teeing to "$HUB"/green/. Keep `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv` GREEN.
- Execute: pgrep -af run_phase_g_dense.py || true; python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense.log (require `[1/8]`→`[8/8]` and SUCCESS banner).
- Verify: python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py --hub "$HUB" --report "$HUB"/analysis/pipeline_verification.json |& tee "$HUB"/analysis/verifier_cli.log; python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py --hub "$HUB" |& tee "$HUB"/analysis/highlights_check.log; pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv | tee "$HUB"/collect/pytest_collect_highlights.log.
- Document: Summarize MS-SSIM/MAE deltas, highlight guard signals, CLI validation findings, and artifact_inventory counts in "$HUB"/summary/summary.md; update docs/fix_plan.md Attempts History plus durable lessons in docs/findings.md (if new guard behaviors emerge); capture Turn Summary + log links in the hub summary.

How-To Map
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
2. export HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-11T001033Z/phase_g_dense_full_execution_real_run
3. mkdir -p "$HUB"/{analysis,cli,collect,green,red,summary}
4. pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_missing_preview -vv | tee "$HUB"/red/pytest_highlights_missing_preview.log || true
5. pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_preview_mismatch -vv | tee "$HUB"/red/pytest_highlights_preview_mismatch.log || true
6. pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_delta_mismatch -vv | tee "$HUB"/red/pytest_highlights_delta_mismatch.log || true
7. Apply the test + verifier code edits described above
8. pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_missing_preview -vv | tee "$HUB"/green/pytest_highlights_missing_preview.log
9. pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_preview_mismatch -vv | tee "$HUB"/green/pytest_highlights_preview_mismatch.log
10. pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_delta_mismatch -vv | tee "$HUB"/green/pytest_highlights_delta_mismatch.log
11. pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_complete -vv | tee "$HUB"/green/pytest_highlights_complete.log
12. pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_cli_logs_complete -vv | tee "$HUB"/green/pytest_cli_logs_complete.log
13. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/green/pytest_orchestrator_dense_exec.log
14. pgrep -af run_phase_g_dense.py || true (ensure no stale process)
15. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense.log
16. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py --hub "$HUB" --report "$HUB"/analysis/pipeline_verification.json |& tee "$HUB"/analysis/verifier_cli.log
17. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py --hub "$HUB" |& tee "$HUB"/analysis/highlights_check.log
18. pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv | tee "$HUB"/collect/pytest_collect_highlights.log
19. Document deltas/highlights/CLI/metadata in "$HUB"/summary/summary.md and sync docs/fix_plan.md + docs/findings.md

Pitfalls To Avoid
- Skipping `AUTHORITATIVE_CMDS_DOC` export breaks CONFIG-001 sequencing; keep it in the shell before running CLIs.
- Do not delete old hubs; this loop relies on deterministic evidence under 2025-11-11T001033Z (TYPE-PATH-001 compliance).
- Keep pytest RED logs before editing code; overwriting them violates the TDD record.
- Ensure highlight formatting retains ± signs and precision (MS-SSIM 3 decimals, MAE 6); str() or f"{value}" will drop leading zeros.
- When editing `validate_metrics_delta_highlights`, retain existing preview/JSON validation order so CLI errors stay actionable.
- The orchestrator run must reach `[8/8]`; abort and log blocker if any subcommand fails (no partial data reuse).
- Do not install or upgrade packages—environment changes are outside this focus (Environment Freeze rule).
- Keep artifact_inventory paths POSIX-relative; avoid writing absolute paths when touching hub files.

If Blocked
- If pytest selectors fail due to missing metadata fields even after edits, capture the exact traceback/log under `$HUB/analysis/` and update docs/fix_plan.md Attempts History with the failure signature.
- If `run_phase_g_dense.py` halts mid-phase, stop immediately, archive blocker log, and summarize the failing phase plus command in `$HUB/summary/summary.md`; rerun only after the supervisor provides new guidance.

Findings Applied (Mandatory)
- POLICY-001 — PyTorch dependency already satisfied; no optional-skip excuses if torch imports surface.
- CONFIG-001 — AUTHORITATIVE_CMDS_DOC export + orchestrator CLI maintain legacy bridge ordering; never bypass.
- DATA-001 — Treat Phase C NPZ layout as canonical; rely on MetadataManager guards instead of manual shape edits.
- TYPE-PATH-001 — Hub inventory + CLI validator expect POSIX-relative paths; keep Path objects throughout scripts/tests.
- STUDY-001 — Report MS-SSIM/MAE deltas vs Baseline/PtyChi with correct signs in summary.
- PHASEC-METADATA-001 — Preserve metadata compliance table from `summarize_phase_g_outputs()` when documenting run results.
- TEST-CLI-001 — Maintain explicit RED/GREEN fixtures for CLI/log validation so filename regressions fail fast.

Pointers
- docs/DEVELOPER_GUIDE.md:603 — TDD methodology and Red→Green guardrails.
- docs/findings.md:16 — STUDY-001 context for reporting MS-SSIM/MAE deltas.
- docs/findings.md:22 — PHASEC-METADATA-001 requirements for Phase C metadata checks.
- docs/findings.md:23 — TEST-CLI-001 per-phase CLI filename and sentinel expectations.
- docs/development/TEST_SUITE_INDEX.md:62 — Phase G orchestrator test coverage and selectors.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py:309 — highlight validator to modify in this loop.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:1 — orchestrator CLI invoked during Execute/Verify.

Next Up (optional)
1. If time remains, extend `verify_dense_pipeline_artifacts.py` with amplitude delta checks so highlights must mention both amplitude/phase when future spec requires it.
2. Begin sparse-view parity run once dense evidence is archived and highlight guard remains green.

Doc Sync Plan (Conditional)
- Not needed beyond the collect-only artifact already captured in this Do Now (tests unchanged name-wise).

Mapped Tests Guardrail
- `pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv` must succeed after code edits; treat any collection failure as a hard stop before calling the loop complete.
