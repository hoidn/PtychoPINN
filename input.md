Summary: Add the missing delta-preview helper (correct precision + new pytest) and rerun the dense Phase G pipeline to capture verifier evidence under the 2025-11-11T003351Z hub.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_phase_g_dense_orchestrator.py::test_persist_delta_highlights_creates_preview -vv; pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv; pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_complete -vv; pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-11T003351Z/phase_g_dense_full_execution_real_run/

Do Now (hard validity contract)
- Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::persist_delta_highlights — factor the inline delta formatter (lines 985-1058) into a helper that formats MS-SSIM with ±0.000 and MAE with ±0.000000, writes both `metrics_delta_highlights.txt` and the missing `metrics_delta_highlights_preview.txt`, and returns the structured `delta_summary` dict so `main()` can serialize JSON + stdout banners from the same data.
- Implement: tests/study/test_phase_g_dense_orchestrator.py::test_persist_delta_highlights_creates_preview — add a RED→GREEN test that feeds synthetic aggregate metrics (PtychoPINN/Baseline/PtyChi) through the new helper, asserts the highlight lines contain signed values with correct precision, verifies the preview file exists with four phase-only lines, and checks the returned numeric deltas (e.g., Baseline.mae.phase == -0.000025).
- Validate: run `pytest tests/study/test_phase_g_dense_orchestrator.py::test_persist_delta_highlights_creates_preview -vv`, `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv`, `pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_complete -vv`, and `pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv`, archiving logs under `$HUB`/{green,collect}.
- Execute: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md and HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-11T003351Z/phase_g_dense_full_execution_real_run; mkdir -p "$HUB"/{analysis,cli,collect,green,red,summary}; python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense.log (require `[1/8]`→`[8/8]` and SUCCESS banner).
- Verify: python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py --hub "$HUB" --report "$HUB"/analysis/pipeline_verification.json |& tee "$HUB"/analysis/verifier_cli.log; python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py --hub "$HUB" |& tee "$HUB"/analysis/highlights_check.log; pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv | tee "$HUB"/collect/pytest_collect_highlights.log.
- Document: summarize MS-SSIM/MAE deltas (phase emphasis), highlight preview/validator status, CLI log coverage, and artifact_inventory counts in "$HUB"/summary/summary.md; update docs/fix_plan.md Attempts History + galph Turn Summary with artifact links and note any new durable lessons.

How-To Map
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
2. export HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-11T003351Z/phase_g_dense_full_execution_real_run
3. mkdir -p "$HUB"/{analysis,cli,collect,green,red,summary}
4. pytest tests/study/test_phase_g_dense_orchestrator.py::test_persist_delta_highlights_creates_preview -vv | tee "$HUB"/red/pytest_delta_preview_helper_red.log || true
5. Implement the helper + test changes (keep helper pure so tests can import it).
6. pytest tests/study/test_phase_g_dense_orchestrator.py::test_persist_delta_highlights_creates_preview -vv | tee "$HUB"/green/pytest_delta_preview_helper_green.log
7. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/green/pytest_orchestrator_exec.log
8. pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_complete -vv | tee "$HUB"/green/pytest_highlights_complete.log
9. pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv | tee "$HUB"/collect/pytest_collect_highlights.log
10. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense.log
11. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py --hub "$HUB" --report "$HUB"/analysis/pipeline_verification.json |& tee "$HUB"/analysis/verifier_cli.log
12. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py --hub "$HUB" |& tee "$HUB"/analysis/highlights_check.log
13. pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv | tee "$HUB"/collect/pytest_collect_highlights_after_run.log
14. Update "$HUB"/summary/summary.md plus docs/fix_plan.md and galph_memory.md; attach Turn Summary + artifact inventory snapshot.

Pitfalls To Avoid
- Do not bypass `AUTHORITATIVE_CMDS_DOC`; CONFIG-001 requires the bridge before any legacy import.
- Keep hub-relative POSIX paths when writing inventory/previews (TYPE-PATH-001 guard + verifier rely on them).
- Preview lines must include the formatted phase deltas; leaving them out will fail `missing_preview_values`.
- Apply metric-specific precision: MS-SSIM → ±0.000, MAE → ±0.000000 (both amplitude + phase) or the validator + checker will remain RED.
- Avoid reusing the old 2025-11-11T001033Z hub; all new artifacts belong under 2025-11-11T003351Z.
- Watch long-running pipeline logs for early blockers; if `[8/8]` never appears, stop and capture blocker evidence instead of rerunning blindly.
- No package installs or env tweaks (Environment Freeze); resolve issues inside repo only.
- Keep new helper purely functional so unit tests stay fast; no file I/O in the test beyond the helper output directory.

If Blocked
- If the new helper test still fails after implementation (e.g., precision mismatch), capture the pytest output under `$HUB/red/` and update docs/fix_plan.md with the failure signature before stopping.
- If the dense pipeline aborts mid-phase, archive `$HUB/cli/run_phase_g_dense.log` plus the offending phase log, summarize blocker details in `$HUB/summary/summary.md`, and mark the focus blocked in docs/fix_plan.md so the supervisor can triage.

Findings Applied (Mandatory)
- POLICY-001 — PyTorch backend remains mandatory for PtyChi comparisons; no skipping torch deps.
- CONFIG-001 — Run orchestrator + helper scripts with AUTHORITATIVE_CMDS_DOC exported so legacy modules see synchronized params.
- DATA-001 — Trust generated Phase C NPZs; do not hand-edit diffraction/object data when debugging highlights.
- TYPE-PATH-001 — All hub artifacts must stay POSIX-relative (inventory + verifier expect this).
- STUDY-001 — Report MS-SSIM/MAE deltas (phase emphasis) with explicit signs when summarizing results.
- TEST-CLI-001 — Preview + highlights must align in content/precision so CLI validation fails fast on regressions.

Pointers
- docs/findings.md:16 — STUDY-001 reporting rules for MS-SSIM/MAE deltas.
- docs/findings.md:23 — TEST-CLI-001 preview/log enforcement details.
- docs/TESTING_GUIDE.md:320 — Phase G delta artifact expectations + orchestrator usage.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:985 — current inline highlight formatter to be refactored.
- tests/study/test_phase_g_dense_artifacts_verifier.py:2108 — preview/precision expectations that the helper must satisfy.

Next Up (optional)
1. After dense run evidence lands, schedule the sparse-view pipeline for parity.
2. Extend highlights checker to emit CSV so future regressions can diff numeric deltas automatically.

Doc Sync Plan (Conditional)
- After the new pytest lands, append the helper/test entry to docs/development/TEST_SUITE_INDEX.md (Phase G section) and mention the preview artifact workflow in docs/TESTING_GUIDE.md if wording changes; run `pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv` (already mapped) and archive the log under `$HUB`/collect/ for traceability before updating docs.

Mapped Tests Guardrail
- Keep `pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv` GREEN; if collection fails after code changes, stop and fix (or document the block) before claiming the loop complete.
