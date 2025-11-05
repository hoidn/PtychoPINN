Summary: Add an analyze-digest failure regression test, then rerun the dense Phase C→G pipeline and generate the metrics digest under the new hub.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Mapped tests:
  - pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_flags_failures -vv
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T010500Z/phase_g_dense_full_execution_real_run/

Do Now:
- Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
- Implement: tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_flags_failures (lock exit-code + banner behavior when analyze_dense_metrics sees failed jobs)
- Validate: pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_flags_failures -vv
- Validate: pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview -vv
- Execute: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T010500Z/phase_g_dense_full_execution_real_run --dose 1000 --view dense --splits train test --clobber | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T010500Z/phase_g_dense_full_execution_real_run/cli/run_phase_g_dense_cli.log
- Run: python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py --metrics plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T010500Z/phase_g_dense_full_execution_real_run/analysis/metrics_summary.json --highlights plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T010500Z/phase_g_dense_full_execution_real_run/analysis/aggregate_highlights.txt --output plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T010500Z/phase_g_dense_full_execution_real_run/analysis/metrics_digest.md | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T010500Z/phase_g_dense_full_execution_real_run/analysis/metrics_digest.log
- Capture: Mirror MS-SSIM/MAE deltas + digest path into plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T010500Z/phase_g_dense_full_execution_real_run/summary/summary.md and update docs/fix_plan.md Attempts History with exit codes + artifact links.
- Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T010500Z/phase_g_dense_full_execution_real_run/

How-To Map:
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
2. export HUB=plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T010500Z/phase_g_dense_full_execution_real_run
3. mkdir -p "$HUB"/{plan,collect,red,green,cli,analysis,summary}
4. pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_flags_failures -vv | tee "$HUB"/red/pytest_analyze_dense_failures.log  # expect RED (test missing) until implemented
5. Implement tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_flags_failures (fixtures for n_failed > 0 + stdout/stderr assertions)
6. pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_flags_failures -vv | tee "$HUB"/green/pytest_analyze_dense_failures.log
7. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview -vv | tee "$HUB"/green/pytest_highlights_preview.log
8. pytest tests/study/test_phase_g_dense_metrics_report.py -k report_phase_g_dense_metrics -vv | tee "$HUB"/collect/pytest_report_helper_regression.log
9. pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_flags_failures --collect-only -vv | tee "$HUB"/green/pytest_analyze_dense_collect.log
10. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber | tee "$HUB"/cli/run_phase_g_dense_cli.log
11. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py --metrics "$HUB"/analysis/metrics_summary.json --highlights "$HUB"/analysis/aggregate_highlights.txt --output "$HUB"/analysis/metrics_digest.md | tee "$HUB"/analysis/metrics_digest.log
12. Append MS-SSIM/MAE deltas + digest link to "$HUB"/summary/summary.md (prepend latest Turn Summary) and replicate exit codes/artifacts into docs/fix_plan.md.

Pitfalls To Avoid:
- Do not skip the initial RED run; capture failure output for the new test in red/.
- Keep AUTHORITATIVE_CMDS_DOC exported before every pytest/pipeline command to satisfy CONFIG-001 guard.
- Use Path objects in the new test (TYPE-PATH-001) when staging tmp fixtures.
- Ensure analyze_dense_metrics fixtures set n_failed>0 to exercise the failure-handling path.
- Run pipeline with --clobber to avoid stale Phase C/F outputs blocking hub preparation.
- Monitor long run with occasional `pgrep -fl run_phase_g_dense.py` if it appears hung; log results if abnormal.
- Treat any non-zero exit from pipeline or digest as blocker; capture blocker.log and halt further processing.
- Keep all logs/artifacts inside the hub; no outputs at repo root.
- Preserve highlights file produced by reporting helper; digest depends on it.
- Do not modify stable core modules (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`).

If Blocked:
- Save failing pytest output to "$HUB"/red/ with selector in filename and summarize traceback in docs/fix_plan.md.
- If pipeline aborts, record return code + failing command in "$HUB"/analysis/blocker.log, keep CLI log, and mark attempt blocked in docs/fix_plan.md and galph_memory.md before exiting.
- If digest inputs missing, capture directory tree via `find "$HUB" -maxdepth 3 -type f` into "$HUB"/analysis/tree.txt and log reason for absence in Attempts History.

Findings Applied (Mandatory):
- POLICY-001 — Keep PyTorch dependency expectations; do not skip torch-backed steps when executing comparisons.
- CONFIG-001 — Export AUTHORITATIVE_CMDS_DOC before pytest/pipeline invocations so legacy consumers observe synchronized config.
- DATA-001 — Stage NPZ fixtures via MetadataManager patterns; confirm no `_metadata` leakage in analyze test fixtures.
- TYPE-PATH-001 — Use `Path` for filesystem interactions in tests/CLI wrappers.
- OVERSAMPLING-001 — Verify digest highlights note overlap statistics; watch for neighbor_count warnings in logs.

Pointers:
- tests/study/test_phase_g_dense_metrics_report.py:420 — Existing reporting helper tests to mirror structure for the new analyze regression.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py:120 — Failure exit code guard for `n_failed > 0`.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:640 — Pipeline command sequence and highlights preview.
- docs/TESTING_GUIDE.md:268 — Phase G orchestrator workflow + AUTHORITATIVE_CMDS_DOC guidance.
- docs/findings.md:1 — POLICY-001 / DATA-001 / TYPE-PATH-001 references.

Next Up (optional):
- Run the sparse view dense pipeline (train/test) with identical digest workflow.

Doc Sync Plan (Conditional):
- After GREEN tests, update docs/TESTING_GUIDE.md §2.5 and docs/development/TEST_SUITE_INDEX.md with the new `test_analyze_dense_metrics_flags_failures` selector; attach `pytest ... --collect-only` log (step 9) under `$HUB`/green/.

Mapped Tests Guardrail:
- Confirm `pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_flags_failures --collect-only -vv` reports ≥1 collected test (artifact in step 9). If collection fails, patch before proceeding.

Hard Gate:
- Do not close the focus until pipeline exits 0, `analysis/metrics_summary.json`, `aggregate_report.md`, `aggregate_highlights.txt`, and `metrics_digest.md` all exist with non-empty contents, and docs/fix_plan.md captures MS-SSIM/MAE deltas + artifact links alongside exit codes.
