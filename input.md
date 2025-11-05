Summary: Monitor the in-flight dense Phase C→G run and, once it finishes, capture verified MS-SSIM/MAE deltas plus provenance artifacts.
Mode: Perf
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 - Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Mapped tests:
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T115706Z/phase_g_dense_full_execution_real_run/

Do Now:
- Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
- Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py::main — after the dense pipeline exits, run the highlights consistency verifier against the new metrics bundle and archive its log under analysis/.
- Document: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T115706Z/phase_g_dense_full_execution_real_run/summary/summary.md — add runtime, CONFIG-001/DATA-001/TYPE-PATH-001 notes, MS-SSIM & MAE deltas, and artifact links once metrics land.
- Validate: pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
- Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T115706Z/phase_g_dense_full_execution_real_run/

How-To Map:
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
2. HUB=plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T115706Z/phase_g_dense_full_execution_real_run
3. pgrep -fl run_phase_g_dense.py || true
4. pgrep -fl studies.fly64_dose_overlap || true
5. tail -n 40 "$HUB"/cli/phase_c_generation.log
6. Wait until steps 3 and 4 report no active processes before moving forward; if anything is still running, re-check every 5–10 minutes.
7. test -f "$HUB"/analysis/metrics_summary.json && test -f "$HUB"/analysis/metrics_delta_summary.json && test -f "$HUB"/analysis/metrics_delta_highlights.txt && test -f "$HUB"/analysis/metrics_delta_highlights_preview.txt && test -f "$HUB"/analysis/metrics_digest.md && test -f "$HUB"/analysis/aggregate_report.md && test -f "$HUB"/analysis/aggregate_highlights.txt
8. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py --hub "$PWD/$HUB" | tee "$HUB"/analysis/highlights_consistency_check.log
9. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py --metrics "$PWD/$HUB"/analysis/metrics_summary.json --highlights "$PWD/$HUB"/analysis/aggregate_highlights.txt --output "$PWD/$HUB"/analysis/metrics_digest.md | tee "$HUB"/analysis/metrics_digest_refresh.log
10. find "$HUB" -maxdepth 3 -type f | sort > "$HUB"/analysis/artifact_inventory.txt
11. Update "$HUB"/summary/summary.md and docs/fix_plan.md with runtime, guardrail evidence, MS-SSIM/MAE deltas (PtychoPINN vs Baseline/PtyChi), and log references.
12. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/green/pytest_orchestrator_dense_exec_post_run.log

Pitfalls To Avoid:
- Do not terminate the generation command prematurely; let the background job finish naturally before verifying artifacts.
- Stay in the same repo root used to launch the run so "$PWD/$HUB" resolves correctly (previous `$PWD` expansion bug produced `/plans/...`).
- Keep AUTHORITATIVE_CMDS_DOC exported for every helper invocation to preserve CONFIG-001 ordering.
- Never edit metrics JSON or highlights by hand; rely on orchestrator outputs and the verification scripts.
- If metrics files are missing, don’t rerun immediately—first review cli logs to understand which phase failed.
- Ensure PyTorch dependency remains available; do not modify environment or install packages mid-run.
- Treat highlights_consistency_check.log as a hard gate; resolve mismatches rather than editing text outputs.
- Preserve GPU availability for the active job; avoid launching new compute-heavy processes until the pipeline completes.

If Blocked:
- Capture the failing command (full CLI, exit code, stdio snippet) into "$HUB"/cli/ with a timestamped filename.
- Summarize the failure mode in summary.md and docs/fix_plan.md, marking the ledger attempt as blocked with the remediation path.
- If the pipeline never exits, record the stuck PID list and the most recent log tail, then halt further commands and document the timeout.

Findings Applied (Mandatory):
- POLICY-001 — PyTorch remains mandatory; later phases rely on torch-backed baselines.
- CONFIG-001 — Maintain AUTHORITATIVE_CMDS_DOC exports before invoking any legacy-touching helpers.
- DATA-001 — Accept only orchestrator-emitted NPZs validated by the built-in checks; report any missing keys immediately.
- TYPE-PATH-001 — Use resolved paths ("$PWD/$HUB") whenever invoking helper scripts to avoid string-path regressions.
- OVERSAMPLING-001 — Leave the dense grouping configuration untouched; verify K≥C assumptions via the generated datasets.
- STUDY-001 — Capture and report MS-SSIM/MAE deltas comparing PtychoPINN, Baseline, and PtyChi.

Pointers:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T115706Z/phase_g_dense_full_execution_real_run/plan/plan.md:1
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:531
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py:1
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py:1
- tests/study/test_phase_g_dense_orchestrator.py:856
- docs/TESTING_GUIDE.md:333

Next Up (optional):
- If dense view completes cleanly, repeat the workflow for the sparse overlap view to finish the dose/overlap matrix.

Doc Sync Plan (Conditional):
- Not applicable; mapped pytest selector unchanged and already documented.

Mapped Tests Guardrail:
- pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest --collect-only -vv must keep collecting; fix collection issues before closing the loop.

Hard Gate:
- Do not mark the attempt complete until the pipeline exits 0, all analysis artifacts exist, highlights_consistency_check.log reports matches, metrics_digest.md reflects the new run, and summary.md + docs/fix_plan.md capture the MS-SSIM/MAE deltas with artifact links.
