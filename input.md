Summary: Run the dense Phase C→G pipeline with --clobber to capture real MS-SSIM/MAE deltas and archive the evidence bundle.
Mode: none
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Mapped tests:
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T170500Z/phase_g_dense_full_execution_real_run/

Do Now:
- Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
- Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main — run the dense Phase C→G pipeline with --clobber to generate fresh MS-SSIM/MAE deltas; no code edits expected, capture artifacts under $HUB.
- Document: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T170500Z/phase_g_dense_full_execution_real_run/summary.md — record runtime, provenance checks, and extracted deltas.
- Validate: pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
- Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T170500Z/phase_g_dense_full_execution_real_run/

How-To Map:
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
2. export HUB=plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T170500Z/phase_g_dense_full_execution_real_run
3. mkdir -p "$HUB"/{plan,summary,analysis,cli,collect,red,green}
4. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/green/pytest_orchestrator_dense_exec_recheck.log
5. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense.log
6. test -f "$HUB"/analysis/metrics_summary.json && test -f "$HUB"/analysis/metrics_delta_summary.json && test -f "$HUB"/analysis/metrics_delta_highlights.txt && test -f "$HUB"/analysis/metrics_digest.md && test -f "$HUB"/analysis/aggregate_report.md && test -f "$HUB"/analysis/aggregate_highlights.txt
7. cat "$HUB"/analysis/metrics_delta_highlights.txt | tee "$HUB"/analysis/metrics_delta_highlights_preview.txt
8. find "$HUB" -maxdepth 3 -type f | sort > "$HUB"/analysis/artifact_inventory.txt
9. Manually update "$HUB"/summary/summary.md and docs/fix_plan.md with runtime, MS-SSIM/MAE deltas (PtychoPINN vs Baseline/PtyChi), provenance confirmation, and artifact links.

Pitfalls To Avoid:
- Keep AUTHORITATIVE_CMDS_DOC exported before the pipeline run to satisfy CONFIG-001.
- Use --clobber so stale artifacts from prior hubs cannot leak into the evidence bundle.
- Do not modify metrics JSON/summary files by hand; rely on pipeline output and only read/record values.
- Ensure highlights text matches the JSON deltas; investigate if mismatched before signing off.
- Capture the full CLI log via tee; no silent runs or partial transcripts.
- Budget 2–4 hours for the pipeline; if you must interrupt, document the partial state and mark the attempt blocked.
- Avoid editing production modules beyond orchestrator invocation; this loop is evidence-only unless failures require fixes.

If Blocked:
- Preserve the failing CLI log under "$HUB"/cli/ and summarize the error signature (command + excerpt) in summary.md and docs/fix_plan.md, marking the focus blocked.
- If pytest fails, keep the RED log under "$HUB"/red/`date +%H%M%S`_pytest.log and diagnose before re-running the pipeline.
- When artifacts are missing, copy any partial JSON into "$HUB"/analysis/`problem_*`.json and describe the gap in summary.md with proposed fix.

Findings Applied (Mandatory):
- POLICY-001 — PyTorch dependency is mandatory; pipeline relies on torch>=2.2 during summarize/report steps.
- CONFIG-001 — Export AUTHORITATIVE_CMDS_DOC to guarantee params.cfg synchronization before orchestrator imports legacy modules.
- DATA-001 — Accept only metrics JSON outputs produced by the pipeline; no manual editing of schema or dtypes.
- TYPE-PATH-001 — Keep artifact paths relative within logs and summaries; no absolute paths in success messaging.
- OVERSAMPLING-001 — Dense overlap parameters stay fixed; do not tweak design constants during evidence run.
- STUDY-001 — Report MS-SSIM/MAE deltas comparing PtychoPINN vs Baseline/PtyChi for the dense fly64 condition.

Pointers:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:861 — Delta aggregation + highlights emission block to monitor during run.
- tests/study/test_phase_g_dense_orchestrator.py:856 — Regression selector ensuring highlights artifact persists.
- docs/TESTING_GUIDE.md:336 — Phase G dense evidence checklist and highlights verification steps.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T170500Z/plan/plan.md:1 — Supervisor plan for this hub.

Next Up (optional):
- Prepare sparse view Phase G evidence run once dense metrics are archived.

Doc Sync Plan (Conditional):
- Not required; selectors unchanged and already registered.

Mapped Tests Guardrail:
- pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest --collect-only -vv must continue to collect; if it drops, author a fix before proceeding.

Hard Gate:
- Do not close the loop until the dense pipeline exits 0 with refreshed artifacts, highlights text matches JSON deltas, summary.md + docs/fix_plan.md carry the recorded MS-SSIM/MAE values, and the GREEN pytest log is archived under the new hub.
