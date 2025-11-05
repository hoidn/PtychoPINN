Summary: Relaunch the dense Phase C->G pipeline in the new 2025-11-05T111247Z hub to capture MS-SSIM/MAE deltas and supporting artifacts.
Mode: none
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 - Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Mapped tests:
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T111247Z/phase_g_dense_full_execution_real_run/

Do Now:
- Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
- Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main - execute the dense Phase C->G pipeline with --clobber under the 2025-11-05T111247Z hub to produce the full metrics bundle.
- Document: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T111247Z/phase_g_dense_full_execution_real_run/summary/summary.md - record runtime, provenance checks, MS-SSIM/MAE deltas, and artifact links.
- Validate: pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
- Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T111247Z/phase_g_dense_full_execution_real_run/

How-To Map:
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
2. HUB=plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T111247Z/phase_g_dense_full_execution_real_run
3. pgrep -fl run_phase_g_dense.py || true
4. pgrep -fl studies.fly64_dose_overlap || true
5. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/green/pytest_orchestrator_dense_exec_recheck.log
6. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$PWD/$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense.log
7. test -f "$HUB"/analysis/metrics_summary.json && test -f "$HUB"/analysis/metrics_delta_summary.json && test -f "$HUB"/analysis/metrics_delta_highlights.txt && test -f "$HUB"/analysis/metrics_delta_highlights_preview.txt && test -f "$HUB"/analysis/metrics_digest.md && test -f "$HUB"/analysis/aggregate_report.md && test -f "$HUB"/analysis/aggregate_highlights.txt
8. find "$HUB" -maxdepth 3 -type f | sort > "$HUB"/analysis/artifact_inventory.txt
9. cat "$HUB"/analysis/metrics_delta_highlights.txt | tee "$HUB"/analysis/metrics_delta_highlights_preview.txt
10. Update "$HUB"/summary/summary.md and docs/fix_plan.md with runtime, provenance checklist (CONFIG-001/DATA-001), MS-SSIM & MAE deltas (PtychoPINN vs Baseline/PtyChi), and artifact links.

Pitfalls To Avoid:
- Launch commands from the repo root so "$PWD/$HUB" resolves inside this checkout (prevents spilling into PtychoPINN2).
- Confirm no lingering orchestrator processes before relaunching; terminate them if `pgrep` returns PIDs.
- Always pass --clobber so stale artifacts from previous hubs cannot contaminate the new run.
- Do not hand-edit metrics JSON or summaries; rely on orchestrator outputs only.
- Compare highlights text against JSON deltas before sign-off; resolve mismatches instead of editing outputs.
- Keep AUTHORITATIVE_CMDS_DOC exported for every command to satisfy CONFIG-001 and legacy bridge rules.
- Expect multi-hour runtime; keep the pipeline attached to `tee` so logs are captured continuously.

If Blocked:
- Preserve failing CLI logs under "$HUB"/cli/ with timestamps and summarize the error signature (command, exit code, snippet) in summary.md and docs/fix_plan.md.
- For pytest failures, store the RED log under "$HUB"/red/`date +%H%M%S`_pytest.log and triage before reattempting the pipeline.
- If metrics artifacts are missing after a successful exit, copy partial outputs into "$HUB"/analysis/problem_*`, describe gaps plus next steps in summary.md, and mark the ledger entry blocked with mitigation steps.

Findings Applied (Mandatory):
- POLICY-001 - Torch>=2.2 remains mandatory; the orchestrator imports torch-backed helpers during Phase F/G.
- CONFIG-001 - Export AUTHORITATIVE_CMDS_DOC so params.cfg stays synchronized before legacy consumers run.
- DATA-001 - Accept only pipeline-emitted DATA-001 compliant outputs; validator logs confirm metadata integrity.
- TYPE-PATH-001 - Preserve Path usage in command definitions and artifact references.
- OVERSAMPLING-001 - Keep dense overlap constants unchanged; rely on existing design parameters.
- STUDY-001 - Capture MS-SSIM/MAE deltas for PtychoPINN vs Baseline/PtyChi and document them in summary/docs.

Pointers:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:738 - Stage orchestration for full dense execution path and command sequence.
- tests/study/test_phase_g_dense_orchestrator.py:856 - Regression ensuring analyze_dense_metrics.py runs after reporting helper.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T111247Z/phase_g_dense_full_execution_real_run/plan/plan.md:1 - Supervisor plan and checklist for this hub.
- docs/TESTING_GUIDE.md:333 - Phase G delta persistence expectations (JSON + highlights) used for verification.

Next Up (optional):
- Run sparse-view dense pipeline evidence after dense metrics land.

Doc Sync Plan (Conditional):
- Not required; selectors unchanged and registries already cover the orchestrator regression.

Mapped Tests Guardrail:
- pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest --collect-only -vv must continue to collect; repair before closing the loop if collection count drops.

Hard Gate:
- Do not mark complete until the pipeline exits 0, the metrics/digest artifacts above exist, highlights text matches JSON deltas, summary.md and docs/fix_plan.md record MS-SSIM/MAE values with artifact links, and the GREEN pytest log is archived under the new hub.
