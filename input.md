Summary: Execute the dense Phase C->G pipeline in the new 2025-11-05T115706Z hub and capture verified MS-SSIM/MAE deltas plus provenance logs.
Mode: Perf
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 - Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Mapped tests:
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T115706Z/phase_g_dense_full_execution_real_run/

Do Now:
- Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
- Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main — run the dense Phase C->G pipeline with --clobber under the 2025-11-05T115706Z hub to produce the full metrics bundle.
- Document: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T115706Z/phase_g_dense_full_execution_real_run/summary/summary.md — record runtime, provenance checks, MS-SSIM/MAE deltas, and artifact links.
- Validate: pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
- Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T115706Z/phase_g_dense_full_execution_real_run/

How-To Map:
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
2. HUB=plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T115706Z/phase_g_dense_full_execution_real_run
3. pgrep -fl run_phase_g_dense.py || true
4. pgrep -fl studies.fly64_dose_overlap || true
5. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/green/pytest_orchestrator_dense_exec_recheck.log
6. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$PWD/$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense.log
7. test -f "$HUB"/analysis/metrics_summary.json && test -f "$HUB"/analysis/metrics_delta_summary.json && test -f "$HUB"/analysis/metrics_delta_highlights.txt && test -f "$HUB"/analysis/metrics_delta_highlights_preview.txt && test -f "$HUB"/analysis/metrics_digest.md && test -f "$HUB"/analysis/aggregate_report.md && test -f "$HUB"/analysis/aggregate_highlights.txt
8. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py --hub "$PWD/$HUB" | tee "$HUB"/analysis/highlights_consistency_check.log
9. find "$HUB" -maxdepth 3 -type f | sort > "$HUB"/analysis/artifact_inventory.txt
10. Update "$HUB"/summary/summary.md and docs/fix_plan.md with runtime, CONFIG-001/DATA-001/TYPE-PATH-001 notes, MS-SSIM & MAE deltas (PtychoPINN vs Baseline/PtyChi), and artifact links.

Pitfalls To Avoid:
- Launch all commands from the repo root so "$PWD/$HUB" resolves correctly inside this checkout.
- Terminate any lingering orchestrator processes before rerunning; duplicates corrupt artifact directories.
- Always pass --clobber to guarantee stale artifacts do not leak into the new hub.
- Never hand-edit metrics JSON or highlights; rely on emitted orchestrator outputs and the check script.
- Keep AUTHORITATIVE_CMDS_DOC exported for every command to respect CONFIG-001 legacy bridge ordering.
- Expect multi-hour runtime; keep the pipeline attached to tee so logs stream into the hub.
- If pytest fails, capture the RED log under "$HUB"/red/ and fix before proceeding to the long run.
- Treat the highlights consistency check as a hard gate; resolve differences instead of editing text files.
- Do not relocate artifacts outside the hub; summary/doc updates should link into the timestamped directory.

If Blocked:
- Preserve failing CLI logs under "$HUB"/cli/ with timestamps and summarize the error signature (command, exit code, snippet) in summary.md and docs/fix_plan.md.
- For pytest failures, store the RED log under "$HUB"/red/$(date -u +%H%M%SZ)_pytest.log and triage before reattempting the pipeline.
- If orchestrator exits 0 but artifacts are missing, copy partial outputs into "$HUB"/analysis/problem_*", describe gaps plus mitigation steps in summary.md, and mark docs/fix_plan.md as blocked with follow-up actions.

Findings Applied (Mandatory):
- POLICY-001 — Torch>=2.2 remains mandatory; pipeline stages rely on PyTorch-backed helpers.
- CONFIG-001 — Export AUTHORITATIVE_CMDS_DOC before invoking orchestrator so params.cfg stays synchronized.
- DATA-001 — Accept only pipeline-emitted DATA-001 compliant outputs; validator logs confirm metadata integrity.
- TYPE-PATH-001 — Maintain Path usage in orchestrator and check scripts to avoid string-path regressions.
- OVERSAMPLING-001 — Leave dense overlap constants unchanged; verify K≥C assumptions via existing datasets.
- STUDY-001 — Capture and report MS-SSIM/MAE deltas for PtychoPINN vs Baseline/PtyChi in summary/docs.

Pointers:
- `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T115706Z/phase_g_dense_full_execution_real_run/plan/plan.md:1`
- `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:531`
- `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py:1`
- `tests/study/test_phase_g_dense_orchestrator.py:856`
- `docs/TESTING_GUIDE.md:333`

Next Up (optional):
- Run sparse-view dense pipeline evidence after dense metrics land.

Doc Sync Plan (Conditional):
- Not required; mapped selectors unchanged and registry entries already cover the orchestrator regression.

Mapped Tests Guardrail:
- pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest --collect-only -vv must keep collecting; repair before closing the loop if collection count drops.

Hard Gate:
- Do not mark complete until the pipeline exits 0, the metrics/digest artifacts exist, highlights validation passes, summary.md and docs/fix_plan.md record MS-SSIM/MAE deltas with artifact links, and the GREEN pytest log is archived under the new hub.
