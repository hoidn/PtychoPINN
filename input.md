Summary: Run the dense Phase C→G pipeline in the new 2025-11-09T190500Z hub to capture fresh MS-SSIM/MAE deltas for the dense dose=1000 condition.
Mode: none
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Mapped tests:
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T190500Z/phase_g_dense_full_execution_real_run/

Do Now:
- Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
- Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main — execute the dense Phase C→G pipeline with --clobber under the 2025-11-09T190500Z hub to produce the full metrics bundle.
- Document: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T190500Z/phase_g_dense_full_execution_real_run/summary/summary.md — record runtime, provenance checks, MS-SSIM/MAE deltas, and artifact links.
- Validate: pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
- Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T190500Z/phase_g_dense_full_execution_real_run/

How-To Map:
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
2. HUB=plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T190500Z/phase_g_dense_full_execution_real_run
3. pgrep -fl run_phase_g_dense.py || true
4. pgrep -fl studies.fly64_dose_overlap || true
5. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/green/pytest_orchestrator_dense_exec_recheck.log
6. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$PWD/$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense.log
7. test -f "$HUB"/analysis/metrics_summary.json && test -f "$HUB"/analysis/metrics_delta_summary.json && test -f "$HUB"/analysis/metrics_delta_highlights.txt && test -f "$HUB"/analysis/metrics_digest.md && test -f "$HUB"/analysis/aggregate_report.md && test -f "$HUB"/analysis/aggregate_highlights.txt
8. find "$HUB" -maxdepth 3 -type f | sort > "$HUB"/analysis/artifact_inventory.txt
9. cat "$HUB"/analysis/metrics_delta_highlights.txt | tee "$HUB"/analysis/metrics_delta_highlights_preview.txt
10. Update "$HUB"/summary/summary.md and docs/fix_plan.md with runtime, provenance checklist (CONFIG-001/DATA-001), MS-SSIM & MAE deltas (PtychoPINN vs Baseline/PtyChi), and artifact links.

Pitfalls To Avoid:
- Run commands from the repo root so `$PWD/$HUB` resolves inside this checkout (prevents spilling into PtychoPINN2).
- Ensure no lingering orchestrator processes before relaunching; kill them if `pgrep` returns results.
- Always pass --clobber so stale artifacts from 170500Z cannot contaminate the new hub.
- Do not edit generated metrics JSON/Markdown manually; rely on orchestrator outputs only.
- Verify highlights text matches delta JSON before signing off; investigate mismatches instead of editing outputs.
- Expect long runtime (Phase E baseline + dense training); leave the pipeline attached to tee so logs are captured continuously.
- Keep AUTHORITATIVE_CMDS_DOC exported for every command to satisfy CONFIG-001 and legacy bridge requirements.

If Blocked:
- Preserve failing CLI logs under "$HUB"/cli/ with timestamps and summarize the error signature (command, exit code, snippet) in summary.md and docs/fix_plan.md.
- For pytest failures, store the RED log under "$HUB"/red/`date +%H%M%S`_pytest.log and triage before reattempting the pipeline.
- If metrics artifacts are missing after a supposedly successful run, copy partial JSON/text into "$HUB"/analysis/problem_* and describe gaps plus next steps in summary.md; mark the ledger entry blocked with mitigation.

Findings Applied (Mandatory):
- POLICY-001 — Torch>=2.2 remains required; Phase G pipeline imports torch-backed helpers during summarization.
- CONFIG-001 — Export AUTHORITATIVE_CMDS_DOC to keep params.cfg synchronized before legacy modules run.
- DATA-001 — Accept only pipeline-emitted DATA-001 compliant outputs; validate generated train/test splits implicitly via validator logs.
- TYPE-PATH-001 — Use Path-safe commands and keep artifact paths relative in success banners and summaries.
- OVERSAMPLING-001 — Do not alter dense overlap constants; use the existing design parameters.
- STUDY-001 — Capture MS-SSIM/MAE deltas for PtychoPINN vs Baseline/PtyChi and record them in summary/docs.

Pointers:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:700 — Command list & success banner logic for metrics bundle.
- tests/study/test_phase_g_dense_orchestrator.py:856 — Regression selector asserting digest + delta outputs.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T190500Z/phase_g_dense_full_execution_real_run/plan/plan.md:1 — Supervisor plan with full checklist for this hub.
- docs/TESTING_GUIDE.md:336 — Dense Phase G evidence workflow and verification steps.

Next Up (optional):
- Kick off sparse view dense pipeline evidence run once dense metrics are archived.

Doc Sync Plan (Conditional):
- Not required; selectors unchanged and registry entries already cover the orchestrator regression.

Mapped Tests Guardrail:
- pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest --collect-only -vv must continue to collect; author repairs before closing the loop if collection count drops.

Hard Gate:
- Do not mark the loop complete until the dense pipeline exits 0, the full metrics bundle is present, highlights text matches the JSON deltas, summary.md plus docs/fix_plan.md capture MS-SSIM/MAE values with artifact links, and the GREEN pytest log is archived under the new hub.
