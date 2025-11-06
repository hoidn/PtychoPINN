Summary: Shepherd the relaunched dense Phase C→G run to completion and capture comparison evidence (highlights, metrics digest, MS-SSIM/MAE deltas) for dose=1000.
Mode: Perf
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 - Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Mapped tests:
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T115706Z/phase_g_dense_full_execution_real_run/

Do Now:
- Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
- Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main — monitor the live relaunch (PID 2478561) via `cli/run_phase_g_dense_relaunch_2025-11-06T074519Z.log`; if the process exits before `[8/8]`, relaunch with `--clobber`, capture blocker evidence, and rerun until all Phase D–G artifacts land under `analysis/`.
- Document: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T115706Z/phase_g_dense_full_execution_real_run/summary/summary.md — append runtime, guardrail evidence (CONFIG-001/DATA-001/TYPE-PATH-001), MS-SSIM and MAE deltas vs Baseline/PtyChi, artifact inventory pointers, refreshed highlights/digest logs, and mirror the attempt in docs/fix_plan.md.
- Validate: pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
- Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T115706Z/phase_g_dense_full_execution_real_run/

How-To Map:
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
2. HUB=/home/ollie/Documents/PtychoPINN2/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T115706Z/phase_g_dense_full_execution_real_run
3. LOG="$HUB"/cli/run_phase_g_dense_relaunch_2025-11-06T074519Z.log
4. test -d "$HUB" && echo "Hub ready: $HUB" || { echo "Missing hub"; exit 1; }
5. if ! pgrep -f "plans/active/.*/run_phase_g_dense.py" >/dev/null; then \
       RUN_LOG="$HUB"/cli/run_phase_g_dense_relaunch_$(date -u +%Y-%m-%dT%H%M%SZ).log; \
       python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py \
         --hub "$HUB" \
         --dose 1000 \
         --view dense \
         --splits train test \
         --clobber \
         |& tee "$RUN_LOG"; \
       LOG="$RUN_LOG"; \
   fi
6. tail -n 40 "$LOG" | rg "\[[1-8]/8\]" || echo "No stage banner yet"
7. wait_for_completion() { while pgrep -f "plans/active/.*/run_phase_g_dense.py" >/dev/null; do sleep 120; tail -n 5 "$LOG"; done; }
8. wait_for_completion
9. if ! rg -q "\[8/8\]" "$LOG"; then echo "Pipeline incomplete"; exit 1; fi
10. for f in metrics_summary.json metrics_delta_summary.json metrics_delta_highlights.txt metrics_delta_highlights_preview.txt metrics_digest.md aggregate_report.md aggregate_highlights.txt; do test -f "$HUB"/analysis/"$f" || { echo "Missing $f"; exit 1; }; done
11. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py --hub "$PWD/$HUB" | tee "$HUB"/analysis/highlights_consistency_check_$(date -u +%Y-%m-%dT%H%M%SZ).log
12. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py --metrics "$PWD/$HUB"/analysis/metrics_summary.json --highlights "$PWD/$HUB"/analysis/aggregate_highlights.txt --output "$PWD/$HUB"/analysis/metrics_digest.md | tee "$HUB"/analysis/metrics_digest_refresh_$(date -u +%Y-%m-%dT%H%M%SZ).log
13. find "$HUB" -maxdepth 3 -type f | sort > "$HUB"/analysis/artifact_inventory_$(date -u +%Y-%m-%dT%H%M%SZ).txt
14. Edit "$HUB"/summary/summary.md with runtimes, guardrail evidence, MS-SSIM/MAE deltas vs Baseline & PtyChi, highlight/digest log links, artifact inventory reference, and mirror attempt details in docs/fix_plan.md.
15. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/green/pytest_orchestrator_dense_exec_post_run_$(date -u +%Y-%m-%dT%H%M%SZ).log

Pitfalls To Avoid:
- Do not interrupt the orchestrator mid-phase; let the background process run to completion or restart cleanly with `--clobber` if it fails.
- Keep AUTHORITATIVE_CMDS_DOC exported before any helper invocation per CONFIG-001.
- Use absolute `$PWD/$HUB` paths when invoking helpers to satisfy TYPE-PATH-001 safeguards.
- Treat highlights_consistency_check failures as hard blockers; do not hand-edit analysis outputs.
- Avoid manual deletion of NPZ outputs; rely on orchestrator `--clobber` to manage stale artifacts.
- Preserve GPU availability for this run; defer other GPU-heavy workloads until `[8/8]` logs.
- Refresh metrics_digest.md via the helper script instead of manual edits to maintain provenance.
- Record MS-SSIM/MAE deltas with artifact links in both summary.md and docs/fix_plan.md per STUDY-001.
- Leave dense view grouping parameters untouched to respect OVERSAMPLING-001 constraints.
- Keep hub directory structure stable; do not relocate `analysis/` or `data/` while the run is active.

If Blocked:
- Capture failing command + exit code and append the last 40 log lines to "$HUB"/cli/blocker_$(date -u +%Y%m%dT%H%M%SZ).log.
- Note blocker details, hypotheses, and next steps in "$HUB"/summary/summary.md and docs/fix_plan.md, marking the attempt `blocked` with artifacts.
- If GPU exhaustion or OOM occurs, record `nvidia-smi` snapshot timestamps only (no dumps) and halt further actions until supervisor guidance.

Findings Applied (Mandatory):
- POLICY-001 — Preserve PyTorch baseline tooling; no dependency changes during this loop.
- CONFIG-001 — Keep AUTHORITATIVE_CMDS_DOC exported before any CLI helper or subprocess runs.
- DATA-001 — Rely on validator outputs; do not hand-edit NPZ contents.
- TYPE-PATH-001 — Use `$PWD/$HUB` absolute paths and avoid moving hub directories during execution.
- OVERSAMPLING-001 — Maintain dense view parameters (K=7, gridsize=2) untouched.
- STUDY-001 — Report MS-SSIM/MAE deltas vs Baseline/PtyChi with provenance links in summary & ledger.

Pointers:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:598
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py:1
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py:1
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T115706Z/phase_g_dense_full_execution_real_run/cli/run_phase_g_dense_relaunch_2025-11-06T074519Z.log:1
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T115706Z/phase_g_dense_full_execution_real_run/summary/summary.md:1
- docs/TESTING_GUIDE.md:333
- docs/findings.md:8

Next Up (optional):
- If dense run finishes quickly, stage sparse view relaunch plan with updated hub prep checklist.

Doc Sync Plan (Conditional):
- Not applicable; mapped pytest selector unchanged.

Mapped Tests Guardrail:
- pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest --collect-only -vv should continue collecting 1 test; investigate immediately if collection count changes.

Hard Gate:
- Do not mark complete until the orchestrator log shows `[8/8]`, Phase D–G artifacts exist, highlights consistency passes, metrics_digest.md is refreshed, summary/fix_plan updated with MS-SSIM/MAE deltas and provenance, and the mapped pytest selector passes with a new log archived under `green/`.
