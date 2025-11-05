Summary: Let the in-flight dense Phase C→G run finish, then capture highlights/digest evidence and sync docs/tests for the dense dose=1000 view.
Mode: Perf
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 - Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Mapped tests:
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T115706Z/phase_g_dense_full_execution_real_run/

Do Now:
- Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
- Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main — monitor the live rerun (PID 2278335) until the log shows `[8/8]` and all Phase D–G artifacts land; if the process stops early, relaunch with `--clobber`, capture blocker evidence, and rerun to completion with TYPE-PATH-001 safe paths.
- Document: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T115706Z/phase_g_dense_full_execution_real_run/summary/summary.md — append runtime, guardrail notes (CONFIG-001/DATA-001/TYPE-PATH-001), MS-SSIM and MAE deltas vs Baseline/PtyChi, artifact inventory pointers, and link refreshed logs; reflect the same attempt in docs/fix_plan.md.
- Validate: pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
- Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T115706Z/phase_g_dense_full_execution_real_run/

How-To Map:
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
2. HUB=/home/ollie/Documents/PtychoPINN2/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T115706Z/phase_g_dense_full_execution_real_run
3. test -d "$HUB" && echo "Hub ready: $HUB" || { echo "Missing hub"; exit 1; }
4. pgrep -fl "run_phase_g_dense" || echo "No orchestrator process detected"
5. if ! pgrep -f "plans/active/.*/run_phase_g_dense.py" >/dev/null; then \
       python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py \
         --hub "$HUB" \
         --dose 1000 \
         --view dense \
         --splits train test \
         --clobber \
         |& tee "$HUB"/cli/run_phase_g_dense_full_$(date -u +%Y-%m-%dT%H%M%SZ).log; \
   fi
6. tail -n 40 "$HUB"/cli/run_phase_g_dense_full_*.log | rg "\[[1-8]/8\]"
7. wait_for_completion() { while pgrep -f "plans/active/.*/run_phase_g_dense.py" >/dev/null; do sleep 120; tail -n 5 "$HUB"/cli/run_phase_g_dense_full_*.log; done; }
8. wait_for_completion
9. if ! rg -q "\[8/8\]" "$HUB"/cli/run_phase_g_dense_full_*.log; then echo "Pipeline incomplete"; exit 1; fi
10. test -f "$HUB"/analysis/metrics_summary.json && test -f "$HUB"/analysis/metrics_delta_summary.json && test -f "$HUB"/analysis/metrics_delta_highlights.txt && test -f "$HUB"/analysis/metrics_digest.md && test -f "$HUB"/analysis/aggregate_report.md
11. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py --hub "$HUB" | tee "$HUB"/analysis/highlights_consistency_check.log
12. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py --metrics "$HUB"/analysis/metrics_summary.json --highlights "$HUB"/analysis/aggregate_highlights.txt --output "$HUB"/analysis/metrics_digest.md | tee "$HUB"/analysis/metrics_digest_refresh_$(date -u +%Y-%m-%dT%H%M%SZ).log
13. find "$HUB" -maxdepth 3 -type f | sort > "$HUB"/analysis/artifact_inventory_$(date -u +%Y-%m-%dT%H%M%SZ).txt
14. Edit "$HUB"/summary/summary.md with runtime, guardrail evidence, MS-SSIM/MAE deltas vs Baseline/PtyChi, highlight/digest links, artifact inventory reference, and mention refreshed pytest log; mirror attempt details in docs/fix_plan.md.
15. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/green/pytest_orchestrator_dense_exec_post_run_$(date -u +%Y-%m-%dT%H%M%SZ).log

Pitfalls To Avoid:
- Do not interrupt the orchestrator mid-phase; let the background process finish naturally or relaunch with `--clobber` if it fails.
- Keep AUTHORITATIVE_CMDS_DOC exported before running any helper scripts (CONFIG-001 guard).
- Use absolute hub paths (`$HUB`) when invoking helpers to avoid the `/plans` permission regression (TYPE-PATH-001).
- Treat highlights_consistency_check.log failures as hard blockers—fix data instead of editing outputs.
- Do not delete NPZs manually; use orchestrator `--clobber` to clear stale artifacts.
- Ensure GPU resources remain dedicated to this run; avoid starting other GPU-heavy jobs concurrently.
- Refresh metrics_digest.md via the helper script instead of editing by hand, preserving provenance logs.
- When editing summary.md/docs/fix_plan.md, record MS-SSIM/MAE deltas with full provenance paths per STUDY-001 expectations.
- Preserve existing skip summaries and manifests; do not overwrite unless the orchestrator regenerates them.
- Keep repo root as CWD so relative script imports resolve correctly.

If Blocked:
- Capture failing command + exit code and append the last 40 log lines to `$HUB/cli/blocker_$(date -u +%Y%m%dT%H%M%SZ).log`.
- Note blocker details, hypotheses, and next steps in `$HUB/summary/summary.md` and docs/fix_plan.md, marking the attempt `blocked` with artifacts.
- If GPU exhaustion or OOM occurs, record `nvidia-smi` snapshot (timestamps only, no full dumps) and halt further actions until supervisor guidance.

Findings Applied (Mandatory):
- POLICY-001 — Preserve PyTorch baseline tooling; no changes to torch deps during this loop.
- CONFIG-001 — Keep AUTHORITATIVE_CMDS_DOC exported before any CLI helper or subprocess runs.
- DATA-001 — Rely on validator output logs; do not hand-edit NPZ contents.
- TYPE-PATH-001 — Use `$PWD/$HUB` absolute paths and avoid moving hub directories during execution.
- OVERSAMPLING-001 — Leave dense view grouping parameters untouched; orchestrator enforces K=7, gridsize=2.
- STUDY-001 — Report MS-SSIM/MAE deltas vs Baseline/PtyChi with artifact links in summary & ledger.

Pointers:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:598
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py:1
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py:1
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T115706Z/phase_g_dense_full_execution_real_run/summary/summary.md:1
- docs/TESTING_GUIDE.md:333
- docs/findings.md:42

Next Up (optional):
- If dense run assets land quickly, stage sparse view rerun plan using the same workflow.

Doc Sync Plan (Conditional):
- Not applicable; mapped pytest selector unchanged.

Mapped Tests Guardrail:
- pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest --collect-only -vv should continue collecting 1 test; investigate immediately if collection count changes.

Hard Gate:
- Do not mark complete until the orchestrator log shows `[8/8]`, Phase D–G artifacts exist, highlights consistency passes, metrics_digest.md is refreshed, summary/fix_plan updated with MS-SSIM/MAE deltas and provenance, and the mapped pytest selector passes with a new log archived under `green/`.
