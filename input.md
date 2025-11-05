Summary: Re-launch the dense dose=1000 Phase C→G pipeline and capture verified MS-SSIM/MAE deltas plus highlights once artifacts land.
Mode: Perf
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 - Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Mapped tests:
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T115706Z/phase_g_dense_full_execution_real_run/

Do Now:
- Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
- Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main — re-launch the dense Phase C→G pipeline with TYPE-PATH-001 safe `"$PWD/$HUB"` hub path, confirm the log shows all 8 stages and Phase D–G outputs land under the hub.
- Document: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T115706Z/phase_g_dense_full_execution_real_run/summary/summary.md — append runtime, guardrail notes (CONFIG-001/DATA-001/TYPE-PATH-001), MS-SSIM & MAE deltas, artifact inventory, and link the new orchestrator log; mirror the update in docs/fix_plan.md Attempts History.
- Validate: pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
- Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T115706Z/phase_g_dense_full_execution_real_run/

How-To Map:
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
2. HUB=plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T115706Z/phase_g_dense_full_execution_real_run
3. test -d "$HUB/data/phase_c" && echo "Phase C assets present" || { echo "missing Phase C"; exit 1; }
4. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py \
   --hub "$PWD/$HUB" \
   --dose 1000 \
   --view dense \
   --splits train test \
   --clobber \
   |& tee "$HUB"/cli/run_phase_g_dense_full_2025-11-05T123500Z.log
5. grep -F "[8/8]" "$HUB"/cli/run_phase_g_dense_full_2025-11-05T123500Z.log
6. test -f "$HUB"/analysis/metrics_summary.json && test -f "$HUB"/analysis/metrics_delta_summary.json && test -f "$HUB"/analysis/metrics_delta_highlights.txt && test -f "$HUB"/analysis/metrics_delta_highlights_preview.txt && test -f "$HUB"/analysis/metrics_digest.md && test -f "$HUB"/analysis/aggregate_report.md && test -f "$HUB"/analysis/aggregate_highlights.txt
7. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py --hub "$PWD/$HUB" | tee "$HUB"/analysis/highlights_consistency_check.log
8. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py --metrics "$PWD/$HUB"/analysis/metrics_summary.json --highlights "$PWD/$HUB"/analysis/aggregate_highlights.txt --output "$PWD/$HUB"/analysis/metrics_digest.md | tee "$HUB"/analysis/metrics_digest_refresh.log
9. find "$HUB" -maxdepth 3 -type f | sort > "$HUB"/analysis/artifact_inventory.txt
10. Update "$HUB"/summary/summary.md and docs/fix_plan.md with runtime, guardrail evidence (CONFIG-001/DATA-001/TYPE-PATH-001), MS-SSIM/MAE deltas vs Baseline/PtyChi, highlights log links, and provenance for metrics_delta files.
11. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/green/pytest_orchestrator_dense_exec_post_run.log

Pitfalls To Avoid:
- Do not forget `AUTHORITATIVE_CMDS_DOC`; CONFIG-001 requires it before every helper invocation.
- Keep the hub path absolute via `"$PWD/$HUB"` or the script will try to write under `/plans` again (TYPE-PATH-001 regression).
- Allow the orchestrator to finish all eight phases; do not interrupt Phase E training even if it runs for several hours.
- Treat highlights_consistency_check.log as a hard gate—resolve mismatches rather than editing text outputs.
- Never hand-edit metrics JSON or highlights files; only use provided helpers.
- Preserve GPU availability for the run; avoid concurrent heavy GPU workloads.
- If `grep -F "[8/8]"` fails, inspect the log before re-running—capture the failure snippet into `cli/` with a timestamped filename.
- Do not delete prior Phase C outputs manually; rely on `--clobber` to manage hub hygiene.
- Keep the repo root as CWD so `$PWD/$HUB` resolves correctly for every helper script.
- Avoid mixing TensorFlow/PyTorch env tweaks mid-run; environment is frozen.

If Blocked:
- Capture the failing command, exit code, and the last ~40 lines of its log into `$HUB/cli/blocker_$(date -u +%Y%m%dT%H%M%SZ).log`.
- Note the failure and remediation hypothesis in `$HUB/summary/summary.md` plus docs/fix_plan.md, and mark the attempt `blocked` with the captured evidence.
- If the orchestrator stalls >30 min with no log progress, record active PIDs (`pgrep -fl run_phase_g_dense`) and the latest log tail, then halt further steps until supervisor guidance.

Findings Applied (Mandatory):
- POLICY-001 — PyTorch baseline assets remain required; ensure the orchestrator leaves PyTorch installs untouched.
- CONFIG-001 — Always export AUTHORITATIVE_CMDS_DOC before invoking CLI helpers or runners.
- DATA-001 — Trust only orchestrator-emitted NPZs validated during the run; verify DATA-001 guard outputs remain green.
- TYPE-PATH-001 — Use resolved hub paths (`"$PWD/$HUB"`) to avoid `/plans` permission errors.
- OVERSAMPLING-001 — Keep the dense view configuration (K=7, gridsize=2) intact; do not alter grouping parameters on rerun.
- STUDY-001 — Report MS-SSIM/MAE deltas across PtychoPINN, Baseline, and PtyChi with citation links in summary.md.

Pointers:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:700
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py:1
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py:1
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T115706Z/phase_g_dense_full_execution_real_run/summary/summary.md:1
- docs/TESTING_GUIDE.md:333
- docs/findings.md:42

Next Up (optional):
- If dense view succeeds quickly, queue the sparse view rerun under the same workflow.

Doc Sync Plan (Conditional):
- Not applicable; mapped pytest selector unchanged.

Mapped Tests Guardrail:
- pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest --collect-only -vv must continue to collect; resolve collection issues immediately.

Hard Gate:
- Do not mark complete until the orchestrator log shows `[8/8]`, all Phase D–G artifacts exist, highlights consistency passes, metrics_digest.md refreshed, summary/fix_plan updated with MS-SSIM/MAE deltas, and the mapped pytest selector passes with a new post-run log in `green/`.
