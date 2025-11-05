Summary: Automate Phase G delta highlights and run the dense pipeline for real MS-SSIM/MAE evidence with provenance intact.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Mapped tests:
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands -vv
  - pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_success_digest -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T150500Z/phase_g_dense_full_execution_real_run/

Do Now:
- Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
- Implement: tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest — demand an auto-generated metrics_delta_highlights.txt with four signed delta lines before touching the orchestrator.
- Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main — emit metrics_delta_highlights.txt alongside the delta JSON and mention it in the success banner without breaking provenance metadata.
- Document: docs/TESTING_GUIDE.md — record the highlights artifact and verification steps for dense Phase G runs.
- Validate: pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
- Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T150500Z/phase_g_dense_full_execution_real_run/

How-To Map:
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
2. export HUB=plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T150500Z/phase_g_dense_full_execution_real_run
3. mkdir -p "$HUB"/{plan,summary,red,green,collect,cli,analysis}
4. Update tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest to assert metrics_delta_highlights.txt exists with the four expected delta lines (signed, 3-decimal) and is referenced in stdout.
5. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/red/pytest_orchestrator_delta_highlights_red.log  # expect RED until orchestrator writes highlights.
6. Modify plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main to write "$HUB"/analysis/metrics_delta_highlights.txt during delta computation and include the relative path in the success banner.
7. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/green/pytest_orchestrator_delta_highlights_green.log
8. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands -vv | tee "$HUB"/green/pytest_collect_only.log
9. pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_success_digest -vv | tee "$HUB"/green/pytest_analyze_success.log
10. Edit docs/TESTING_GUIDE.md Phase G section to describe metrics_delta_highlights.txt and verification steps; rerun Step 9 if doc formatting impacts behavior.
11. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense.log
12. test -f "$HUB"/analysis/metrics_summary.json && test -f "$HUB"/analysis/metrics_delta_summary.json && test -f "$HUB"/analysis/metrics_delta_highlights.txt && test -f "$HUB"/analysis/metrics_digest.md
13. cat "$HUB"/analysis/metrics_delta_highlights.txt | tee "$HUB"/analysis/metrics_delta_highlights_preview.txt
14. find "$HUB" -maxdepth 3 -type f | sort > "$HUB"/analysis/artifact_inventory.txt
15. Update "$HUB"/summary/summary.md with RED→GREEN test logs, MS-SSIM/MAE deltas, metadata confirmation, and artifact links; mirror the deltas, highlights path, and provenance in docs/fix_plan.md Attempts History.

Pitfalls To Avoid:
- Keep AUTHORITATIVE_CMDS_DOC exported before every orchestrator invocation (CONFIG-001).
- Preserve metrics_delta_summary.json structure and provenance fields; highlights file must not mutate the JSON payload.
- Serialize highlights paths relative to the hub (TYPE-PATH-001); no absolute paths in stdout or artifacts.
- Maintain the delta line format (+/-0.000) and ordering so existing digest parsing stays valid.
- Ensure the highlights file is regenerated on each run (overwrite, no stale content) and captured under analysis/.
- Do not touch core physics/model modules or non-study initiatives while implementing.
- Capture CLI/test logs under the hub; nothing should be written to repo root or /tmp beyond ephemeral scratch (delete before commit).

If Blocked:
- Save failing pytest output to "$HUB"/red/pytest_orchestrator_delta_highlights_red.log and summarize the assertion gap in summary.md plus docs/fix_plan.md.
- On pipeline failure, keep "$HUB"/cli/run_phase_g_dense.log, stash any partial artifacts under "$HUB"/analysis/, and document the failure (command, error excerpt) in summary.md and docs/fix_plan.md with next unblock steps.
- If highlights content cannot be generated due to missing metrics, dump the JSON payload to "$HUB"/analysis/blocker_metrics_summary.json and mark the ledger entry blocked.

Findings Applied (Mandatory):
- POLICY-001 — PyTorch dependency is mandatory; pipeline assumes torch>=2.2 available.
- CONFIG-001 — Enforce `update_legacy_dict` via AUTHORITATIVE_CMDS_DOC export before orchestrator commands.
- DATA-001 — metrics JSON artifacts must remain compliant with documented schema.
- TYPE-PATH-001 — New highlights path and banner output must use relative POSIX serialization.
- OVERSAMPLING-001 — Dense configuration parameters remain unchanged during evidence run.
- STUDY-001 — Record MS-SSIM/MAE deltas comparing PtychoPINN vs Baseline/PtyChi for fly64 dense view.

Pointers:
- tests/study/test_phase_g_dense_orchestrator.py:856 — Exec-mode test to extend with highlights assertions.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:830 — Delta computation block where highlights persistence should live.
- docs/TESTING_GUIDE.md:331 — Phase G delta persistence documentation to refresh with highlights artifact guidance.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py:1 — Reference digest expectations when validating highlight content.
- docs/findings.md:8 — Findings POLICY-001 / CONFIG-001 / DATA-001 / TYPE-PATH-001 / OVERSAMPLING-001 / STUDY-001 governing this work.

Next Up (optional):
- After dense evidence lands, schedule the sparse view pipeline run to complete Phase G coverage.

Doc Sync Plan (Conditional):
- Not required; selectors unchanged, only strengthened assertions.

Mapped Tests Guardrail:
- pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest --collect-only -vv must collect (>0); fix immediately if it drops.

Hard Gate:
- Do not finish until metrics_delta_highlights.txt is generated automatically (and archived), provenance metadata remains correct, all mapped selectors pass with logs captured, the dense pipeline exits 0 with refreshed artifacts, and docs/fix_plan.md plus summary.md include MS-SSIM/MAE deltas and highlight references for traceability.
