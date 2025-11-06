Summary: Extend the dense Phase G verifier for delta artifacts and rerun the dense pipeline to capture MS-SSIM/MAE evidence with full provenance.
Mode: Perf
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Mapped tests:
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
  - pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_success_digest -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T095003Z/phase_g_dense_full_execution_real_run/

Do Now:
- Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
- Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py::main — add enforcement for `metrics_delta_summary.json` (generated_at/source_metrics fields and numeric delta keys) plus four-line `metrics_delta_highlights.txt` validation while retaining Phase C compliance checks.
- Execute: python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$RUN_LOG" (expect clean `[1/8]`→`[8/8]` with no blockers).
- Validate: pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/green/pytest_orchestrator_dense_exec_$(date -u +%Y-%m-%dT%H%M%SZ).log && pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_success_digest -vv | tee "$HUB"/green/pytest_metrics_digest_success_$(date -u +%Y-%m-%dT%H%M%SZ).log && python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py --hub "$HUB" --report "$HUB"/analysis/pipeline_verification.json | tee "$HUB"/analysis/pipeline_verification_$(date -u +%Y-%m-%dT%H%M%SZ).log
- Document: Refresh `$HUB`/summary/summary.md, docs/fix_plan.md, and docs/findings.md (if new lessons) with MS-SSIM/MAE deltas, metadata compliance status, delta provenance, and artifact inventory references.
- Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T095003Z/phase_g_dense_full_execution_real_run/

How-To Map:
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
2. export HUB="$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T095003Z/phase_g_dense_full_execution_real_run"
3. mkdir -p "$HUB"/{analysis,cli,collect,green,red,summary}
4. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest --collect-only -q
5. pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_success_digest --collect-only -q
6. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
7. pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_success_digest -vv
8. Implement verify_dense_pipeline_artifacts.py delta checks (ensure `generated_at` ISO-8601 UTC, `source_metrics` points to an existing file within `$HUB`, and both vs_Baseline/vs_PtyChi MS-SSIM/MAE deltas include amplitude/phase pairs).
9. RUN_LOG="$HUB"/cli/run_phase_g_dense_dense_view_$(date -u +%Y-%m-%dT%H%M%SZ).log
10. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$RUN_LOG"
11. rg "\[[1-8]/8\]" "$RUN_LOG"
12. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py --hub "$HUB" --report "$HUB"/analysis/pipeline_verification.json | tee "$HUB"/analysis/pipeline_verification_$(date -u +%Y-%m-%dT%H%M%SZ).log
13. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py --metrics "$HUB"/analysis/metrics_summary.json --highlights "$HUB"/analysis/aggregate_highlights.txt --output "$HUB"/analysis/metrics_digest.md | tee "$HUB"/analysis/metrics_digest_refresh_$(date -u +%Y-%m-%dT%H%M%SZ).log
14. python -m json.tool "$HUB"/analysis/metrics_delta_summary.json > "$HUB"/analysis/metrics_delta_summary_pretty.json
15. find "$HUB" -maxdepth 4 -type f | sort > "$HUB"/analysis/artifact_inventory_$(date -u +%Y-%m-%dT%H%M%SZ).txt
16. Update "$HUB"/summary/summary.md, docs/fix_plan.md, and docs/findings.md (if applicable) with deltas/compliance status; stage only initiative + doc changes.

Pitfalls To Avoid:
- Do not reuse prior hubs; all fresh evidence must land under 2025-11-06T095003Z.
- Keep GPU resources idle while the dense run executes to avoid CUDA contention.
- Treat any missing `[x/8]` progress or non-zero exit as a blocker—capture logs and stop.
- Ensure the delta JSON includes `generated_at` UTC and `source_metrics` relative to the hub; missing keys must fail the verifier.
- Keep Phase C NPZ `_metadata` intact; never delete or regenerate data/phase_c manually.
- Delta highlights must list exactly four lines (MS-SSIM/MAE vs Baseline/PtyChi); gaps signal analyzer failure.
- Archive all pytest/verifier logs with UTC timestamps under the correct subdirectories.
- Do not modify production modules outside the initiative scripts/docs scope.
- Run the verifier only after the pipeline completes to avoid false negatives.

If Blocked:
- Capture the failing command, exit code, and tail -n 40 of the relevant log into "$HUB"/analysis/blocker_$(date -u +%Y%m%dT%H%M%SZ).log.
- Update "$HUB"/summary/summary.md and docs/fix_plan.md marking the attempt `blocked`, citing the blocker log.
- Stop reruns; await supervisor guidance with logs attached.

Findings Applied (Mandatory):
- POLICY-001 — Honor PyTorch dependency expectations when calling verifier/analyzer helpers.
- CONFIG-001 — Ensure CONFIG-001 bridge runs before any legacy consumers during pipeline execution.
- DATA-001 — Verify NPZ key/dtype contracts when checking Phase C + Phase G artifacts.
- STUDY-001 — Record MS-SSIM/MAE deltas against Baseline and PtyChi with provenance in summary/docs.
- OVERSAMPLING-001 — Preserve dense overlap parameters; do not alter grouping constants.
- TYPE-PATH-001 — Normalize hub-relative paths and enforce UTC timestamps in generated artifacts.
- PHASEC-METADATA-001 — Confirm metadata compliance flags propagate through metrics_summary and verifier checks.

Pointers:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py:1
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:930
- tests/study/test_phase_g_dense_orchestrator.py:856
- tests/study/test_phase_g_dense_metrics_report.py:299
- docs/findings.md:22
- docs/TESTING_GUIDE.md:331

Next Up (optional):
- Draft sparse-view rerun checklist once dense evidence is captured.

Doc Sync Plan (Conditional):
- None — no new or renamed pytest selectors expected this loop.

Mapped Tests Guardrail:
- Steps 4-5 ensure both selectors collect (>0) before execution.

Hard Gate:
- Do not mark complete until `[8/8]` appears in the run log, the enhanced verifier passes, metrics bundle + delta artifacts populate analysis/, MS-SSIM/MAE deltas are documented in summary/docs, and both mapped pytest logs are archived under `$HUB`/green/.
