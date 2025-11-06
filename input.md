Summary: Gather end-to-end dense Phase C→G evidence and capture MS-SSIM/MAE + metadata compliance results for the fly64 dose study.
Mode: Perf
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Mapped tests:
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
  - pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_success_digest -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T091223Z/phase_g_dense_full_execution_real_run/

Do Now:
- Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
- Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py::main — add initiative-scoped checker that validates the post-run metrics bundle (summary/delta/highlights/digest) and enforces Phase C metadata compliance flags before documentation.
- Execute: python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$RUN_LOG" (expect clean `[1/8]`→`[8/8]` sequence with no blockers).
- Document: Update "$HUB"/summary/summary.md and docs/fix_plan.md with MS-SSIM/MAE deltas, metadata compliance outcomes, highlights digest provenance, checker outputs, and artifact inventory references.
- Validate: pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/green/pytest_orchestrator_dense_exec_$(date -u +%Y-%m-%dT%H%M%SZ).log && pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_success_digest -vv | tee "$HUB"/green/pytest_metrics_digest_success_$(date -u +%Y-%m-%dT%H%M%SZ).log && python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py --hub "$HUB" --report "$HUB"/analysis/pipeline_verification.json | tee "$HUB"/analysis/pipeline_verification_$(date -u +%Y-%m-%dT%H%M%SZ).log
- Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T091223Z/phase_g_dense_full_execution_real_run/

How-To Map:
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
2. export HUB="$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T091223Z/phase_g_dense_full_execution_real_run"
3. mkdir -p "$HUB"/{analysis,cli,collect,green,red,summary}
4. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest --collect-only -q
5. pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_success_digest --collect-only -q
6. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
7. pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_success_digest -vv
8. Implement verify_dense_pipeline_artifacts.py (script builds success/failure summary for metrics bundle, metadata compliance flags, expected file roster)
9. RUN_LOG="$HUB"/cli/run_phase_g_dense_dense_view_$(date -u +%Y-%m-%dT%H%M%SZ).log
10. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$RUN_LOG"
11. rg "\[[1-8]/8\]" "$RUN_LOG"
12. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py --hub "$HUB" --report "$HUB"/analysis/pipeline_verification.json | tee "$HUB"/analysis/pipeline_verification_$(date -u +%Y-%m-%dT%H%M%SZ).log
13. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py --metrics "$HUB"/analysis/metrics_summary.json --highlights "$HUB"/analysis/aggregate_highlights.txt --output "$HUB"/analysis/metrics_digest.md | tee "$HUB"/analysis/metrics_digest_refresh_$(date -u +%Y-%m-%dT%H%M%SZ).log
14. find "$HUB" -maxdepth 4 -type f | sort > "$HUB"/analysis/artifact_inventory_$(date -u +%Y-%m-%dT%H%M%SZ).txt
15. Update "$HUB"/summary/summary.md, docs/fix_plan.md, and docs/findings.md if new durable lessons emerge.
16. git status -sb (ensure only initiative + doc updates staged before handoff)

Pitfalls To Avoid:
- Do not reuse prior hubs; all fresh evidence must land under 2025-11-06T091223Z with absolute `$HUB`.
- Keep GPU free while the long run executes; avoid concurrent CUDA/TensorFlow jobs.
- Treat any `[x/8]` stall or non-zero return as a blocker—capture logs and stop retries.
- Ensure metadata compliance table lists all dose×split combos; missing splits indicate hub corruption.
- Do not edit production modules; confine code changes to initiative scripts and docs.
- Preserve existing `_metadata` inside Phase C NPZ files; never nuke data/phase_c by hand.
- Capture pytest and checker logs with UTC timestamps beneath `$HUB`/green or `$HUB`/analysis respectively.
- Run verify script only after pipeline completes; earlier invocation will fail due to missing files.
- Keep delta JSON/highlights in sync with digest; rerun analyzer if the pipeline regenerates them.

If Blocked:
- Capture the failing command, exit code, and tail -n 40 of relevant log into "$HUB"/analysis/blocker_$(date -u +%Y%m%dT%H%M%SZ).log.
- Update "$HUB"/summary/summary.md and docs/fix_plan.md marking the attempt `blocked`, citing the blocker log.
- Stop reruns; wait for supervisor direction with logs attached.

Findings Applied (Mandatory):
- POLICY-001 (docs/findings.md:8) — honor PyTorch dependency assumptions while orchestrating analysis helpers.
- CONFIG-001 (docs/findings.md:10) — ensure CONFIG-001 bridge remains intact before Phase C→G stages consume params.cfg.
- DATA-001 (docs/findings.md:14) — enforce NPZ key/dtype contracts when verifying Phase C + Phase G artifacts.
- STUDY-001 (docs/findings.md:16) — report MS-SSIM/MAE deltas against Baseline and PtyChi with provenance in summary/docs.
- OVERSAMPLING-001 (docs/findings.md:17) — leave dense overlap parameters untouched during rerun.
- TYPE-PATH-001 (docs/findings.md:21) — normalize hub-relative paths in scripts and logs.
- PHASEC-METADATA-001 (docs/findings.md:22) — confirm metadata compliance flagging for patched Phase C outputs.

Pointers:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:820
- tests/study/test_phase_g_dense_orchestrator.py:856
- tests/study/test_phase_g_dense_metrics_report.py:299
- docs/findings.md:22
- docs/TESTING_GUIDE.md:331

Next Up (optional):
- Prepare sparse-view rerun checklist once dense evidence is published.

Doc Sync Plan (Conditional):
- None — no new or renamed pytest selectors expected this loop.

Mapped Tests Guardrail:
- Steps 4-5 ensure both mapped selectors collect (>0) via --collect-only before execution.

Hard Gate:
- Do not consider the focus complete until `[8/8]` appears in the run log, verify script passes, metrics bundle + metadata compliance artifacts populate analysis/, MS-SSIM/MAE deltas are documented in summary/docs, and both mapped pytest logs are archived under `$HUB`/green/.
