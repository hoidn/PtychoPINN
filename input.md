Summary: Extend the dense Phase C→G pipeline run so metadata compliance is captured in summaries while gathering fresh MS-SSIM/MAE evidence.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Mapped tests:
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_summarize_phase_g_outputs -vv
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T084736Z/phase_g_dense_post_metadata_fix/

Do Now:
- Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
- Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::summarize_phase_g_outputs — persist Phase C metadata compliance (per dose + split) in metrics_summary.json/metrics_summary.md and update tests/study/test_phase_g_dense_orchestrator.py::test_summarize_phase_g_outputs to assert the new field.
- Execute: python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$RUN_LOG" (expect `[1/8]`→`[8/8]` with no blocker).
- Document: Update "$HUB"/summary/summary.md plus docs/fix_plan.md with MS-SSIM/MAE deltas, metadata compliance summary, highlights/digest provenance, and new artifact inventory.
- Validate: pytest tests/study/test_phase_g_dense_orchestrator.py::test_summarize_phase_g_outputs -vv | tee "$HUB"/green/pytest_summarize_phase_g_outputs_$(date -u +%Y-%m-%dT%H%M%SZ).log && pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/green/pytest_phase_g_dense_exec_$(date -u +%Y-%m-%dT%H%M%SZ).log
- Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T084736Z/phase_g_dense_post_metadata_fix/

How-To Map:
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
2. export HUB="$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T084736Z/phase_g_dense_post_metadata_fix"
3. mkdir -p "$HUB"/summary "$HUB"/cli "$HUB"/analysis "$HUB"/green
4. pytest tests/study/test_phase_g_dense_orchestrator.py::test_summarize_phase_g_outputs --collect-only -q
5. pytest tests/study/test_phase_g_dense_orchestrator.py::test_summarize_phase_g_outputs -vv || true  # establish RED before edits
6. Implement metadata summary extension + test assertion (edit files noted above)
7. pytest tests/study/test_phase_g_dense_orchestrator.py::test_summarize_phase_g_outputs -vv
8. RUN_LOG="$HUB"/cli/run_phase_g_dense_post_metadata_fix_$(date -u +%Y-%m-%dT%H%M%SZ).log
9. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$RUN_LOG"
10. rg "\[[1-8]/8\]" "$RUN_LOG"
11. for dose in 1000 10000 100000; do for split in train test; do test -f "$HUB"/data/phase_c/dose_${dose}/patched_${split}.npz || { echo "Missing Phase C ${dose} ${split}"; exit 1; }; done; done
12. for f in metrics_summary.json metrics_delta_summary.json metrics_delta_highlights.txt metrics_delta_highlights_preview.txt aggregate_highlights.txt aggregate_report.md metrics_digest.md; do test -f "$HUB"/analysis/"$f" || { echo "Missing $f"; exit 1; }; done
13. rg '"phase_c_metadata' "$HUB"/analysis/metrics_summary.json
14. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py --hub "$HUB" | tee "$HUB"/analysis/highlights_consistency_check_$(date -u +%Y-%m-%dT%H%M%SZ).log
15. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py --metrics "$HUB"/analysis/metrics_summary.json --highlights "$HUB"/analysis/aggregate_highlights.txt --output "$HUB"/analysis/metrics_digest.md | tee "$HUB"/analysis/metrics_digest_refresh_$(date -u +%Y-%m-%dT%H%M%SZ).log
16. find "$HUB" -maxdepth 3 -type f | sort > "$HUB"/analysis/artifact_inventory_$(date -u +%Y-%m-%dT%H%M%SZ).txt
17. pytest tests/study/test_phase_g_dense_orchestrator.py::test_summarize_phase_g_outputs -vv | tee "$HUB"/green/pytest_summarize_phase_g_outputs_final_$(date -u +%Y-%m-%dT%H%M%SZ).log
18. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/green/pytest_phase_g_dense_exec_final_$(date -u +%Y-%m-%dT%H%M%SZ).log
19. Edit "$HUB"/summary/summary.md and docs/fix_plan.md with metrics deltas, metadata compliance evidence, highlight digest provenance, pytest log names, and artifact inventory reference.

Pitfalls To Avoid:
- Do not reuse the 2025-11-05 hub; new run must land under the 2025-11-06T084736Z directory to keep provenance clean.
- Keep all helper invocations Path-normalized (`$HUB` absolute); avoid relative hub paths inside scripts.
- Ensure metadata compliance summary covers every dose×split pair; no partial lists or silent skips.
- Do not relax highlight consistency guard—treat any mismatch as a blocker and capture logs.
- Avoid editing production modules outside plans/active; scope changes to orchestrator helpers and tests only.
- Preserve `_metadata` in patched NPZ files; never delete or mutate Phase C outputs manually.
- Capture pytest logs under `$HUB`/green/ with UTC timestamps to satisfy execution proof requirements.
- Include MS-SSIM + MAE deltas versus both Baseline and PtyChi in the summary update per STUDY-001.
- Keep GPU free for the pipeline run; delay other GPU-heavy jobs until `[8/8]` completes.
- Record any failures immediately in blocker.log before rerunning commands.

If Blocked:
- Append the failing command, exit code, and last 40 log lines to "$HUB"/analysis/blocker_$(date -u +%Y%m%dT%H%M%SZ).log.
- Update "$HUB"/summary/summary.md and docs/fix_plan.md with blocker details, marking the attempt `blocked` and linking the log.
- Stop further reruns; wait for supervisor guidance with the blocker evidence ready.

Findings Applied (Mandatory):
- POLICY-001 (docs/findings.md:8) — Maintain PyTorch dependency assumptions while running the orchestrator helpers.
- CONFIG-001 (docs/findings.md:10) — Do not disturb CONFIG-001 bridging; metadata summary must assume params.cfg already synced.
- DATA-001 (docs/findings.md:14) — Metadata compliance check must respect the DATA-001 NPZ contract.
- STUDY-001 (docs/findings.md:16) — Summaries must report MS-SSIM/MAE deltas against Baseline and PtyChi with provenance.
- OVERSAMPLING-001 (docs/findings.md:17) — Leave dense overlap parameters untouched during rerun.
- TYPE-PATH-001 (docs/findings.md:21) — Normalize hub paths before filesystem operations.
- PHASEC-METADATA-001 (docs/findings.md:22) — Capture the new patched layout metadata check in summaries to prevent regressions.

Pointers:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:283
- tests/study/test_phase_g_dense_orchestrator.py:109
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T081826Z/phase_c_metadata_guard_blocker/summary/summary.md:1
- docs/findings.md:22
- docs/TESTING_GUIDE.md:333

Next Up (optional):
- Stage sparse-view rerun checklist once dense pipeline artifacts are published.

Doc Sync Plan (Conditional):
- After the new metadata summary assertion passes, keep selectors unchanged; no registry updates required beyond artifact logging.

Mapped Tests Guardrail:
- Confirm both mapped pytest selectors collect exactly one test with `--collect-only` before final validation.

Hard Gate:
- Do not mark the focus complete until `[8/8]` dense pipeline evidence, metadata compliance summary, highlights consistency, MS-SSIM/MAE deltas, and both mapped pytest logs are archived under the new hub path.
