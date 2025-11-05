Summary: Add provenance metadata to the Phase G delta JSON and run the dense pipeline to capture real MS-SSIM/MAE evidence.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Mapped tests:
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands -vv
  - pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_success_digest -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T130500Z/phase_g_dense_full_execution_real_run/

Do Now:
- Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
- Implement: tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest — assert metadata fields (`generated_at`, `source_metrics`) inside metrics_delta_summary.json before touching the orchestrator.
- Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main — persist the metadata fields when writing metrics_delta_summary.json, keeping TYPE-PATH-001 compliance.
- Document: docs/TESTING_GUIDE.md — describe the enriched metrics_delta_summary.json schema and provenance checks.
- Validate: pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
- Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T130500Z/phase_g_dense_full_execution_real_run/

How-To Map:
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
2. export HUB=plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T130500Z/phase_g_dense_full_execution_real_run
3. mkdir -p "$HUB"/{plan,summary,red,green,collect,cli,analysis}
4. Update tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest to seed a stale metrics_delta_summary.json, require `generated_at` + `source_metrics`, and expect overwrite.
5. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/red/pytest_orchestrator_delta_metadata_red.log  # expect failure until orchestrator emits metadata.
6. Modify plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main to add metadata (UTC timestamp + relative source path) when writing metrics_delta_summary.json and to unlink stale files when metrics_summary generation fails.
7. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/green/pytest_orchestrator_delta_metadata_green.log
8. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands -vv | tee "$HUB"/green/pytest_collect_only.log
9. pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_success_digest -vv | tee "$HUB"/green/pytest_analyze_success.log
10. Edit docs/TESTING_GUIDE.md Phase G section with metadata schema; rerun step 9 if doctest linting touches behaviors.
11. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense.log
12. test -f "$HUB"/analysis/metrics_summary.json && test -f "$HUB"/analysis/metrics_delta_summary.json && test -f "$HUB"/analysis/metrics_digest.md
13. python -m json.tool "$HUB"/analysis/metrics_delta_summary.json > "$HUB"/analysis/metrics_delta_summary.pretty.json
14. rg "\"ms_ssim\"" -A2 "$HUB"/analysis/metrics_delta_summary.pretty.json > "$HUB"/analysis/metrics_delta_highlights.txt && rg "\"mae\"" -A2 "$HUB"/analysis/metrics_delta_summary.pretty.json >> "$HUB"/analysis/metrics_delta_highlights.txt
15. find "$HUB" -maxdepth 3 -type f | sort > "$HUB"/analysis/artifact_inventory.txt
16. Update "$HUB"/summary/summary.md with test pass counts, metadata fields, MS-SSIM/MAE deltas, and artifact links; mirror the deltas + provenance in docs/fix_plan.md Attempts History.

Pitfalls To Avoid:
- Keep AUTHORITATIVE_CMDS_DOC exported before every pytest/CLI invocation (CONFIG-001).
- Do not reformat existing delta stdout strings; metadata enrichments must not alter user-facing deltas.
- Generate metadata using UTC (`timezone.utc`) to avoid local skew and ensure deterministic tests.
- Serialize `source_metrics` as a relative POSIX path (TYPE-PATH-001); no absolute paths in JSON.
- Ensure the stale JSON unlink only fires when metrics_summary generation fails; real runs must retain new data.
- Capture pipeline logs under `$HUB/cli/`; nothing leaks to repo root.
- Avoid editing core physics/model modules (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`).
- Leave reporting helper command order untouched so tests remain stable.

If Blocked:
- Archive RED pytest output under `$HUB/red/` and describe assertion deltas in summary.md plus docs/fix_plan.md.
- If metadata fields remain missing after implementation, dump the offending JSON to `$HUB/analysis/blocker_metrics_delta.json` and mark the ledger entry as blocked with details.
- On pipeline crash, stop immediately, keep `$HUB/cli/run_phase_g_dense.log`, and record the failing command + stack trace in `$HUB/analysis/blocker.txt` and docs/fix_plan.md Attempts History.

Findings Applied (Mandatory):
- POLICY-001 — PyTorch dependency enforced; dense pipeline assumes torch available.
- CONFIG-001 — AUTHORITATIVE_CMDS_DOC export precedes orchestrator invocations to sync params.cfg bridging.
- DATA-001 — metrics_summary.json and enriched metrics_delta_summary.json must follow documented schema contracts.
- TYPE-PATH-001 — All new paths (metadata, banner, highlights) must use pathlib with forward slashes and relative serialization.
- OVERSAMPLING-001 — Dense configuration remains unchanged; no edits to gridsize/grouping during evidence run.
- STUDY-001 — Record MS-SSIM/MAE deltas comparing PtychoPINN vs Baseline/PtyChi for fly64 dense study evidence.

Pointers:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:881 — Delta JSON persistence block to extend with metadata.
- tests/study/test_phase_g_dense_orchestrator.py:1070 — Exec-mode assertions to enhance with metadata expectations.
- docs/TESTING_GUIDE.md:331 — Phase G delta summary documentation to refresh for new schema.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py:1 — Reference for digest expectations when summarizing pipeline output.
- docs/findings.md:8 — Findings POLICY-001 / CONFIG-001 / DATA-001 / TYPE-PATH-001 / OVERSAMPLING-001 / STUDY-001 governing this work.

Next Up (optional):
- Execute the sparse view Phase G pipeline once dense evidence and metadata verification land.

Doc Sync Plan (Conditional):
- Not required; no new tests added or renamed this loop.

Mapped Tests Guardrail:
- pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest --collect-only -vv must collect (>0); fix selector issues immediately if it drops.

Hard Gate:
- Do not end the loop until metadata-enhanced metrics_delta_summary.json is generated, success banner surfaces the JSON path, all mapped selectors pass with logs archived, the dense pipeline exits 0 with fresh artifacts, and docs/fix_plan.md plus summary.md capture MS-SSIM/MAE deltas, metadata fields, and artifact locations.
