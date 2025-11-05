Summary: Persist the Phase G delta metrics to a JSON artifact and run the dense pipeline to capture real MS-SSIM/MAE evidence.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Mapped tests:
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands -vv
  - pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_success_digest -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T110500Z/phase_g_dense_full_execution_real_run/

Do Now:
- Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
- Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main — persist the computed MS-SSIM/MAE deltas to analysis/metrics_delta_summary.json and print the saved path in the success banner.
- Implement: tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest — assert metrics_delta_summary.json creation and validate its contents while keeping existing delta stdout checks.
- Validate: pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
- Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T110500Z/phase_g_dense_full_execution_real_run/

How-To Map:
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
2. export HUB=plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T110500Z/phase_g_dense_full_execution_real_run
3. mkdir -p "$HUB"/{plan,summary,red,green,collect,cli,analysis}
4. Update tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest to require metrics_delta_summary.json (file existence + JSON keys) before touching run_phase_g_dense.py.
5. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/red/pytest_orchestrator_delta_json_red.log  # expect failure until JSON persistence lands
6. Modify plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main to write metrics_delta_summary.json under analysis/, include both Baseline and PtyChi deltas for MS-SSIM/MAE, use null for missing values, and append a success-banner line with the JSON path.
7. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/green/pytest_orchestrator_delta_json_green.log
8. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands -vv | tee "$HUB"/green/pytest_collect_only.log
9. pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_success_digest -vv | tee "$HUB"/green/pytest_analyze_success.log
10. Update docs/TESTING_GUIDE.md Phase G section to mention metrics_delta_summary.json (location, schema basics, usage); rerun `pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_success_digest -vv` if wording adjustments affect docs gating.
11. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense.log
12. test -f "$HUB"/analysis/metrics_summary.json && test -f "$HUB"/analysis/metrics_delta_summary.json && test -f "$HUB"/analysis/metrics_digest.md
13. python -m json.tool "$HUB"/analysis/metrics_delta_summary.json > "$HUB"/analysis/metrics_delta_summary.pretty.json
14. rg "ms_ssim" "$HUB"/analysis/metrics_delta_summary.pretty.json | tee "$HUB"/analysis/metrics_delta_highlights.txt
15. find "$HUB" -maxdepth 3 -type f | sort > "$HUB"/analysis/artifact_inventory.txt
16. Update "$HUB"/summary/summary.md with test pass counts, delta values from metrics_delta_summary.json, and artifact links; capture the same MS-SSIM/MAE figures in docs/fix_plan.md Attempts History.

Pitfalls To Avoid:
- Do not change the existing 3-decimal formatting for stdout deltas; JSON should store raw floats, not formatted strings.
- Keep Path handling TYPE-PATH-001 compliant—no os.path.join or bare strings for analysis paths.
- Ensure the new success-banner line references the JSON path relative to the hub so logs stay portable.
- Avoid writing JSON when metrics_summary.json is missing; emit a warning and skip to keep pipeline resilient.
- Massive CLI logs belong under "$HUB"/cli/, not the repo root.
- Preserve AUTHORITATIVE_CMDS_DOC in the environment before every pytest or CLI call.
- Do not modify core TensorFlow/Torch modules (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`).
- Leave aggregate_report.md generation flow untouched; the new JSON must be additive, not replace highlights/digest outputs.

If Blocked:
- Archive failing pytest output under "$HUB"/red/" and document the assertion mismatch in summary.md plus docs/fix_plan.md.
- If metrics_delta_summary.json remains empty or malformed after the pipeline run, keep the CLI log, capture the JSON contents into analysis/blocker.json, and mark the attempt blocked with details in summary + ledger.
- On pipeline crash, stop immediately, retain "$HUB"/cli/run_phase_g_dense.log", and note the failing command and stack trace in analysis/blocker.txt and docs/fix_plan.md Attempts History.

Findings Applied (Mandatory):
- POLICY-001 — PyTorch remains available for backend helpers during Phase G comparisons.
- CONFIG-001 — AUTHORITATIVE_CMDS_DOC export precedes orchestrator commands to sync legacy config bridges.
- DATA-001 — metrics_summary.json/metrics_delta_summary.json must respect documented aggregate schema.
- TYPE-PATH-001 — All new paths (JSON, banner) rely on pathlib with forward slashes.
- OVERSAMPLING-001 — Dense configuration stays unchanged while running the pipeline.
- STUDY-001 — Record MS-SSIM/MAE deltas to contrast PtychoPINN with Baseline/PtyChi for the fly64 synthetic study.

Pointers:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:824 — Current delta stdout helper location to extend with JSON persistence.
- tests/study/test_phase_g_dense_orchestrator.py:880 — Exec-mode test ready for JSON assertions.
- docs/TESTING_GUIDE.md:300 — Phase G workflow documentation to update with the new artifact.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py:1 — Digest behavior reference for evidence expectations.
- docs/findings.md:9 — Active findings covering POLICY-001 / CONFIG-001 / DATA-001 / TYPE-PATH-001 / OVERSAMPLING-001 / STUDY-001.

Next Up (optional):
- Run the sparse view Phase G pipeline once dense evidence is archived.

Mapped Tests Guardrail:
- pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest --collect-only -vv must report the test; fix selector collection before completing the loop if it drops to zero.

Hard Gate:
- Do not close the loop until metrics_delta_summary.json is generated with numeric deltas, the success banner surfaces its path, all mapped selectors pass with logs archived, the dense pipeline exits 0 with fresh metrics artifacts, and docs/fix_plan.md plus summary.md capture the MS-SSIM/MAE figures and artifact locations.
