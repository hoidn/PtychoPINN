Summary: Surface the Phase G metrics digest paths in the orchestrator success banner and generate fresh dense run evidence with the new digest automation.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Mapped tests:
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands -vv
  - pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_success_digest -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T070500Z/phase_g_dense_full_execution_real_run/

Do Now:
- Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
- Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main (surface metrics_digest.md + cli/metrics_digest_cli.log in the success banner after adding a failing assertion to the orchestrator exec test)
- Validate: pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
- Validate: pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands -vv
- Validate: pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_success_digest -vv
- Execute: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T070500Z/phase_g_dense_full_execution_real_run --dose 1000 --view dense --splits train test --clobber | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T070500Z/phase_g_dense_full_execution_real_run/cli/run_phase_g_dense_cli.log
- Verify: tail -n 40 plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T070500Z/phase_g_dense_full_execution_real_run/cli/run_phase_g_dense_cli.log > plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T070500Z/phase_g_dense_full_execution_real_run/analysis/cli_tail.txt && find plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T070500Z/phase_g_dense_full_execution_real_run -maxdepth 3 -type f | sort > plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T070500Z/phase_g_dense_full_execution_real_run/analysis/artifact_inventory.txt
- Update: Prepend metrics deltas + artifact links to plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T070500Z/phase_g_dense_full_execution_real_run/summary/summary.md, refresh docs/TESTING_GUIDE.md §2.5 and docs/development/TEST_SUITE_INDEX.md if selector inventory changes, and log Attempt in docs/fix_plan.md
- Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T070500Z/phase_g_dense_full_execution_real_run/

How-To Map:
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
2. export HUB=plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T070500Z/phase_g_dense_full_execution_real_run
3. mkdir -p "$HUB"/{plan,collect,red,green,cli,analysis,summary}
4. Edit tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest to assert stdout contains both "Metrics digest (Markdown):" and "Metrics digest log:" (expect RED) and run pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/red/pytest_digest_exec.log
5. Run pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands -vv | tee "$HUB"/red/pytest_collect_only.log  # should still pass; keep log for regression guard
6. Update plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main so the success banner prints "Metrics digest (Markdown): {metrics_digest_md}" and "Metrics digest log: {analyze_digest_log}" via pathlib string conversion
7. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/green/pytest_digest_exec.log
8. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands -vv | tee "$HUB"/green/pytest_collect_only.log
9. pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_success_digest -vv | tee "$HUB"/green/pytest_analyze_success.log
10. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber | tee "$HUB"/cli/run_phase_g_dense_cli.log
11. tail -n 40 "$HUB"/cli/run_phase_g_dense_cli.log > "$HUB"/analysis/cli_tail.txt
12. find "$HUB" -maxdepth 3 -type f | sort > "$HUB"/analysis/artifact_inventory.txt && cp "$HUB"/analysis/metrics_digest.md "$HUB"/analysis/metrics_digest_preview.md
13. Prepend the Turn Summary + metrics deltas to "$HUB"/summary/summary.md and update docs/fix_plan.md Attempt entry; attach pytest collect-only log if selector inventory changed before updating docs/TESTING_GUIDE.md §2.5 and docs/development/TEST_SUITE_INDEX.md
14. git status --short > "$HUB"/summary/git_status.txt

Pitfalls To Avoid:
- Forgetting to capture RED evidence before touching run_phase_g_dense.py breaks TDD requirements.
- Running pytest without AUTHORITATIVE_CMDS_DOC causes CONFIG-001 guard failures.
- Editing collect-only output formatting beyond added banner lines risks breaking existing command order assertions.
- Allowing run_phase_g_dense.py to print Path objects without str() conversion violates TYPE-PATH-001 expectations in tests.
- Skipping the dense pipeline because it is long leaves this attempt incomplete; document and halt only if execution fails with captured logs.
- Writing artifacts outside "$HUB" violates storage policy; keep digests/logs under the hub.
- Do not modify core modules (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`).
- Keep stdout assertions ASCII-safe; no emojis in new strings.
- If the pipeline exits non-zero, stop immediately, log blocker details, and mark attempt blocked.

If Blocked:
- Archive failing pytest output under "$HUB"/red/ with selector in filename, summarize the failure in summary.md, and log the block in docs/fix_plan.md + galph_memory.md.
- If the dense pipeline aborts, keep the full CLI log, capture the failing command snippet into "$HUB"/analysis/blocker.txt", and mark the attempt blocked pending rerun.
- When digest files are missing, run `find "$HUB" -maxdepth 3 -type f` to document state, stash under analysis/, and stop further actions until resolved.

Findings Applied (Mandatory):
- POLICY-001 — PyTorch dependency remains mandatory; no environment downgrade while running orchestrator helpers.
- CONFIG-001 — Export AUTHORITATIVE_CMDS_DOC before invoking orchestrator/tests to satisfy legacy bridge guard.
- DATA-001 — Verify regenerated metrics artifacts align with data contract (metrics_summary.json + highlights contents).
- TYPE-PATH-001 — Normalize Path handling in both banner strings and tests.
- OVERSAMPLING-001 — Dense view relies on K > C; avoid altering grouping assumptions during evidence run.
- STUDY-001 — Track MS-SSIM/MAE deltas to compare against prior fly64 study expectations.

Pointers:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:824 — Success banner block to update with digest paths.
- tests/study/test_phase_g_dense_orchestrator.py:951 — Digest invocation test to tighten stdout assertions.
- docs/TESTING_GUIDE.md:360 — Phase G orchestrator workflow diagram notes digest integration.
- docs/development/TEST_SUITE_INDEX.md:62 — Registry entry for Phase G orchestrator selectors.
- docs/findings.md:8 — POLICY-001 / CONFIG-001 / DATA-001 / TYPE-PATH-001 / OVERSAMPLING-001 entries for guardrails.

Next Up (optional):
- Prepare sparse view digest automation once dense evidence is captured and analyzed.

Doc Sync Plan (Conditional):
- After GREEN, rerun pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest --collect-only -vv > "$HUB"/collect/pytest_digest_exec_collect.log and update docs/TESTING_GUIDE.md §2.5 plus docs/development/TEST_SUITE_INDEX.md if assertions changed.

Mapped Tests Guardrail:
- Ensure pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest --collect-only -vv reports ≥1 collected test; adjust selector or add test before completion otherwise.

Hard Gate:
- Attempt is incomplete unless run_phase_g_dense.py exits 0, `$HUB` contains metrics_summary.json/aggregate_report.md/aggregate_highlights.txt/metrics_digest.md from this run, banner strings mention both digest artifacts, docs/fix_plan.md logs the attempt, and summary.md records MS-SSIM/MAE deltas with artifact links.
