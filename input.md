Summary: Add a success-path digest regression, then run the dense Phase C→G pipeline to produce fresh metrics + digest evidence under the 030500Z hub.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Mapped tests:
  - pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_success_digest -vv
  - pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_flags_failures -vv
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T030500Z/phase_g_dense_full_execution_real_run/

Do Now:
- Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
- Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py::generate_digest (add explicit success banner when n_failed == 0 and wire matching regression test in tests/study/test_phase_g_dense_metrics_report.py)
- Validate: pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_success_digest -vv
- Validate: pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_flags_failures -vv
- Validate: pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview -vv
- Execute: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T030500Z/phase_g_dense_full_execution_real_run --dose 1000 --view dense --splits train test --clobber | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T030500Z/phase_g_dense_full_execution_real_run/cli/run_phase_g_dense_cli.log
- Run: python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py --metrics plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T030500Z/phase_g_dense_full_execution_real_run/analysis/metrics_summary.json --highlights plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T030500Z/phase_g_dense_full_execution_real_run/analysis/aggregate_highlights.txt --output plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T030500Z/phase_g_dense_full_execution_real_run/analysis/metrics_digest.md | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T030500Z/phase_g_dense_full_execution_real_run/analysis/metrics_digest.log
- Update: Prepend latest MS-SSIM/MAE deltas + digest/log links to plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T030500Z/phase_g_dense_full_execution_real_run/summary/summary.md and add corresponding Attempt entry in docs/fix_plan.md (include exit codes, findings, artifact paths).
- Sync Docs: Append new selector coverage for both analyze_dense_metrics regressions to docs/TESTING_GUIDE.md §2.5 and docs/development/TEST_SUITE_INDEX.md after collecting pytest --collect-only evidence.
- Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T030500Z/phase_g_dense_full_execution_real_run/

How-To Map:
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
2. export HUB=plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T030500Z/phase_g_dense_full_execution_real_run
3. mkdir -p "$HUB"/{plan,collect,red,green,cli,analysis,summary}
4. pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_success_digest -vv | tee "$HUB"/red/pytest_analyze_dense_success.log  # expect RED until success banner implemented
5. Implement success banner in analyze_dense_metrics.py::generate_digest and add the new success-path regression test in tests/study/test_phase_g_dense_metrics_report.py
6. pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_success_digest -vv | tee "$HUB"/green/pytest_analyze_dense_success.log
7. pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_flags_failures -vv | tee "$HUB"/green/pytest_analyze_dense_failures.log
8. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview -vv | tee "$HUB"/green/pytest_highlights_preview.log
9. pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_success_digest --collect-only -vv | tee "$HUB"/collect/pytest_analyze_dense_success_collect.log
10. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber | tee "$HUB"/cli/run_phase_g_dense_cli.log
11. find "$HUB" -maxdepth 3 -type f | sort > "$HUB"/analysis/artifact_inventory.txt  # sanity-check expected files present
12. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py --metrics "$HUB"/analysis/metrics_summary.json --highlights "$HUB"/analysis/aggregate_highlights.txt --output "$HUB"/analysis/metrics_digest.md | tee "$HUB"/analysis/metrics_digest.log
13. cat "$HUB"/analysis/metrics_digest.md > "$HUB"/analysis/metrics_digest_preview.txt  # capture digest snapshot for artifacts
14. Update summary.md (prepend Turn Summary) with deltas + digest path; update docs/fix_plan.md Attempts History with execution results and findings references.
15. Update docs/TESTING_GUIDE.md §2.5 and docs/development/TEST_SUITE_INDEX.md with the new selector, referencing `$HUB` logs; capture diff-ready rationale in summary.md.
16. git status --short >> "$HUB"/summary/git_status.txt  # record touched files pre-commit handoff

Pitfalls To Avoid:
- Do not skip the RED run for the new success digest test; capture the failing log in red/ before implementing the banner.
- Keep AUTHORITATIVE_CMDS_DOC exported before every pytest or pipeline command to satisfy CONFIG-001.
- Use `Path` objects inside the new test fixtures (TYPE-PATH-001) and avoid hard-coded string concatenation.
- Ensure the success banner string matches the test expectation exactly; include emoji only if ASCII safe.
- Run the pipeline with --clobber to clear stale phase outputs before regeneration.
- Monitor the long-running pipeline; if it hangs, capture `pgrep -fl run_phase_g_dense.py` output and log the blocker before exiting.
- Do not move digest files outside the hub; keep digest/log/summary within `$HUB`.
- Treat any non-zero exit code from pipeline or analyze script as blocker; write exit status to summary and docs.
- Remember to refresh docs/TESTING_GUIDE.md and TEST_SUITE_INDEX.md once the new selector is GREEN; skip only with documented justification.
- Avoid editing core physics modules (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`).

If Blocked:
- Save failing pytest output under `$HUB`/red/ with selector in filename and note traceback in docs/fix_plan.md.
- If pipeline aborts, preserve `$HUB`/cli/run_phase_g_dense_cli.log, append exit code + failing command to `$HUB`/analysis/blocker.log, and mark the attempt blocked in docs/fix_plan.md and galph_memory.md.
- If analyze script inputs missing, capture `find "$HUB" -maxdepth 3 -type f` into `$HUB`/analysis/tree.txt, record reason in summary.md, and halt further steps.

Findings Applied (Mandatory):
- POLICY-001 — Enforce PyTorch dependency expectations during comparisons and digest generation.
- CONFIG-001 — Export AUTHORITATIVE_CMDS_DOC before CLI/test runs to keep legacy consumers synchronized.
- DATA-001 — Maintain NPZ/data contract compliance throughout regenerated Phase C assets.
- TYPE-PATH-001 — Normalize filesystem interactions via Path objects in code/tests.
- OVERSAMPLING-001 — Validate overlap metrics in digest vs. design thresholds.

Pointers:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py:120 — Success/failure banner logic to extend for `n_failed == 0`.
- tests/study/test_phase_g_dense_metrics_report.py:187 — Existing failure-path regression to mirror for success-path coverage.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:640 — Pipeline command sequence and highlights preview hook.
- docs/TESTING_GUIDE.md:268 — Phase G orchestrator workflow + AUTHORITATIVE_CMDS_DOC guard instructions.
- docs/findings.md:8 — POLICY-001 / CONFIG-001 / DATA-001 / TYPE-PATH-001 / OVERSAMPLING-001 ledger entries.

Next Up (optional):
- After dense pipeline completes, rerun workflow for sparse view to compare overlap sensitivity.

Doc Sync Plan (Conditional):
- After GREEN, update docs/TESTING_GUIDE.md §2.5 and docs/development/TEST_SUITE_INDEX.md with `test_analyze_dense_metrics_success_digest` and attach `$HUB`/collect/pytest_analyze_dense_success_collect.log as evidence; keep logs in reports/2025-11-09T030500Z/.../collect/.

Mapped Tests Guardrail:
- Ensure `pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_success_digest --collect-only -vv` reports ≥1 collected test (artifact step 9). If collection fails, adjust selector before proceeding.

Hard Gate:
- Do not mark this focus done until the pipeline exits 0, the hub contains `metrics_summary.json`, `aggregate_report.md`, `aggregate_highlights.txt`, and `metrics_digest.md`, docs/fix_plan.md logs exit codes + findings, and summary.md records MS-SSIM/MAE deltas with digest link.
