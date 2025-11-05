Summary: Integrate analyze_dense_metrics into the dense Phase C→G pipeline and rerun the dose=1000 workflow to capture fresh digest evidence.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Mapped tests:
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands -vv
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
  - pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_success_digest -vv
  - pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_flags_failures -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T050500Z/phase_g_dense_full_execution_real_run/

Do Now:
- Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
- Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main (invoke analyze_dense_metrics.py automatically after metrics summarization, emit digest paths in success banner, and update orchestrator tests to cover the new command + execution)
- Validate: pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands -vv
- Validate: pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
- Validate: pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_success_digest -vv
- Validate: pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_flags_failures -vv
- Execute: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T050500Z/phase_g_dense_full_execution_real_run --dose 1000 --view dense --splits train test --clobber | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T050500Z/phase_g_dense_full_execution_real_run/cli/run_phase_g_dense_cli.log
- Verify: find plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T050500Z/phase_g_dense_full_execution_real_run -maxdepth 3 -type f | sort > plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T050500Z/phase_g_dense_full_execution_real_run/analysis/artifact_inventory.txt
- Update: Prepend MS-SSIM/MAE deltas + digest/log links to plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T050500Z/phase_g_dense_full_execution_real_run/summary/summary.md and add corresponding attempt entry in docs/fix_plan.md (include exit codes, findings, artifact paths).
- Sync Docs: After GREEN + pipeline success, refresh docs/TESTING_GUIDE.md §2.5 and docs/development/TEST_SUITE_INDEX.md with the new orchestrator digest integration test; cite collect-only evidence.
- Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T050500Z/phase_g_dense_full_execution_real_run/

How-To Map:
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
2. export HUB=plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T050500Z/phase_g_dense_full_execution_real_run
3. mkdir -p "$HUB"/{plan,collect,red,green,cli,analysis,summary}
4. Update tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands to expect the analyze_dense_metrics command, then run pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands -vv | tee "$HUB"/red/pytest_collect_only.log  # expect RED until orchestrator invokes analyze step
5. Author tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest mirroring the highlights preview guard and confirming the analyze script runs (stub subprocess to avoid real execution); run pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/red/pytest_exec_digest.log  # expect RED before code change
6. Modify plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main to insert analyze_dense_metrics.py into the command sequence (collect-only + real execution) and print digest path in success summary.
7. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands -vv | tee "$HUB"/green/pytest_collect_only.log
8. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/green/pytest_exec_digest.log
9. pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_success_digest -vv | tee "$HUB"/green/pytest_analyze_success.log
10. pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_flags_failures -vv | tee "$HUB"/green/pytest_analyze_failures.log
11. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest --collect-only -vv | tee "$HUB"/collect/pytest_exec_digest_collect.log
12. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber | tee "$HUB"/cli/run_phase_g_dense_cli.log
13. tail -n 40 "$HUB"/cli/run_phase_g_dense_cli.log > "$HUB"/analysis/run_tail.txt  # capture completion banner + digest path
14. find "$HUB" -maxdepth 3 -type f | sort > "$HUB"/analysis/artifact_inventory.txt
15. cat "$HUB"/analysis/metrics_digest.md > "$HUB"/analysis/metrics_digest_preview.md
16. Prepend Turn Summary with MS-SSIM/MAE deltas + artifact links to "$HUB"/summary/summary.md and record attempt in docs/fix_plan.md (exit codes, findings, digest path).
17. Update docs/TESTING_GUIDE.md §2.5 and docs/development/TEST_SUITE_INDEX.md with the new orchestrator digest test once GREEN; note evidence files in summary.md.
18. git status --short > "$HUB"/summary/git_status.txt

Pitfalls To Avoid:
- Keep AUTHORITATIVE_CMDS_DOC exported before every pytest/pipeline run (CONFIG-001).
- Stub subprocess calls inside new test to avoid executing full pipeline logic; route to fake analyze command.
- Ensure --collect-only output lists analyze_dense_metrics.py and metrics_digest.md; update expectation strings precisely.
- Do not double-run analyze_dense_metrics.py manually unless pipeline exits early; digest should appear from orchestrator.
- Capture RED evidence before altering run_phase_g_dense.py to satisfy TDD guardrails.
- Preserve ASCII-only success banner additions; avoid emoji regressions.
- Treat any non-zero exit codes as blockers and log them immediately in summary.md + docs/fix_plan.md.
- Keep all generated artifacts under "$HUB" to satisfy storage directives; no root-level logs.
- Remain device/dtype neutral in tests (TYPE-PATH-001 compliance).
- Do not modify core physics modules (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`).

If Blocked:
- Save failing pytest output under "$HUB"/red/ with selector in filename and document traceback in docs/fix_plan.md.
- If pipeline aborts, keep "$HUB"/cli/run_phase_g_dense_cli.log, write exit code plus failing command to "$HUB"/analysis/blocker.log, and mark attempt blocked in docs/fix_plan.md + galph_memory.md.
- When analyze digest missing, capture `find "$HUB" -maxdepth 3 -type f` into "$HUB"/analysis/tree.txt, log reason in summary.md, and halt further steps.
- If new test cannot stub subprocess safely, document limitation, revert the stub changes, and mark focus blocked pending alternative approach.

Findings Applied (Mandatory):
- POLICY-001 — PyTorch dependency remains enforced for comparison helpers.
- CONFIG-001 — Legacy bridge requires AUTHORITATIVE_CMDS_DOC export before orchestrators/tests.
- DATA-001 — Verify regenerated datasets maintain contract adherence.
- TYPE-PATH-001 — Use pathlib consistently in orchestrator/tests.
- OVERSAMPLING-001 — Confirm dense overlap metrics align with design.
- STUDY-001 — Track fly64 study deltas when updating summary.

Pointers:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:760 — Success banner + CLI summary section to extend with digest info.
- tests/study/test_phase_g_dense_orchestrator.py:29 — Collect-only command inventory test to tighten for analyze step.
- tests/study/test_phase_g_dense_metrics_report.py:234 — Analyze digest regression fixtures for success/failure paths.
- docs/TESTING_GUIDE.md:298 — Phase G orchestrator workflow instructions and env guard.
- docs/findings.md:8 — POLICY-001 / CONFIG-001 / DATA-001 / TYPE-PATH-001 / OVERSAMPLING-001 ledger entries.

Next Up (optional):
- Repeat the digest automation for sparse view once dense pipeline evidence is stable.

Doc Sync Plan (Conditional):
- After GREEN, run pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest --collect-only -vv (artifact step 11) and update docs/TESTING_GUIDE.md §2.5 plus docs/development/TEST_SUITE_INDEX.md with the new selector, citing "$HUB"/collect/pytest_exec_digest_collect.log.

Mapped Tests Guardrail:
- Ensure pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest --collect-only -vv reports ≥1 collected test; adjust selector before pipeline execution if collection fails.

Hard Gate:
- Do not mark this attempt done until run_phase_g_dense.py exits 0 with analyze digest integration confirmed, "$HUB"/analysis contains metrics_summary.json, aggregate_report.md, aggregate_highlights.txt, and metrics_digest.md produced by the pipeline, docs/fix_plan.md logs exit codes + findings, and summary.md records MS-SSIM/MAE deltas referencing the new digest.
