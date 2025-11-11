Summary: Hook run_phase_g_dense into the new ssim_grid helper, prove the orchestration change with targeted pytest, then execute one full dense run that emits verified MS-SSIM/MAE deltas plus documentation updates.
Mode: Perf
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands -vv; pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-11T235500Z/phase_g_dense_run_with_ssim_grid/

Do Now (hard validity contract)
- Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main — after the analyze_dense_metrics.py invocation, call plans/.../bin/ssim_grid.py with `--hub <hub>` and log to `cli/ssim_grid_cli.log`, then surface `analysis/ssim_grid_summary.md` + log locations in the success banner (TYPE-PATH-001 compliance).
- Implement: tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest — extend the stubbed run_command tracking so the test asserts the new ssim_grid.py command fires after analyze_dense_metrics.py and produces `analysis/ssim_grid_summary.md`; update the collect-only test to list the helper/log as well.
- Execute: python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber (with AUTHORITATIVE_CMDS_DOC exported) so the counted Phase C→G run plus ssim_grid summary land under this loop’s hub.
- Validate: pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands -vv && pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv (capture logs to $HUB/green/pytest_phase_g_dense_{collect_only,exec}.log after fixes).
- Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-11T235500Z/phase_g_dense_run_with_ssim_grid/ (store plan, summary, CLI logs, pytest outputs, verification report, MS-SSIM/MAE digest, doc diff notes).

How-To Map
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md; export HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-11T235500Z/phase_g_dense_run_with_ssim_grid
2. Modify run_phase_g_dense.py to append:
   - helper = repo_root / "plans/active/.../bin/ssim_grid.py"
   - log = hub/"cli/ssim_grid_cli.log"
   - cmd = [sys.executable, helper, "--hub", str(hub)]
   - run_command(cmd, log)
   - append summary/log lines to success banner (POSIX relative paths)
3. Update tests/study/test_phase_g_dense_orchestrator.py:
   - collect-only test: assert stdout contains "ssim_grid.py" and "ssim_grid_cli.log"
   - exec test: extend stub_run_command to recognize the helper, create analysis/ssim_grid_summary.md, and assert call order reporting helper → analyze_digest → ssim_grid
4. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands -vv | tee "$HUB"/green/pytest_phase_g_dense_collect_only.log
5. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/green/pytest_phase_g_dense_exec.log
6. pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -vv | tee "$HUB"/collect/pytest_collect_phase_g_dense.log (guardrail proof)
7. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber > "$HUB"/cli/run_phase_g_dense_stdout.log 2>&1
8. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py --hub "$HUB" --report "$HUB"/analysis/verification_report.json --dose 1000 --view dense | tee "$HUB"/analysis/verify_dense_stdout.log
9. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py --hub "$HUB" | tee "$HUB"/analysis/check_dense_highlights.log (stays green); rerun ssim_grid.py manually if the orchestrator stops early.
10. Update docs/TESTING_GUIDE.md (Phase G Delta Metrics Persistence) and docs/development/TEST_SUITE_INDEX.md (Phase G section) to describe the preview-only artifact, ±0.000/±0.000000 precision, ssim_grid helper, and new pytest selector; log diffs in $HUB/summary.
11. Summarize MS-SSIM/MAE deltas + preview guard status in "$HUB"/summary/summary.md and link CLI + verification logs; update docs/fix_plan.md + galph_memory.md before handing back.

Pitfalls To Avoid
- Do not skip AUTHORITATIVE_CMDS_DOC or legacy params bridge errors will recur (CONFIG-001).
- Keep hub-relative paths in banners/logs (TYPE-PATH-001); no absolute /tmp entries in success output.
- ssim_grid helper must remain phase-only (PREVIEW-PHASE-001); fail fast if the preview file mentions "amplitude".
- Capture both pytest selectors’ full logs under $HUB/green; missing RED evidence is acceptable only if tests stayed green immediately.
- Use --clobber only after archiving previous Phase C outputs; never delete without the orchestrator’s prepare_hub guard.
- Dense run can take time; if it fails, stop and archive blocker.log rather than re-running blindly.
- Don’t forget doc/test registry updates; stale instructions in TESTING_GUIDE/TEST_SUITE_INDEX are blocking the next authorization gate.
- Keep new helper integration isolated to plans/active bin; do not touch protected modules under ptycho/.

If Blocked
- If run_phase_g_dense fails, capture `$HUB`/analysis/blocker.log + the offending CLI log, note the failing command in summary.md, and pause further execution until resolved.
- If pytest selectors fail due to new ordering expectations, keep the failing log under `$HUB`/red/`, summarize the failure text, and stop before running the pipeline.
- If the dense run cannot start (e.g., missing Phase C inputs), record the missing resource + command in summary.md and update docs/fix_plan.md Attempts History as blocked.

Findings Applied (Mandatory)
- POLICY-001 — PyTorch baseline recon remains required; keep torch>=2.2 available during dense runs.
- CONFIG-001 — Always export AUTHORITATIVE_CMDS_DOC before invoking legacy-aware helpers.
- DATA-001 — `verify_dense_pipeline_artifacts.py` must see DATA-001 compliant NPZ/JSON outputs; capture report JSON as proof.
- TYPE-PATH-001 — Success banner + doc updates must cite relative `analysis/...` and `cli/...` paths only.
- STUDY-001 — Report MS-SSIM/MAE deltas with ± signs (phase emphasis) inside ssim_grid_summary.md and summary.md.
- TEST-CLI-001 — Preserve full CLI logs + pytest red/green runs under the hub so orchestrator/validator coverage stays reproducible.
- PREVIEW-PHASE-001 — ssim_grid integration must fail fast on amplitude contamination; preview guard evidence belongs in summary/logs.

Pointers
- docs/fix_plan.md:24 — Active focus metadata + guardrails.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:1 — Orchestrator code that needs the ssim_grid hook.
- tests/study/test_phase_g_dense_orchestrator.py:1 — Collect-only + exec tests to update for the new helper.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/ssim_grid.py:1 — Helper to be invoked by the orchestrator.
- docs/TESTING_GUIDE.md:331 — Phase G Delta Metrics Persistence section (precision + preview text currently stale).
- docs/development/TEST_SUITE_INDEX.md:1 — Add/refresh the ssim_grid smoke test entry.

Next Up (optional)
1. Run the sparse-view pipeline once the dense run is archived, reusing the ssim_grid summary workflow for cross-view comparisons.
2. Automate SSIM grid aggregation across multiple hubs (loop over `reports/*/phase_g_dense_*`) once at least two counted runs exist.

Doc Sync Plan (Conditional)
- After code/tests pass, update docs/TESTING_GUIDE.md §Phase G Delta Metrics Persistence with the MAE ±0.000000 rule, preview-only artifact description, and the new ssim_grid helper step.
- Update docs/development/TEST_SUITE_INDEX.md (Study → Phase G section) to list `tests/study/test_ssim_grid.py::test_smoke_ssim_grid` and reference the orchestrator integration.
- Capture `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -vv` output in `$HUB/collect/pytest_collect_phase_g_dense.log` to prove selectors still collect.

Mapped Tests Guardrail
- `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv` must collect (>0); if collection fails, stop, capture the collect-only log, and fix the selector before proceeding.

Hard Gate
- Do not mark this loop complete until the dense run produces `analysis/ssim_grid_summary.md`, `analysis/metrics_delta_summary.json`, preview/digest artifacts, and a passing `verification_report.json`, all referenced in summary.md with linked CLI logs.
