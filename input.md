Summary: Emit a stdout highlights preview from the Phase G orchestrator and capture a fresh dense Phase C→G run with full metrics/highlights evidence.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Mapped tests:
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview -vv
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands -vv
  - pytest tests/study/test_phase_g_dense_metrics_report.py::test_report_phase_g_dense_metrics -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T110500Z/phase_g_dense_full_execution_real_run/

Do Now:
- Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
- Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main — emit an "Aggregate highlights" preview (reading analysis/aggregate_highlights.txt) after the reporting helper runs in execution mode; extend tests/study/test_phase_g_dense_orchestrator.py to cover the preview.
- Validate: pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview -vv
- Execute: pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands -vv && pytest tests/study/test_phase_g_dense_metrics_report.py::test_report_phase_g_dense_metrics -vv
- Execute: stdbuf -oL -eL python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T110500Z/phase_g_dense_full_execution_real_run --dose 1000 --view dense --splits train test --clobber
- Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T110500Z/phase_g_dense_full_execution_real_run/

How-To Map:
1. `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`
2. RED preview proof: `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T110500Z/phase_g_dense_full_execution_real_run/red/pytest_highlights_preview_red.log`
3. Update `tests/study/test_phase_g_dense_orchestrator.py` to add/adjust fixtures so execution mode stubs write deterministic highlights text and assert preview lines in captured stdout.
4. Adjust `bin/run_phase_g_dense.py::main` to read `analysis/aggregate_highlights.txt` (Path-based, execution mode only) after the reporting helper command succeeds; print bannered preview to stdout and guard against missing/empty files.
5. GREEN preview proof: `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T110500Z/phase_g_dense_full_execution_real_run/green/pytest_highlights_preview_green.log`
6. Regression (collect-only): `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T110500Z/phase_g_dense_full_execution_real_run/green/pytest_collect_only_green.log`
7. Helper guard: `pytest tests/study/test_phase_g_dense_metrics_report.py::test_report_phase_g_dense_metrics -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T110500Z/phase_g_dense_full_execution_real_run/green/pytest_report_helper_green.log`
8. Selector inventory: `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T110500Z/phase_g_dense_full_execution_real_run/collect/pytest_phase_g_orchestrator_collect.log`
9. Dense run: `stdbuf -oL -eL python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T110500Z/phase_g_dense_full_execution_real_run --dose 1000 --view dense --splits train test --clobber | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T110500Z/phase_g_dense_full_execution_real_run/cli/run_phase_g_dense_cli.log`
10. Sanity check outputs: ensure `analysis/metrics_summary.json`, `analysis/metrics_summary.md`, `analysis/aggregate_report.md`, `analysis/aggregate_highlights.txt`, and CLI logs exist; verify highlights deltas stay within ±0.05 MS-SSIM and ±0.01 MAE thresholds.
11. Documentation: update `summary/summary.md` with highlights + metrics, append Turn Summary block (top), refresh `docs/fix_plan.md`, and log findings adherence.

Pitfalls To Avoid:
- Do not emit the highlights preview during collect-only runs; tests expect no extra stdout there.
- Use `Path` objects for all filesystem reads/writes (TYPE-PATH-001 compliance).
- Capture RED logs before implementation; missing RED evidence blocks sign-off.
- Keep stdout preview concise (≤6 lines) to avoid drowning CLI output.
- Treat validator or CLI failures as blockers; write details to `analysis/blocker.log`.
- Avoid editing core TensorFlow/PyTorch modules; confine changes to initiative scripts/tests.
- Ensure `AUTHORITATIVE_CMDS_DOC` stays exported for every shell session before the orchestrator run.
- Do not rely on stale artifacts—always run with `--clobber` and confirm hub paths match this loop.
- Validate MS-SSIM/MAE deltas inside thresholds; flag larger swings immediately in summary + ledger.

If Blocked:
- Archive failing pytest or CLI output under `red/` and note the command in `summary/summary.md`.
- Document blocker details + mitigation attempts in `docs/fix_plan.md` Attempts history and ping supervisor via galph_memory (state=switch_focus if recurring).
- Do not downgrade or skip tests; halt until guidance is provided.

Findings Applied (Mandatory):
- POLICY-001 — PyTorch dependency must remain active; report ImportError signals immediately.
- CONFIG-001 — Preserve legacy bridge order in orchestrator before any downstream phases.
- DATA-001 — Dense run must honor dataset/metadata validators; investigate any violations.
- TYPE-PATH-001 — Normalize all filesystem paths with `Path` to avoid string path regressions.
- OVERSAMPLING-001 — Dense overlap parameters fixed; highlight unexpected metric deviations.

Pointers:
- docs/findings.md:8 — POLICY-001 PyTorch dependency policy.
- docs/findings.md:10 — CONFIG-001 legacy bridge requirement.
- docs/findings.md:14 — DATA-001 NPZ/data contract guardrails.
- docs/findings.md:21 — TYPE-PATH-001 path normalization lessons.
- docs/development/TEST_SUITE_INDEX.md:60 — Phase G orchestrator selectors.
- docs/TESTING_GUIDE.md:280 — Phase G helper workflow details.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:189 — Phase G objectives checklist.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T110500Z/phase_g_dense_full_execution_real_run/plan/plan.md — Current loop plan + guardrails.

Next Up (optional):
- If highlights preview + dense run complete quickly, prep sparse-view runbook for the same preview workflow.

Doc Sync Plan (Conditional):
- After GREEN + collect-only, archive `collect/pytest_phase_g_orchestrator_collect.log` and update `docs/development/TEST_SUITE_INDEX.md` plus `docs/TESTING_GUIDE.md` if selector names/descriptions change with the new preview test.

Mapped Tests Guardrail:
- Confirm `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -vv` reports ≥1 test; capture the log in `collect/pytest_phase_g_orchestrator_collect.log`.

Hard Gate:
- If any mapped selector collects 0 or the dense pipeline exits non-zero, stop immediately, log artifacts, and mark the attempt blocked rather than complete.
