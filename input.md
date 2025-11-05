Summary: Add a key MS-SSIM/MAE delta summary to the Phase G orchestrator stdout and rerun the dense pipeline to capture real metrics evidence.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Mapped tests:
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands -vv
  - pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_success_digest -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T090500Z/phase_g_dense_full_execution_real_run/

Do Now:
- Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
- Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main — emit a formatted MS-SSIM/MAE delta block sourced from analysis/metrics_summary.json (drive via updated orchestrator exec test)
- Validate: pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
- Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T090500Z/phase_g_dense_full_execution_real_run/

How-To Map:
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
2. export HUB=plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T090500Z/phase_g_dense_full_execution_real_run
3. Ensure "$HUB" exists with plan/collect/red/green/cli/analysis/summary (directories already scaffolded; create if missing).
4. Update tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest to seed metrics_summary.json in the stub summarizer and assert the four new delta lines; run RED: `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/red/pytest_orchestrator_delta_red.log` (expect failure until implementation).
5. Implement helper in plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py (after analyze digest) to load "$HUB"/analysis/metrics_summary.json, compute MS-SSIM/MAE deltas (PtychoPINN - Baseline / PtyChi), and print the formatted block with f"{value:+.3f}"; guard missing data by printing "N/A".
6. GREEN targeted tests:
   - `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/green/pytest_orchestrator_delta_green.log`
   - `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands -vv | tee "$HUB"/green/pytest_collect_only.log`
   - `pytest tests/study/test_phase_g_dense_metrics_report.py::test_analyze_dense_metrics_success_digest -vv | tee "$HUB"/green/pytest_analyze_success.log`
7. Kick off the dense pipeline run (long): `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense.log` (keep terminal session open; expect 2–4 h runtime).
8. After completion, verify artifacts exist (`metrics_summary.json`, `aggregate_report.md`, `aggregate_highlights.txt`, `metrics_digest.md`, `cli/metrics_digest_cli.log`) and extract the new stdout delta block: `rg "Δ" "$HUB"/cli/run_phase_g_dense.log > "$HUB"/analysis/metric_delta_block.txt`.
9. Capture artifact inventory: `find "$HUB" -maxdepth 3 -type f | sort > "$HUB"/analysis/artifact_inventory.txt` and copy digest preview `cp "$HUB"/analysis/metrics_digest.md "$HUB"/analysis/metrics_digest_preview.md`.
10. Summarize MS-SSIM/MAE delta values and key observations in "$HUB"/analysis/metrics_highlights.txt (include numbers from the delta block and digest) and update "$HUB"/summary/summary.md with pass/fail counts + artifact links.
11. Update docs/fix_plan.md Attempts History (add 2025-11-09T090500Z execution details), then stage summary/doc edits after verifying `git status`.

Pitfalls To Avoid:
- TDD guard: capture the RED failure log before touching run_phase_g_dense.py.
- Forgetting AUTHORITATIVE_CMDS_DOC breaks CONFIG-001 guards in orchestrator tests and CLI.
- Do not let delta helper raise if Baseline/PtyChi keys are missing; print "N/A" and continue so pipeline evidence still ships.
- Keep new stdout lines ASCII with fixed precision (±0.000); avoid emoji/symbols that may break CI log parsing.
- Ensure helper reads from `analysis/metrics_summary.json` relative to hub (TYPE-PATH-001 compliance); no hard-coded cwd assumptions.
- Pipeline log must stay under `$HUB/cli/`; do not leave large logs at repo root.
- abort pipeline immediately if exit code non-zero; archive `cli/run_phase_g_dense.log` and document blocker in summary + docs/fix_plan.md.
- No edits to core modules (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`).

If Blocked:
- Store failing pytest output under `$HUB/red/`, note selector + failure in summary.md, and log the block in docs/fix_plan.md plus galph_memory.md.
- If the pipeline aborts, keep the full CLI log, capture the failing command snippet into `$HUB/analysis/blocker.txt`, and mark the attempt blocked pending rerun.
- When metrics_summary.json is missing or malformed, snapshot `$HUB/analysis` via `find` into artifact_inventory, skip helper execution, and document the issue before exiting.

Findings Applied (Mandatory):
- POLICY-001 — PyTorch dependency stays enabled; orchestrator helpers rely on torch-ready environment.
- CONFIG-001 — Always export AUTHORITATIVE_CMDS_DOC before pytest/pipeline invocations to keep legacy bridge aligned.
- DATA-001 — Verify regenerated metrics_summary.json and digest respect the documented schema.
- TYPE-PATH-001 — Use Path objects and deterministic string formatting for banner output and tests.
- OVERSAMPLING-001 — Dense scenario assumes K > C; avoid altering group configuration during evidence run.
- STUDY-001 — Record MS-SSIM/MAE deltas to track PtychoPINN vs Baseline/PtyChi performance consistency.

Pointers:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:800 — Success banner + digest printing block to extend with delta helper call.
- tests/study/test_phase_g_dense_orchestrator.py:856 — Exec-mode integration test to tighten with delta assertions and stub summary data.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py:1 — Digest script behavior reference for success/failure semantics.
- docs/TESTING_GUIDE.md:300 — Phase G testing workflow and selector registry expectations.
- docs/findings.md:8 — POLICY-001/CONFIG-001/DATA-001/TYPE-PATH-001/OVERSAMPLING-001 guidance.

Next Up (optional):
- After dense evidence lands, queue sparse view pipeline run with identical delta summary helper.

Mapped Tests Guardrail:
- `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest --collect-only -vv` must report ≥1 collected test; author or adjust selector before exiting if collection drops to zero.

Hard Gate:
- Do not close loop unless the new delta block prints in both stdout and CLI log, all mapped selectors pass (GREEN logs saved), the dense pipeline exits 0 with fresh metrics_summary.json/digest artifacts under `$HUB`, summary.md captures MS-SSIM/MAE deltas with artifact links, and docs/fix_plan.md records this attempt.
