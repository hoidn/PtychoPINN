# Phase G Dense Pipeline Completion — Supervisor Plan (2025-11-05T125421Z)

## Focus
- Initiative: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
- Scope: Finish the dense dose=1000 Phase C→G pipeline run that is currently executing under hub `2025-11-05T115706Z/phase_g_dense_full_execution_real_run`, gather verification artifacts, and sync summaries/ledger.

## Current Status Snapshot
- Background orchestrator process (`python plans/.../run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber`) is still running (PIDs 2278335, 2278339, 2278340) and actively emitting Phase C outputs.
- `phase_c_generation.log` shows completion for dose 1e3 and is currently generating dose 1e4 datasets; Phase D–G artifacts are not present yet.
- No new logs beyond `phase_c_generation.log` exist; `analysis/` directory remains empty.
- Prior attempt (2025-11-05T123500Z) documented that the previous background run terminated after Phase C; this rerun is the live replacement.

## Key Findings Applied
- POLICY-001: PyTorch baseline tooling must remain untouched by supervisor actions.
- CONFIG-001: Ensure `AUTHORITATIVE_CMDS_DOC` stays exported before any helper invocation.
- DATA-001: Rely on validator outputs in Phase C logs, no manual patching of NPZs.
- TYPE-PATH-001: Keep hub resolved via `$PWD/$HUB` and do not move underlying directories while the run is active.
- OVERSAMPLING-001: Maintain dense view parameters (K=7, gridsize=2) as emitted by orchestrator.
- STUDY-001: Final summary must report MS-SSIM/MAE deltas vs Baseline and PtyChi once metrics are available.

## Expectations for Ralph
1. Confirm the background run finishes all eight orchestrator stages (`grep '[8/8]'` in `cli/run_phase_g_dense_full_*.log`), or re-launch with `--clobber` if it aborts.
2. After completion, run highlight/digest helpers and the dedicated check script to validate consistency.
3. Capture Phase G metrics deltas (JSON + highlights), refresh `metrics_digest.md`, and update `summary/summary.md` with runtime + guardrail evidence.
4. Run the mapped pytest selector `tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv` and archive results under `green/`.
5. Update `docs/fix_plan.md` Attempts History and `summary/summary.md` with MS-SSIM/MAE deltas, artifact inventory, and provenance.

## Artifacts for This Loop
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T125421Z/phase_g_dense_pipeline_completion_handoff/summary.md — supervisor notes & turn summary
- Latest hub for execution evidence: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T115706Z/phase_g_dense_full_execution_real_run/
