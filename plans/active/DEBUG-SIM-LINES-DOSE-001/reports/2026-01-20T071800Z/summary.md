### Turn Summary
Extended `run_phase_c2_scenario.py` so `run_metadata.json` now exposes explicit `training_history_path`/`training_summary_path` entries (relative to the scenario hub), reran the baked gs1_ideal/gs2_ideal runs, and captured the new history JSON/Markdown summaries with NaN detection embedded in both metadata and Markdown tables.
Regenerated the gs1/gs2 reassembly telemetry (CLI log + JSON/Markdown) to confirm padded canvases remain at 828/826 px with `fits_canvas=true`, and reran the synthetic helpers CLI smoke selector (collect + targeted test) to guard the plan-local runner.
Next: inspect the gs1 history vs gs2 to isolate the first NaN stage (if any) and decide whether additional diagnostics or PyTorch parity probes are required before Phase C4.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T071800Z/ (gs*_ideal_runner.log, history.json/history_summary.json, gs*_ideal_training_summary.md, reassembly_cli.log, pytest logs)

### Turn Summary
Instrumented the Phase C2 runner with JSON-safe history capture, NaN warnings, and Markdown summaries while wiring run_metadata to the new artifacts.
Replayed the gs1_ideal and gs2_ideal stable profiles plus reassembly telemetry; both produced complete history logs (no NaNs) and reassembly limits stayed `fits_canvas=true`.
Next: Investigate why gs1_ideal still collapses despite clean telemetry (Phase C3 diagnostics on training dynamics vs coordinate handling).
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T071800Z/ (gs1_ideal_runner.log, gs2_ideal_runner.log, reassembly_cli.log)

### Turn Summary
Scoped Phase C3 to add per-epoch telemetry to the Phase C2 runner so gs1_ideal vs gs2_ideal NaNs can be diagnosed without touching production modules.
Marked C2b complete, updated the implementation plan + fix plan attempts, and refreshed input.md with the new instrumentation Do Now plus pytest/reassembly commands.
Next: implement the runner instrumentation, rerun both scenarios under the new hub, and capture history.json/history_summary plus the reassembly + pytest evidence.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T071800Z/
