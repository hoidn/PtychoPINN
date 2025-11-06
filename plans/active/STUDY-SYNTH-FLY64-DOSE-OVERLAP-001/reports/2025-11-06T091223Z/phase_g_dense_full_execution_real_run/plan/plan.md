# Dense Phase G Evidence Run Plan (2025-11-06T091223Z)

## Objectives
- Produce a fresh dense Phase C→G pipeline run (`--dose 1000 --view dense --splits train test --clobber`) that reaches `[8/8]` without blockers.
- Capture the resulting metrics bundle (metrics_summary.json, metrics_delta_summary.json, metrics_delta_highlights.txt, metrics_digest.md, aggregate_highlights.txt, aggregate_report.md) and verify Phase C metadata compliance.
- Document MS-SSIM/MAE deltas, highlights provenance, and checker outputs in summary.md and docs/fix_plan.md.

## Tasks
1. Implement `plans/.../bin/verify_dense_pipeline_artifacts.py::main` to validate the metrics bundle + metadata compliance signals.
2. Run mapped pytest selectors to ensure orchestrator helpers and digest reporters remain green.
3. Execute `run_phase_g_dense.py --hub "$HUB" --clobber --dose 1000 --view dense --splits train test` capturing CLI log (expect `[1/8]`→`[8/8]`).
4. Run the verification script and refresh analyze_dense_metrics digest; store logs under `$HUB`/analysis/ with UTC timestamps.
5. Generate artifact inventory and update summary.md + docs/fix_plan.md with metrics deltas, metadata compliance, and artifact references.

## Deliverables
- `$HUB`/cli/run_phase_g_dense_dense_view_<timestamp>.log
- `$HUB`/analysis/pipeline_verification.json (+ log)
- `$HUB`/analysis/metrics_digest.md (refreshed)
- `$HUB`/analysis/artifact_inventory_<timestamp>.txt
- `$HUB`/summary/summary.md capturing MS-SSIM/MAE deltas + provenance
- docs/fix_plan.md Attempts History update for this run
