# Dense Phase G Evidence Run Plan (2025-11-06T095003Z)

## Objectives
- Extend the Phase G artifact verifier to cover the delta bundle (`metrics_delta_summary.json`, `metrics_delta_highlights.txt`) with provenance and structure checks.
- Execute a fresh dense Phase C→G pipeline run (`--dose 1000 --view dense --splits train test --clobber`) that completes `[1/8]→[8/8]` cleanly.
- Validate the produced artifacts via pytest selectors, the enhanced verifier, and digest refresh; capture MS-SSIM/MAE deltas plus metadata compliance evidence in documentation.

## Tasks
1. Update `plans/.../bin/verify_dense_pipeline_artifacts.py::main` to:
   - Require `analysis/metrics_delta_summary.json` with `generated_at`, `source_metrics`, and `deltas.{vs_Baseline,vs_PtyChi}.{ms_ssim,mae}.{amplitude,phase}` keys.
   - Confirm `analysis/metrics_delta_highlights.txt` exists and contains four expected delta lines (MS-SSIM/MAE vs Baseline/PtyChi).
   - Validate the `source_metrics` path points to an existing file beneath the hub.
2. Run mapped pytest selectors (collect-only first) to ensure orchestrator + digest tests exercise current code paths.
3. Launch `run_phase_g_dense.py --hub "$HUB" --clobber --dose 1000 --view dense --splits train test`, capturing CLI log with UTC timestamp and verifying `[1/8]→[8/8]`.
4. Execute the verifier and digest refresh scripts; persist JSON reports and logs under `$HUB`/analysis/ with UTC timestamps.
5. Generate artifact inventory, then update `$HUB`/summary/summary.md, `docs/fix_plan.md`, and `docs/findings.md` (if new lessons) with MS-SSIM/MAE deltas, metadata compliance status, and artifact references.

## Deliverables
- `$HUB`/plan/plan.md (this document) and `$HUB`/summary/summary.md turn summary.
- `$HUB/cli/run_phase_g_dense_dense_view_<UTC>.log` with `[1/8]→[8/8]` evidence.
- `$HUB/analysis/pipeline_verification.json` plus verifier log showing PASS.
- `$HUB/analysis/metrics_digest.md` (refreshed) with log plus `$HUB/analysis/metrics_delta_summary.json` and `$HUB/analysis/metrics_delta_highlights.txt`.
- `$HUB/analysis/artifact_inventory_<UTC>.txt` enumerating produced files.
- Ledger updates noting attempt outcomes and referencing this hub.
