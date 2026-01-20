# DEBUG-SIM-LINES-DOSE-001 Summary

### Turn Summary
Implemented `_update_max_position_jitter_from_offsets()` with padded-size parity handling, wired it into the workflow container factory, and aligned the SIM-LINES reassembly telemetry to use the new jitter updates.
Resolved the integration test failure caused by odd padded sizes by enforcing N-parity in the required canvas calculation, then added pytest coverage and refreshed test docs.
Re-ran the targeted workflow selector, the integration marker, and the gs1/gs2 custom reassembly CLI to confirm `fits_canvas=True` with zero loss.
Next: run the Phase C2 gs1/gs2 ideal telemetry or move to the inference smoke validation once this padded-size update is accepted.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T060900Z/ (reassembly_cli.log, pytest_integration.log)

# DEBUG-SIM-LINES-DOSE-001 Summary

### Turn Summary
Authored `bin/reassembly_limits_report.py` to replay the SIM-LINES snapshot, log padded-size inputs, and probe `reassemble_whole_object()` with both the legacy `get_padded_size()` and a spec-sized canvas.
Captured evidence that gs1_custom already needs ≈828 px canvases (vs 74 px padded) and gs2_custom requires ≈831 px (vs 78 px), with dummy reassembly sums losing 95–100 % of the signal when `size` stays at the legacy value.
Next: use the B4 telemetry to plan the Phase C fix for the padded-size math or open a stabilization initiative if it touches shared modules.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T050500Z/ (reassembly_cli.log, reassembly_gs1_custom.json, reassembly_gs2_custom.json)

### Turn Summary
Advanced Phase C planning for the reassembly fix: mined the B4 telemetry, marked Phase B complete in the plan, and scoped C1 around updating `create_ptycho_data_container()` to expand `params.cfg['max_position_jitter']` based on actual grouped offsets (with pytest coverage) so `get_padded_size()` meets the spec requirement.
Captured the trimmed-down Do Now for Ralph (jitter updater + regression test) and refreshed docs/fix_plan.md/galph_memory with the new focus; no new artifacts this loop.
Next: Ralph updates the workflow helper, adds the targeted test, and reruns the existing selectors plus the SIM-LINES CLI guard to verify padded-size math is correct end-to-end.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T060156Z/ (planning notes only)

### Turn Summary
Reviewed the Phase B3 grouping telemetry and confirmed gs2 offsets climb to ~382 px while the legacy padded-size (N + (gridsize−1)·offset + buffer) still evaluates to only 78 px, so reassembly cannot cover the scan trajectory.
Scoped Phase B4 around a `reassembly_limits_report.py` helper that rebuilds the nongrid simulation from the Phase A snapshot, contrasts observed offsets vs `get_padded_size()`, and runs a sum-preservation probe using `reassemble_whole_object()` with `size=get_padded_size()` vs the required canvas.
Next: implement the new CLI, run it for gs1_custom and gs2_custom (train/test subsets) with JSON+Markdown outputs plus the reassembly sum ratios, and rerun the synthetic_helpers CLI smoke pytest guard while archiving logs under the new hub.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T050500Z/ (reserved for reassembly limits evidence + pytest log)

### Turn Summary
Extended `bin/grouping_summary.py` so the grouping telemetry now includes overall mean/std plus per-axis coordinate stats and nn-index ranges, then reran gs1 default/gs2 default/gs2 neighbor-count=1 so B3 has refreshed evidence.
Captured JSON+Markdown summaries for all three scenarios along with the CLI stream that records the expected neighbor-count failure signature, and the pytest guard stayed green.
Next: mine the per-axis offset spread + nn-index histograms to decide whether B4 needs more grouping probes or if we can pivot directly to the reassembly experiment.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T041700Z/ (grouping_cli.log, grouping_gs2_custom_default.json, pytest_sim_lines_pipeline_import.log)

### Turn Summary
Reviewed the Phase B2 artifacts and confirmed legacy vs sim_lines probe normalization is numerically identical (≤5e-7 deltas), so normalization is no longer a suspect.
Updated the working plan + fix ledger, scoped Phase B3 around richer grouping telemetry (per-axis offset stats + nn-index ranges), and prepared a new artifacts hub for the gs1/gs2 + neighbor-count runs.
Next: extend `bin/grouping_summary.py` with the new stats, rerun it for the three scenarios, and archive the CLI log plus pytest guard under the fresh hub.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T041700Z/ (planned grouping telemetry + pytest logs)

### Turn Summary
Built the new `grouping_summary.py` plan-local CLI so we can replay the SIM-LINES nongrid pipeline and emit JSON/Markdown grouping stats for any override set.
Captured 1000/1000 grouped samples for both SIM-LINES train/test splits and recorded the expected 'only 2 points for 4-channel groups' failure signature for the dose_experiments-style gridsize=2 probe, then reran the synthetic helpers CLI smoke test.
Next: analyze these summaries to decide which grouping/probe experiments should anchor Phase B2 and whether additional overrides are required.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T020000Z/ (grouping_sim_lines_default.json, grouping_dose_experiments_legacy.json, pytest_sim_lines_pipeline_import.log)

### Turn Summary
Analyzed the SIM-LINES-4X snapshot JSON vs legacy dose_experiments defaults to identify divergent parameters (photons, gridsize, grouping) and mapped them into a comparison draft for Phase A4.
Recorded diffs in the plan, updated the compliance checklist, and confirmed existing artifacts cover A1/A3; still need actionable code tasks for the comparison CLI.
Next: prepare a Do Now for Phase A4 with implementable instructions (comparison helper + logging) or pivot if dependencies block.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T003217Z/ (comparison_draft.md placeholder)

2026-01-13: Drafted the phased debugging plan to isolate sim_lines_4x vs dose_experiments discrepancies.
2026-01-16: Captured the SIM-LINES-4X parameter snapshot (new CLI) plus the legacy `dose_experiments` tree/script for comparison and reran the synthetic helpers CLI smoke test to prove the pipeline import path is healthy.
