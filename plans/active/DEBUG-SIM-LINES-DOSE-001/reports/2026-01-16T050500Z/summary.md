### Turn Summary
Reviewed the Phase B3 grouping telemetry and confirmed gs2 offsets climb to ~382 px while the legacy padded-size (N + (gridsize−1)·offset + buffer) still evaluates to only 78 px, so reassembly cannot cover the scan trajectory.
Scoped Phase B4 around a `reassembly_limits_report.py` helper that rebuilds the nongrid simulation from the Phase A snapshot, contrasts observed offsets vs `get_padded_size()`, and runs a sum-preservation probe using `reassemble_whole_object()` with `size=get_padded_size()` vs the required canvas.
Next: implement the new CLI, run it for gs1_custom and gs2_custom (train/test subsets) with JSON+Markdown outputs plus the reassembly sum ratios, and rerun the synthetic_helpers CLI smoke pytest guard while archiving logs under the new hub.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T050500Z/ (reserved for reassembly limits evidence + pytest log)
