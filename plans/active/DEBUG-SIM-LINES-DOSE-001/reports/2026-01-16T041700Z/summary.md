### Turn Summary
Extended `bin/grouping_summary.py` so the grouping telemetry now includes overall mean/std plus per-axis coordinate stats and nn-index ranges, then reran gs1 default/gs2 default/gs2 neighbor-count=1 so B3 has refreshed evidence.
Captured JSON+Markdown summaries for all three scenarios along with the CLI stream that records the expected neighbor-count failure signature, and the pytest guard stayed green.
Next: mine the per-axis offset spread + nn-index histograms to decide whether B4 needs more grouping probes or if we can pivot directly to the reassembly experiment.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T041700Z/ (grouping_cli.log, grouping_gs2_custom_default.json, pytest_sim_lines_pipeline_import.log)
