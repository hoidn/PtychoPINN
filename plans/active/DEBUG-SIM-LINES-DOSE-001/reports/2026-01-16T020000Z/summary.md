### Turn Summary
Built the new `grouping_summary.py` plan-local CLI so we can replay the SIM-LINES nongrid pipeline and emit JSON/Markdown grouping stats for any override set.
Captured 1000/1000 grouped samples for both SIM-LINES train/test splits and recorded the expected 'only 2 points for 4-channel groups' failure signature for the dose_experiments-style gridsize=2 probe, then reran the synthetic helpers CLI smoke test.
Next: analyze these summaries to decide which grouping/probe experiments should anchor Phase B2 and whether additional overrides are required.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T020000Z/ (grouping_sim_lines_default.json, grouping_dose_experiments_legacy.json, pytest_sim_lines_pipeline_import.log)
