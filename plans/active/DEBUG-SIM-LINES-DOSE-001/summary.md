# DEBUG-SIM-LINES-DOSE-001 Summary

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
