### Turn Summary
Implemented the SIM-LINES-4X parameter snapshot CLI and captured the JSON output so we can compare nongrid defaults against the legacy dose_experiments runbook.
Archived the git tree plus notebook scan and reran the sim_lines pipeline import smoke pytest to prove the CLI import path still works after the new helper landed.
Next: line up the Phase A4 comparison and prep the Phase B differential experiments using the captured metadata.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/ (sim_lines_4x_params_snapshot.json, dose_experiments_tree.txt, dose_experiments_param_scan.md, pytest_sim_lines_pipeline_import.log)

### Turn Summary
Framed the Phase A Do Now around a new sim_lines_4x parameter snapshot tool plus legacy dose_experiments inventory so we can compare pipelines apples-to-apples.
Updated docs/fix_plan.md Attempts, rewrote input.md with detailed implementation/test instructions, and recorded the new artifacts hub for Ralph.
Next: implement `collect_sim_lines_4x_params.py`, capture the git history outputs, and run the sim_lines CLI smoke test while archiving all logs.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/ (input.md, fix_plan update log)
