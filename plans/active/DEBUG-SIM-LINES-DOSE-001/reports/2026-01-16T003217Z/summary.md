### Turn Summary
Created `compare_sim_lines_params.py` to diff the SIM-LINES-4X snapshot against the legacy dose_experiments defaults and emitted Markdown/JSON artifacts with scenario-level parameter deltas.
Captured `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v` to prove the helper tree still imports cleanly and logged the evidence in docs/fix_plan.md.
Next: Use the diff to pick the highest-risk delta (grid vs nongrid or probe normalization) for Phase B experiments, or extend the parser if additional parameters need to be surfaced.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T003217Z/ (comparison_draft.md, comparison_diff.json, pytest_sim_lines_pipeline_import.log)

### Turn Summary
Analyzed the SIM-LINES-4X snapshot JSON vs legacy dose_experiments defaults to identify divergent parameters (photons, gridsize, grouping) and mapped them into a comparison draft for Phase A4.
Recorded diffs in the plan, updated the compliance checklist, and confirmed existing artifacts cover A1/A3; still need actionable code tasks for the comparison CLI.
Next: prepare a Do Now for Phase A4 with implementable instructions (comparison helper + logging) or pivot if dependencies block.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T003217Z/ (comparison_draft.md placeholder)
