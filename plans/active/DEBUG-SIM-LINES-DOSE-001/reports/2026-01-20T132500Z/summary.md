### Turn Summary
Added stage-mean/ratio derivation and “largest drop” detection to `bin/analyze_intensity_bias.py`, so the analyzer now quantifies raw → grouped → normalized → prediction behavior per spec and flags where amplitude collapses.
Ran `python bin/analyze_intensity_bias.py ...` for the gs1_ideal + gs2_ideal hubs to refresh `bias_summary.json/.md` with the new tables/bullets and captured the CLI log under this artifacts hub.
Re-ran `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v` to keep the guard green.
Next: mine the stage-ratio telemetry (grouped→normalized ≈0.56) to decide what normalization code needs remediation before touching loaders or loss weights.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T132500Z/ (bias_summary.json, bias_summary.md, analyze_intensity_bias.log, pytest_cli_smoke.log)

### Turn Summary
Deduped the duplicated 2026-01-20T121500Z attempts entry in docs/fix_plan.md and recorded the doc hygiene request along with a new plan note.
Opened the 2026-01-20T132500Z hub plus plan/summary updates so the ratio-diagnostics increment has a landing zone and traceable record.
Next: implement the analyzer stage-ratio extension, rerun it for gs1_ideal + gs2_ideal, and capture the CLI pytest guard.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T132500Z/
