### Turn Summary
Prepped Phase B1 by closing out the A4 diff work, refreshing the plan/ledger, and scoping a plan-local grouping_summary CLI so we can compare SIM-LINES vs dose_experiments grouping behavior without touching production code.
Documented the new artifacts hub plus JSON/Markdown expectations and rewrote input.md with exact CLI runs + pytest evidence so Ralph can build the tool and capture both parameter regimes.
Next: Ralph implements grouping_summary.py, runs the sim_lines_default and dose_experiments_legacy modes, and archives the summaries alongside the CLI smoke-test log.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T020000Z/
