### Turn Summary (B0f planning 2026-01-20T102300Z)
Scoped B0f isolation test to run gs1_custom (gridsize=1 + custom probe) and compare metrics against the gs1_ideal/gs2_ideal baselines that now both show identical failure patterns (~-2.3 amplitude bias, ~0.1 pearson_r) after C4f CONFIG-001 bridging.
Decision tree: if gs1_custom shows similar metrics → failure is workflow-wide (H-LOSS-WIRING); if significantly better → failure is ideal-probe-specific (H-PROBE-IDEAL-REGRESSION).
Next: Ralph executes the Phase C2 runner for gs1_custom, runs the multi-scenario analyzer, and records the decision branch in this summary.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T102300Z/ (gs1_custom_runner.log, bias_summary.md, pytest_cli_smoke.log expected)
