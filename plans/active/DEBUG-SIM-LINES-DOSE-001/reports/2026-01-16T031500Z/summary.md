### Turn Summary
Built the plan-local probe normalization CLI to compare legacy `set_default_probe()` against the sim_lines normalization path, generated JSON/Markdown stats for all four gs1/gs2 × custom/ideal scenarios, and logged the CLI runs.
Captured identical normalization factors/amplitude stats between the two branches (max L2 delta ≈5e-7) plus the full CLI log, then reran the synthetic helpers CLI smoke test for guardrail coverage.
Next: interpret the per-scenario probe stats to decide whether to proceed with grouping/reassembly experiments (Phase B3/B4) or document probe normalization as non-root-cause.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T031500Z/ (probe_stats_* files, probe_normalization_cli.log, pytest_sim_lines_pipeline_import.log)

### Turn Summary
Captured Phase B1 evidence showing nongrid SIM-LINES generates the requested 1000/1000 groups while dose_experiments-style overrides fail with “Dataset has only 2 points but 4 coordinates per group requested,” proving the KDTree grouping limit is the immediate divergence.
Scoped Phase B2 around a probe-normalization comparison CLI that reproduces both the legacy `set_default_probe()` path and the sim_lines `make_probe`/`normalize_probe_guess` workflow so we can quantify amplitude/scale deltas per scenario.
Next: implement `bin/probe_normalization_report.py`, emit JSON/Markdown stats for all four scenarios, and rerun the synthetic helpers CLI smoke guard before moving to probe-focused experiments.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T031500Z/ (analysis.md)
