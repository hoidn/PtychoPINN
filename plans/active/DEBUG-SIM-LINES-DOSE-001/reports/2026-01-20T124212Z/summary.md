### Turn Summary
Executed the `capture_dose_normalization.py` CLI with dose_experiments_param_scan.md defaults (gridsize=2, probe_scale=4, neighbor_count=5) to generate normalization parity telemetry.
The CLI completed successfully: dataset-derived intensity_scale=262.78 vs closed-form fallback=988.21 (ratio=0.266); stage flow shows raw=1.41→grouped=1.45→normalized=0.38→container=0.38 with grouped→normalized as the largest drop (ratio=0.26, ~74% amplitude reduction).
All artifacts generated: capture_config.json, capture_summary.md, dose_normalization_stats.{json,md}, intensity_stats.{json,md}. pytest smoke test passed (1/1).
Next: Compare dose_legacy_gs2 normalization stats against sim_lines gs1_ideal/gs2_ideal runs via the analyzer to identify which parameters diverge.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T124212Z/dose_normalization/ (dose_normalization_stats.json, capture_summary.md, pytest_cli_smoke.log)
