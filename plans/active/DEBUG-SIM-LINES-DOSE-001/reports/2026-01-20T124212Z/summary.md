### Turn Summary
Implemented capture_dose_normalization.py CLI that loads dose_experiments defaults (gridsize=2, probe_scale=4, neighbor_count=5) and captures stage telemetry for normalization parity analysis.
Ran dose_legacy_gs2: dataset intensity_scale=262.78 vs fallback=988.21 (ratio=0.266); largest drop at grouped→normalized (ratio=0.26), confirming ~74% amplitude reduction at normalize_data.
Next: compare dose_legacy_gs2 stats with sim_lines gs1_ideal/gs2_ideal to identify which normalization parameters diverge.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T124212Z/dose_normalization/ (capture_config.json, dose_normalization_stats.json, dose_normalization_stats.md)
