### Turn Summary
Processed reviewer override identifying critical regressions: D4f dataset_intensity_stats handling removed, Phase C canvas jitter guard deleted, metrics evaluation helper deleted, loss-weight constraint violated.
Updated fix_plan.md, implementation.md (added REGRESSION RECOVERY section), and findings.md with regression notes. Wrote input.md delegating REG-2 (calculate_intensity_scale) fix to Ralph as first priority.
Next: Ralph restores dataset-derived intensity scale priority in train_pinn.py, then continues to REG-3 (canvas guard) and REG-4 (metrics helper).
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-22T233342Z/
