### Turn Summary
Verified regression status: 6/7 tests fail in TestIntensityScale, confirming REG-2 (calculate_intensity_scale) is still broken.
The prior input.md delegated the fix but implementation hasn't landed yet; current code only uses closed-form fallback and accesses `.X` (breaks lazy loading).
Next: Ralph implements the 3-priority order fix (dataset_intensity_stats → _X_np → fallback), runs tests, then proceeds to REG-3/REG-4.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-22T234600Z/
