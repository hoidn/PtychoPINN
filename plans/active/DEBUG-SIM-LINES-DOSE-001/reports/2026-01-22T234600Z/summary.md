### Turn Summary

Restored the 4-priority `calculate_intensity_scale()` implementation per specs/spec-ptycho-core.md §Normalization Invariants.
The function now checks: (1) dataset_intensity_stats, (2) _X_np NumPy backing, (3) .X tensor, (4) closed-form fallback.
All 7 TestIntensityScale tests pass (7/7), plus the CLI smoke test.
Next: Continue with REG-3 to restore `_update_max_position_jitter_from_offsets()`.

Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-22T234600Z/logs/
- pytest_train_pinn.log (7 passed)
- pytest_train_pinn_collect.log (7 tests)
- pytest_cli_smoke.log (1 passed)
