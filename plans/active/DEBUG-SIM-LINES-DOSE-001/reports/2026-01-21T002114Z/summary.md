### Turn Summary
Implemented dataset-derived intensity scale formula in `ptycho/train_pinn.py::calculate_intensity_scale()` per spec-ptycho-core.md Normalization Invariants.
The fix computes `s = sqrt(nphotons / E_batch[sum_xy |X|^2])` from actual data statistics instead of always using the closed-form fallback.
Created 4 regression tests in `tests/test_train_pinn.py::TestIntensityScale` covering dataset-derived, fallback, rank-3, and multi-channel cases; all pass.
Next: run gs2_ideal scenario with the fixed scale to verify the dataset/fallback ratio collapses to 1.0 and amplitude bias is reduced.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T002114Z/logs/ (pytest_test_train_pinn.log, pytest_cli_smoke.log)
