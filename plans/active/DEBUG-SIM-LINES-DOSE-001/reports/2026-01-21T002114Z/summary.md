### Turn Summary (2026-01-21T004000Z)
Verified D4c dataset-derived intensity scale implementation - all 4 TestIntensityScale tests pass, CLI smoke test passes.
Ran gs2_ideal scenario with fresh evidence: dataset_scale=577.74 vs fallback_scale=988.21 (ratio=0.585) confirms correct computation. The 0.58 ratio is expected behavior reflecting actual data statistics vs theoretical assumptions - not a bug.
Remaining amplitude bias (prediction_to_truth=6.6x) stems from model/loss architecture, not intensity scale computation.
Next: Mark D4c complete; investigate loss wiring or model architecture for the remaining amplitude gap.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T002114Z/ (bias_summary.json, gs2_ideal/, logs/)

### Turn Summary (2026-01-21T002114Z - prior)
Implemented dataset-derived intensity scale formula in `ptycho/train_pinn.py::calculate_intensity_scale()` per spec-ptycho-core.md Normalization Invariants.
The fix computes `s = sqrt(nphotons / E_batch[sum_xy |X|^2])` from actual data statistics instead of always using the closed-form fallback.
Created 4 regression tests in `tests/test_train_pinn.py::TestIntensityScale` covering dataset-derived, fallback, rank-3, and multi-channel cases; all pass.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T002114Z/logs/ (pytest_test_train_pinn.log, pytest_cli_smoke.log)
