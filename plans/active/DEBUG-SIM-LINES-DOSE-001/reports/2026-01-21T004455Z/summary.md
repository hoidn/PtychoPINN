### Turn Summary (2026-01-21T004455Z)
Implemented D4d lazy-container fix in `calculate_intensity_scale()` to prefer `_X_np` (NumPy) over `.X` (TensorFlow tensor), keeping intensity scale computation CPU-bound and avoiding `_tensor_cache` population.
Added `test_lazy_container_does_not_materialize` to TestIntensityScale that asserts `_tensor_cache` stays empty after invoking `calculate_intensity_scale()` on a lazy container.
All 5 TestIntensityScale tests pass; CLI smoke test passes; docs/findings.md PINN-CHUNKED-001 updated with evidence.
Next: Mark D4d complete; gs2_ideal scenario can be re-run to verify no regression if needed.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T004455Z/ (pytest_test_train_pinn.log, pytest_cli_smoke.log)
