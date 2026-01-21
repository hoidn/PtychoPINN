### Turn Summary
Implemented D5b forward-pass IntensityScaler tracing in `run_phase_c2_scenario.py` to capture scale diagnostics.
Training hit NaN loss at epoch 1, but the telemetry confirms scales match (100.00% match between model_exp_log_scale and external_intensity_scale).
The ~6.5x amplitude gap is NOT caused by a scale mismatch; next step is to investigate training stability or missing post-inference rescaling.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T030000Z/ (gs2_ideal/run_metadata.json, logs/*.log)
