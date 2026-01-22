### Turn Summary
Extended D6 training label telemetry in `run_phase_c2_scenario.py` to capture Y_amp/Y_I stats and compute `label_vs_truth_analysis` block; fixed Keras 3.x API compatibility issue in `tf_helper.py` (replaced deprecated `tf.keras.metrics.mean_absolute_error` with raw TensorFlow operations).
D6 telemetry confirms training labels (`Y_amp` mean=2.71) match ground truth (mean=2.71) to 0.05%, proving the amplitude gap is NOT in label scaling; the gap is entirely in model output (`output_vs_truth_ratio=0.12`).
Next: Investigate loss function formulation or model architecture for amplitude attenuation root cause.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-22T021500Z/ (intensity_stats.json, bias_summary.md, logs/*.log)
