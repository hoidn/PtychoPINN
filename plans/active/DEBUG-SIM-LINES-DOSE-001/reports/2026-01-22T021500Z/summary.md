### Turn Summary
Extended `record_training_label_stats()` to capture Y_amp (amplitude), Y_I (intensity), and improved label_vs_truth_analysis with amplitude gap metrics.
Training run blocked by Keras 3 API incompatibility in `ptycho/tf_helper.py` (`tf.keras.metrics.mean_absolute_error` removed); cannot modify core module without authorization.
Next: Request authorization to fix Keras 3 API in tf_helper.py, or revert realspace_weight to 0 to unblock training.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-22T021500Z/ (README.md, logs/)
