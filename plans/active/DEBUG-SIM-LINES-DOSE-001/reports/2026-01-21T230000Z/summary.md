### Turn Summary
Implemented Phase D6 training label stats instrumentation: added `compute_dataset_intensity_stats()` to loader.py with backward-compatible API, restored ptycho/cache.py (removed in prior revert), created `record_training_label_stats()` function for NumPy-backed container inspection, and extended `write_intensity_stats_outputs()` with label_vs_truth_analysis sections.
Training scenario run blocked by Keras 3.x API incompatibility in tf_helper.py:1476 (`tf.keras.metrics.mean_absolute_error` was removed); this is a core module issue outside initiative scope.
Next: Either run scenarios in Keras 2.x environment or request maintainer fix for the Keras 3.x compatibility issue.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T230000Z/ (logs/gs1_ideal_runner.log with error trace)
