### Turn Summary
Implemented Phase D5 train/test intensity-scale parity instrumentation: runner now computes dataset_intensity_stats for both splits before training, derives per-split scales, and persists split_intensity_stats to run_metadata.json.
Main problem was quantifying how train vs test raw diffraction statistics diverge; solved by adding compute_dataset_intensity_stats calls on diff3d arrays (PINN-CHUNKED-001 compliant) with 5% deviation threshold flagging.
Evidence shows gs1_ideal at 3.17% deviation (within tolerance) and gs2_ideal at 5.96% (exceeds 5%). Amplitude bias persists, pointing to forward-pass scale handling as next investigation target.
Next: D5b forward-pass instrumentation to trace IntensityScaler output vs model prediction.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T024500Z/ (bias_summary.md, logs/*.log)
