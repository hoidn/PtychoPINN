### Turn Summary
Reviewed D5 evidence showing model predictions are ~6.5x smaller than ground truth (full_chain_product=18.571). IntensityScaler state matches params.cfg but dataset-derived scale is ~2x higher.
Analysis traced the shrinkage to the forward pass: inputs are externally scaled by intensity_scale, divided by IntensityScaler, processed, then multiplied by IntensityScaler_inv. The cancellation appears correct but outputs remain underscaled.
Scoped Phase D5b to add forward-pass diagnostic telemetry without modifying core modules — logging external scale vs model exp_log_scale, input/output means, and amplification ratios.
Next: Ralph instruments the runner with D5b diagnostics, reruns gs2_ideal, and captures the diagnostic metadata.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T030000Z/ (reserved for upcoming evidence)
