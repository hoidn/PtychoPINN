### Turn Summary
Re-ran gs2_ideal scenario to capture valid forward-pass diagnostics (prior T030000Z run had NaN training).
Key finding: IntensityScaler matches perfectly (100%), model amplifies input 7.45× but truth requires 31.8× — output_vs_truth_ratio=0.234 confirms predictions are ~4.3× smaller than ground truth.
The amplitude gap is NOT caused by IntensityScaler mismatch; root cause is elsewhere (likely training targets or missing post-inference rescaling).
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T210000Z/ (gs2_ideal/run_metadata.json, logs/gs2_ideal_runner.log)

## D5b Forward-Pass Diagnostics Summary

### Key Metrics (from run_metadata.json::forward_pass_diagnostics)

| Metric | Value |
| --- | ---: |
| external_intensity_scale | 273.350281 |
| model_exp_log_scale | 273.350270 |
| scale_match_pct | 99.9999961% |
| scale_discrepancy_flag | false |
| input_mean | 0.0851 |
| output_mean | 0.6341 |
| amplification_ratio | 7.452 |
| ground_truth_mean | 2.708 |
| output_vs_truth_ratio | 0.234 |

### Interpretation

1. **IntensityScaler is NOT the source of the amplitude gap**
   - `model_exp_log_scale` matches `external_intensity_scale` to 7 significant figures
   - No scale mismatch at inference time

2. **Forward-pass amplification is working**
   - Model amplifies normalized inputs by ~7.45× (`0.085 → 0.634`)
   - This is the expected behavior of the reconstruction pipeline

3. **Gap is between model output and ground truth**
   - `output_vs_truth_ratio = 0.234` → predictions are 23.4% of truth magnitude
   - The ~4.3× gap (1/0.234) indicates model learns to produce smaller outputs than the target labels

4. **Root cause hypotheses**
   - Training targets (Y_amp/Y_I) may be scaled differently than inference ground truth
   - The model may learn to minimize loss with suboptimal amplitude scaling
   - There may be a missing post-inference rescaling step that the original dose_experiments pipeline had

### Spec Compliance

Per `specs/spec-ptycho-core.md §Normalization Invariants`:
- The IntensityScaler symmetry (`exp(log_scale)` → forward scaling, inverse at output) is preserved
- The bundle intensity_scale matches params.cfg (delta < 1e-4)
- The deviation is in the model output magnitude, not the scaling layers

### Training Health

- Epochs completed: 5/5 (no EarlyStopping)
- NaN status: **None** (all metrics finite)
- Final loss: -252,244 (pred_intensity_loss dominant at 99.7%)
