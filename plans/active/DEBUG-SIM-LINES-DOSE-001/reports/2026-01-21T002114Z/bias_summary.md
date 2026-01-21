# Intensity Bias Summary

**Spec Reference:** `specs/spec-ptycho-core.md §Normalization Invariants`

- Generated at: 2026-01-21T00:39:38.280648+00:00
- Scenario count: 1

| Scenario | Amp bias mean | Amp median | Phase bias mean | Phase median | Bundle scale | Legacy scale | Δscale | Scale mode | Training NaN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| gs2_ideal | -2.299 | -2.531 | 0.132 | 0.000 | 494.105 | 494.105 | 0.000 | least_squares | No |

## Scenario: gs2_ideal
- Base directory: `/home/ollie/Documents/tmp/PtychoPINN/plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T002114Z/gs2_ideal`
- Intensity scale: bundle 494.105 vs legacy 494.105 (Δ=0.000)
- normalize_data gain: 0.577
- Prediction scale: mode=least_squares value=2.141 source=least_squares
  * Note: least_squares=2.141
- Training NaNs: no

### Loss Composition

**Spec Reference:** `specs/spec-ptycho-workflow.md §Loss and Optimization`

- Epochs trained: 5
- Total loss (final): -896848.812 (min=-896848.812, max=-883211.062)
- Learning rate: last=0.001000 (min=0.001000, max=0.001000)

| Loss Component | Final Value | Contribution | Has NaN |
| --- | ---: | ---: | --- |
| `pred_intensity_loss` | -893161.750 | 99.6% | No |
| `intensity_scaler_inv_loss` | 8.493 | -0.0% | No |
| `trimmed_obj_loss` | 0.000 | -0.0% | No |

**Dominant loss term:** `pred_intensity_loss`
  - Dominance ratio vs next: 105162.0×
**Inactive components (≈0):** `trimmed_obj_loss`
  - Per `specs/spec-ptycho-workflow.md §Loss and Optimization`: `trimmed_obj_loss=0` indicates `realspace_weight=0` (TV/MAE disabled)

### Intensity Scale Comparison

**Spec Reference:** `specs/spec-ptycho-core.md §Normalization Invariants`

Per the spec, two compliant intensity scale calculation modes are allowed:
1. **Dataset-derived (preferred):** `s = sqrt(nphotons / E_batch[Σ_xy |Ψ|²])`
2. **Closed-form fallback:** `s ≈ sqrt(nphotons) / (N/2)`

If the dataset-derived scale differs significantly from the fallback value (988.21), this indicates the actual data statistics differ from the assumed model.

| Property | Value |
| --- | ---: |
| Dataset-derived scale | 577.738 |
| Fallback scale | 988.212 |
| nphotons | 1000000000.000 |
| N (patch size) | 64 |
| E_batch[Σ|Ψ|²] | 2995.974 |
| Delta (dataset - fallback) | -410.474 |
| Ratio (dataset / fallback) | 0.585 |

⚠️ **Dataset vs fallback scale mismatch:** ratio=0.585 indicates that the actual mean intensity per sample (2995.974) differs from the assumed (N/2)² = 1024.
   - The dataset-derived scale is **smaller** than the fallback, meaning the raw diffraction has higher average intensity than assumed → predictions may be overscaled.

### IntensityScaler State

**Spec Reference:** `specs/spec-ptycho-core.md §Normalization Invariants`

Per the architecture, `IntensityScaler` and `IntensityScaler_inv` layers use a shared `log_scale` tf.Variable. The effective scaling factor is `exp(log_scale)`. If this diverges from the recorded bundle/params.cfg value, it may indicate double-scaling or a training-time drift that contributes to amplitude bias.

| Property | Value |
| --- | ---: |
| log_scale (raw) | 6.203 |
| exp(log_scale) | 494.105 |
| trainable | True |
| params.cfg intensity_scale | 494.105 |
| params.cfg trainable | True |
| delta (exp - cfg) | -0.000 |
| ratio (exp / cfg) | 1.000 |

### Training Container X Stats

| Metric | Value |
| --- | ---: |
| shape | [64, 64, 64, 4] |
| dtype | float32 |
| min | 0.000 |
| max | 15.892 |
| mean | 0.085 |
| std | 0.493 |
| nan_count | 0 |

- Inference canvas: padded=822 required=817 fits_canvas=True
- **Largest drop:** Grouped diffraction → Container X (ratio=0.577, Δ=-0.423)
  - Per `specs/spec-ptycho-core.md §Normalization Invariants`: symmetry SHALL hold for X_scaled = s · X

### Stage Means
| Stage | Mean |
| --- | ---: |
| Raw diffraction | 0.146 |
| Grouped diffraction | 0.147 |
| Grouped X (normalized) | 0.085 |
| Container X | 0.085 |

### Stage Ratios

Per `specs/spec-ptycho-core.md §Normalization Invariants`: normalized→prediction and prediction→truth deltas indicate where amplitude collapses in the pipeline.

| Transition | Ratio |
| --- | ---: |
| Raw → grouped | 1.010 |
| Grouped → normalized | 0.577 |
| Normalized → prediction | 4.811 |
| Prediction → truth | 6.616 |

### Normalization Invariant Check

**Spec Reference:** `specs/spec-ptycho-core.md §Normalization Invariants`

Per the spec, symmetry SHALL hold: `X_scaled = s · X`. The product of stage ratios (raw→truth) should ideally equal 1.0 for symmetric normalization.

| Stage Transition | Individual Ratio | Cumulative Product |
| --- | ---: | ---: |
| Raw → grouped | 1.010 | 1.010 |
| Grouped → normalized | 0.577 | 0.583 |
| Normalized → prediction | 4.811 | 2.807 |
| Prediction → truth | 6.616 | 18.571 |

**Full chain product (raw→truth):** 18.571
**Deviation from unity:** 17.571
**Tolerance:** 0.05 (5%)
**Passes tolerance:** ❌ No

⚠️ **Symmetry violated:** The normalization chain does not preserve amplitude as required by `specs/spec-ptycho-core.md §Normalization Invariants`.

**Stage Deviation Breakdown** (sorted by impact):

| Stage | Ratio | Deviation | Effect |
| --- | ---: | ---: | --- |
| prediction_to_truth | 6.616 | 5.616 | gain |
| normalized_to_prediction | 4.811 | 3.811 | gain |
| grouped_to_normalized | 0.577 | 0.423 | loss |
| raw_to_grouped | 1.010 | 0.010 | gain |

**Primary deviation source:** `prediction_to_truth` (ratio=6.616, deviation=5.616)


### Prediction ↔ Truth Scaling
| Metric | Value |
| --- | ---: |
| Ratio mean | 2.182 |
| Ratio median | 2.121 |
| Ratio p05 | 1.465 |
| Ratio p95 | 3.046 |
| Ratio count | 21737 |

* Best scalar: ratio_median (2.121)
* MAE baseline 2.369 → 2.369; RMSE baseline 2.559 → 2.559
| Scalar | Value | Scaled MAE | Scaled RMSE |
| --- | ---: | ---: | ---: |
| Ratio mean | 2.182 | 2.369 | 2.559 |
| Ratio median | 2.121 | 2.369 | 2.559 |
| Ratio p05 | 1.465 | 2.431 | 2.582 |
| Ratio p95 | 3.046 | 2.468 | 2.600 |
| Least squares | 2.141 | 2.369 | 2.559 |

### Amplitude Bias
* mean=-2.299, median=-2.531, p05=-3.696, p95=0.257
* MAE=2.369, RMSE=2.559, max_abs=4.768, pearson_r=0.136

### Phase Bias
* mean=0.132, median=0.000, p05=0.000, p95=0.947
* MAE=0.132, RMSE=0.352, max_abs=1.073

### Normalization Stage Stats
| Stage | Source | Count | Mean | Std | Min | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| raw_diffraction | RawData | 512 | 0.146 | 0.843 | 0.000 | 27.523 |
| grouped_diffraction | RawData.generate_grouped_data | 64 | 0.147 | 0.853 | 0.000 | 27.523 |
| grouped_X_full | normalize_data | 64 | 0.085 | 0.493 | 0.000 | 15.892 |
| container_X | PtychoDataContainer | n/a | 0.085 | 0.493 | 0.000 | 15.892 |

---
