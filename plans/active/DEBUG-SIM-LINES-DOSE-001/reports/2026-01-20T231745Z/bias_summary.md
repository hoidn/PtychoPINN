# Intensity Bias Summary

**Spec Reference:** `specs/spec-ptycho-core.md §Normalization Invariants`

- Generated at: 2026-01-20T23:27:34.968596+00:00
- Scenario count: 2

| Scenario | Amp bias mean | Amp median | Phase bias mean | Phase median | Bundle scale | Legacy scale | Δscale | Scale mode | Training NaN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| gs2_base | -2.673 | -2.667 | 0.000 | 0.000 | 988.212 | 988.212 | 0.000 | least_squares | Yes |
| gs2_ne60 | -2.307 | -2.531 | 0.125 | 0.000 | 988.212 | 988.212 | 0.000 | least_squares | No |

## Scenario: gs2_base
- Base directory: `/home/ollie/Documents/tmp/PtychoPINN/plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T231745Z/gs2_ideal`
- Intensity scale: bundle 988.212 vs legacy 988.212 (Δ=0.000)
- normalize_data gain: 0.577
- Prediction scale: mode=least_squares value=n/a source=None
- Training NaNs: YES (metrics: intensity_scaler_inv_loss, loss, pred_intensity_loss, train_loss, val_intensity_scaler_inv_loss, val_loss, val_pred_intensity_loss)

### Loss Composition

**Spec Reference:** `specs/spec-ptycho-workflow.md §Loss and Optimization`

- Epochs trained: 4
- Total loss (final): nan (min=nan, max=nan)
- Learning rate: last=0.000500 (min=0.000500, max=0.001000)

| Loss Component | Final Value | Contribution | Has NaN |
| --- | ---: | ---: | --- |
| `pred_intensity_loss` | nan | n/a | Yes |
| `intensity_scaler_inv_loss` | nan | n/a | Yes |
| `trimmed_obj_loss` | 0.000 | n/a | No |

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
| log_scale (raw) | 6.896 |
| exp(log_scale) | 988.212 |
| trainable | True |
| params.cfg intensity_scale | 988.212 |
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
- **Largest drop:** Container X → Prediction (ratio=0.000, Δ=-1.000)
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
| Normalized → prediction | 0.000 |
| Prediction → truth | n/a |

### Normalization Invariant Check

**Spec Reference:** `specs/spec-ptycho-core.md §Normalization Invariants`

Per the spec, symmetry SHALL hold: `X_scaled = s · X`. The product of stage ratios (raw→truth) should ideally equal 1.0 for symmetric normalization.

| Stage Transition | Individual Ratio | Cumulative Product |
| --- | ---: | ---: |
| Raw → grouped | 1.010 | 1.010 |
| Grouped → normalized | 0.577 | 0.583 |
| Normalized → prediction | 0.000 | 0.000 |
| Prediction → truth | n/a | n/a |

**Full chain product (raw→truth):** n/a
**Deviation from unity:** n/a
**Tolerance:** 0.05 (5%)
**Passes tolerance:** ⚠️ N/A

**Stage Deviation Breakdown** (sorted by impact):

| Stage | Ratio | Deviation | Effect |
| --- | ---: | ---: | --- |
| normalized_to_prediction | 0.000 | 1.000 | loss |
| grouped_to_normalized | 0.577 | 0.423 | loss |
| raw_to_grouped | 1.010 | 0.010 | gain |

**Primary deviation source:** `normalized_to_prediction` (ratio=0.000, deviation=1.000)


### Prediction ↔ Truth Scaling
| Metric | Value |
| --- | ---: |
| Ratio mean | n/a |
| Ratio median | n/a |
| Ratio p05 | n/a |
| Ratio p95 | n/a |
| Ratio count | 0 |

* Best scalar: n/a (n/a)
* MAE baseline 2.673 → n/a; RMSE baseline 2.749 → n/a
| Scalar | Value | Scaled MAE | Scaled RMSE |
| --- | ---: | ---: | ---: |
| Ratio mean | n/a | n/a | n/a |
| Ratio median | n/a | n/a | n/a |
| Ratio p05 | n/a | n/a | n/a |
| Ratio p95 | n/a | n/a | n/a |
| Least squares | n/a | n/a | n/a |

### Amplitude Bias
* mean=-2.673, median=-2.667, p05=-3.742, p95=-1.619
* MAE=2.673, RMSE=2.749, max_abs=4.768, pearson_r=n/a

### Phase Bias
* mean=0.000, median=0.000, p05=0.000, p95=0.000
* MAE=0.000, RMSE=0.000, max_abs=0.000

### Normalization Stage Stats
| Stage | Source | Count | Mean | Std | Min | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| raw_diffraction | RawData | 512 | 0.146 | 0.843 | 0.000 | 27.523 |
| grouped_diffraction | RawData.generate_grouped_data | 64 | 0.147 | 0.853 | 0.000 | 27.523 |
| grouped_X_full | normalize_data | 64 | 0.085 | 0.493 | 0.000 | 15.892 |
| container_X | PtychoDataContainer | n/a | 0.085 | 0.493 | 0.000 | 15.892 |

---

## Scenario: gs2_ne60
- Base directory: `/home/ollie/Documents/tmp/PtychoPINN/plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T231745Z/gs2_ideal_nepochs60`
- Intensity scale: bundle 988.212 vs legacy 988.212 (Δ=0.000)
- normalize_data gain: 0.577
- Prediction scale: mode=least_squares value=1.823 source=least_squares
  * Note: least_squares=1.823
- Training NaNs: no

### Loss Composition

**Spec Reference:** `specs/spec-ptycho-workflow.md §Loss and Optimization`

- Epochs trained: 50
- Total loss (final): -3929596.000 (min=-3929627.750, max=-3838297.500)
- Learning rate: last=0.000100 (min=0.000100, max=0.001000)

| Loss Component | Final Value | Contribution | Has NaN |
| --- | ---: | ---: | --- |
| `pred_intensity_loss` | -3901522.750 | 99.3% | No |
| `intensity_scaler_inv_loss` | 13.257 | -0.0% | No |
| `trimmed_obj_loss` | 0.000 | -0.0% | No |

**Dominant loss term:** `pred_intensity_loss`
  - Dominance ratio vs next: 294309.4×
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
| log_scale (raw) | 6.896 |
| exp(log_scale) | 988.212 |
| trainable | True |
| params.cfg intensity_scale | 988.212 |
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
| Normalized → prediction | 4.718 |
| Prediction → truth | 6.746 |

### Normalization Invariant Check

**Spec Reference:** `specs/spec-ptycho-core.md §Normalization Invariants`

Per the spec, symmetry SHALL hold: `X_scaled = s · X`. The product of stage ratios (raw→truth) should ideally equal 1.0 for symmetric normalization.

| Stage Transition | Individual Ratio | Cumulative Product |
| --- | ---: | ---: |
| Raw → grouped | 1.010 | 1.010 |
| Grouped → normalized | 0.577 | 0.583 |
| Normalized → prediction | 4.718 | 2.753 |
| Prediction → truth | 6.746 | 18.571 |

**Full chain product (raw→truth):** 18.571
**Deviation from unity:** 17.571
**Tolerance:** 0.05 (5%)
**Passes tolerance:** ❌ No

⚠️ **Symmetry violated:** The normalization chain does not preserve amplitude as required by `specs/spec-ptycho-core.md §Normalization Invariants`.

**Stage Deviation Breakdown** (sorted by impact):

| Stage | Ratio | Deviation | Effect |
| --- | ---: | ---: | --- |
| prediction_to_truth | 6.746 | 5.746 | gain |
| normalized_to_prediction | 4.718 | 3.718 | gain |
| grouped_to_normalized | 0.577 | 0.423 | loss |
| raw_to_grouped | 1.010 | 0.010 | gain |

**Primary deviation source:** `prediction_to_truth` (ratio=6.746, deviation=5.746)


### Prediction ↔ Truth Scaling
| Metric | Value |
| --- | ---: |
| Ratio mean | 1.928 |
| Ratio median | 1.862 |
| Ratio p05 | 1.214 |
| Ratio p95 | 2.848 |
| Ratio count | 21737 |

* Best scalar: least_squares (1.823)
* MAE baseline 2.378 → 2.378; RMSE baseline 2.563 → 2.563
| Scalar | Value | Scaled MAE | Scaled RMSE |
| --- | ---: | ---: | ---: |
| Ratio mean | 1.928 | 2.380 | 2.564 |
| Ratio median | 1.862 | 2.378 | 2.563 |
| Ratio p05 | 1.214 | 2.444 | 2.588 |
| Ratio p95 | 2.848 | 2.512 | 2.634 |
| Least squares | 1.823 | 2.378 | 2.563 |

### Amplitude Bias
* mean=-2.307, median=-2.531, p05=-3.696, p95=0.183
* MAE=2.378, RMSE=2.563, max_abs=4.768, pearson_r=0.140

### Phase Bias
* mean=0.125, median=0.000, p05=0.000, p95=0.937
* MAE=0.125, RMSE=0.335, max_abs=1.158

### Normalization Stage Stats
| Stage | Source | Count | Mean | Std | Min | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| raw_diffraction | RawData | 512 | 0.146 | 0.843 | 0.000 | 27.523 |
| grouped_diffraction | RawData.generate_grouped_data | 64 | 0.147 | 0.853 | 0.000 | 27.523 |
| grouped_X_full | normalize_data | 64 | 0.085 | 0.493 | 0.000 | 15.892 |
| container_X | PtychoDataContainer | n/a | 0.085 | 0.493 | 0.000 | 15.892 |

---
