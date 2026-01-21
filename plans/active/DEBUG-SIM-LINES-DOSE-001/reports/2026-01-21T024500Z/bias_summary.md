# Intensity Bias Summary

**Spec Reference:** `specs/spec-ptycho-core.md §Normalization Invariants`

- Generated at: 2026-01-21T02:52:25.868378+00:00
- Scenario count: 2

| Scenario | Amp bias mean | Amp median | Phase bias mean | Phase median | Bundle scale | Legacy scale | Δscale | Scale mode | Training NaN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| gs1_ideal | -2.682 | -2.671 | 0.000 | 0.000 | 558.293 | 558.293 | 0.000 | least_squares | Yes |
| gs2_ideal | -2.295 | -2.531 | 0.114 | 0.000 | 273.350 | 273.350 | 0.000 | least_squares | No |

## Scenario: gs1_ideal
- Base directory: `/home/ollie/Documents/tmp/PtychoPINN/plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T024500Z/gs1_ideal`
- Intensity scale: bundle 558.293 vs legacy 558.293 (Δ=0.000)
- normalize_data gain: 0.560
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
| Dataset-derived scale | 576.596 |
| Fallback scale | 988.212 |
| nphotons | 1000000000.000 |
| N (patch size) | 64 |
| E_batch[Σ|Ψ|²] | 3007.859 |
| Delta (dataset - fallback) | -411.616 |
| Ratio (dataset / fallback) | 0.583 |

⚠️ **Dataset vs fallback scale mismatch:** ratio=0.583 indicates that the actual mean intensity per sample (3007.859) differs from the assumed (N/2)² = 1024.
   - The dataset-derived scale is **smaller** than the fallback, meaning the raw diffraction has higher average intensity than assumed → predictions may be overscaled.

### Train/Test Intensity Scale Parity

**Spec Reference:** `specs/spec-ptycho-core.md §Normalization Invariants`

Per the spec, intensity scale is computed as `s = sqrt(nphotons / E_batch[Σ_xy |Ψ|²])`. If train and test splits have different raw diffraction statistics, their dataset-derived scales will differ, potentially contributing to bias during inference.

| Split | E_batch[Σ|Ψ|²] | n_samples | Dataset Scale |
| --- | ---: | ---: | ---: |
| Train | 3208.303 | 256 | 558.293 |
| Test | 3007.859 | 256 | 576.596 |

- nphotons: 1000000000.000
- Train/Test scale ratio: 0.968
- Train/Test scale delta: -18.302
- Deviation from parity: 3.17%

✅ Train/Test scale parity is within 5% tolerance.

### IntensityScaler State

**Spec Reference:** `specs/spec-ptycho-core.md §Normalization Invariants`

Per the architecture, `IntensityScaler` and `IntensityScaler_inv` layers use a shared `log_scale` tf.Variable. The effective scaling factor is `exp(log_scale)`. If this diverges from the recorded bundle/params.cfg value, it may indicate double-scaling or a training-time drift that contributes to amplitude bias.

| Property | Value |
| --- | ---: |
| log_scale (raw) | 6.325 |
| exp(log_scale) | 558.293 |
| trainable | True |
| params.cfg intensity_scale | 558.293 |
| params.cfg trainable | True |
| delta (exp - cfg) | 0.000 |
| ratio (exp / cfg) | 1.000 |

### Training Container X Stats

| Metric | Value |
| --- | ---: |
| shape | [64, 64, 64, 1] |
| dtype | float32 |
| min | 0.000 |
| max | 15.049 |
| mean | 0.085 |
| std | 0.493 |
| nan_count | 0 |

- Inference canvas: padded=828 required=828 fits_canvas=True
- **Largest drop:** Container X → Prediction (ratio=0.000, Δ=-1.000)
  - Per `specs/spec-ptycho-core.md §Normalization Invariants`: symmetry SHALL hold for X_scaled = s · X

### Stage Means
| Stage | Mean |
| --- | ---: |
| Raw diffraction | 0.147 |
| Grouped diffraction | 0.153 |
| Grouped X (normalized) | 0.085 |
| Container X | 0.085 |

### Stage Ratios

Per `specs/spec-ptycho-core.md §Normalization Invariants`: normalized→prediction and prediction→truth deltas indicate where amplitude collapses in the pipeline.

| Transition | Ratio |
| --- | ---: |
| Raw → grouped | 1.041 |
| Grouped → normalized | 0.560 |
| Normalized → prediction | 0.000 |
| Prediction → truth | n/a |

### Normalization Invariant Check

**Spec Reference:** `specs/spec-ptycho-core.md §Normalization Invariants`

Per the spec, symmetry SHALL hold: `X_scaled = s · X`. The product of stage ratios (raw→truth) should ideally equal 1.0 for symmetric normalization.

| Stage Transition | Individual Ratio | Cumulative Product |
| --- | ---: | ---: |
| Raw → grouped | 1.041 | 1.041 |
| Grouped → normalized | 0.560 | 0.583 |
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
| grouped_to_normalized | 0.560 | 0.440 | loss |
| raw_to_grouped | 1.041 | 0.041 | gain |

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
* MAE baseline 2.682 → n/a; RMSE baseline 2.757 → n/a
| Scalar | Value | Scaled MAE | Scaled RMSE |
| --- | ---: | ---: | ---: |
| Ratio mean | n/a | n/a | n/a |
| Ratio median | n/a | n/a | n/a |
| Ratio p05 | n/a | n/a | n/a |
| Ratio p95 | n/a | n/a | n/a |
| Least squares | n/a | n/a | n/a |

### Amplitude Bias
* mean=-2.682, median=-2.671, p05=-3.754, p95=-1.637
* MAE=2.682, RMSE=2.757, max_abs=4.768, pearson_r=n/a

### Phase Bias
* mean=0.000, median=0.000, p05=0.000, p95=0.000
* MAE=0.000, RMSE=0.000, max_abs=0.000

### Normalization Stage Stats
| Stage | Source | Count | Mean | Std | Min | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| raw_diffraction | RawData | 256 | 0.147 | 0.844 | 0.000 | 26.896 |
| grouped_diffraction | RawData.generate_grouped_data | 64 | 0.153 | 0.880 | 0.000 | 26.896 |
| grouped_X_full | normalize_data | 64 | 0.085 | 0.493 | 0.000 | 15.049 |
| container_X | PtychoDataContainer | n/a | 0.085 | 0.493 | 0.000 | 15.049 |

---

## Scenario: gs2_ideal
- Base directory: `/home/ollie/Documents/tmp/PtychoPINN/plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T024500Z/gs2_ideal`
- Intensity scale: bundle 273.350 vs legacy 273.350 (Δ=0.000)
- normalize_data gain: 0.577
- Prediction scale: mode=least_squares value=2.017 source=least_squares
  * Note: least_squares=2.017
- Training NaNs: no

### Loss Composition

**Spec Reference:** `specs/spec-ptycho-workflow.md §Loss and Optimization`

- Epochs trained: 5
- Total loss (final): -252199.297 (min=-252244.469, max=-245504.344)
- Learning rate: last=0.001000 (min=0.001000, max=0.001000)

| Loss Component | Final Value | Contribution | Has NaN |
| --- | ---: | ---: | --- |
| `pred_intensity_loss` | -251269.422 | 99.6% | No |
| `intensity_scaler_inv_loss` | 4.927 | -0.0% | No |
| `trimmed_obj_loss` | 0.000 | -0.0% | No |

**Dominant loss term:** `pred_intensity_loss`
  - Dominance ratio vs next: 50998.5×
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

### Train/Test Intensity Scale Parity

**Spec Reference:** `specs/spec-ptycho-core.md §Normalization Invariants`

Per the spec, intensity scale is computed as `s = sqrt(nphotons / E_batch[Σ_xy |Ψ|²])`. If train and test splits have different raw diffraction statistics, their dataset-derived scales will differ, potentially contributing to bias during inference.

| Split | E_batch[Σ|Ψ|²] | n_samples | Dataset Scale |
| --- | ---: | ---: | ---: |
| Train | 3388.024 | 512 | 543.284 |
| Test | 2995.974 | 512 | 577.738 |

- nphotons: 1000000000.000
- Train/Test scale ratio: 0.940
- Train/Test scale delta: -34.454
- Deviation from parity: 5.96%

⚠️ **Train/Test scale deviation exceeds 5%**: This indicates the raw diffraction statistics differ significantly between splits, which may contribute to inference bias.

### IntensityScaler State

**Spec Reference:** `specs/spec-ptycho-core.md §Normalization Invariants`

Per the architecture, `IntensityScaler` and `IntensityScaler_inv` layers use a shared `log_scale` tf.Variable. The effective scaling factor is `exp(log_scale)`. If this diverges from the recorded bundle/params.cfg value, it may indicate double-scaling or a training-time drift that contributes to amplitude bias.

| Property | Value |
| --- | ---: |
| log_scale (raw) | 5.611 |
| exp(log_scale) | 273.350 |
| trainable | True |
| params.cfg intensity_scale | 273.350 |
| params.cfg trainable | True |
| delta (exp - cfg) | 0.000 |
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
| Normalized → prediction | 4.851 |
| Prediction → truth | 6.561 |

### Normalization Invariant Check

**Spec Reference:** `specs/spec-ptycho-core.md §Normalization Invariants`

Per the spec, symmetry SHALL hold: `X_scaled = s · X`. The product of stage ratios (raw→truth) should ideally equal 1.0 for symmetric normalization.

| Stage Transition | Individual Ratio | Cumulative Product |
| --- | ---: | ---: |
| Raw → grouped | 1.010 | 1.010 |
| Grouped → normalized | 0.577 | 0.583 |
| Normalized → prediction | 4.851 | 2.831 |
| Prediction → truth | 6.561 | 18.571 |

**Full chain product (raw→truth):** 18.571
**Deviation from unity:** 17.571
**Tolerance:** 0.05 (5%)
**Passes tolerance:** ❌ No

⚠️ **Symmetry violated:** The normalization chain does not preserve amplitude as required by `specs/spec-ptycho-core.md §Normalization Invariants`.

**Stage Deviation Breakdown** (sorted by impact):

| Stage | Ratio | Deviation | Effect |
| --- | ---: | ---: | --- |
| prediction_to_truth | 6.561 | 5.561 | gain |
| normalized_to_prediction | 4.851 | 3.851 | gain |
| grouped_to_normalized | 0.577 | 0.423 | loss |
| raw_to_grouped | 1.010 | 0.010 | gain |

**Primary deviation source:** `prediction_to_truth` (ratio=6.561, deviation=5.561)


### Prediction ↔ Truth Scaling
| Metric | Value |
| --- | ---: |
| Ratio mean | 2.021 |
| Ratio median | 1.984 |
| Ratio p05 | 1.405 |
| Ratio p95 | 2.743 |
| Ratio count | 21737 |

* Best scalar: ratio_median (1.984)
* MAE baseline 2.363 → 2.363; RMSE baseline 2.557 → 2.557
| Scalar | Value | Scaled MAE | Scaled RMSE |
| --- | ---: | ---: | ---: |
| Ratio mean | 2.021 | 2.363 | 2.557 |
| Ratio median | 1.984 | 2.363 | 2.557 |
| Ratio p05 | 1.405 | 2.423 | 2.579 |
| Ratio p95 | 2.743 | 2.447 | 2.588 |
| Least squares | 2.017 | 2.363 | 2.557 |

### Amplitude Bias
* mean=-2.295, median=-2.531, p05=-3.696, p95=0.269
* MAE=2.363, RMSE=2.557, max_abs=4.768, pearson_r=0.135

### Phase Bias
* mean=0.114, median=0.000, p05=0.000, p95=0.819
* MAE=0.114, RMSE=0.304, max_abs=0.934

### Normalization Stage Stats
| Stage | Source | Count | Mean | Std | Min | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| raw_diffraction | RawData | 512 | 0.146 | 0.843 | 0.000 | 27.523 |
| grouped_diffraction | RawData.generate_grouped_data | 64 | 0.147 | 0.853 | 0.000 | 27.523 |
| grouped_X_full | normalize_data | 64 | 0.085 | 0.493 | 0.000 | 15.892 |
| container_X | PtychoDataContainer | n/a | 0.085 | 0.493 | 0.000 | 15.892 |

---
