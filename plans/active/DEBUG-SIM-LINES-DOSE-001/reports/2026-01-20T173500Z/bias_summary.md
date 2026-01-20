# Intensity Bias Summary

**Spec Reference:** `specs/spec-ptycho-core.md §Normalization Invariants`

- Generated at: 2026-01-20T14:48:12.773767+00:00
- Scenario count: 2

| Scenario | Amp bias mean | Amp median | Phase bias mean | Phase median | Bundle scale | Legacy scale | Δscale | Scale mode | Training NaN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| gs2_base | -2.298 | -2.531 | -0.122 | 0.000 | 988.212 | 988.212 | 0.000 | least_squares | No |
| gs2_ne60 | -2.307 | -2.531 | -0.125 | 0.000 | 988.212 | 988.212 | 0.000 | least_squares | No |

## Scenario: gs2_base
- Base directory: `/home/ollie/Documents/tmp/PtychoPINN/plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T173500Z/gs2_ideal`
- Intensity scale: bundle 988.212 vs legacy 988.212 (Δ=0.000)
- normalize_data gain: 0.577
- Prediction scale: mode=least_squares value=1.890 source=least_squares
  * Note: least_squares=1.89
- Training NaNs: no

### Loss Composition

**Spec Reference:** `specs/spec-ptycho-workflow.md §Loss and Optimization`

- Epochs trained: 5
- Total loss (final): -3923681.000 (min=-3923681.000, max=-3887179.000)
- Learning rate: last=0.000500 (min=0.000500, max=0.001000)

| Loss Component | Final Value | Contribution | Has NaN |
| --- | ---: | ---: | --- |
| `pred_intensity_loss` | -3907712.000 | 99.6% | No |
| `intensity_scaler_inv_loss` | 16.754 | -0.0% | No |
| `trimmed_obj_loss` | 0.000 | -0.0% | No |

**Dominant loss term:** `pred_intensity_loss`
  - Dominance ratio vs next: 233239.1×
**Inactive components (≈0):** `trimmed_obj_loss`
  - Per `specs/spec-ptycho-workflow.md §Loss and Optimization`: `trimmed_obj_loss=0` indicates `realspace_weight=0` (TV/MAE disabled)

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
| Normalized → prediction | 4.824 |
| Prediction → truth | 6.598 |

### Normalization Invariant Check

**Spec Reference:** `specs/spec-ptycho-core.md §Normalization Invariants`

Per the spec, symmetry SHALL hold: `X_scaled = s · X`. The product of stage ratios (raw→truth) should ideally equal 1.0 for symmetric normalization.

| Stage Transition | Individual Ratio | Cumulative Product |
| --- | ---: | ---: |
| Raw → grouped | 1.010 | 1.010 |
| Grouped → normalized | 0.577 | 0.583 |
| Normalized → prediction | 4.824 | 2.815 |
| Prediction → truth | 6.598 | 18.571 |

**Full chain product (raw→truth):** 18.571
**Deviation from unity:** 17.571
**Tolerance:** 0.05 (5%)
**Passes tolerance:** ❌ No

⚠️ **Symmetry violated:** The normalization chain does not preserve amplitude as required by `specs/spec-ptycho-core.md §Normalization Invariants`.

**Stage Deviation Breakdown** (sorted by impact):

| Stage | Ratio | Deviation | Effect |
| --- | ---: | ---: | --- |
| prediction_to_truth | 6.598 | 5.598 | gain |
| normalized_to_prediction | 4.824 | 3.824 | gain |
| grouped_to_normalized | 0.577 | 0.423 | loss |
| raw_to_grouped | 1.010 | 0.010 | gain |

**Primary deviation source:** `prediction_to_truth` (ratio=6.598, deviation=5.598)


### Prediction ↔ Truth Scaling
| Metric | Value |
| --- | ---: |
| Ratio mean | 1.915 |
| Ratio median | 1.868 |
| Ratio p05 | 1.300 |
| Ratio p95 | 2.642 |
| Ratio count | 21737 |

* Best scalar: ratio_median (1.868)
* MAE baseline 2.366 → 2.366; RMSE baseline 2.558 → 2.558
| Scalar | Value | Scaled MAE | Scaled RMSE |
| --- | ---: | ---: | ---: |
| Ratio mean | 1.915 | 2.367 | 2.558 |
| Ratio median | 1.868 | 2.366 | 2.558 |
| Ratio p05 | 1.300 | 2.428 | 2.581 |
| Ratio p95 | 2.642 | 2.459 | 2.595 |
| Least squares | 1.890 | 2.366 | 2.558 |

### Amplitude Bias
* mean=-2.298, median=-2.531, p05=-3.696, p95=0.257
* MAE=2.366, RMSE=2.558, max_abs=4.768, pearson_r=0.137

### Phase Bias
* mean=-0.122, median=0.000, p05=-0.893, p95=0.000
* MAE=0.122, RMSE=0.326, max_abs=1.058

### Normalization Stage Stats
| Stage | Source | Count | Mean | Std | Min | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| raw_diffraction | RawData | 512 | 0.146 | 0.843 | 0.000 | 27.523 |
| grouped_diffraction | RawData.generate_grouped_data | 64 | 0.147 | 0.853 | 0.000 | 27.523 |
| grouped_X_full | normalize_data | 64 | 0.085 | 0.493 | 0.000 | 15.892 |
| container_X | PtychoDataContainer | n/a | 0.085 | 0.493 | 0.000 | 15.892 |

---

## Scenario: gs2_ne60
- Base directory: `/home/ollie/Documents/tmp/PtychoPINN/plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T173500Z/gs2_ideal_nepochs60`
- Intensity scale: bundle 988.212 vs legacy 988.212 (Δ=0.000)
- normalize_data gain: 0.577
- Prediction scale: mode=least_squares value=1.796 source=least_squares
  * Note: least_squares=1.796
- Training NaNs: no

### Loss Composition

**Spec Reference:** `specs/spec-ptycho-workflow.md §Loss and Optimization`

- Epochs trained: 49
- Total loss (final): -3929730.000 (min=-3929777.250, max=-3855680.250)
- Learning rate: last=0.000100 (min=0.000100, max=0.001000)

| Loss Component | Final Value | Contribution | Has NaN |
| --- | ---: | ---: | --- |
| `pred_intensity_loss` | -3938092.750 | 100.2% | No |
| `intensity_scaler_inv_loss` | 13.012 | -0.0% | No |
| `trimmed_obj_loss` | 0.000 | -0.0% | No |

**Dominant loss term:** `pred_intensity_loss`
  - Dominance ratio vs next: 302657.3×
**Inactive components (≈0):** `trimmed_obj_loss`
  - Per `specs/spec-ptycho-workflow.md §Loss and Optimization`: `trimmed_obj_loss=0` indicates `realspace_weight=0` (TV/MAE disabled)

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
| Normalized → prediction | 4.712 |
| Prediction → truth | 6.755 |

### Normalization Invariant Check

**Spec Reference:** `specs/spec-ptycho-core.md §Normalization Invariants`

Per the spec, symmetry SHALL hold: `X_scaled = s · X`. The product of stage ratios (raw→truth) should ideally equal 1.0 for symmetric normalization.

| Stage Transition | Individual Ratio | Cumulative Product |
| --- | ---: | ---: |
| Raw → grouped | 1.010 | 1.010 |
| Grouped → normalized | 0.577 | 0.583 |
| Normalized → prediction | 4.712 | 2.749 |
| Prediction → truth | 6.755 | 18.571 |

**Full chain product (raw→truth):** 18.571
**Deviation from unity:** 17.571
**Tolerance:** 0.05 (5%)
**Passes tolerance:** ❌ No

⚠️ **Symmetry violated:** The normalization chain does not preserve amplitude as required by `specs/spec-ptycho-core.md §Normalization Invariants`.

**Stage Deviation Breakdown** (sorted by impact):

| Stage | Ratio | Deviation | Effect |
| --- | ---: | ---: | --- |
| prediction_to_truth | 6.755 | 5.755 | gain |
| normalized_to_prediction | 4.712 | 3.712 | gain |
| grouped_to_normalized | 0.577 | 0.423 | loss |
| raw_to_grouped | 1.010 | 0.010 | gain |

**Primary deviation source:** `prediction_to_truth` (ratio=6.755, deviation=5.755)


### Prediction ↔ Truth Scaling
| Metric | Value |
| --- | ---: |
| Ratio mean | 1.905 |
| Ratio median | 1.844 |
| Ratio p05 | 1.186 |
| Ratio p95 | 2.814 |
| Ratio count | 21737 |

* Best scalar: least_squares (1.796)
* MAE baseline 2.378 → 2.378; RMSE baseline 2.563 → 2.563
| Scalar | Value | Scaled MAE | Scaled RMSE |
| --- | ---: | ---: | ---: |
| Ratio mean | 1.905 | 2.380 | 2.564 |
| Ratio median | 1.844 | 2.379 | 2.563 |
| Ratio p05 | 1.186 | 2.447 | 2.589 |
| Ratio p95 | 2.814 | 2.513 | 2.636 |
| Least squares | 1.796 | 2.378 | 2.563 |

### Amplitude Bias
* mean=-2.307, median=-2.531, p05=-3.696, p95=0.165
* MAE=2.378, RMSE=2.563, max_abs=4.768, pearson_r=0.140

### Phase Bias
* mean=-0.125, median=0.000, p05=-0.934, p95=0.000
* MAE=0.125, RMSE=0.335, max_abs=1.169

### Normalization Stage Stats
| Stage | Source | Count | Mean | Std | Min | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| raw_diffraction | RawData | 512 | 0.146 | 0.843 | 0.000 | 27.523 |
| grouped_diffraction | RawData.generate_grouped_data | 64 | 0.147 | 0.853 | 0.000 | 27.523 |
| grouped_X_full | normalize_data | 64 | 0.085 | 0.493 | 0.000 | 15.892 |
| container_X | PtychoDataContainer | n/a | 0.085 | 0.493 | 0.000 | 15.892 |

---
