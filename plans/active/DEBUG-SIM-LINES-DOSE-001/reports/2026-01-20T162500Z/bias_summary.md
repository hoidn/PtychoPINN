# Intensity Bias Summary

**Spec Reference:** `specs/spec-ptycho-core.md §Normalization Invariants`

- Generated at: 2026-01-20T14:27:58.012247+00:00
- Scenario count: 2

| Scenario | Amp bias mean | Amp median | Phase bias mean | Phase median | Bundle scale | Legacy scale | Δscale | Scale mode | Training NaN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| gs2_base | -2.297 | -2.531 | -0.110 | 0.000 | 988.212 | 988.212 | 0.000 | least_squares | No |
| gs2_ne60 | -2.306 | -2.531 | 0.121 | 0.000 | 988.212 | 988.212 | 0.000 | least_squares | No |

## Scenario: gs2_base
- Base directory: `/home/ollie/Documents/tmp/PtychoPINN/plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T160000Z/gs2_ideal`
- Intensity scale: bundle 988.212 vs legacy 988.212 (Δ=0.000)
- Prediction scale: mode=least_squares value=2.061 source=least_squares
  * Note: least_squares=2.061
- Training NaNs: no

### Loss Composition

**Spec Reference:** `specs/spec-ptycho-workflow.md §Loss and Optimization`

- Epochs trained: 5
- Total loss (final): -3923536.250 (min=-3923994.000, max=-3886826.500)
- Learning rate: last=0.001000 (min=0.001000, max=0.001000)

| Loss Component | Final Value | Contribution | Has NaN |
| --- | ---: | ---: | --- |
| `pred_intensity_loss` | -3892771.500 | 99.2% | No |
| `intensity_scaler_inv_loss` | 16.247 | -0.0% | No |
| `trimmed_obj_loss` | 0.000 | -0.0% | No |

**Dominant loss term:** `pred_intensity_loss`
  - Dominance ratio vs next: 239596.4×
**Inactive components (≈0):** `trimmed_obj_loss`
  - Per `specs/spec-ptycho-workflow.md §Loss and Optimization`: `trimmed_obj_loss=0` indicates `realspace_weight=0` (TV/MAE disabled)

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
| Normalized → prediction | 4.837 |
| Prediction → truth | 6.581 |

### Normalization Invariant Check

**Spec Reference:** `specs/spec-ptycho-core.md §Normalization Invariants`

Per the spec, symmetry SHALL hold: `X_scaled = s · X`. The product of stage ratios (raw→truth) should ideally equal 1.0 for symmetric normalization.

| Stage Transition | Individual Ratio | Cumulative Product |
| --- | ---: | ---: |
| Raw → grouped | 1.010 | 1.010 |
| Grouped → normalized | 0.577 | 0.583 |
| Normalized → prediction | 4.837 | 2.822 |
| Prediction → truth | 6.581 | 18.571 |

**Full chain product (raw→truth):** 18.571
**Deviation from unity:** 17.571
**Tolerance:** 0.05 (5%)
**Passes tolerance:** ❌ No

⚠️ **Symmetry violated:** The normalization chain does not preserve amplitude as required by `specs/spec-ptycho-core.md §Normalization Invariants`.

**Stage Deviation Breakdown** (sorted by impact):

| Stage | Ratio | Deviation | Effect |
| --- | ---: | ---: | --- |
| prediction_to_truth | 6.581 | 5.581 | gain |
| normalized_to_prediction | 4.837 | 3.837 | gain |
| grouped_to_normalized | 0.577 | 0.423 | loss |
| raw_to_grouped | 1.010 | 0.010 | gain |

**Primary deviation source:** `prediction_to_truth` (ratio=6.581, deviation=5.581)


### Prediction ↔ Truth Scaling
| Metric | Value |
| --- | ---: |
| Ratio mean | 2.077 |
| Ratio median | 2.038 |
| Ratio p05 | 1.431 |
| Ratio p95 | 2.824 |
| Ratio count | 21737 |

* Best scalar: ratio_median (2.038)
* MAE baseline 2.364 → 2.364; RMSE baseline 2.558 → 2.558
| Scalar | Value | Scaled MAE | Scaled RMSE |
| --- | ---: | ---: | ---: |
| Ratio mean | 2.077 | 2.364 | 2.558 |
| Ratio median | 2.038 | 2.364 | 2.558 |
| Ratio p05 | 1.431 | 2.425 | 2.579 |
| Ratio p95 | 2.824 | 2.449 | 2.590 |
| Least squares | 2.061 | 2.364 | 2.558 |

### Amplitude Bias
* mean=-2.297, median=-2.531, p05=-3.696, p95=0.254
* MAE=2.364, RMSE=2.558, max_abs=4.768, pearson_r=0.138

### Phase Bias
* mean=-0.110, median=0.000, p05=-0.805, p95=0.000
* MAE=0.110, RMSE=0.293, max_abs=0.991

### Normalization Stage Stats
| Stage | Source | Count | Mean | Std | Min | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| raw_diffraction | RawData | 512 | 0.146 | 0.843 | 0.000 | 27.523 |
| grouped_diffraction | RawData.generate_grouped_data | 64 | 0.147 | 0.853 | 0.000 | 27.523 |
| grouped_X_full | normalize_data | 64 | 0.085 | 0.493 | 0.000 | 15.892 |
| container_X | PtychoDataContainer | n/a | 0.085 | 0.493 | 0.000 | 15.892 |

---

## Scenario: gs2_ne60
- Base directory: `/home/ollie/Documents/tmp/PtychoPINN/plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T140531Z/gs2_ideal_nepochs60`
- Intensity scale: bundle 988.212 vs legacy 988.212 (Δ=0.000)
- normalize_data gain: 0.577
- Prediction scale: mode=least_squares value=1.797 source=least_squares
  * Note: least_squares=1.797
- Training NaNs: no

### Loss Composition

**Spec Reference:** `specs/spec-ptycho-workflow.md §Loss and Optimization`

- Epochs trained: 30
- Total loss (final): -3929574.500 (min=-3929585.250, max=-3876425.000)
- Learning rate: last=0.000100 (min=0.000100, max=0.001000)

| Loss Component | Final Value | Contribution | Has NaN |
| --- | ---: | ---: | --- |
| `pred_intensity_loss` | -3904576.500 | 99.4% | No |
| `intensity_scaler_inv_loss` | 12.801 | -0.0% | No |
| `trimmed_obj_loss` | 0.000 | -0.0% | No |

**Dominant loss term:** `pred_intensity_loss`
  - Dominance ratio vs next: 305010.7×
**Inactive components (≈0):** `trimmed_obj_loss`
  - Per `specs/spec-ptycho-workflow.md §Loss and Optimization`: `trimmed_obj_loss=0` indicates `realspace_weight=0` (TV/MAE disabled)

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
| Normalized → prediction | 4.723 |
| Prediction → truth | 6.739 |

### Normalization Invariant Check

**Spec Reference:** `specs/spec-ptycho-core.md §Normalization Invariants`

Per the spec, symmetry SHALL hold: `X_scaled = s · X`. The product of stage ratios (raw→truth) should ideally equal 1.0 for symmetric normalization.

| Stage Transition | Individual Ratio | Cumulative Product |
| --- | ---: | ---: |
| Raw → grouped | 1.010 | 1.010 |
| Grouped → normalized | 0.577 | 0.583 |
| Normalized → prediction | 4.723 | 2.756 |
| Prediction → truth | 6.739 | 18.571 |

**Full chain product (raw→truth):** 18.571
**Deviation from unity:** 17.571
**Tolerance:** 0.05 (5%)
**Passes tolerance:** ❌ No

⚠️ **Symmetry violated:** The normalization chain does not preserve amplitude as required by `specs/spec-ptycho-core.md §Normalization Invariants`.

**Stage Deviation Breakdown** (sorted by impact):

| Stage | Ratio | Deviation | Effect |
| --- | ---: | ---: | --- |
| prediction_to_truth | 6.739 | 5.739 | gain |
| normalized_to_prediction | 4.723 | 3.723 | gain |
| grouped_to_normalized | 0.577 | 0.423 | loss |
| raw_to_grouped | 1.010 | 0.010 | gain |

**Primary deviation source:** `prediction_to_truth` (ratio=6.739, deviation=5.739)


### Prediction ↔ Truth Scaling
| Metric | Value |
| --- | ---: |
| Ratio mean | 1.895 |
| Ratio median | 1.829 |
| Ratio p05 | 1.200 |
| Ratio p95 | 2.762 |
| Ratio count | 21737 |

* Best scalar: least_squares (1.797)
* MAE baseline 2.377 → 2.377; RMSE baseline 2.563 → 2.563
| Scalar | Value | Scaled MAE | Scaled RMSE |
| --- | ---: | ---: | ---: |
| Ratio mean | 1.895 | 2.380 | 2.563 |
| Ratio median | 1.829 | 2.378 | 2.563 |
| Ratio p05 | 1.200 | 2.444 | 2.588 |
| Ratio p95 | 2.762 | 2.502 | 2.628 |
| Least squares | 1.797 | 2.377 | 2.563 |

### Amplitude Bias
* mean=-2.306, median=-2.531, p05=-3.696, p95=0.185
* MAE=2.377, RMSE=2.563, max_abs=4.768, pearson_r=0.139

### Phase Bias
* mean=0.121, median=0.000, p05=0.000, p95=0.923
* MAE=0.121, RMSE=0.325, max_abs=1.172

### Normalization Stage Stats
| Stage | Source | Count | Mean | Std | Min | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| raw_diffraction | RawData | 512 | 0.146 | 0.843 | 0.000 | 27.523 |
| grouped_diffraction | RawData.generate_grouped_data | 64 | 0.147 | 0.853 | 0.000 | 27.523 |
| grouped_X_full | normalize_data | 64 | 0.085 | 0.493 | 0.000 | 15.892 |
| container_X | PtychoDataContainer | n/a | 0.085 | 0.493 | 0.000 | 15.892 |

---
