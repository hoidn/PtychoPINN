# Intensity Bias Summary

**Spec Reference:** `specs/spec-ptycho-core.md §Normalization Invariants`

- Generated at: 2026-01-20T12:07:53.429993+00:00
- Scenario count: 2

| Scenario | Amp bias mean | Amp median | Phase bias mean | Phase median | Bundle scale | Legacy scale | Δscale | Scale mode | Training NaN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| gs1_ideal | -2.682 | -2.671 | 0.000 | 0.000 | 988.212 | 988.212 | 0.000 | least_squares | Yes |
| dose_legacy_gs2 | -2.604 | -2.594 | 0.008 | 0.000 | 988.212 | 988.212 | 0.000 | none | No |

## Scenario: gs1_ideal
- Base directory: `/home/ollie/Documents/tmp/PtychoPINN/plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T114126Z/gs1_ideal`
- Intensity scale: bundle 988.212 vs legacy 988.212 (Δ=0.000)
- normalize_data gain: 0.560
- Prediction scale: mode=least_squares value=n/a source=None
- Training NaNs: YES (metrics: intensity_scaler_inv_loss, loss, pred_intensity_loss, train_loss, val_intensity_scaler_inv_loss, val_loss, val_pred_intensity_loss)
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
| Transition | Ratio |
| --- | ---: |
| Raw → grouped | 1.041 |
| Grouped → normalized | 0.560 |
| Normalized → prediction | 0.000 |
| Prediction → truth | n/a |


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

## Scenario: dose_legacy_gs2
- Base directory: `/home/ollie/Documents/tmp/PtychoPINN/plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T114126Z/dose_legacy_gs2`
- Intensity scale: bundle 988.212 vs legacy 988.212 (Δ=0.000)
- normalize_data gain: 0.273
- Prediction scale: mode=none value=n/a source=None
- Training NaNs: no
- Inference canvas: padded=820 required=824 fits_canvas=False
- **Largest drop:** Grouped diffraction → Container X (ratio=0.273, Δ=-0.727)
  - Per `specs/spec-ptycho-core.md §Normalization Invariants`: symmetry SHALL hold for X_scaled = s · X

### Stage Means
| Stage | Mean |
| --- | ---: |
| Raw diffraction | 1.363 |
| Grouped diffraction | 1.378 |
| Grouped X (normalized) | 0.376 |
| Container X | 0.376 |

### Stage Ratios
| Transition | Ratio |
| --- | ---: |
| Raw → grouped | 1.011 |
| Grouped → normalized | 0.273 |
| Normalized → prediction | 0.277 |
| Prediction → truth | 26.024 |


### Prediction ↔ Truth Scaling
| Metric | Value |
| --- | ---: |
| Ratio mean | 3.962 |
| Ratio median | 3.920 |
| Ratio p05 | 2.697 |
| Ratio p95 | 5.368 |
| Ratio count | 21161 |

* Best scalar: least_squares (3.902)
* MAE baseline 2.604 → 2.366; RMSE baseline 2.684 → 2.557
| Scalar | Value | Scaled MAE | Scaled RMSE |
| --- | ---: | ---: | ---: |
| Ratio mean | 3.962 | 2.366 | 2.557 |
| Ratio median | 3.920 | 2.366 | 2.557 |
| Ratio p05 | 2.697 | 2.430 | 2.579 |
| Ratio p95 | 5.368 | 2.452 | 2.590 |
| Least squares | 3.902 | 2.366 | 2.557 |

### Amplitude Bias
* mean=-2.604, median=-2.594, p05=-3.695, p95=-1.558
* MAE=2.604, RMSE=2.684, max_abs=4.856, pearson_r=0.168

### Phase Bias
* mean=0.008, median=0.000, p05=0.000, p95=0.066
* MAE=0.008, RMSE=0.024, max_abs=0.181

### Normalization Stage Stats
| Stage | Source | Count | Mean | Std | Min | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| raw_diffraction | RawData | 512 | 1.363 | 1.205 | 0.000 | 6.189 |
| grouped_diffraction | RawData.generate_grouped_data | 64 | 1.378 | 1.210 | 0.000 | 6.189 |
| grouped_X_full | normalize_data | 64 | 0.376 | 0.330 | 0.000 | 1.687 |
| container_X | PtychoDataContainer | n/a | 0.376 | 0.330 | 0.000 | 1.687 |

---
