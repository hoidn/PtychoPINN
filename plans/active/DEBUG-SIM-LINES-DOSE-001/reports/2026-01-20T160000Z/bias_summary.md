# Intensity Bias Summary
- Generated at: 2026-01-20T10:04:09.846915+00:00
- Scenario count: 2

| Scenario | Amp bias mean | Amp median | Phase bias mean | Phase median | Bundle scale | Legacy scale | Δscale | Scale mode | Training NaN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| gs1_ideal | -2.682 | -2.671 | 0.000 | 0.000 | 988.212 | 988.212 | 0.000 | least_squares | Yes |
| gs2_ideal | -2.296 | -2.531 | -0.121 | 0.000 | 988.212 | 988.212 | 0.000 | least_squares | No |

## Scenario: gs1_ideal
- Base directory: `/home/ollie/Documents/tmp/PtychoPINN/plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T160000Z/gs1_ideal`
- Intensity scale: bundle 988.212 vs legacy 988.212 (Δ=0.000)
- Prediction scale: mode=least_squares value=n/a source=None
- Training NaNs: YES (metrics: intensity_scaler_inv_loss, loss, pred_intensity_loss, train_loss, val_intensity_scaler_inv_loss, val_loss, val_pred_intensity_loss)
- Inference canvas: padded=828 required=828 fits_canvas=True
- Largest drop: Container X → Prediction (ratio=0.000, Δ=-1.000)

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

## Scenario: gs2_ideal
- Base directory: `/home/ollie/Documents/tmp/PtychoPINN/plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T160000Z/gs2_ideal`
- Intensity scale: bundle 988.212 vs legacy 988.212 (Δ=0.000)
- Prediction scale: mode=least_squares value=1.911 source=least_squares
  * Note: least_squares=1.911
- Training NaNs: no
- Inference canvas: padded=822 required=817 fits_canvas=True
- Largest drop: Grouped diffraction → Container X (ratio=0.577, Δ=-0.423)

### Stage Means
| Stage | Mean |
| --- | ---: |
| Raw diffraction | 0.146 |
| Grouped diffraction | 0.147 |
| Grouped X (normalized) | 0.085 |
| Container X | 0.085 |

### Stage Ratios
| Transition | Ratio |
| --- | ---: |
| Raw → grouped | 1.010 |
| Grouped → normalized | 0.577 |
| Normalized → prediction | 4.844 |
| Prediction → truth | 6.570 |


### Prediction ↔ Truth Scaling
| Metric | Value |
| --- | ---: |
| Ratio mean | 1.921 |
| Ratio median | 1.882 |
| Ratio p05 | 1.326 |
| Ratio p95 | 2.622 |
| Ratio count | 21737 |

* Best scalar: ratio_median (1.882)
* MAE baseline 2.364 → 2.364; RMSE baseline 2.558 → 2.558
| Scalar | Value | Scaled MAE | Scaled RMSE |
| --- | ---: | ---: | ---: |
| Ratio mean | 1.921 | 2.364 | 2.558 |
| Ratio median | 1.882 | 2.364 | 2.558 |
| Ratio p05 | 1.326 | 2.425 | 2.580 |
| Ratio p95 | 2.622 | 2.451 | 2.590 |
| Least squares | 1.911 | 2.364 | 2.558 |

### Amplitude Bias
* mean=-2.296, median=-2.531, p05=-3.696, p95=0.267
* MAE=2.364, RMSE=2.558, max_abs=4.768, pearson_r=0.135

### Phase Bias
* mean=-0.121, median=0.000, p05=-0.869, p95=0.000
* MAE=0.121, RMSE=0.322, max_abs=1.041

### Normalization Stage Stats
| Stage | Source | Count | Mean | Std | Min | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| raw_diffraction | RawData | 512 | 0.146 | 0.843 | 0.000 | 27.523 |
| grouped_diffraction | RawData.generate_grouped_data | 64 | 0.147 | 0.853 | 0.000 | 27.523 |
| grouped_X_full | normalize_data | 64 | 0.085 | 0.493 | 0.000 | 15.892 |
| container_X | PtychoDataContainer | n/a | 0.085 | 0.493 | 0.000 | 15.892 |

---
