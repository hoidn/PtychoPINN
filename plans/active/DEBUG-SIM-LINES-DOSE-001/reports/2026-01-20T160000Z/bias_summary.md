# Intensity Bias Summary
- Generated at: 2026-01-20T10:18:18.890471+00:00
- Scenario count: 2

| Scenario | Amp bias mean | Amp median | Phase bias mean | Phase median | Bundle scale | Legacy scale | Δscale | Scale mode | Training NaN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| gs1_ideal | -2.291 | -2.524 | 0.131 | 0.000 | 988.212 | 988.212 | 0.000 | least_squares | No |
| gs2_ideal | -2.297 | -2.531 | -0.110 | 0.000 | 988.212 | 988.212 | 0.000 | least_squares | No |

## Scenario: gs1_ideal
- Base directory: `/home/ollie/Documents/tmp/PtychoPINN/plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T160000Z/gs1_ideal`
- Intensity scale: bundle 988.212 vs legacy 988.212 (Δ=0.000)
- Prediction scale: mode=least_squares value=1.710 source=least_squares
  * Note: least_squares=1.71
- Training NaNs: no
- Inference canvas: padded=828 required=828 fits_canvas=True
- Largest drop: Grouped diffraction → Container X (ratio=0.560, Δ=-0.440)

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
| Normalized → prediction | 4.879 |
| Prediction → truth | 6.495 |


### Prediction ↔ Truth Scaling
| Metric | Value |
| --- | ---: |
| Ratio mean | 1.811 |
| Ratio median | 1.749 |
| Ratio p05 | 1.094 |
| Ratio p95 | 2.758 |
| Ratio count | 23065 |

* Best scalar: least_squares (1.710)
* MAE baseline 2.370 → 2.370; RMSE baseline 2.559 → 2.559
| Scalar | Value | Scaled MAE | Scaled RMSE |
| --- | ---: | ---: | ---: |
| Ratio mean | 1.811 | 2.373 | 2.560 |
| Ratio median | 1.749 | 2.371 | 2.559 |
| Ratio p05 | 1.094 | 2.445 | 2.589 |
| Ratio p95 | 2.758 | 2.528 | 2.646 |
| Least squares | 1.710 | 2.370 | 2.559 |

### Amplitude Bias
* mean=-2.291, median=-2.524, p05=-3.702, p95=0.259
* MAE=2.370, RMSE=2.559, max_abs=4.768, pearson_r=0.103

### Phase Bias
* mean=0.131, median=0.000, p05=0.000, p95=0.943
* MAE=0.131, RMSE=0.344, max_abs=1.164

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
- Prediction scale: mode=least_squares value=2.061 source=least_squares
  * Note: least_squares=2.061
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
| Normalized → prediction | 4.837 |
| Prediction → truth | 6.581 |


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
