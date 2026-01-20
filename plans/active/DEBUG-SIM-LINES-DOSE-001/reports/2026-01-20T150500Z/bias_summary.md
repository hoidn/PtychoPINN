# Intensity Bias Summary
- Generated at: 2026-01-20T09:03:58.332183+00:00
- Scenario count: 2

| Scenario | Amp bias mean | Amp median | Phase bias mean | Phase median | Bundle scale | Legacy scale | Δscale | Scale mode | Training NaN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| gs1_ideal | -2.287 | -2.524 | -0.124 | 0.000 | 988.212 | 988.212 | 0.000 | least_squares | No |
| gs2_ideal | -2.296 | -2.531 | 0.116 | 0.000 | 988.212 | 988.212 | 0.000 | least_squares | No |

## Scenario: gs1_ideal
- Base directory: `/home/ollie/Documents/tmp/PtychoPINN/plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T150500Z/gs1_ideal`
- Intensity scale: bundle 988.212 vs legacy 988.212 (Δ=0.000)
- Prediction scale: mode=least_squares value=1.856 source=least_squares
  * Note: least_squares=1.856
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
| Normalized → prediction | 4.930 |
| Prediction → truth | 6.428 |


### Prediction ↔ Truth Scaling
| Metric | Value |
| --- | ---: |
| Ratio mean | 1.924 |
| Ratio median | 1.875 |
| Ratio p05 | 1.199 |
| Ratio p95 | 2.835 |
| Ratio count | 23065 |

* Best scalar: least_squares (1.856)
* MAE baseline 2.364 → 2.364; RMSE baseline 2.557 → 2.557
| Scalar | Value | Scaled MAE | Scaled RMSE |
| --- | ---: | ---: | ---: |
| Ratio mean | 1.924 | 2.365 | 2.557 |
| Ratio median | 1.875 | 2.364 | 2.557 |
| Ratio p05 | 1.199 | 2.440 | 2.586 |
| Ratio p95 | 2.835 | 2.499 | 2.622 |
| Least squares | 1.856 | 2.364 | 2.557 |

### Amplitude Bias
* mean=-2.287, median=-2.524, p05=-3.702, p95=0.264
* MAE=2.364, RMSE=2.557, max_abs=4.768, pearson_r=0.102

### Phase Bias
* mean=-0.124, median=0.000, p05=-0.874, p95=0.000
* MAE=0.124, RMSE=0.323, max_abs=1.108

### Normalization Stage Stats
| Stage | Source | Count | Mean | Std | Min | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| raw_diffraction | RawData | 256 | 0.147 | 0.844 | 0.000 | 26.896 |
| grouped_diffraction | RawData.generate_grouped_data | 64 | 0.153 | 0.880 | 0.000 | 26.896 |
| grouped_X_full | normalize_data | 64 | 0.085 | 0.493 | 0.000 | 15.049 |
| container_X | PtychoDataContainer | n/a | 0.085 | 0.493 | 0.000 | 15.049 |

---

## Scenario: gs2_ideal
- Base directory: `/home/ollie/Documents/tmp/PtychoPINN/plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T150500Z/gs2_ideal`
- Intensity scale: bundle 988.212 vs legacy 988.212 (Δ=0.000)
- Prediction scale: mode=least_squares value=1.987 source=least_squares
  * Note: least_squares=1.987
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
| Normalized → prediction | 4.842 |
| Prediction → truth | 6.573 |


### Prediction ↔ Truth Scaling
| Metric | Value |
| --- | ---: |
| Ratio mean | 1.999 |
| Ratio median | 1.959 |
| Ratio p05 | 1.370 |
| Ratio p95 | 2.739 |
| Ratio count | 21737 |

* Best scalar: ratio_median (1.959)
* MAE baseline 2.365 → 2.364; RMSE baseline 2.558 → 2.558
| Scalar | Value | Scaled MAE | Scaled RMSE |
| --- | ---: | ---: | ---: |
| Ratio mean | 1.999 | 2.365 | 2.558 |
| Ratio median | 1.959 | 2.364 | 2.558 |
| Ratio p05 | 1.370 | 2.426 | 2.580 |
| Ratio p95 | 2.739 | 2.453 | 2.591 |
| Least squares | 1.987 | 2.365 | 2.558 |

### Amplitude Bias
* mean=-2.296, median=-2.531, p05=-3.696, p95=0.266
* MAE=2.365, RMSE=2.558, max_abs=4.768, pearson_r=0.135

### Phase Bias
* mean=0.116, median=0.000, p05=0.000, p95=0.832
* MAE=0.116, RMSE=0.308, max_abs=1.038

### Normalization Stage Stats
| Stage | Source | Count | Mean | Std | Min | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| raw_diffraction | RawData | 512 | 0.146 | 0.843 | 0.000 | 27.523 |
| grouped_diffraction | RawData.generate_grouped_data | 64 | 0.147 | 0.853 | 0.000 | 27.523 |
| grouped_X_full | normalize_data | 64 | 0.085 | 0.493 | 0.000 | 15.892 |
| container_X | PtychoDataContainer | n/a | 0.085 | 0.493 | 0.000 | 15.892 |

---
