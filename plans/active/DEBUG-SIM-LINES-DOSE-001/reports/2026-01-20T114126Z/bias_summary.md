# Intensity Bias Summary

**Spec Reference:** `specs/spec-ptycho-core.md §Normalization Invariants`

- Generated at: 2026-01-20T11:54:06.954003+00:00
- Scenario count: 2

| Scenario | Amp bias mean | Amp median | Phase bias mean | Phase median | Bundle scale | Legacy scale | Δscale | Scale mode | Training NaN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| gs1_ideal | -2.293 | -2.524 | 0.178 | 0.000 | 988.212 | 988.212 | 0.000 | least_squares | No |
| dose_legacy_gs2 | -2.645 | -2.637 | 0.015 | 0.000 | 988.212 | 988.212 | 0.000 | none | No |

## Scenario: gs1_ideal
- Base directory: `/home/ollie/Documents/tmp/PtychoPINN/plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T114126Z/gs1_ideal`
- Intensity scale: bundle 988.212 vs legacy 988.212 (Δ=0.000)
- normalize_data gain: 0.560
- Prediction scale: mode=least_squares value=1.838 source=least_squares
  * Note: least_squares=1.838
- Training NaNs: no
- Inference canvas: padded=828 required=828 fits_canvas=True
- **Largest drop:** Grouped diffraction → Container X (ratio=0.560, Δ=-0.440)
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
| Normalized → prediction | 4.854 |
| Prediction → truth | 6.529 |


### Prediction ↔ Truth Scaling
| Metric | Value |
| --- | ---: |
| Ratio mean | 1.967 |
| Ratio median | 1.899 |
| Ratio p05 | 1.161 |
| Ratio p95 | 3.014 |
| Ratio count | 23065 |

* Best scalar: least_squares (1.838)
* MAE baseline 2.373 → 2.373; RMSE baseline 2.560 → 2.560
| Scalar | Value | Scaled MAE | Scaled RMSE |
| --- | ---: | ---: | ---: |
| Ratio mean | 1.967 | 2.376 | 2.561 |
| Ratio median | 1.899 | 2.374 | 2.561 |
| Ratio p05 | 1.161 | 2.450 | 2.592 |
| Ratio p95 | 3.014 | 2.536 | 2.654 |
| Least squares | 1.838 | 2.373 | 2.560 |

### Amplitude Bias
* mean=-2.293, median=-2.524, p05=-3.702, p95=0.241
* MAE=2.373, RMSE=2.560, max_abs=4.768, pearson_r=0.104

### Phase Bias
* mean=0.178, median=0.000, p05=0.000, p95=1.213
* MAE=0.178, RMSE=0.460, max_abs=1.331

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
- normalize_data gain: 0.284
- Prediction scale: mode=none value=n/a source=None
- Training NaNs: no
- Inference canvas: padded=814 required=805 fits_canvas=True
- **Largest drop:** Container X → Prediction (ratio=0.169, Δ=-0.831)
  - Per `specs/spec-ptycho-core.md §Normalization Invariants`: symmetry SHALL hold for X_scaled = s · X

### Stage Means
| Stage | Mean |
| --- | ---: |
| Raw diffraction | 1.336 |
| Grouped diffraction | 1.316 |
| Grouped X (normalized) | 0.374 |
| Container X | 0.374 |

### Stage Ratios
| Transition | Ratio |
| --- | ---: |
| Raw → grouped | 0.985 |
| Grouped → normalized | 0.284 |
| Normalized → prediction | 0.169 |
| Prediction → truth | 42.764 |


### Prediction ↔ Truth Scaling
| Metric | Value |
| --- | ---: |
| Ratio mean | 3.579 |
| Ratio median | 3.572 |
| Ratio p05 | 2.296 |
| Ratio p95 | 4.894 |
| Ratio count | 12272 |

* Best scalar: ratio_median (3.572)
* MAE baseline 2.645 → 2.521; RMSE baseline 2.727 → 2.667
| Scalar | Value | Scaled MAE | Scaled RMSE |
| --- | ---: | ---: | ---: |
| Ratio mean | 3.579 | 2.521 | 2.667 |
| Ratio median | 3.572 | 2.521 | 2.667 |
| Ratio p05 | 2.296 | 2.566 | 2.682 |
| Ratio p95 | 4.894 | 2.570 | 2.685 |
| Least squares | 3.535 | 2.521 | 2.667 |

### Amplitude Bias
* mean=-2.645, median=-2.637, p05=-3.756, p95=-1.569
* MAE=2.645, RMSE=2.727, max_abs=4.856, pearson_r=0.052

### Phase Bias
* mean=0.015, median=0.000, p05=0.000, p95=0.147
* MAE=0.015, RMSE=0.071, max_abs=2.007

### Normalization Stage Stats
| Stage | Source | Count | Mean | Std | Min | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| raw_diffraction | RawData | 128 | 1.336 | 1.186 | 0.000 | 5.758 |
| grouped_diffraction | RawData.generate_grouped_data | 32 | 1.316 | 1.166 | 0.000 | 5.758 |
| grouped_X_full | normalize_data | 32 | 0.374 | 0.332 | 0.000 | 1.637 |
| container_X | PtychoDataContainer | n/a | 0.374 | 0.332 | 0.000 | 1.637 |

---
