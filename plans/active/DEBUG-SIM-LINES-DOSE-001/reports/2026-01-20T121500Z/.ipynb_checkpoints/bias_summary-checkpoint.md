# Intensity Bias Summary
- Generated at: 2026-01-20T07:42:05.355802+00:00
- Scenario count: 2

| Scenario | Amp bias mean | Amp median | Phase bias mean | Phase median | Bundle scale | Legacy scale | Δscale | Training NaN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| gs1_ideal | -2.487 | -2.538 | 0.161 | 0.000 | 988.212 | 988.212 | 0.000 | No |
| gs2_ideal | -2.673 | -2.667 | 0.000 | 0.000 | 988.212 | 988.212 | 0.000 | Yes |

## Scenario: gs1_ideal
- Base directory: `/home/ollie/Documents/tmp/PtychoPINN/plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T113000Z/gs1_ideal`
- Intensity scale: bundle 988.212 vs legacy 988.212 (Δ=0.000)
- Training NaNs: no
- Inference canvas: padded=828 required=828 fits_canvas=True

### Amplitude Bias
* mean=-2.487, median=-2.538, p05=-3.702, p95=-1.036
* MAE=2.488, RMSE=2.611, max_abs=4.768, pearson_r=0.102

### Phase Bias
* mean=0.161, median=0.000, p05=0.000, p95=1.127
* MAE=0.161, RMSE=0.418, max_abs=1.286

### Normalization Stage Stats
| Stage | Source | Count | Mean | Std | Min | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| raw_diffraction | RawData | 256 | 0.147 | 0.844 | 0.000 | 26.896 |
| grouped_diffraction | RawData.generate_grouped_data | 64 | 0.153 | 0.880 | 0.000 | 26.896 |
| grouped_X_full | normalize_data | 64 | 0.085 | 0.493 | 0.000 | 15.049 |
| container_X | PtychoDataContainer | n/a | 0.085 | 0.493 | 0.000 | 15.049 |

---

## Scenario: gs2_ideal
- Base directory: `/home/ollie/Documents/tmp/PtychoPINN/plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T113000Z/gs2_ideal`
- Intensity scale: bundle 988.212 vs legacy 988.212 (Δ=0.000)
- Training NaNs: YES (metrics: intensity_scaler_inv_loss, loss, pred_intensity_loss, train_loss, val_intensity_scaler_inv_loss, val_loss, val_pred_intensity_loss)
- Inference canvas: padded=822 required=817 fits_canvas=True

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
