# Intensity Bias Summary

**Spec Reference:** `specs/spec-ptycho-core.md §Normalization Invariants`

- Generated at: 2026-01-22T03:09:36.241875+00:00
- Scenario count: 1

| Scenario | Amp bias mean | Amp median | Phase bias mean | Phase median | Bundle scale | Legacy scale | Δscale | Scale mode | Training NaN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| gs1_ideal | -2.287 | -2.524 | -0.129 | 0.000 | 988.212 | 988.212 | 0.000 | least_squares | No |

## Scenario: gs1_ideal
- Base directory: `/home/ollie/Documents/tmp/PtychoPINN/plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-22T021500Z/gs1_ideal`
- Intensity scale: bundle 988.212 vs legacy 988.212 (Δ=0.000)
- normalize_data gain: 0.560
- Prediction scale: mode=least_squares value=1.776 source=least_squares
  * Note: least_squares=1.776
- Training NaNs: no

### Loss Composition

**Spec Reference:** `specs/spec-ptycho-workflow.md §Loss and Optimization`

- Epochs trained: 5
- Total loss (final): -3946549.750 (min=-3946549.750, max=-3895298.750)
- Learning rate: last=0.001000 (min=0.001000, max=0.001000)

| Loss Component | Final Value | Contribution | Has NaN |
| --- | ---: | ---: | --- |
| `pred_intensity_loss` | -3940008.500 | 99.8% | No |
| `intensity_scaler_inv_loss` | 13.917 | -0.0% | No |
| `trimmed_obj_loss` | 2.445 | -0.0% | No |

**Dominant loss term:** `pred_intensity_loss`
  - Dominance ratio vs next: 283101.1×

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
| shape | [64, 64, 64, 1] |
| dtype | float32 |
| min | 0.000 |
| max | 15.049 |
| mean | 0.085 |
| std | 0.493 |
| nan_count | 0 |

- Inference canvas: padded=74 required=828 fits_canvas=False
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

Per `specs/spec-ptycho-core.md §Normalization Invariants`: normalized→prediction and prediction→truth deltas indicate where amplitude collapses in the pipeline.

| Transition | Ratio |
| --- | ---: |
| Raw → grouped | 1.041 |
| Grouped → normalized | 0.560 |
| Normalized → prediction | 4.930 |
| Prediction → truth | 6.428 |

### Normalization Invariant Check

**Spec Reference:** `specs/spec-ptycho-core.md §Normalization Invariants`

Per the spec, symmetry SHALL hold: `X_scaled = s · X`. The product of stage ratios (raw→truth) should ideally equal 1.0 for symmetric normalization.

| Stage Transition | Individual Ratio | Cumulative Product |
| --- | ---: | ---: |
| Raw → grouped | 1.041 | 1.041 |
| Grouped → normalized | 0.560 | 0.583 |
| Normalized → prediction | 4.930 | 2.872 |
| Prediction → truth | 6.428 | 18.463 |

**Full chain product (raw→truth):** 18.463
**Deviation from unity:** 17.463
**Tolerance:** 0.05 (5%)
**Passes tolerance:** ❌ No

⚠️ **Symmetry violated:** The normalization chain does not preserve amplitude as required by `specs/spec-ptycho-core.md §Normalization Invariants`.

**Stage Deviation Breakdown** (sorted by impact):

| Stage | Ratio | Deviation | Effect |
| --- | ---: | ---: | --- |
| prediction_to_truth | 6.428 | 5.428 | gain |
| normalized_to_prediction | 4.930 | 3.930 | gain |
| grouped_to_normalized | 0.560 | 0.440 | loss |
| raw_to_grouped | 1.041 | 0.041 | gain |

**Primary deviation source:** `prediction_to_truth` (ratio=6.428, deviation=5.428)


### Prediction ↔ Truth Scaling
| Metric | Value |
| --- | ---: |
| Ratio mean | 1.842 |
| Ratio median | 1.794 |
| Ratio p05 | 1.150 |
| Ratio p95 | 2.727 |
| Ratio count | 23065 |

* Best scalar: least_squares (1.776)
* MAE baseline 2.364 → 2.364; RMSE baseline 2.557 → 2.557
| Scalar | Value | Scaled MAE | Scaled RMSE |
| --- | ---: | ---: | ---: |
| Ratio mean | 1.842 | 2.366 | 2.557 |
| Ratio median | 1.794 | 2.365 | 2.557 |
| Ratio p05 | 1.150 | 2.439 | 2.586 |
| Ratio p95 | 2.727 | 2.502 | 2.624 |
| Least squares | 1.776 | 2.364 | 2.557 |

### Amplitude Bias
* mean=-2.287, median=-2.524, p05=-3.702, p95=0.274
* MAE=2.364, RMSE=2.557, max_abs=4.768, pearson_r=0.102

### Phase Bias
* mean=-0.129, median=0.000, p05=-0.914, p95=0.000
* MAE=0.129, RMSE=0.338, max_abs=1.126

### Normalization Stage Stats
| Stage | Source | Count | Mean | Std | Min | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| raw_diffraction | RawData | 256 | 0.147 | 0.844 | 0.000 | 26.896 |
| grouped_diffraction | RawData.generate_grouped_data | 64 | 0.153 | 0.880 | 0.000 | 26.896 |
| grouped_X_full | normalize_data | 64 | 0.085 | 0.493 | 0.000 | 15.049 |
| container_X | PtychoDataContainer | n/a | 0.085 | 0.493 | 0.000 | 15.049 |

---
