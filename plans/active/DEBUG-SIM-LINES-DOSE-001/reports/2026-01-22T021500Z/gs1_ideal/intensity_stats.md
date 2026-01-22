# Intensity Statistics

**Spec Reference:** `specs/spec-ptycho-core.md §Normalization Invariants`

- Bundle intensity_scale: 988.211669921875
- Legacy params intensity_scale: 988.211669921875
- bundle minus legacy delta: 0.0
- normalize_data gain: 0.55953
- Stage count: 4

## Intensity Scale Comparison

**Spec Reference:** `specs/spec-ptycho-core.md §Normalization Invariants`

Per the spec, two compliant intensity scale calculation modes are allowed:
1. **Dataset-derived (preferred):** `s = sqrt(nphotons / E_batch[Σ_xy |Ψ|²])`
2. **Closed-form fallback:** `s ≈ sqrt(nphotons) / (N/2)`

If the dataset-derived scale differs significantly from the fallback value (988.21),
this indicates the actual data statistics differ from the assumed model.

| Property | Value |
| --- | ---: |
| Dataset-derived scale | 576.596 |
| Fallback scale | 988.212 |
| nphotons | 1e+09 |
| N (patch size) | 64 |
| E_batch[Σ|Ψ|²] | 3007.86 |
| Delta (dataset - fallback) | -411.616 |
| Ratio (dataset / fallback) | 0.583474 |

⚠️ **Dataset vs fallback scale mismatch:** ratio=0.583474 
indicates that the actual mean intensity per sample (3007.86) 
differs from the assumed (N/2)² = 1024.

## Stage Means

| Stage | Mean |
| --- | ---: |
| Raw diffraction | 0.146682 |
| Grouped diffraction | 0.152742 |
| Grouped X (normalized) | 0.0854637 |
| Container X | 0.0854637 |

## Stage Ratios

| Transition | Ratio |
| --- | ---: |
| Raw diffraction → Grouped diffraction | 1.04131 |
| Grouped diffraction → Grouped X (normalized) | 0.55953 |
| Grouped X (normalized) → Container X | 1 |

## Largest Drop

**Grouped diffraction → Grouped X (normalized)** (ratio=0.55953)

Per `specs/spec-ptycho-core.md §Normalization Invariants`, symmetry SHALL hold:
- Training inputs: `X_scaled = s · X`
- Labels: `Y_amp_scaled = s · X` (amplitude), `Y_int = (s · X)^2` (intensity)

If the ratio deviates significantly from 1.0, investigate whether the normalization
pipeline preserves the intensity_scale symmetry mandated by the spec.

## IntensityScaler State

**Spec Reference:** `specs/spec-ptycho-workflow.md §Loss and Optimization`

Per the architecture, `IntensityScaler` and `IntensityScaler_inv` layers use a shared `log_scale` tf.Variable.
The effective scaling factor is `exp(log_scale)`. If this diverges from the recorded bundle/params.cfg value,
it may indicate double-scaling or a training-time drift that contributes to amplitude bias.

| Property | Value |
| --- | ---: |
| log_scale (raw) | 6.8959 |
| exp(log_scale) | 988.212 |
| trainable | True |
| params.cfg intensity_scale | 988.212 |
| params.cfg trainable | True |
| delta (exp - cfg) | -6.49802e-05 |
| ratio (exp / cfg) | 1 |

## Training Container X Stats

Statistics of the training container's X tensor after normalization.
Per `specs/spec-ptycho-core.md §Normalization Invariants`, this should reflect `X_scaled = s · X`.

| Metric | Value |
| --- | ---: |
| shape | [64, 64, 64, 1] |
| dtype | float32 |
| min | 0 |
| max | 15.0491 |
| mean | 0.0854637 |
| std | 0.492642 |
| nan_count | 0 |

## Training Label Statistics (D6)

**Spec Reference:** `specs/spec-ptycho-core.md §Normalization Invariants`

Training labels fed during model training. Per the spec:
- Labels: `Y_amp_scaled = s · X` (amplitude), `Y_int = (s · X)^2` (intensity)

### Y_amp

| Metric | Value |
| --- | ---: |
| shape | [256, 64, 64, 1] |
| dtype | float32 |
| min | 0 |
| max | 4.60674 |
| mean | 2.70681 |
| std | 0.788543 |

### Y_I

| Metric | Value |
| --- | ---: |
| shape | [256, 64, 64, 1] |
| dtype | float32 |
| min | 0 |
| max | 21.222 |
| mean | 7.94863 |
| std | 3.5842 |

### Y_phi

| Metric | Value |
| --- | ---: |
| shape | [256, 64, 64, 1] |
| dtype | float32 |
| min | 0 |
| max | 0 |
| mean | 0 |
| std | 0 |

### X

| Metric | Value |
| --- | ---: |
| shape | [256, 64, 64, 1] |
| dtype | float32 |
| min | 0 |
| max | 13.8692 |
| mean | 0.0858192 |
| std | 0.49258 |

### Label vs Ground Truth Analysis

| Metric | Value |
| --- | ---: |
| Ground truth amplitude mean | 2.70825 |
| Y_I mean (training label) | 7.94863 |
| X mean (training input) | 0.0858192 |
| Ratio (truth / Y_I) | 0.340719 |
| Ratio (truth / X) | 31.5576 |

**Interpretation:** If `ratio (truth / Y_I)` differs significantly from 1.0,
the training labels are at a different scale than the ground truth used for comparison.
This could explain amplitude bias in reconstructions.

## Per-Stage Statistics

### raw_diffraction

- source: RawData
- count: 256

| Metric | Value |
| --- | --- |
| shape | [256, 64, 64] |
| dtype | float32 |
| min | 0 |
| max | 26.8959 |
| mean | 0.146682 |
| std | 0.84429 |
| finite_count | 1048576 |
| total_count | 1048576 |
| nan_count | 0 |

### grouped_diffraction

- source: RawData.generate_grouped_data
- count: 64
- gridsize: 1

| Metric | Value |
| --- | --- |
| shape | [64, 64, 64, 1] |
| dtype | float32 |
| min | 0 |
| max | 26.8959 |
| mean | 0.152742 |
| std | 0.880456 |
| finite_count | 262144 |
| total_count | 262144 |
| nan_count | 0 |

### grouped_X_full

- source: normalize_data
- count: 64

| Metric | Value |
| --- | --- |
| shape | [64, 64, 64, 1] |
| dtype | float32 |
| min | 0 |
| max | 15.0491 |
| mean | 0.0854637 |
| std | 0.492642 |
| finite_count | 262144 |
| total_count | 262144 |
| nan_count | 0 |

### container_X

- source: PtychoDataContainer
- group_limit: 64

| Metric | Value |
| --- | --- |
| shape | [64, 64, 64, 1] |
| dtype | float32 |
| min | 0 |
| max | 15.0491 |
| mean | 0.0854637 |
| std | 0.492642 |
| finite_count | 262144 |
| total_count | 262144 |
| nan_count | 0 |
