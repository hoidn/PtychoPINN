# Intensity Statistics

**Spec Reference:** `specs/spec-ptycho-core.md §Normalization Invariants`

- Bundle intensity_scale: 494.1046142578125
- Legacy params intensity_scale: 494.1046142578125
- bundle minus legacy delta: 0.0
- normalize_data gain: 0.577416
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
| Dataset-derived scale | 577.738 |
| Fallback scale | 988.212 |
| nphotons | 1e+09 |
| N (patch size) | 64 |
| E_batch[Σ|Ψ|²] | 2995.97 |
| Delta (dataset - fallback) | -410.474 |
| Ratio (dataset / fallback) | 0.58463 |

⚠️ **Dataset vs fallback scale mismatch:** ratio=0.58463 
indicates that the actual mean intensity per sample (2995.97) 
differs from the assumed (N/2)² = 1024.

## Stage Means

| Stage | Mean |
| --- | ---: |
| Raw diffraction | 0.145831 |
| Grouped diffraction | 0.147358 |
| Grouped X (normalized) | 0.085087 |
| Container X | 0.085087 |

## Stage Ratios

| Transition | Ratio |
| --- | ---: |
| Raw diffraction → Grouped diffraction | 1.01047 |
| Grouped diffraction → Grouped X (normalized) | 0.577416 |
| Grouped X (normalized) → Container X | 1 |

## Largest Drop

**Grouped diffraction → Grouped X (normalized)** (ratio=0.577416)

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
| log_scale (raw) | 6.20275 |
| exp(log_scale) | 494.105 |
| trainable | True |
| params.cfg intensity_scale | 494.105 |
| params.cfg trainable | True |
| delta (exp - cfg) | -2.14586e-05 |
| ratio (exp / cfg) | 1 |

## Training Container X Stats

Statistics of the training container's X tensor after normalization.
Per `specs/spec-ptycho-core.md §Normalization Invariants`, this should reflect `X_scaled = s · X`.

| Metric | Value |
| --- | ---: |
| shape | [64, 64, 64, 4] |
| dtype | float32 |
| min | 0 |
| max | 15.8923 |
| mean | 0.085087 |
| std | 0.492707 |
| nan_count | 0 |

## Per-Stage Statistics

### raw_diffraction

- source: RawData
- count: 512

| Metric | Value |
| --- | --- |
| shape | [512, 64, 64] |
| dtype | float32 |
| min | 0 |
| max | 27.5231 |
| mean | 0.145831 |
| std | 0.842717 |
| finite_count | 2097152 |
| total_count | 2097152 |
| nan_count | 0 |

### grouped_diffraction

- source: RawData.generate_grouped_data
- count: 64
- gridsize: 2

| Metric | Value |
| --- | --- |
| shape | [64, 64, 64, 4] |
| dtype | float32 |
| min | 0 |
| max | 27.5231 |
| mean | 0.147358 |
| std | 0.853297 |
| finite_count | 1048576 |
| total_count | 1048576 |
| nan_count | 0 |

### grouped_X_full

- source: normalize_data
- count: 64

| Metric | Value |
| --- | --- |
| shape | [64, 64, 64, 4] |
| dtype | float32 |
| min | 0 |
| max | 15.8923 |
| mean | 0.085087 |
| std | 0.492707 |
| finite_count | 1048576 |
| total_count | 1048576 |
| nan_count | 0 |

### container_X

- source: PtychoDataContainer
- group_limit: 64

| Metric | Value |
| --- | --- |
| shape | [64, 64, 64, 4] |
| dtype | float32 |
| min | 0 |
| max | 15.8923 |
| mean | 0.085087 |
| std | 0.492707 |
| finite_count | 1048576 |
| total_count | 1048576 |
| nan_count | 0 |
