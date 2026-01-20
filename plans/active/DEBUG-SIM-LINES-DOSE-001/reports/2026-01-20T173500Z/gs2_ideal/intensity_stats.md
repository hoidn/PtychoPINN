# Intensity Statistics

**Spec Reference:** `specs/spec-ptycho-core.md §Normalization Invariants`

- Bundle intensity_scale: 988.211669921875
- Legacy params intensity_scale: 988.211669921875
- bundle minus legacy delta: 0.0
- normalize_data gain: 0.577416
- Stage count: 4

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
