# Intensity Statistics

**Spec Reference:** `specs/spec-ptycho-core.md §Normalization Invariants`

- Bundle intensity_scale: 988.211669921875
- Legacy params intensity_scale: 988.211669921875
- bundle minus legacy delta: 0.0
- normalize_data gain: 0.284331
- Stage count: 4

## Stage Means

| Stage | Mean |
| --- | ---: |
| Raw diffraction | 1.3363 |
| Grouped diffraction | 1.31609 |
| Grouped X (normalized) | 0.374206 |
| Container X | 0.374206 |

## Stage Ratios

| Transition | Ratio |
| --- | ---: |
| Raw diffraction → Grouped diffraction | 0.98488 |
| Grouped diffraction → Grouped X (normalized) | 0.284331 |
| Grouped X (normalized) → Container X | 1 |

## Largest Drop

**Grouped diffraction → Grouped X (normalized)** (ratio=0.284331)

Per `specs/spec-ptycho-core.md §Normalization Invariants`, symmetry SHALL hold:
- Training inputs: `X_scaled = s · X`
- Labels: `Y_amp_scaled = s · X` (amplitude), `Y_int = (s · X)^2` (intensity)

If the ratio deviates significantly from 1.0, investigate whether the normalization
pipeline preserves the intensity_scale symmetry mandated by the spec.

## Per-Stage Statistics

### raw_diffraction

- source: RawData
- count: 128

| Metric | Value |
| --- | --- |
| shape | [128, 64, 64] |
| dtype | float32 |
| min | 0 |
| max | 5.75812 |
| mean | 1.3363 |
| std | 1.18587 |
| finite_count | 524288 |
| total_count | 524288 |
| nan_count | 0 |

### grouped_diffraction

- source: RawData.generate_grouped_data
- count: 32
- gridsize: 2

| Metric | Value |
| --- | --- |
| shape | [32, 64, 64, 4] |
| dtype | float32 |
| min | 0 |
| max | 5.75812 |
| mean | 1.31609 |
| std | 1.16631 |
| finite_count | 524288 |
| total_count | 524288 |
| nan_count | 0 |

### grouped_X_full

- source: normalize_data
- count: 32

| Metric | Value |
| --- | --- |
| shape | [32, 64, 64, 4] |
| dtype | float32 |
| min | 0 |
| max | 1.63721 |
| mean | 0.374206 |
| std | 0.331617 |
| finite_count | 524288 |
| total_count | 524288 |
| nan_count | 0 |

### container_X

- source: PtychoDataContainer
- group_limit: 32

| Metric | Value |
| --- | --- |
| shape | [32, 64, 64, 4] |
| dtype | float32 |
| min | 0 |
| max | 1.63721 |
| mean | 0.374206 |
| std | 0.331617 |
| finite_count | 524288 |
| total_count | 524288 |
| nan_count | 0 |
