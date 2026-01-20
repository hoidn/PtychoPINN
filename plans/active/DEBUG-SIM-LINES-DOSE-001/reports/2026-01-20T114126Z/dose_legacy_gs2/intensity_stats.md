# Intensity Statistics

**Spec Reference:** `specs/spec-ptycho-core.md §Normalization Invariants`

- Bundle intensity_scale: 988.211669921875
- Legacy params intensity_scale: 988.211669921875
- bundle minus legacy delta: 0.0
- normalize_data gain: 0.272574
- Stage count: 4

## Stage Means

| Stage | Mean |
| --- | ---: |
| Raw diffraction | 1.36333 |
| Grouped diffraction | 1.3783 |
| Grouped X (normalized) | 0.375688 |
| Container X | 0.375688 |

## Stage Ratios

| Transition | Ratio |
| --- | ---: |
| Raw diffraction → Grouped diffraction | 1.01098 |
| Grouped diffraction → Grouped X (normalized) | 0.272574 |
| Grouped X (normalized) → Container X | 1 |

## Largest Drop

**Grouped diffraction → Grouped X (normalized)** (ratio=0.272574)

Per `specs/spec-ptycho-core.md §Normalization Invariants`, symmetry SHALL hold:
- Training inputs: `X_scaled = s · X`
- Labels: `Y_amp_scaled = s · X` (amplitude), `Y_int = (s · X)^2` (intensity)

If the ratio deviates significantly from 1.0, investigate whether the normalization
pipeline preserves the intensity_scale symmetry mandated by the spec.

## Per-Stage Statistics

### raw_diffraction

- source: RawData
- count: 512

| Metric | Value |
| --- | --- |
| shape | [512, 64, 64] |
| dtype | float32 |
| min | 0 |
| max | 6.18935 |
| mean | 1.36333 |
| std | 1.20534 |
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
| max | 6.18935 |
| mean | 1.3783 |
| std | 1.21045 |
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
| max | 1.68706 |
| mean | 0.375688 |
| std | 0.329937 |
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
| max | 1.68706 |
| mean | 0.375688 |
| std | 0.329937 |
| finite_count | 1048576 |
| total_count | 1048576 |
| nan_count | 0 |
