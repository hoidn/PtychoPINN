# Intensity Statistics

**Spec Reference:** `specs/spec-ptycho-core.md §Normalization Invariants`

- Bundle intensity_scale: 988.211669921875
- Legacy params intensity_scale: 988.211669921875
- bundle minus legacy delta: 0.0
- normalize_data gain: 0.55953
- Stage count: 4

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
