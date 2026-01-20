# Intensity Statistics

**Spec Reference:** `specs/spec-ptycho-core.md §Normalization Invariants`

- Bundle intensity_scale: 262.77606567063026
- Legacy params intensity_scale: 988.2117688026185
- bundle minus legacy delta: -725.4357031319882
- normalize_data gain: 0.259373
- Stage count: 4

## Stage Means

| Stage | Mean |
| --- | ---: |
| Raw diffraction | 1.41235 |
| Grouped diffraction | 1.45007 |
| Grouped X (normalized) | 0.376111 |
| Container X | 0.377452 |

## Stage Ratios

| Transition | Ratio |
| --- | ---: |
| Raw diffraction → Grouped diffraction | 1.02671 |
| Grouped diffraction → Grouped X (normalized) | 0.259373 |
| Grouped X (normalized) → Container X | 1.00357 |

## Largest Drop

**Grouped diffraction → Grouped X (normalized)** (ratio=0.259373)

Per `specs/spec-ptycho-core.md §Normalization Invariants`, symmetry SHALL hold:
- Training inputs: `X_scaled = s · X`
- Labels: `Y_amp_scaled = s · X` (amplitude), `Y_int = (s · X)^2` (intensity)

If the ratio deviates significantly from 1.0, investigate whether the normalization
pipeline preserves the intensity_scale symmetry mandated by the spec.

## Per-Stage Statistics

### raw_diffraction

- n_images: 1024
- source: simulate_nongrid_raw_data

| Metric | Value |
| --- | --- |
| shape | [1024, 64, 64] |
| dtype | float32 |
| min | 0 |
| max | 6.18749 |
| mean | 1.41235 |
| std | 1.24134 |
| finite_count | 4194304 |
| total_count | 4194304 |
| nan_count | 0 |

### grouped_diffraction

- n_groups: 64
- gridsize: 2
- neighbor_count: 5

| Metric | Value |
| --- | --- |
| shape | [64, 64, 64, 4] |
| dtype | float32 |
| min | 0 |
| max | 5.93069 |
| mean | 1.45007 |
| std | 1.2702 |
| finite_count | 1048576 |
| total_count | 1048576 |
| nan_count | 0 |

### grouped_X_full

- normalization: normalize_data
- N: 64

| Metric | Value |
| --- | --- |
| shape | [64, 64, 64, 4] |
| dtype | float32 |
| min | 0 |
| max | 1.53826 |
| mean | 0.376111 |
| std | 0.329455 |
| finite_count | 1048576 |
| total_count | 1048576 |
| nan_count | 0 |

### container_X

- source: PtychoDataContainer

| Metric | Value |
| --- | --- |
| shape | [64, 64, 64, 4] |
| dtype | float32 |
| min | 0 |
| max | 1.41793 |
| mean | 0.377452 |
| std | 0.327917 |
| finite_count | 1048576 |
| total_count | 1048576 |
| nan_count | 0 |
