# Intensity Statistics

- Recorded intensity_scale: 988.211669921875
- Stage count: 4

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
