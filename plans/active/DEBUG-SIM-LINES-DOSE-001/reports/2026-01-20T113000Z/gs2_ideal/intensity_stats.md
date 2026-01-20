# Intensity Statistics

- Bundle intensity_scale: 988.211669921875
- Legacy params intensity_scale: 988.211669921875
- bundle minus legacy delta: 0.0
- Stage count: 4

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
