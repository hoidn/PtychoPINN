# Intensity Statistics

- Bundle intensity_scale: 988.211669921875
- Legacy params intensity_scale: 988.211669921875
- bundle minus legacy delta: 0.0
- Stage count: 4

### raw_diffraction

- source: RawData
- count: 256

| Metric | Value |
| --- | --- |
| shape | [256, 64, 64] |
| dtype | float32 |
| min | 0 |
| max | 6.03929 |
| mean | 1.36803 |
| std | 1.20664 |
| finite_count | 1048576 |
| total_count | 1048576 |
| nan_count | 0 |

### grouped_diffraction

- source: RawData.generate_grouped_data
- count: 128
- gridsize: 1

| Metric | Value |
| --- | --- |
| shape | [128, 64, 64, 1] |
| dtype | float32 |
| min | 0 |
| max | 6.03929 |
| mean | 1.38058 |
| std | 1.21393 |
| finite_count | 524288 |
| total_count | 524288 |
| nan_count | 0 |

### grouped_X_full

- source: normalize_data
- count: 128

| Metric | Value |
| --- | --- |
| shape | [128, 64, 64, 1] |
| dtype | float32 |
| min | 0 |
| max | 1.64256 |
| mean | 0.37549 |
| std | 0.330162 |
| finite_count | 524288 |
| total_count | 524288 |
| nan_count | 0 |

### container_X

- source: PtychoDataContainer
- group_limit: 128

| Metric | Value |
| --- | --- |
| shape | [128, 64, 64, 1] |
| dtype | float32 |
| min | 0 |
| max | 1.64256 |
| mean | 0.37549 |
| std | 0.330162 |
| finite_count | 524288 |
| total_count | 524288 |
| nan_count | 0 |
