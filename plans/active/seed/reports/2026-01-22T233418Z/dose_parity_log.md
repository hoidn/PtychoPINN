# D0 Parity Log

## Metadata

- **Scenario ID:** PGRID-20250826-P1E5-T1024
- **Dataset Root:** `photon_grid_study_20250826_152459`
- **Baseline Params:** `photon_grid_study_20250826_152459/results_p1e5/train_1024/trial_1/baseline_run/08-26-2025-16.38.17_baseline_gs1/08-26-2025-16.38.17_baseline_gs1/params.dill`
- **Timestamp:** 2026-01-22T23:39:38.709055+00:00
- **Git SHA:** `7f266736c8e84215a55556da34c558910a40f950`

## Key Parameters

| Parameter | Value |
|-----------|-------|
| N | 64 |
| gridsize | 1 |
| nimgs_train | 9 |
| nimgs_test | 3 |
| batch_size | 16 |
| nepochs | 50 |
| mae_weight | 0.0 |
| nll_weight | 1.0 |
| default_probe_scale | 0.7 |
| intensity_scale.trainable | True |
| probe.trainable | False |
| probe.mask | False |
| label | baseline_gs1 |
| output_prefix | photon_grid_study_20250826_152459/results_p1e5/train_1024/trial_1/baseline_run/08-26-2025-16.38.17_baseline_gs1/ |
| nphotons_in_params | 1000000000.0 |
| timestamp | 08/26/2025, 16:38:17 |
| intensity_scale_value | 988.2117309570312 |

## Metrics (train, test)

| Metric | Train | Test |
|--------|-------|------|
| mae | 0.00955534 | 5.38331e-07 |
| ms_ssim | 0.924828 | 0.920607 |
| psnr | 71.3178 | 158.059 |
| mse | 0.00480062 | 1.01674e-11 |
| frc50 | 6 | nan |

## Probe Flags

- **probe.trainable:** False
- **probe.mask:** False
- **intensity_scale.trainable:** True
- **intensity_scale_value:** 988.2117309570312

## Dataset Files

| File | Photon Dose | Patterns | Diff Shape | SHA256 (truncated) |
|------|-------------|----------|------------|--------------------|
| data_p1e3.npz | 1e3 | 5000 | [5000, 64, 64] | f9fa3f9f3f1cf8fc... |
| data_p1e4.npz | 1e4 | 5000 | [5000, 64, 64] | 1cce1fe9596a8229... |
| data_p1e5.npz | 1e5 | 5000 | [5000, 64, 64] | 01007daf8afc67aa... |
| data_p1e6.npz | 1e6 | 5000 | [5000, 64, 64] | 95cfd6aee6b2c061... |
| data_p1e7.npz | 1e7 | 5000 | [5000, 64, 64] | 9902ae24e90d2fa6... |
| data_p1e8.npz | 1e8 | 5000 | [5000, 64, 64] | 56b4f66a92aa28b2... |
| data_p1e9.npz | 1e9 | 5000 | [5000, 64, 64] | 3e1f229af34525a7... |

**Total datasets:** 7

## Stage-Level Stats by Dataset

### data_p1e3.npz (photon dose: 1e3)

#### Raw Diffraction

| Stat | Value |
|------|-------|
| min | 0 |
| max | 0.083666 |
| mean | 0.00708807 |
| std | 0.0139204 |
| median | 0 |
| p01 | 0 |
| p10 | 0 |
| p90 | 0.0316228 |
| p99 | 0.0447214 |
| nonzero_fraction | 0.210662 |
| count | 20480000 |

#### Normalized Diffraction

| Stat | Value |
|------|-------|
| min | 0 |
| max | 1 |
| mean | 0.0847186 |
| std | 0.16638 |
| median | 0 |
| p01 | 0 |
| p10 | 0 |
| p90 | 0.377964 |
| p99 | 0.534522 |
| nonzero_fraction | 0.210662 |
| count | 20480000 |

#### Grouped Intensity

| Stat | Value |
|------|-------|
| n_unique_scans | 1 |
| n_patterns | 5000 |
| min | 0.00708807 |
| max | 0.00708807 |
| mean | 0.00708807 |
| std | 0 |
| median | 0.00708807 |


### data_p1e4.npz (photon dose: 1e4)

#### Raw Diffraction

| Stat | Value |
|------|-------|
| min | 0 |
| max | 0.0458258 |
| mean | 0.013336 |
| std | 0.00813896 |
| median | 0.0141421 |
| p01 | 0 |
| p10 | 0 |
| p90 | 0.0223607 |
| p99 | 0.0282843 |
| nonzero_fraction | 0.785277 |
| count | 20480000 |

#### Normalized Diffraction

| Stat | Value |
|------|-------|
| min | 0 |
| max | 1 |
| mean | 0.291015 |
| std | 0.177607 |
| median | 0.308607 |
| p01 | 0 |
| p10 | 0 |
| p90 | 0.48795 |
| p99 | 0.617213 |
| nonzero_fraction | 0.785277 |
| count | 20480000 |

#### Grouped Intensity

| Stat | Value |
|------|-------|
| n_unique_scans | 1 |
| n_patterns | 5000 |
| min | 0.013336 |
| max | 0.013336 |
| mean | 0.013336 |
| std | 0 |
| median | 0.013336 |


### data_p1e5.npz (photon dose: 1e5)

#### Raw Diffraction

| Stat | Value |
|------|-------|
| min | 0 |
| max | 0.0320936 |
| mean | 0.0142134 |
| std | 0.00649034 |
| median | 0.0164317 |
| p01 | 0 |
| p10 | 0 |
| p90 | 0.0189737 |
| p99 | 0.0240832 |
| nonzero_fraction | 0.85482 |
| count | 20480000 |

#### Normalized Diffraction

| Stat | Value |
|------|-------|
| min | 0 |
| max | 1 |
| mean | 0.442873 |
| std | 0.202231 |
| median | 0.511992 |
| p01 | 0 |
| p10 | 0 |
| p90 | 0.591198 |
| p99 | 0.750404 |
| nonzero_fraction | 0.85482 |
| count | 20480000 |

#### Grouped Intensity

| Stat | Value |
|------|-------|
| n_unique_scans | 1 |
| n_patterns | 5000 |
| min | 0.0142134 |
| max | 0.0142134 |
| mean | 0.0142134 |
| std | 0 |
| median | 0.0142134 |


### data_p1e6.npz (photon dose: 1e6)

#### Raw Diffraction

| Stat | Value |
|------|-------|
| min | 0 |
| max | 0.0273679 |
| mean | 0.0143016 |
| std | 0.0062933 |
| median | 0.0168819 |
| p01 | 0 |
| p10 | 0 |
| p90 | 0.0177764 |
| p99 | 0.0241454 |
| nonzero_fraction | 0.881294 |
| count | 20480000 |

#### Normalized Diffraction

| Stat | Value |
|------|-------|
| min | 0 |
| max | 1 |
| mean | 0.522569 |
| std | 0.229952 |
| median | 0.616853 |
| p01 | 0 |
| p10 | 0 |
| p90 | 0.649535 |
| p99 | 0.882253 |
| nonzero_fraction | 0.881294 |
| count | 20480000 |

#### Grouped Intensity

| Stat | Value |
|------|-------|
| n_unique_scans | 1 |
| n_patterns | 5000 |
| min | 0.0143016 |
| max | 0.0143016 |
| mean | 0.0143016 |
| std | 0 |
| median | 0.0143016 |


### data_p1e7.npz (photon dose: 1e7)

#### Raw Diffraction

| Stat | Value |
|------|-------|
| min | 0 |
| max | 0.0258902 |
| mean | 0.0143206 |
| std | 0.00624994 |
| median | 0.0169823 |
| p01 | 0 |
| p10 | 0.000316228 |
| p90 | 0.0173609 |
| p99 | 0.0241247 |
| nonzero_fraction | 0.916567 |
| count | 20480000 |

#### Normalized Diffraction

| Stat | Value |
|------|-------|
| min | 0 |
| max | 1 |
| mean | 0.55313 |
| std | 0.241402 |
| median | 0.655938 |
| p01 | 0 |
| p10 | 0.0122142 |
| p90 | 0.670559 |
| p99 | 0.931809 |
| nonzero_fraction | 0.916567 |
| count | 20480000 |

#### Grouped Intensity

| Stat | Value |
|------|-------|
| n_unique_scans | 1 |
| n_patterns | 5000 |
| min | 0.0143206 |
| max | 0.0143206 |
| mean | 0.0143206 |
| std | 0 |
| median | 0.0143206 |


### data_p1e8.npz (photon dose: 1e8)

#### Raw Diffraction

| Stat | Value |
|------|-------|
| min | 0 |
| max | 0.0255507 |
| mean | 0.0143267 |
| std | 0.00623597 |
| median | 0.0170097 |
| p01 | 0 |
| p10 | 0.000360555 |
| p90 | 0.0172893 |
| p99 | 0.0241228 |
| nonzero_fraction | 0.960384 |
| count | 20480000 |

#### Normalized Diffraction

| Stat | Value |
|------|-------|
| min | 0 |
| max | 1 |
| mean | 0.560714 |
| std | 0.244062 |
| median | 0.665723 |
| p01 | 0 |
| p10 | 0.0141113 |
| p90 | 0.676666 |
| p99 | 0.944114 |
| nonzero_fraction | 0.960384 |
| count | 20480000 |

#### Grouped Intensity

| Stat | Value |
|------|-------|
| n_unique_scans | 1 |
| n_patterns | 5000 |
| min | 0.0143267 |
| max | 0.0143267 |
| mean | 0.0143267 |
| std | 0 |
| median | 0.0143267 |


### data_p1e9.npz (photon dose: 1e9)

#### Raw Diffraction

| Stat | Value |
|------|-------|
| min | 0 |
| max | 0.0254112 |
| mean | 0.0143281 |
| std | 0.00623264 |
| median | 0.017015 |
| p01 | 0 |
| p10 | 0.00034641 |
| p90 | 0.0172985 |
| p99 | 0.0241221 |
| nonzero_fraction | 0.984447 |
| count | 20480000 |

#### Normalized Diffraction

| Stat | Value |
|------|-------|
| min | 0 |
| max | 1 |
| mean | 0.56385 |
| std | 0.245271 |
| median | 0.669586 |
| p01 | 0 |
| p10 | 0.0136322 |
| p90 | 0.680743 |
| p99 | 0.949272 |
| nonzero_fraction | 0.984447 |
| count | 20480000 |

#### Grouped Intensity

| Stat | Value |
|------|-------|
| n_unique_scans | 1 |
| n_patterns | 5000 |
| min | 0.0143281 |
| max | 0.0143281 |
| mean | 0.0143281 |
| std | 0 |
| median | 0.0143281 |


---
*Full details in `dose_parity_log.json`*
