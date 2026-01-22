# Dose Baseline Summary

**Scenario ID:** PGRID-20250826-P1E5-T1024

**Dataset Root:** `photon_grid_study_20250826_152459`

**Baseline Params:** `photon_grid_study_20250826_152459/results_p1e5/train_1024/trial_1/baseline_run/08-26-2025-16.38.17_baseline_gs1/08-26-2025-16.38.17_baseline_gs1/params.dill`

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
| frc50 | 6 | nan |
| mse | 0.00480062 | 1.01674e-11 |

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

**Full JSON:** See `dose_baseline_summary.json` in the same directory.
