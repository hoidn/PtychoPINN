# Phase G Metrics Summary

**Total Jobs:** 1  
**Successful:** 1  
**Failed:** 0  

---

## Aggregate Metrics

Summary statistics across all jobs per model.

### Pty-chi (pty-chi)

**MS-SSIM:**

| Statistic | Amplitude | Phase |
|-----------|-----------|-------|
| Mean | 0.617 | 0.829 |
| Best | 0.617 | 0.829 |

**MAE:**

| Statistic | Amplitude | Phase |
|-----------|-----------|-------|
| Mean | 0.070 | 0.128 |

### PtychoPINN

**MS-SSIM:**

| Statistic | Amplitude | Phase |
|-----------|-----------|-------|
| Mean | 0.190 | 0.039 |
| Best | 0.190 | 0.039 |

**MAE:**

| Statistic | Amplitude | Phase |
|-----------|-----------|-------|
| Mean | 0.041 | 0.242 |

---

## Job: dense/test (dose=1000)

### PtychoPINN

| Metric | Amplitude | Phase | Value |
|--------|-----------|-------|-------|
| mae | 0.0414 | 0.2422 |  |
| mse | 0.0027 | 0.0749 |  |
| psnr | 73.8598 | 59.3834 |  |
| ssim | 0.2216 | 0.7302 |  |
| ms_ssim | 0.1902 | 0.0392 |  |
| frc50 | 1.0000 | 1.0000 |  |
| computation_time_s |  |  | 7.1077 |

### Baseline

| Metric | Amplitude | Phase | Value |
|--------|-----------|-------|-------|
| mae |  |  |  |
| mse |  |  |  |
| psnr |  |  |  |
| ssim |  |  |  |
| ms_ssim |  |  |  |
| frc50 |  |  |  |
| computation_time_s |  |  | 12.6594 |

### Pty-chi (pty-chi)

| Metric | Amplitude | Phase | Value |
|--------|-----------|-------|-------|
| mae | 0.0696 | 0.1283 |  |
| mse | 0.0073 | 0.0275 |  |
| psnr | 69.5256 | 63.7347 |  |
| ssim | 0.3006 | 0.8483 |  |
| ms_ssim | 0.6168 | 0.8294 |  |
| frc50 | 1.0000 | 1.0000 |  |
| registration_offset_dy |  |  | 0.1000 |
| registration_offset_dx |  |  | -1.2200 |

---

## Phase C Metadata Compliance

Validation of Phase C NPZ files for _metadata and canonical transformations.

| Dose | Split | Compliant | Has Metadata | Has Canonical Transform | Path |
|------|-------|-----------|--------------|-------------------------|------|
| dose_1000 | test | ✓ | ✓ | ✓ | data/phase_c/dose_1000/patched_test.npz |
| dose_1000 | train | ✓ | ✓ | ✓ | data/phase_c/dose_1000/patched_train.npz |

