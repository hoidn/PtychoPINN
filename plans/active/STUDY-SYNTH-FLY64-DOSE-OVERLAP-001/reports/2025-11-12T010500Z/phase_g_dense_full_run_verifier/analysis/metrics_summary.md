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
| Mean | 0.645 | 0.866 |
| Best | 0.645 | 0.866 |

**MAE:**

| Statistic | Amplitude | Phase |
|-----------|-----------|-------|
| Mean | 0.068 | 0.114 |

### PtychoPINN

**MS-SSIM:**

| Statistic | Amplitude | Phase |
|-----------|-----------|-------|
| Mean | 0.004 | 0.048 |
| Best | 0.004 | 0.048 |

**MAE:**

| Statistic | Amplitude | Phase |
|-----------|-----------|-------|
| Mean | 0.401 | 0.242 |

---

## Job: dense/test (dose=1000)

### PtychoPINN

| Metric | Amplitude | Phase | Value |
|--------|-----------|-------|-------|
| mae | 0.4013 | 0.2415 |  |
| mse | 0.1780 | 0.0754 |  |
| psnr | 55.6255 | 59.3590 |  |
| ssim | 0.0932 | 0.7273 |  |
| ms_ssim | 0.0037 | 0.0476 |  |
| frc50 | 1.0000 | 1.0000 |  |
| computation_time_s |  |  | 20.4702 |

### Baseline

| Metric | Amplitude | Phase | Value |
|--------|-----------|-------|-------|
| mae |  |  |  |
| mse |  |  |  |
| psnr |  |  |  |
| ssim |  |  |  |
| ms_ssim |  |  |  |
| frc50 |  |  |  |
| computation_time_s |  |  | 16.2981 |

### Pty-chi (pty-chi)

| Metric | Amplitude | Phase | Value |
|--------|-----------|-------|-------|
| mae | 0.0678 | 0.1140 |  |
| mse | 0.0070 | 0.0217 |  |
| psnr | 69.6919 | 64.7570 |  |
| ssim | 0.3152 | 0.8727 |  |
| ms_ssim | 0.6447 | 0.8656 |  |
| frc50 | 1.0000 | 1.0000 |  |
| registration_offset_dy |  |  | 1.4400 |
| registration_offset_dx |  |  | 0.1000 |

---

## Phase C Metadata Compliance

Validation of Phase C NPZ files for _metadata and canonical transformations.

| Dose | Split | Compliant | Has Metadata | Has Canonical Transform | Path |
|------|-------|-----------|--------------|-------------------------|------|
| dose_1000 | test | ✓ | ✓ | ✓ | data/phase_c/dose_1000/patched_test.npz |
| dose_1000 | train | ✓ | ✓ | ✓ | data/phase_c/dose_1000/patched_train.npz |

