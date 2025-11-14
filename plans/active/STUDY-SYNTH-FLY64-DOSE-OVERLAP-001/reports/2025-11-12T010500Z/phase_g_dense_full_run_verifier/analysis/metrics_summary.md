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
| Mean | 0.628 | 0.839 |
| Best | 0.628 | 0.839 |

**MAE:**

| Statistic | Amplitude | Phase |
|-----------|-----------|-------|
| Mean | 0.069 | 0.123 |

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
| psnr | 55.6255 | 59.3589 |  |
| ssim | 0.0932 | 0.7273 |  |
| ms_ssim | 0.0037 | 0.0476 |  |
| frc50 | 1.0000 | 1.0000 |  |
| computation_time_s |  |  | 20.6057 |

### Baseline

| Metric | Amplitude | Phase | Value |
|--------|-----------|-------|-------|
| mae |  |  |  |
| mse |  |  |  |
| psnr |  |  |  |
| ssim |  |  |  |
| ms_ssim |  |  |  |
| frc50 |  |  |  |
| computation_time_s |  |  | 16.3219 |

### Pty-chi (pty-chi)

| Metric | Amplitude | Phase | Value |
|--------|-----------|-------|-------|
| mae | 0.0689 | 0.1228 |  |
| mse | 0.0071 | 0.0253 |  |
| psnr | 69.5884 | 64.0996 |  |
| ssim | 0.3081 | 0.8558 |  |
| ms_ssim | 0.6280 | 0.8392 |  |
| frc50 | 1.0000 | 1.0000 |  |
| registration_offset_dy |  |  | 0.1800 |
| registration_offset_dx |  |  | -0.2800 |

---

## Phase C Metadata Compliance

Validation of Phase C NPZ files for _metadata and canonical transformations.

| Dose | Split | Compliant | Has Metadata | Has Canonical Transform | Path |
|------|-------|-----------|--------------|-------------------------|------|
| dose_1000 | test | ✓ | ✓ | ✓ | data/phase_c/dose_1000/patched_test.npz |
| dose_1000 | train | ✓ | ✓ | ✓ | data/phase_c/dose_1000/patched_train.npz |

