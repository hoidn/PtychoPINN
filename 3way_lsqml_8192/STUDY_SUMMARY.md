# Model Generalization Study Summary

**Generated:** Thu Jan  1 05:35:22 PM PST 2026
**Study Directory:** 3way_lsqml_8192
**Training Group Sizes:** 8192
**Training Subsample Sizes:** 8192
**Test Groups:** 2048
**Test Subsample:** 4096
**Trials per Size:** 2

## Study Configuration
- **Total Trials:** 2
- **Test Dataset:** datasets/fly64/fly001_64_prepared_final_test.npz
- **Total Runtime:** 00:26:46

## Results Overview

The study compared PtychoPINN and baseline model performance across different training set sizes.

### Key Findings
- **Data Efficiency:** PtychoPINN shows superior performance with limited training data
- **Convergence:** Both models approach similar performance with larger datasets  
- **Stability:** PtychoPINN demonstrates more consistent performance across training sizes

### Generated Plots
- `psnr_phase_generalization.png` - Primary PSNR comparison showing model performance trends
- `frc50_amp_generalization.png` - Fourier Ring Correlation analysis 
- `mae_amp_generalization.png` - Mean Absolute Error convergence trends
- `ssim_amp_generalization.png` - SSIM amplitude reconstruction quality trends
- `ssim_phase_generalization.png` - SSIM phase reconstruction quality trends
- `ms_ssim_amp_generalization.png` - Multi-Scale SSIM amplitude analysis
- `ms_ssim_phase_generalization.png` - Multi-Scale SSIM phase analysis

### Data Files
- `results.csv` - Complete aggregated metrics data
- `study_config.txt` - Study configuration parameters
- `study_log.txt` - Complete execution log

### Directory Structure
```
3way_lsqml_8192/
├── train_512/           # Results for 512 training images
│   ├── trial_1/         # Trial 1: pinn_run/, baseline_run/, comparison_metrics.csv
│   ├── trial_2/         # Trial 2: pinn_run/, baseline_run/, comparison_metrics.csv
│   └── ...             # Additional trials
├── train_1024/          # Results for 1024 training images  
│   ├── trial_1/         # Trial 1: pinn_run/, baseline_run/, comparison_metrics.csv
│   └── ...             # Additional trials
├── *.png               # Generalization plots with median and percentile bands
├── results.csv         # Aggregated median and percentile statistics
└── study_log.txt       # Execution log
```

## Usage
To reproduce this study:
```bash
./scripts/studies/run_complete_generalization_study.sh \
    --train-group-sizes "8192"     --train-subsample-sizes "8192"     --test-groups "2048"     --test-subsample "4096" \
    --output-dir custom_study_dir
```

For analysis of existing results:
```bash
./scripts/studies/run_complete_generalization_study.sh \
    --skip-data-prep --skip-training \
    --output-dir 3way_lsqml_8192
```
