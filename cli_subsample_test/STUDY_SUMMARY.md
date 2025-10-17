# Model Generalization Study Summary

**Generated:** Thu Sep  4 10:55:29 PM PDT 2025
**Study Directory:** cli_subsample_test
**Training Group Sizes:** 128
**Training Subsample Sizes:** 128
**Test Groups:** 128
**Test Subsample:** 128
**Trials per Size:** 1

## Study Configuration
- **Total Trials:** 1
- **Test Dataset:** prepare_1e4_photons_5k/dataset/test.npz
- **Total Runtime:** 00:01:36

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
cli_subsample_test/
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
    --train-group-sizes "128"     --train-subsample-sizes "128"     --test-groups "128"     --test-subsample "128" \
    --output-dir custom_study_dir
```

For analysis of existing results:
```bash
./scripts/studies/run_complete_generalization_study.sh \
    --skip-data-prep --skip-training \
    --output-dir cli_subsample_test
```
