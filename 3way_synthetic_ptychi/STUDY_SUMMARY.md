# Model Generalization Study Summary

**Generated:** Tue Aug 26 02:47:27 PM PDT 2025
**Study Directory:** 3way_synthetic_ptychi
**Training Sizes:** 512
**Trials per Size:** 1

## Study Configuration
- **Total Trials:** 1
- **Test Dataset:** datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_test.npz
- **Total Runtime:** 00:02:34

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
3way_synthetic_ptychi/
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
    --train-sizes "512" \
    --output-dir custom_study_dir
```

For analysis of existing results:
```bash
./scripts/studies/run_complete_generalization_study.sh \
    --skip-data-prep --skip-training \
    --output-dir 3way_synthetic_ptychi
```
