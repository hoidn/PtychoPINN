# Model Generalization Studies - Quick Reference

This directory contains tools for conducting comprehensive model generalization studies comparing PtychoPINN and baseline performance across different training set sizes.

## Quick Commands

### Complete Automated Study
```bash
# Full study with default settings (4-8 hours)
./run_complete_generalization_study.sh

# Quick test study (1-2 hours) 
./run_complete_generalization_study.sh --train-sizes "256 512"

# Statistical multi-trial study (robust results with uncertainty)
./run_complete_generalization_study.sh \
    --train-sizes "512 1024 2048" \
    --num-trials 3 \
    --output-dir robust_study

# Custom study with parallel training
./run_complete_generalization_study.sh \
    --train-sizes "512 1024 2048 4096" \
    --parallel-jobs 2 \
    --output-dir my_study
```

### Partial Workflows
```bash
# Skip dataset preparation (use existing data)
./run_complete_generalization_study.sh \
    --skip-data-prep \
    --test-data tike_outputs/fly001_final_downsampled/fly001_final_downsampled_data_transposed.npz

# Analysis only (no training)
./run_complete_generalization_study.sh \
    --skip-data-prep --skip-training \
    --output-dir existing_results

# Manual training for single size
python ../training/train.py \
    --train_data_file data_train.npz \
    --test_data_file data_test.npz \
    --n_images 1024 \
    --output_dir train_1024_pinn \
    --nepochs 50
```

### Results Analysis
```bash
# Generate generalization plots
python aggregate_and_plot_results.py study_dir/ \
    --metric psnr --part phase \
    --output psnr_generalization.png

# Filter out failed trials with poor phase quality
python aggregate_and_plot_results.py study_dir/ \
    --ms-ssim-phase-threshold 0.25

# Disable MS-SSIM filtering to include all trials
python aggregate_and_plot_results.py study_dir/ \
    --ms-ssim-phase-threshold 0

# Compare specific models
python ../compare_models.py \
    --pinn_dir pinn_model_dir \
    --baseline_dir baseline_model_dir \
    --test_data test.npz \
    --output_dir comparison_results
```

## Key Files

| File | Purpose |
|------|---------|
| `run_complete_generalization_study.sh` | **Main script** - Complete automated study |
| `aggregate_and_plot_results.py` | Generate publication-ready generalization plots |
| `run_generalization_study.sh` | Legacy manual study script |
| `QUICK_REFERENCE.md` | This quick reference |

## Output Structure

```
study_output/
├── train_512/           # Results for 512 training images
│   ├── trial_1/         # First training run
│   ├── trial_2/         # Second training run (if --num-trials > 1)
│   └── trial_N/         # Nth training run
├── train_1024/          # Results for 1024 training images  
│   ├── trial_1/
│   └── trial_N/
├── psnr_phase_generalization.png    # 📊 Main result plot (mean ± percentiles)
├── frc50_amp_generalization.png     # 📊 FRC analysis (mean ± percentiles)
├── mae_amp_generalization.png       # 📊 Error trends (mean ± percentiles)
├── results.csv                      # 📋 Statistical aggregation (mean, p25, p75)
└── STUDY_SUMMARY.md                 # 📄 Summary report
```

## Study Configurations

| Configuration | Training Sizes | Trials | Runtime | Disk Space | Use Case |
|---------------|----------------|--------|---------|------------|----------|
| **Quick Test** | 512 | 2 | 1-2 hours | ~15GB | Testing, validation |
| **Standard** | 512, 1024, 2048 | 3 | 4-6 hours | ~60GB | Publication results |
| **Extended** | 512, 1024, 2048, 4096 | 5 | 8-12 hours | ~100GB | Comprehensive research |
| **Legacy Single** | 512, 1024, 2048, 4096 | 1 | 4-6 hours | ~50GB | Backward compatibility |

## Multi-Trial Statistical Analysis

The `--num-trials` flag enables robust statistical analysis by training multiple models per configuration:

```bash
# Single trial (legacy mode) - shows mean performance
./run_complete_generalization_study.sh --train-sizes "512 1024"

# Multi-trial mode - shows mean ± percentiles for robust statistics  
./run_complete_generalization_study.sh --train-sizes "512 1024" --num-trials 3
```

**Benefits of Multi-Trial Analysis:**
- **Robust Statistics**: Mean and percentiles instead of single-point estimates
- **Uncertainty Quantification**: Shows performance variability across training runs
- **Outlier Resistance**: Less sensitive to random initialization effects
- **Publication Quality**: Professional plots with uncertainty bands

## Enhanced NaN Handling & MS-SSIM Filtering

The aggregation script now includes robust handling of failed trials and quality-based filtering:

### NaN Handling
- **Automatic Exclusion**: Failed metrics (NaN values) are automatically excluded from statistics
- **Preservation**: NaN values are preserved in data, not replaced with zeros/inf
- **Logging**: Reports how many NaN trials were excluded for each metric

### MS-SSIM Phase Filtering
Use `--ms-ssim-phase-threshold` to exclude poor-quality trials:

```bash
# Default: Exclude trials with MS-SSIM (phase) < 0.3
python aggregate_and_plot_results.py study_dir/

# Custom threshold: Stricter quality requirement
python aggregate_and_plot_results.py study_dir/ --ms-ssim-phase-threshold 0.5

# Include all trials (no filtering)
python aggregate_and_plot_results.py study_dir/ --ms-ssim-phase-threshold 0
```

**Use Cases:**
- **Failed Reconstructions**: Automatically exclude trials that completely failed
- **Quality Control**: Filter out trials with poor perceptual quality
- **Robust Statistics**: Ensure aggregated results represent successful runs only

## Common Workflows

### 🚀 First-Time User
```bash
# Run a quick test to validate setup
./run_complete_generalization_study.sh \
    --train-sizes "512" \
    --num-trials 2 \
    --output-dir validation_study

# Check results
ls validation_study/
open validation_study/psnr_phase_generalization.png
```

### 📊 Research Publication  
```bash
# Full study for paper with statistical robustness
./run_complete_generalization_study.sh \
    --train-sizes "512 1024 2048 4096" \
    --num-trials 5 \
    --output-dir paper_results_$(date +%Y%m%d)

# Generate all key plots (automatically shows mean ± percentiles)
cd paper_results_*/
python ../aggregate_and_plot_results.py . --metric psnr --part phase
python ../aggregate_and_plot_results.py . --metric frc50 --part amp  
python ../aggregate_and_plot_results.py . --metric mae --part amp
```

### 🔧 Model Development
```bash
# Quick iteration testing
./run_complete_generalization_study.sh \
    --train-sizes "512 1024" \
    --parallel-jobs 2 \
    --output-dir dev_test_v1

# Compare against baseline
./run_complete_generalization_study.sh \
    --skip-data-prep \
    --output-dir dev_test_v2 \
    --test-data tike_outputs/fly001_final_downsampled/fly001_final_downsampled_data_transposed.npz
```

### 📈 Analysis of Existing Results
```bash
# Re-analyze existing study results
./run_complete_generalization_study.sh \
    --skip-data-prep --skip-training \
    --output-dir path/to/existing/results

# Custom metric analysis
python aggregate_and_plot_results.py existing_results/ \
    --metric mse --part phase \
    --output custom_analysis.png
```

## Tool Capabilities & Limitations

| Tool | Controls Training Size | Controls Test Size | Use Case |
|------|----------------------|-------------------|----------|
| `run_complete_generalization_study.sh` | ✅ `--train-sizes` | ✅ (via dataset) | Full studies |
| `compare_models.py` | ❌ (post-training) | ❌ (uses full dataset) | Existing models |
| `aggregate_and_plot_results.py` | N/A | N/A | Result visualization |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **GPU out of memory** | `--parallel-jobs 1` or reduce training sizes |
| **Dataset not found** | Run `bash ../../prepare.sh` first |
| **Training fails** | Check individual logs in `train_SIZE/*/training.log` |
| **Missing comparison files** | Re-run comparison manually with `../compare_models.py` |
| **Plot generation fails** | Verify all `comparison_metrics.csv` files exist |

## Dependencies

- Python 3.10+ with PtychoPINN environment
- GPU with ≥8GB VRAM (recommended)
- ~50GB free disk space for full study
- Required packages: `tensorflow`, `numpy`, `pandas`, `matplotlib`

## Help & Documentation

```bash
# Script help
./run_complete_generalization_study.sh --help

# Detailed documentation
cat ../../docs/studies/GENERALIZATION_STUDY_GUIDE.md

# Check environment
python -c "import ptycho; print('✓ PtychoPINN ready')"
```

## Study Phases

1. **Dataset Preparation** (~30 min) - Generate 20K training + test images
2. **Model Training** (~3-6 hours) - Train PtychoPINN + baseline for each size  
3. **Model Comparison** (~15 min) - Calculate metrics and create visualizations
4. **Results Aggregation** (~5 min) - Generate generalization plots and summary

## Expected Results

- **PtychoPINN**: Superior performance with limited training data
- **Baseline**: Requires more data but approaches PtychoPINN performance
- **Both models**: Performance plateaus around 2048-4096 training images
- **Key insight**: PtychoPINN provides better data efficiency for ptychography

---

**💡 Tip**: Start with a quick test study (`--train-sizes "256 512"`) to validate your setup before running the full study.