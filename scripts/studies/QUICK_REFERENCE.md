# Model Generalization Studies - Quick Reference

This directory contains tools for conducting comprehensive model generalization studies comparing PtychoPINN and baseline performance across different training set sizes.

## Quick Commands

### Complete Automated Study
```bash
# Full study with default settings (4-8 hours)
./run_complete_generalization_study.sh

# Quick test study (1-2 hours) 
./run_complete_generalization_study.sh --train-sizes "256 512"

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
├── train_1024/          # Results for 1024 training images  
├── train_2048/          # Results for 2048 training images
├── train_4096/          # Results for 4096 training images
├── psnr_phase_generalization.png    # 📊 Main result plot
├── frc50_amp_generalization.png     # 📊 FRC analysis
├── mae_amp_generalization.png       # 📊 Error trends
├── results.csv                      # 📋 Aggregated data
└── STUDY_SUMMARY.md                 # 📄 Summary report
```

## Study Configurations

| Configuration | Training Sizes | Runtime | Disk Space | Use Case |
|---------------|----------------|---------|------------|----------|
| **Quick Test** | 256, 512 | 1-2 hours | ~20GB | Testing, validation |
| **Standard** | 512, 1024, 2048, 4096 | 4-6 hours | ~50GB | Publication results |
| **Extended** | 256, 512, 1024, 2048, 4096, 8192 | 8-12 hours | ~75GB | Comprehensive research |

## Common Workflows

### 🚀 First-Time User
```bash
# Run a quick test to validate setup
./run_complete_generalization_study.sh \
    --train-sizes "256 512" \
    --output-dir validation_study

# Check results
ls validation_study/
open validation_study/psnr_phase_generalization.png
```

### 📊 Research Publication  
```bash
# Full study for paper
./run_complete_generalization_study.sh \
    --train-sizes "512 1024 2048 4096" \
    --output-dir paper_results_$(date +%Y%m%d)

# Generate all key plots
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