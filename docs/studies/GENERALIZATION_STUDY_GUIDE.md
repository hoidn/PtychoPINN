# Complete Model Generalization Study Guide

This guide provides comprehensive instructions for running and analyzing model generalization studies comparing PtychoPINN and baseline performance across different training set sizes.

## Quick Start

### One-Command Complete Study
```bash
# Run complete study with default settings
./scripts/studies/run_complete_generalization_study.sh

# Custom study with specific training sizes
./scripts/studies/run_complete_generalization_study.sh \
    --train-sizes "256 512 1024 2048" \
    --parallel-jobs 2 \
    --output-dir my_generalization_study
```

### Estimated Resources
- **Time**: 4-8 hours (depending on hardware and training sizes)
- **Disk Space**: ~50GB for full study with 4 training sizes
- **GPU Memory**: ≥8GB VRAM recommended
- **CPU**: Parallel training can utilize multiple cores

## Detailed Workflow

### Step 1: Environment Setup

```bash
# Activate conda environment
conda activate ptycho

# Verify installation
python -c "import ptycho; print('PtychoPINN ready')"

# Navigate to project root
cd /path/to/PtychoPINN
```

### Step 2: Dataset Preparation

The study requires large-scale training and test datasets. This step generates 20,000 total images (10,000 train + 10,000 test).

#### Automatic Preparation
```bash
# Generate datasets automatically (included in complete study)
# Default behavior (backward compatible)
bash scripts/prepare.sh

# With custom parameters for specific studies
bash scripts/prepare.sh --input-file path/to/reconstruction.npz --output-dir studies/my_dataset --sim-images 20000 --sim-photons 1e9
```

#### Manual Preparation (if needed)
```bash
# Step-by-step dataset creation
python scripts/tools/pad_to_even_tool.py input.npz padded.npz
python scripts/tools/transpose_rename_convert_tool.py padded.npz transposed.npz
python scripts/tools/prepare_data_tool.py transposed.npz interpolated.npz --interpolate --zoom-factor 2.0
python scripts/simulation/simulate_and_save.py \
    --input-file interpolated.npz \
    --output-file simulated.npz \
    --n-images 20000 \
    --n-photons 1e9
python scripts/tools/downsample_data_tool.py simulated.npz downsampled.npz --crop-factor 2 --bin-factor 2
python scripts/tools/split_dataset_tool.py downsampled.npz final_dir --split-fraction 0.5
```

**Expected Output:**
- `datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_train.npz` (training data)
- `tike_outputs/fly001_final_downsampled/fly001_final_downsampled_data_transposed.npz` (test data)

### Step 3: Model Training

Train both PtychoPINN and baseline models for each training set size.

#### Individual Training Commands

**PtychoPINN Training:**
```bash
python scripts/training/train.py \
    --train_data_file datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_train.npz \
    --test_data_file tike_outputs/fly001_final_downsampled/fly001_final_downsampled_data_transposed.npz \
    --n_images 1024 \
    --output_dir study_results/train_1024/pinn_run \
    --nepochs 50
```

**Baseline Training:**
```bash
python scripts/run_baseline.py \
    --train_data datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_train.npz \
    --test_data tike_outputs/fly001_final_downsampled/fly001_final_downsampled_data_transposed.npz \
    --n_images 1024 \
    --output_dir study_results/train_1024/baseline_run \
    --epochs 50
```

#### Parallel Training
```bash
# Train multiple sizes in parallel (requires sufficient GPU memory)
./scripts/studies/run_complete_generalization_study.sh \
    --skip-data-prep \
    --skip-comparison \
    --parallel-jobs 2 \
    --train-sizes "512 1024"
```

**Training Outputs:**
- `train_SIZE/pinn_run/wts.h5.zip` - PtychoPINN trained model
- `train_SIZE/baseline_run/baseline_model.h5` - Baseline trained model
- `train_SIZE/*/history.dill` - Training history and metrics

### Step 4: Model Comparison

Generate quantitative metrics and visual comparisons for each training size.

```bash
python scripts/compare_models.py \
    --pinn_dir study_results/train_1024/pinn_run \
    --baseline_dir study_results/train_1024/baseline_run \
    --test_data tike_outputs/fly001_final_downsampled/fly001_final_downsampled_data_transposed.npz \
    --output_dir study_results/train_1024
```

**Comparison Outputs:**
- `comparison_metrics.csv` - Quantitative metrics (PSNR, MAE, MSE, FRC50)
- `comparison_plot.png` - Side-by-side visual comparison

### Step 5: Results Aggregation

Create publication-ready generalization plots from all training sizes.

```bash
# Primary PSNR phase plot
python scripts/studies/aggregate_and_plot_results.py \
    study_results/ \
    --metric psnr \
    --part phase \
    --output psnr_phase_generalization.png

# Additional metrics
python scripts/studies/aggregate_and_plot_results.py \
    study_results/ \
    --metric frc50 \
    --part amp \
    --output frc50_amp_generalization.png

python scripts/studies/aggregate_and_plot_results.py \
    study_results/ \
    --metric mae \
    --part amp \
    --output mae_amp_generalization.png
```

## Study Configurations

### Small-Scale Study (Testing)
```bash
./scripts/studies/run_complete_generalization_study.sh \
    --train-sizes "256 512" \
    --output-dir small_scale_test
```
- **Runtime**: ~2 hours
- **Disk**: ~20GB
- **Purpose**: Quick validation and testing

### Standard Study (Recommended)
```bash
./scripts/studies/run_complete_generalization_study.sh \
    --train-sizes "512 1024 2048 4096" \
    --parallel-jobs 2 \
    --output-dir standard_generalization_study
```
- **Runtime**: ~6 hours
- **Disk**: ~50GB  
- **Purpose**: Publication-quality results

### Extended Study (Research)
```bash
./scripts/studies/run_complete_generalization_study.sh \
    --train-sizes "256 512 1024 2048 4096 8192" \
    --parallel-jobs 4 \
    --output-dir extended_generalization_study
```
- **Runtime**: ~12 hours
- **Disk**: ~75GB
- **Purpose**: Comprehensive analysis with more data points

## Partial Execution

### Use Existing Dataset
```bash
./scripts/studies/run_complete_generalization_study.sh \
    --skip-data-prep \
    --test-data tike_outputs/fly001_final_downsampled/fly001_final_downsampled_data_transposed.npz
```

### Skip Training (Analysis Only)
```bash
./scripts/studies/run_complete_generalization_study.sh \
    --skip-data-prep \
    --skip-training \
    --output-dir existing_study_results
```

### Resume Interrupted Study
```bash
# Continue from where training left off
./scripts/studies/run_complete_generalization_study.sh \
    --skip-data-prep \
    --output-dir interrupted_study_results \
    --test-data tike_outputs/fly001_final_downsampled/fly001_final_downsampled_data_transposed.npz
```

## Output Structure

```
study_output_directory/
├── datasets/                                    # Training/test data (if generated)
│   └── fly001_reconstructed_prepared/
│       ├── *_train.npz                         # Training dataset  
│       └── *_test.npz                          # Test dataset
├── train_512/                                  # Results for 512 training images
│   ├── pinn_run/                              # PtychoPINN outputs
│   │   ├── wts.h5.zip                         # Trained model
│   │   ├── history.dill                       # Training history
│   │   └── params.dill                        # Model parameters
│   ├── baseline_run/                          # Baseline outputs  
│   │   ├── baseline_model.h5                  # Trained model
│   │   └── recon.dill                         # Reconstruction
│   ├── comparison_metrics.csv                 # Quantitative comparison
│   └── comparison_plot.png                    # Visual comparison
├── train_1024/                                # Results for 1024 training images
├── train_2048/                                # Results for 2048 training images  
├── train_4096/                                # Results for 4096 training images
├── psnr_phase_generalization.png              # Primary generalization plot
├── frc50_amp_generalization.png               # FRC analysis plot
├── mae_amp_generalization.png                 # MAE trend plot
├── results.csv                                # Aggregated metrics data
├── study_config.txt                           # Study configuration
├── study_log.txt                              # Complete execution log
└── STUDY_SUMMARY.md                           # Automated summary report
```

## Troubleshooting Common Issues

### Quick Troubleshooting Checklist

| Issue | Solution |
|-------|----------|
| **Not enough training images** | Increase dataset size in `prepare.sh` or use existing larger dataset |
| **Training interrupted** | Resume individual training manually, then continue with comparison |
| **Compare models failed** | Ensure both model directories contain proper model files |
| **Workflow hierarchy confusion** | Use top-level `run_complete_generalization_study.sh` for full control |

### Common Failure Modes

**"Not enough training images" Error:**
- **Symptom**: Study fails when requesting more images than dataset contains
- **Cause**: Dataset has fewer images than requested training size
- **Solution**: Use existing large dataset `datasets/fly/fly001_transposed.npz` (~10k images)
- **Alternative**: Increase simulation size with `scripts/prepare.sh --sim-images 40000` (may hit GPU memory limits)

**Training Interrupted or Failed:**
- **Symptom**: Missing model files in `train_SIZE/trial_N/pinn_run/` or `baseline_run/`
- **Cause**: Training crashed due to GPU memory, timeout, or other issues
- **Solution**: Resume individual training step:
  ```bash
  python scripts/training/train.py \
      --train_data_file data_train.npz \
      --test_data_file data_test.npz \
      --n_images SIZE \
      --output_dir train_SIZE/trial_N/pinn_run
  ```

**Compare Models Failed:**
- **Symptom**: Missing `comparison_metrics.csv` files in trial directories
- **Cause**: Model comparison step failed due to missing files or incompatible formats
- **Solution**: Run comparison manually:
  ```bash
  python scripts/compare_models.py \
      --pinn_dir train_SIZE/trial_N/pinn_run \
      --baseline_dir train_SIZE/trial_N/baseline_run \
      --test_data test_data.npz \
      --output_dir train_SIZE/trial_N
  ```

**Tool Selection Confusion:**
- **Problem**: Using wrong tool for intended workflow (e.g., `compare_models.py` with training size control)
- **Solution**: Follow tool capabilities matrix in `scripts/studies/QUICK_REFERENCE.md`
- **Best Practice**: Use `run_complete_generalization_study.sh` for full workflow control

### Recovery Patterns

**Partial Study Recovery:**
If a study partially completes, you can resume from where it left off:
```bash
# Resume training from specific point
./run_complete_generalization_study.sh \
    --skip-data-prep \
    --train-sizes "2048 4096" \
    --output-dir existing_study_dir

# Skip training, just regenerate plots  
./run_complete_generalization_study.sh \
    --skip-data-prep --skip-training \
    --output-dir existing_study_dir
```

**Individual Component Recovery:**
```bash
# Regenerate specific comparison
python scripts/compare_models.py \
    --pinn_dir study_dir/train_1024/trial_1/pinn_run \
    --baseline_dir study_dir/train_1024/trial_1/baseline_run \
    --test_data test_data.npz \
    --output_dir study_dir/train_1024/trial_1

# Regenerate all plots
python scripts/studies/aggregate_and_plot_results.py study_dir/
```

## Interpreting Results

### Key Metrics

**PSNR (Peak Signal-to-Noise Ratio)**
- Higher values = better reconstruction quality
- Measured in decibels (dB)
- Separate values for amplitude and phase

**MAE (Mean Absolute Error)**  
- Lower values = better reconstruction accuracy
- Direct pixel-wise comparison with ground truth
- Scale: typically 0.01-0.05 for good reconstructions

**FRC50 (Fourier Ring Correlation)**
- Higher values = better spatial resolution
- Measures correlation in frequency domain
- Scale: typically 10-50 pixels

**MSE (Mean Squared Error)**
- Lower values = better reconstruction accuracy  
- More sensitive to outliers than MAE
- Scale: typically 0.0001-0.01

### Typical Patterns

**PtychoPINN Characteristics:**
- Strong performance with limited data (512-1024 images)
- Stable performance across training sizes
- Leverages physics constraints effectively

**Baseline Characteristics:**
- Requires more training data for competitive performance
- Shows larger improvement with increased data
- More traditional deep learning scaling behavior

**Expected Trends:**
- Both models improve with more training data
- Performance gains diminish after ~2048 training images
- PtychoPINN maintains advantage in data-limited scenarios

## Troubleshooting

### Common Issues

**GPU Memory Errors**
```bash
# Reduce batch size or train sequentially
./scripts/studies/run_complete_generalization_study.sh \
    --parallel-jobs 1 \
    --train-sizes "512 1024"
```

**Dataset Not Found**
```bash
# Verify dataset preparation completed
ls -la datasets/fly001_reconstructed_prepared/

# Re-run preparation if needed (with custom parameters)
bash scripts/prepare.sh --output-dir studies/recovery_dataset --sim-images 20000
```

**Training Failures**
```bash
# Check individual training logs
tail -f study_output/train_1024/pinn_run/training.log
tail -f study_output/train_1024/baseline_run/training.log
```

**Missing Comparison Files**
```bash
# Manually run comparison for specific size
python scripts/compare_models.py \
    --pinn_dir study_output/train_1024/pinn_run \
    --baseline_dir study_output/train_1024/baseline_run \
    --test_data datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_test.npz \
    --output_dir study_output/train_1024
```

### Performance Optimization

**GPU Utilization**
- Use `nvidia-smi` to monitor GPU usage
- Increase batch size if memory allows
- Enable mixed precision training for speed

**Disk I/O**
- Use SSD storage for datasets if possible
- Consider data preprocessing caching
- Monitor disk space during large studies

**Parallelization**
- Limit parallel jobs based on GPU memory
- Use `--parallel-jobs 1` for large training sizes
- Monitor system resources with `htop`

## Advanced Usage

### Custom Metrics
```python
# Add custom evaluation in compare_models.py
def custom_metric(target, pred):
    # Your custom evaluation logic
    return metric_value

# Then update comparison CSV generation
```

### Custom Training Configurations
```bash
# Modify training parameters in the automated script
# Or run individual training commands with custom args
python scripts/training/train.py \
    --train_data_file data.npz \
    --n_images 1024 \
    --nepochs 100 \
    --learning_rate 0.0001 \
    --output_dir custom_run
```

### Visualization Customization
```python
# Modify scripts/studies/aggregate_and_plot_results.py
# To change plot styling, add metrics, or modify layouts
```

## Integration with Research Workflows

### Paper Results
1. Run standard study with 4+ training sizes
2. Generate all three key plots (PSNR, FRC, MAE)
3. Extract key statistics from `results.csv`
4. Use `STUDY_SUMMARY.md` for methods description

### Model Development
1. Use small-scale studies for rapid iteration
2. Compare architectural changes across training sizes
3. Validate improvements with full-scale study

### Hyperparameter Optimization
1. Run studies with different model configurations
2. Compare generalization behavior across settings
3. Identify robust configurations across data regimes

## Citation

When using this generalization study framework in research, please cite:

```bibtex
@misc{ptychopinn_generalization_study,
    title={PtychoPINN Model Generalization Study Framework},
    author={[Your Name]},
    year={2025},
    note={Available at: https://github.com/your-repo/PtychoPINN}
}
```