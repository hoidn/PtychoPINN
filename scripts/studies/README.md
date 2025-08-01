# Model Generalization Studies

This directory contains tools and workflows for conducting systematic studies of model generalization across different training set sizes and evaluation conditions.

> **⚠️ Important Limitation for `gridsize > 1` Studies**
>
> The current data loading pipeline selects training images **sequentially**. When using a `gridsize` greater than 1, this means the model is trained on a small, spatially localized (and therefore biased) region of the object.
>
> - **Do not** shuffle the dataset before subsampling, as this will create physically meaningless neighbor groups.
> - For rigorous results with `gridsize > 1`, it is strongly recommended to prepare separate, smaller, but complete datasets for each training size, rather than relying on the `--n-train-images` flag.
> - Results from studies using this flag with `gridsize > 1` should be interpreted as measuring performance on a localized subset, not general performance across the entire object.

## Overview

The generalization study workflow enables researchers to:
- Train models with varying training dataset sizes
- Evaluate model performance across different test conditions
- Generate comprehensive analysis plots showing performance trends
- Compare PtychoPINN vs baseline model generalization characteristics

## Key Scripts

### `run_complete_generalization_study.sh`
Master orchestration script that automates the complete generalization study workflow with support for both synthetic and experimental datasets.

**Purpose:** Trains multiple model pairs (PtychoPINN + baseline) across a range of training set sizes, then aggregates results into comprehensive analysis plots. Supports both auto-generated synthetic datasets and user-provided experimental datasets.

**Usage:**
```bash
# Automatic synthetic data generation (default mode)
./scripts/studies/run_complete_generalization_study.sh [options]

# Experimental dataset mode
./scripts/studies/run_complete_generalization_study.sh \
    --train-data <train_data.npz> \
    --test-data <test_data.npz> \
    --skip-data-prep \
    [options]
```

**Key Options:**
- `--train-sizes "512 1024 2048"`: Space-separated list of training set sizes to test
- `--test-sizes "1024 2048 4096"`: **NEW** Space-separated list of test set sizes (must match number of train sizes)
- `--num-trials N`: Number of trials per training size for statistical robustness (default: 5)
- `--train-data PATH`: Path to training dataset (for experimental data mode)
- `--test-data PATH`: Path to test dataset (for experimental data mode)
- `--skip-data-prep`: Skip synthetic data generation, use provided datasets
- `--output-dir DIR`: Output directory (default: timestamped directory)
- `--parallel-jobs N`: Number of parallel training jobs (default: 1)
- `--add-tike-arm`: **NEW** Enable 3-way comparison including Tike iterative reconstruction
- `--tike-iterations N`: **NEW** Number of Tike iterations (default: 1000, use 100-200 for quick tests)

### `run_generalization_study.sh` (Legacy)
Legacy script for manual generalization studies. Use `run_complete_generalization_study.sh` for new studies.

### `aggregate_and_plot_results.py`
Analysis script that processes results from multiple training runs and generates visualization plots.

**Purpose:** Collects metrics from multiple comparison directories, aggregates the data, and creates publication-quality plots showing model performance trends.

**Usage:**
```bash
python scripts/studies/aggregate_and_plot_results.py <study_output_dir> [--output-plot results.png]
```

## Workflow Modes

The generalization study script supports two primary modes:

### Mode 1: Synthetic Data Generation (Default)
Automatically generates large-scale synthetic datasets for controlled studies.

**Best for:** Controlled experiments, publication studies, baseline comparisons

```bash
# Full synthetic study with default settings
./scripts/studies/run_complete_generalization_study.sh

# Custom synthetic study
./scripts/studies/run_complete_generalization_study.sh \
    --train-sizes "512 1024 2048 4096" \
    --num-trials 3 \
    --output-dir my_synthetic_study
```

### Mode 2: Experimental Data Analysis
Uses existing experimental datasets for real-world validation studies.

**Best for:** Validating models on real data, comparing experimental vs synthetic performance

```bash
# Basic experimental study
./scripts/studies/run_complete_generalization_study.sh \
    --train-data "datasets/fly64/fly001_64_train_converted.npz" \
    --test-data "datasets/fly64/fly001_64_train_converted.npz" \
    --skip-data-prep \
    --train-sizes "512 1024" \
    --output-dir experimental_study

# Multi-trial experimental study for robust statistics
./scripts/studies/run_complete_generalization_study.sh \
    --train-data "datasets/fly64/fly64_top_half_shuffled.npz" \
    --test-data "datasets/fly64/fly001_64_train_converted.npz" \
    --skip-data-prep \
    --train-sizes "512 1024 2048" \
    --num-trials 3 \
    --output-dir robust_experimental_study
```

### Mode 3: Three-Way Comparison Studies (NEW)
Compare PtychoPINN, Baseline, and Tike iterative reconstruction for comprehensive algorithm evaluation.

**Best for:** Algorithm benchmarking, comparing ML vs traditional methods, research publications

```bash
# Quick 3-way comparison study
./scripts/studies/run_complete_generalization_study.sh \
    --add-tike-arm \
    --tike-iterations 100 \
    --train-sizes "512 1024" \
    --num-trials 2 \
    --output-dir quick_3way_study

# Full 3-way research study
./scripts/studies/run_complete_generalization_study.sh \
    --add-tike-arm \
    --tike-iterations 1000 \
    --train-sizes "512 1024 2048 4096" \
    --num-trials 3 \
    --train-data "datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_train.npz" \
    --test-data "datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_test.npz" \
    --skip-data-prep \
    --output-dir research_3way_study
```

**3-Way Study Output:**
- **2x4 comparison plots**: PtychoPINN, Baseline, Tike, Ground Truth
- **Complete metrics CSV**: All methods evaluated on identical test subsets
- **Fair evaluation**: Automatic test data subsampling ensures equal evaluation conditions
- **Timing data**: Computation time comparison across all three methods

## Complete Workflow Examples

### Synthetic Data Study
```bash
# Full publication-quality study with synthetic data
./scripts/studies/run_complete_generalization_study.sh \
    --train-sizes "512 1024 2048 4096" \
    --num-trials 5 \
    --output-dir publication_study_$(date +%Y%m%d)

# Results: publication_study_YYYYMMDD/
# -> train_512/trial_1/, trial_2/, ..., trial_5/
# -> train_1024/trial_1/, trial_2/, ..., trial_5/
# -> psnr_phase_generalization.png (with uncertainty bands)
# -> STUDY_SUMMARY.md
```

### Experimental Data Study
```bash
# Study with experimental fly64 dataset
./scripts/studies/run_complete_generalization_study.sh \
    --train-data "datasets/fly64/fly001_64_train_converted.npz" \
    --test-data "datasets/fly64/fly001_64_train_converted.npz" \
    --skip-data-prep \
    --train-sizes "512 1024 2048" \
    --num-trials 3 \
    --output-dir fly64_study

# Results: fly64_study/
# -> train_512/trial_1/, trial_2/, trial_3/
# -> generalization plots with experimental data validation
```

## Output Directory Structure

After running a generalization study, the output directory will contain:

```
study_output/
├── train_512/                      # Results for 512 training images
│   ├── trial_1/                    # First training run
│   │   ├── pinn_run/               # PtychoPINN model outputs
│   │   │   ├── wts.h5.zip          # Trained model weights
│   │   │   ├── history.dill        # Training history
│   │   │   └── params.dill         # Training configuration
│   │   ├── baseline_run/           # Baseline model outputs
│   │   │   └── baseline_model.h5   # Trained baseline model
│   │   ├── comparison_plot.png     # Side-by-side comparison
│   │   └── comparison_metrics.csv  # Quantitative metrics
│   ├── trial_2/                    # Second training run (if --num-trials > 1)
│   └── trial_N/                    # Additional trials
├── train_1024/                     # Results for 1024 training images
│   ├── trial_1/
│   └── trial_N/
├── train_2048/                     # Results for 2048 training images
│   ├── trial_1/
│   └── trial_N/
├── psnr_phase_generalization.png   # 📊 Primary generalization plot (mean ± percentiles)
├── frc50_amp_generalization.png    # 📊 FRC analysis plot  
├── mae_amp_generalization.png      # 📊 Error trends plot
├── ssim_amp_generalization.png     # 📊 SSIM amplitude analysis
├── ssim_phase_generalization.png   # 📊 SSIM phase analysis
├── ms_ssim_amp_generalization.png  # 📊 Multi-Scale SSIM amplitude
├── ms_ssim_phase_generalization.png # 📊 Multi-Scale SSIM phase
├── results.csv                     # 📋 Aggregated median and percentile statistics
├── study_config.txt                # Configuration parameters
├── study_log.txt                   # Complete execution log
└── STUDY_SUMMARY.md                # 📄 Executive summary report
```

## Key Metrics Tracked

The study tracks comprehensive metrics across training set sizes with statistical robustness:

### Core Metrics
- **Mean Absolute Error (MAE)**: Average pixel-wise error
- **Mean Squared Error (MSE)**: Squared pixel-wise error  
- **Peak Signal-to-Noise Ratio (PSNR)**: Image quality metric
- **Fourier Ring Correlation (FRC50)**: Spatial frequency resolution metric

### Advanced Perceptual Metrics
- **SSIM (Structural Similarity)**: Perceptual similarity for amplitude and phase
- **MS-SSIM (Multi-Scale SSIM)**: Multi-resolution perceptual analysis
- **Registration Offsets**: Alignment quality assessment

### Statistical Analysis
- **Multi-Trial Support**: Runs multiple training instances per configuration
- **Robust Statistics**: Reports mean, 25th percentile, and 75th percentile
- **Uncertainty Quantification**: Visualizes performance variability
- **NaN Handling**: Automatically excludes failed trials from aggregation

## Experimental Data Requirements

### For Experimental Datasets:
1. **Preprocessing Required**: Use `transpose_rename_convert_tool.py` for raw experimental data
2. **Shuffling Critical**: For gridsize=1 studies, shuffle datasets with `shuffle_dataset_tool.py`
3. **Format Compliance**: Must follow [Data Contracts](../../docs/data_contracts.md) specification
4. **Sufficient Size**: Ensure dataset has enough images for largest training size requested

### Recommended Experimental Datasets:
- **fly64**: `datasets/fly64/fly001_64_train_converted.npz` (10,304 images)
- **fly64 (spatial bias)**: `datasets/fly64/fly64_top_half_shuffled.npz` (5,172 images, shuffled)

## Usage Tips

### Study Design
- **Training Set Sizes**: Choose sizes that span 1-2 orders of magnitude to observe clear trends
- **Multi-Trial Robustness**: Use `--num-trials 3-5` for publication-quality results
- **Resource Planning**: Each trial can take 1-2 hours; plan accordingly for multiple trials
- **Decoupled Train/Test Sizes**: **NEW** Use `--test-sizes` to evaluate models on different test set sizes than training:
  ```bash
  # Train on small sets, test on larger sets to assess generalization
  ./run_complete_generalization_study.sh \
      --train-sizes "256 512 1024" \
      --test-sizes "512 1024 2048"
  ```
  This allows studying how models trained on limited data perform on larger test sets

### Performance Optimization  
- **Parallel Jobs**: Use `--parallel-jobs` carefully to avoid GPU memory conflicts
- **Disk Space**: Allocate ~50-100GB for full studies with multiple trials
- **GPU Memory**: Monitor memory usage, especially with large training sizes

### Result Interpretation
- **Statistical Significance**: Multi-trial studies provide uncertainty bands in plots
- **Model Comparison**: PtychoPINN typically shows superior data efficiency
- **Convergence Analysis**: Both models often converge at larger dataset sizes

## Requirements

- All scripts in this directory require the main PtychoPINN environment to be activated
- Input datasets must conform to the [Data Contracts](../../docs/data_contracts.md) specification
- Sufficient disk space for multiple model checkpoints and intermediate results