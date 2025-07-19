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

### `run_generalization_study.sh`
Master orchestration script that automates the complete generalization study workflow.

**Purpose:** Trains multiple model pairs (PtychoPINN + baseline) across a range of training set sizes, then aggregates results into comprehensive analysis plots.

**Usage:**
```bash
./scripts/studies/run_generalization_study.sh <train_data.npz> <test_data.npz> <output_dir> [options]
```

**Options:**
- `--n-train-sizes`: Space-separated list of training set sizes to test
- `--n-test-images`: Number of test images to use for evaluation (default: 500)

### `aggregate_and_plot_results.py`
Analysis script that processes results from multiple training runs and generates visualization plots.

**Purpose:** Collects metrics from multiple comparison directories, aggregates the data, and creates publication-quality plots showing model performance trends.

**Usage:**
```bash
python scripts/studies/aggregate_and_plot_results.py <study_output_dir> [--output-plot results.png]
```

## Complete Workflow Example

Here's a step-by-step example of running a complete generalization study:

```bash
# 1. Prepare your dataset (if needed)
./scripts/prepare.sh datasets/raw/my_data.npz datasets/prepared/

# 2. Run the complete generalization study
./scripts/studies/run_generalization_study.sh \
    datasets/prepared/my_data_train.npz \
    datasets/prepared/my_data_test.npz \
    generalization_study_results \
    --n-train-sizes "128 256 512 1024 2048" \
    --n-test-images 500

# 3. View results
ls generalization_study_results/
# -> n_train_128/  n_train_256/  n_train_512/  n_train_1024/  n_train_2048/
# -> generalization_study_results.png
# -> aggregated_metrics.csv
```

## Output Directory Structure

After running a generalization study, the output directory will contain:

```
generalization_study_results/
├── n_train_128/                    # Results for 128 training images
│   ├── pinn_run/                   # PtychoPINN model outputs
│   ├── baseline_run/               # Baseline model outputs
│   ├── comparison_plot.png         # Side-by-side comparison
│   └── comparison_metrics.csv      # Quantitative metrics
├── n_train_256/                    # Results for 256 training images
│   └── ...
├── n_train_512/                    # Results for 512 training images
│   └── ...
├── generalization_study_results.png # Final aggregated plot
└── aggregated_metrics.csv         # Combined metrics from all runs
```

## Key Metrics Tracked

The study tracks several important metrics across training set sizes:

- **Mean Absolute Error (MAE)**: Average pixel-wise error
- **Mean Squared Error (MSE)**: Squared pixel-wise error
- **Peak Signal-to-Noise Ratio (PSNR)**: Image quality metric
- **Fourier Ring Correlation (FRC)**: Spatial frequency resolution metric

## Integration with Main Workflows

This generalization study builds on the core comparison workflow (`run_comparison.sh`) by:
1. Running multiple instances with different `--n-train-images` values
2. Collecting and aggregating the resulting metrics
3. Generating trend analysis plots

The individual comparison runs use the same configuration file (`configs/comparison_config.yaml`) to ensure consistent hyperparameters across all experiments.

## Usage Tips

- **Training Set Sizes**: Choose sizes that span 1-2 orders of magnitude to observe clear trends
- **Test Set Size**: Use a consistent, sufficiently large test set (≥500 images) for reliable metrics
- **Resource Planning**: Each training run can take several hours; plan accordingly for multiple runs
- **Reproducibility**: Results are deterministic given the same input data and training configuration

## Requirements

- All scripts in this directory require the main PtychoPINN environment to be activated
- Input datasets must conform to the [Data Contracts](../../docs/data_contracts.md) specification
- Sufficient disk space for multiple model checkpoints and intermediate results