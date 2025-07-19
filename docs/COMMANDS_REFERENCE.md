# PtychoPINN Commands Reference

Quick reference for common workflows and command patterns in PtychoPINN.

## 📋 Quick Navigation
- [Data Preparation](#data-preparation)
- [Training](#training) 
- [Inference](#inference)
- [Evaluation & Comparison](#evaluation--comparison)
- [Studies & Experiments](#studies--experiments)
- [Parameter Interpretation](#parameter-interpretation)

---

## Data Preparation

### Dataset Conversion and Preprocessing
```bash
# Convert raw data (REQUIRED FIRST STEP FOR RAW EXPERIMENTAL DATA)
python scripts/tools/transpose_rename_convert_tool.py datasets/raw/data.npz datasets/converted/data.npz

# Prepare datasets (smoothing, interpolation, etc.)
python scripts/tools/prepare_data_tool.py --input datasets/converted/data.npz --output datasets/prepared/ --smooth

# Shuffle dataset for randomized sampling
python scripts/tools/shuffle_dataset_tool.py --input datasets/fly/fly001.npz --output datasets/fly/fly001_shuffled.npz

# Split dataset into train/test
python scripts/tools/split_dataset_tool.py datasets/fly/fly001.npz output_dir/ --split-fraction 0.8

# Visualize dataset structure and contents
python scripts/tools/visualize_dataset.py datasets/fly/fly001.npz dataset_visualization.png
```

### Data Format Verification
```bash
# Quick data inspection
python scripts/inspect_ptycho_data.py datasets/fly/fly001.npz

# Verify data contracts compliance
python -c "
import numpy as np
data = np.load('datasets/fly/fly001.npz')
print('Keys:', list(data.keys()))
print('Shapes:', {k: data[k].shape for k in data.keys()})
print('Data types:', {k: data[k].dtype for k in data.keys()})
"
```

---

## Training

### Basic Training Patterns
```bash
# Traditional gridsize=1 training
ptycho_train --train_data_file datasets/fly/fly001.npz --n_images 5000 --gridsize 1 --nepochs 100 --output_dir training_run_gs1

# Grouping-aware gridsize=2 training  
ptycho_train --train_data_file datasets/fly/fly001.npz --n_images 1250 --gridsize 2 --nepochs 100 --output_dir training_run_gs2

# Configuration-based training
ptycho_train --config configs/fly_config.yaml --output_dir config_based_run
```

### Full-Scale Training Examples
```bash
# Train on complete dataset (gridsize=1) - traditional approach
ptycho_train --train_data_file datasets/fly/fly001_transposed.npz --n_images 10304 --gridsize 1 --nepochs 50 --output_dir full_validation_gs1

# Train on complete dataset (gridsize=2) - uses grouping-aware subsampling
ptycho_train --train_data_file datasets/fly/fly001_transposed.npz --n_images 2576 --gridsize 2 --nepochs 50 --output_dir full_validation_gs2
```

### Quick Verification Training
```bash
# Quick test with minimal epochs for verification
ptycho_train --train_data_file datasets/fly/fly001_transposed.npz --n_images 50 --gridsize 1 --nepochs 1 --output_dir quick_test_gs1
ptycho_train --train_data_file datasets/fly/fly001_transposed.npz --n_images 25 --gridsize 2 --nepochs 1 --output_dir quick_test_gs2
```

---

## Inference

### Basic Inference Patterns
```bash
# Standard inference (model_path should point to directory containing wts.h5.zip)
ptycho_inference --model_path training_run/ --test_data datasets/fly/test.npz --output_dir inference_results

# Inference with specific number of test images
ptycho_inference --model_path training_run/ --test_data datasets/fly/test.npz --n_images 512 --output_dir inference_512

# Gridsize-aware inference (must match training gridsize)
ptycho_inference --model_path training_run_gs2/ --test_data datasets/fly/test.npz --n_images 128 --gridsize 2 --output_dir inference_gs2
```

### Full-Scale Inference Examples
```bash
# Test gridsize=1 model on 1024 individual images
ptycho_inference --model_path full_validation_gs1 --test_data datasets/fly/fly001_transposed.npz --n_images 1024 --output_dir final_test_gs1

# Test gridsize=2 model on 256 groups (1024 total patterns)
ptycho_inference --model_path full_validation_gs2 --test_data datasets/fly/fly001_transposed.npz --n_images 256 --gridsize 2 --output_dir final_test_gs2
```

---

## Evaluation & Comparison

### Model Comparison Workflows
```bash
# Basic model comparison
python scripts/compare_models.py \
    --pinn_dir training_pinn/ \
    --baseline_dir training_baseline/ \
    --test_data datasets/fly/test.npz \
    --output_dir comparison_results

# Comparison with debug visualization
python scripts/compare_models.py \
    --pinn_dir training_pinn/ \
    --baseline_dir training_baseline/ \
    --test_data datasets/fly/test.npz \
    --output_dir comparison_debug \
    --save-debug-images

# Skip automatic registration (if needed)
python scripts/compare_models.py \
    --pinn_dir training_pinn/ \
    --baseline_dir training_baseline/ \
    --test_data datasets/fly/test.npz \
    --output_dir comparison_no_reg \
    --skip-registration
```

### Complete Training + Comparison Pipeline
```bash
# Run complete comparison workflow (gridsize=1, default)
./scripts/run_comparison.sh datasets/fly/train.npz datasets/fly/test.npz comparison_output

# Comparison with specific training/test sizes (gridsize=1)
./scripts/run_comparison.sh \
    datasets/fly/train.npz \
    datasets/fly/test.npz \
    comparison_sized \
    --n-train-images 2048 \
    --n-test-images 512
```

### GridSize=2 Comparison Workflow

**Important:** The comparison script uses a hardcoded configuration file. For `gridsize=2`, you must create a custom configuration.

```bash
# --- COMPLETE GRIDSIZE=2 COMPARISON WORKFLOW ---

# Step 1: Back up and modify configuration file
cp configs/comparison_config.yaml configs/comparison_config.yaml.bak

cat << EOF > configs/comparison_config.yaml
# configs/comparison_config.yaml (MODIFIED FOR GRIDSIZE=2)
N: 64
gridsize: 2
n_filters_scale: 2
model_type: 'pinn'
amp_activation: 'sigmoid'
object_big: true
probe_big: false
probe_mask: true
pad_object: true
probe_scale: 10.0
nepochs: 50
batch_size: 16
output_dir: "comparison_outputs"
train_data_file: "dummy_train.npz"
test_data_file: "dummy_test.npz"
n_images: 512
nphotons: 1000.0
nll_weight: 1.0
mae_weight: 0.0
probe_trainable: true
intensity_scale_trainable: true
positions_provided: true
EOF

echo "Configuration updated for gridsize=2."

# Step 2: Run comparison with group-based parameters
# Training: 10,000 total patterns = 2500 groups (10000 ÷ 4 patterns/group)
# Testing: 1024 groups = 4096 total patterns
./scripts/run_comparison.sh \
    datasets/fly/fly001_transposed.npz \
    datasets/fly/fly001_transposed.npz \
    comparison_gs2_10k_train \
    --n-train-images 2500 \
    --n-test-images 1024

echo "Comparison run finished."

# Step 3: Restore original configuration
mv configs/comparison_config.yaml.bak configs/comparison_config.yaml
echo "Original configuration file restored."

# Step 4: Verify results in log files
echo "Check comparison_gs2_10k_train/pinn_run/ for log messages confirming:"
echo "- Parameter interpretation: --n-images=2500 refers to neighbor groups"
echo "- Using grouping-aware subsampling strategy for gridsize=2"
```

---

## Studies & Experiments

### Generalization Studies
```bash
# Complete generalization study with multiple training sizes
./scripts/studies/run_complete_generalization_study.sh \
    --train-data datasets/fly64/fly001_64_train_converted.npz \
    --test-data datasets/fly64/fly001_64_train_converted.npz \
    --output-dir generalization_study_results \
    --train-sizes "512 1024 2048 4096" \
    --test-size 1024

# Multi-trial statistical study
./scripts/studies/run_complete_generalization_study.sh \
    --train-data datasets/fly64/fly001_64_train_converted.npz \
    --test-data datasets/fly64/fly001_64_train_converted.npz \
    --output-dir multi_trial_study \
    --train-sizes "512 1024 2048" \
    --test-size 1024 \
    --num-trials 3

# Quick generalization study (fewer sizes)
./scripts/studies/run_complete_generalization_study.sh \
    --train-data datasets/fly64/fly001_64_train_converted.npz \
    --test-data datasets/fly64/fly001_64_train_converted.npz \
    --output-dir quick_gen_study \
    --train-sizes "512 1024" \
    --test-size 512
```

### Results Analysis and Visualization
```bash
# Aggregate and plot generalization study results
python scripts/studies/aggregate_and_plot_results.py generalization_study_results --output final_plots/plot.png

# Extract metrics from individual comparison results
python -c "
import pandas as pd
import glob
# Read all comparison_metrics.csv files from study subdirectories
files = glob.glob('generalization_study_results/train_*/comparison_metrics.csv')
for f in files:
    df = pd.read_csv(f)
    print(f'{f}: MS-SSIM Phase = {df.loc[df[\"Metric\"] == \"ms_ssim_phase\", \"PtychoPINN\"].values[0]}')
"
```

### Simulation Studies
```bash
# Generate synthetic data for controlled experiments
python scripts/simulation/simulate_and_save.py \
    --input-file datasets/fly/fly001_transposed.npz \
    --output-file sim_outputs/synthetic_data.npz \
    --n-images 2000 \
    --gridsize 1 \
    --visualize

# Run simulation with synthetic line patterns
python scripts/simulation/run_with_synthetic_lines.py \
    --output-dir simulation_outputs \
    --probe-size 64 \
    --n-images 2000
```

---

## Parameter Interpretation

### Understanding n_images Parameter
The `--n_images` parameter is interpreted differently based on the `gridsize` setting:

| GridSize | n_images Value | Interpretation | Total Patterns | Example Command |
|----------|----------------|----------------|-----------------|-----------------|
| 1        | 10304          | Individual images | 10304 | `--n_images 10304 --gridsize 1` |
| 1        | 1024           | Individual images | 1024 | `--n_images 1024 --gridsize 1` |
| 2        | 2576           | Neighbor groups | ~10304 (2576×4) | `--n_images 2576 --gridsize 2` |
| 2        | 256            | Neighbor groups | 1024 (256×4) | `--n_images 256 --gridsize 2` |

### Parameter Equivalence Examples
```bash
# These commands train on approximately the same amount of data:

# Traditional approach: 1024 individual patterns
ptycho_train --train_data_file dataset.npz --n_images 1024 --gridsize 1

# Grouping-aware approach: 256 groups = 1024 total patterns  
ptycho_train --train_data_file dataset.npz --n_images 256 --gridsize 2
```

### GridSize=2 Calculation Examples
```bash
# For 10,000 total patterns with gridsize=2:
# 10,000 ÷ 4 patterns/group = 2500 groups
ptycho_train --train_data_file dataset.npz --n_images 2500 --gridsize 2

# For 1024 groups with gridsize=2:
# 1024 groups × 4 patterns/group = 4096 total patterns
ptycho_inference --model_path model/ --test_data dataset.npz --n_images 1024 --gridsize 2
```

**📚 For complete gridsize=2 comparison workflow, see [GridSize=2 Comparison Workflow](#gridsize2-comparison-workflow)**

### Log Message Interpretation
When running commands, look for these key log messages:

```
# Gridsize=1 (traditional)
INFO - Parameter interpretation: --n-images=1024 refers to individual images (gridsize=1)

# Gridsize=2 (grouping-aware)  
INFO - Parameter interpretation: --n-images=256 refers to neighbor groups (gridsize=2, total patterns=1024)
INFO - Groups cache loaded from dataset.g2k4.groups_cache.npz
INFO - Using 18058 cached groups
INFO - Randomly sampling 256 groups from 18058 available groups
```

---

## Performance and Debugging

### Memory and Performance Monitoring
```bash
# Monitor GPU memory during training
nvidia-smi -l 1

# Monitor system resources
htop

# Training with verbose logging
ptycho_train --train_data_file dataset.npz --n_images 1000 --verbose --output_dir debug_run
```

### Cache Management
```bash
# Check for existing cache files
find . -name "*.groups_cache.npz"

# Remove cache files to force regeneration
rm datasets/fly/fly001_transposed.g2k4.groups_cache.npz

# Cache files are automatically named: <dataset_name>.g<gridsize>k<overlap_factor>.groups_cache.npz
```

### Common Troubleshooting Commands
```bash
# Verify environment setup
ptycho_train --help

# Test with minimal data for debugging
ptycho_train --train_data_file dataset.npz --n_images 10 --nepochs 1 --output_dir debug_test

# Check data format and shapes
python -c "
import numpy as np
data = np.load('dataset.npz')
for key in ['probeGuess', 'objectGuess', 'diffraction', 'xcoords', 'ycoords']:
    if key in data:
        print(f'{key}: {data[key].shape} {data[key].dtype}')
    else:
        print(f'{key}: MISSING')
"
```

---

## Configuration File Examples

### Basic Training Configuration
```yaml
# save as configs/basic_training.yaml
model:
  N: 64
  gridsize: 1
  model_type: pinn

data:
  train_data_file: "datasets/fly/fly001_transposed.npz"
  n_images: 5000

training:
  batch_size: 16
  nepochs: 100
  output_dir: "config_training_run"
```

### Grouping-Aware Configuration
```yaml
# save as configs/grouping_aware.yaml
model:
  N: 64
  gridsize: 2
  model_type: pinn

data:
  train_data_file: "datasets/fly/fly001_transposed.npz"
  n_images: 1250  # Will create 1250 groups (5000 total patterns)

training:
  batch_size: 16
  nepochs: 100
  output_dir: "grouping_aware_run"
```

### Usage with Configuration Files
```bash
# Train using configuration file
ptycho_train --config configs/basic_training.yaml

# Override specific parameters
ptycho_train --config configs/basic_training.yaml --nepochs 50 --output_dir modified_run
```

---

## Quick Reference Summary

### Most Common Commands
```bash
# Standard training workflow
ptycho_train --train_data_file dataset.npz --n_images 2000 --nepochs 50 --output_dir my_training
ptycho_inference --model_path my_training/ --test_data dataset.npz --n_images 500 --output_dir my_inference

# Model comparison
python scripts/compare_models.py --pinn_dir pinn_run/ --baseline_dir baseline_run/ --test_data dataset.npz --output_dir comparison/

# Generalization study
./scripts/studies/run_complete_generalization_study.sh --train-data dataset.npz --test-data dataset.npz --output-dir study/ --train-sizes "512,1024,2048"
```

### Key Parameter Guidelines
- **Always specify `--output_dir`** to avoid overwriting results
- **Match `--gridsize`** between training and inference  
- **Use `--n_images`** values that make sense for your `gridsize`
- **Check log messages** to verify parameter interpretation
- **Monitor cache creation** for `gridsize > 1` on first run

For detailed explanations and troubleshooting, see the full documentation in the [Developer Guide](DEVELOPER_GUIDE.md) and [Tool Selection Guide](TOOL_SELECTION_GUIDE.md).