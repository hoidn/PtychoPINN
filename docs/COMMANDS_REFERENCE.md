# PtychoPINN Commands Reference

Quick reference for essential PtychoPINN workflows.

## 📋 Quick Navigation
- [Data Preparation](#data-preparation)
- [Training](#training) 
- [Inference](#inference)
- [Model Comparison](#model-comparison)
- [Studies](#studies)
- [Parameter Guide](#parameter-guide)

---

## Data Preparation

```bash
# Convert raw experimental data (REQUIRED FIRST STEP)
python scripts/tools/transpose_rename_convert_tool.py raw_data.npz converted_data.npz

# Shuffle dataset for randomized sampling
python scripts/tools/shuffle_dataset_tool.py input.npz output.npz

# Split into train/test
python scripts/tools/split_dataset_tool.py dataset.npz output_dir/ --split-fraction 0.8

# Visualize dataset
python scripts/tools/visualize_dataset.py dataset.npz output.png
```

---

## Training

```bash
# Basic training
ptycho_train --train_data_file dataset.npz --n_images 2000 --nepochs 50 --output_dir my_run

# With configuration file
ptycho_train --config configs/my_config.yaml

# GridSize=2 training (grouping-aware subsampling)
ptycho_train --train_data_file dataset.npz --n_images 500 --gridsize 2 --nepochs 50 --output_dir gs2_run
```

### Parameter Interpretation
- **GridSize=1**: `--n_images` = individual diffraction patterns
- **GridSize=2**: `--n_images` = neighbor groups (each group = 4 patterns)

| GridSize | n_images | Total Patterns | Example |
|----------|----------|----------------|---------|
| 1 | 1000 | 1000 | Individual images |
| 2 | 250 | 1000 | 250 groups × 4 patterns |

---

## Inference

```bash
# Basic inference
ptycho_inference --model_path trained_model/ --test_data test.npz --output_dir inference_out

# With specific number of test patterns
ptycho_inference --model_path trained_model/ --test_data test.npz --n_images 500 --output_dir inference_out

# GridSize=2 inference (must match training gridsize)
ptycho_inference --model_path gs2_model/ --test_data test.npz --n_images 125 --gridsize 2 --output_dir gs2_inference
```

---

## Model Comparison

```bash
# Compare two trained models
python scripts/compare_models.py \
    --pinn_dir pinn_model/ \
    --baseline_dir baseline_model/ \
    --test_data test.npz \
    --output_dir comparison_out

# Complete training + comparison workflow
./scripts/run_comparison.sh train.npz test.npz output_dir

# With specific training/test sizes
./scripts/run_comparison.sh train.npz test.npz output_dir --n-train-images 2000 --n-test-images 500
```

---

## Studies

```bash
# Generalization study
./scripts/studies/run_complete_generalization_study.sh \
    --train-data dataset.npz \
    --test-data dataset.npz \
    --output-dir study_results \
    --train-sizes "512 1024 2048" \
    --test-size 512

# Multi-trial study
./scripts/studies/run_complete_generalization_study.sh \
    --train-data dataset.npz \
    --test-data dataset.npz \
    --output-dir multi_trial_study \
    --train-sizes "512 1024" \
    --num-trials 3

# Plot results
python scripts/studies/aggregate_and_plot_results.py study_results --output plots/
```

---

## Parameter Guide

### Common Configurations
```bash
# Quick test (minimal epochs)
ptycho_train --train_data_file dataset.npz --n_images 100 --nepochs 1 --output_dir test_run

# Full training
ptycho_train --train_data_file dataset.npz --n_images 5000 --nepochs 100 --output_dir full_run

# Production training with validation
ptycho_train --train_data_file train.npz --test_data_file test.npz --n_images 5000 --nepochs 100 --output_dir production_run
```

### Key Guidelines
- **Always specify `--output_dir`** to avoid overwriting results
- **Match `--gridsize`** between training and inference
- **Use reasonable `--n_images`** values for your dataset size
- **Monitor training logs** for parameter interpretation messages

### Log Message Examples
```
# GridSize=1
INFO - Parameter interpretation: --n-images=1000 refers to individual images (gridsize=1)

# GridSize=2  
INFO - Parameter interpretation: --n-images=250 refers to neighbor groups (gridsize=2, total patterns=1000)
INFO - Using grouping-aware subsampling strategy for gridsize=2
```

---

## Quick Troubleshooting

```bash
# Check dataset format
python -c "import numpy as np; data=np.load('dataset.npz'); print(list(data.keys())); print({k: data[k].shape for k in data.keys()})"

# Verify environment
ptycho_train --help

# Monitor training progress
tail -f output_dir/train_debug.log

# Check GPU usage
nvidia-smi
```

For detailed explanations, see the [Developer Guide](DEVELOPER_GUIDE.md) and [Tool Selection Guide](TOOL_SELECTION_GUIDE.md).