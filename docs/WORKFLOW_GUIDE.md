# PtychoPINN Workflow Guide

This guide documents the complete workflows for using PtychoPINN, from data preparation through inference and model comparison.

## Core Workflow: Train → Infer → Compare

The typical PtychoPINN workflow consists of three main stages:

1. **Training**: Create and train models using `ptycho_train`
2. **Inference**: Run a trained model using `ptycho_inference`
3. **Comparison**: Compare multiple models using `compare_models.py`

## 🎯 Quick Start: Complete Workflow

```bash
# Prepare your data (if needed)
python scripts/tools/transpose_rename_convert_tool.py raw_data.npz converted_data.npz

# Train a model
ptycho_train --train_data_file converted_data.npz --output_dir my_model --nepochs 50

# Run inference with the trained model
ptycho_inference --model_path my_model --test_data converted_data.npz --output_dir my_model_infer

# (Optional) Compare multiple models
python scripts/compare_models.py \
    --pinn_dir my_model \
    --baseline_dir other_model \
    --test_data converted_data.npz \
    --output_dir model_comparison
```

## 📋 When to Use Each Tool

### Use `ptycho_train` when:
- Creating new trained models
- Experimenting with different architectures or parameters
- Training on new datasets

### Use `ptycho_inference` when:
- Running a single trained model to produce reconstructions
- Performing backend-specific smoke checks
- Creating quick per-model visuals
- Debugging model loading/inference behavior

### Use `compare_models.py` when:
- Comparing multiple models head-to-head
- Benchmarking PtychoPINN vs Baseline vs Tike
- Running systematic model comparisons
- You want side-by-side performance analysis

## 🔄 Common Workflow Patterns

### Pattern 1: Model Development Workflow

```bash
# 1. Train initial model
ptycho_train --train_data_file datasets/train.npz --output_dir model_v1 --nepochs 25

# 2. Quick inference check
ptycho_inference --model_path model_v1 --test_data datasets/test.npz --output_dir model_v1_infer

# 3. Iterate with different parameters
ptycho_train --config configs/improved_config.yaml --output_dir model_v2 --nepochs 50

# 4. Compare versions
python scripts/compare_models.py \
    --pinn_dir model_v2 \
    --baseline_dir model_v1 \
    --test_data datasets/test.npz \
    --output_dir v1_vs_v2_comparison
```

### Pattern 2: Systematic Study Workflow

```bash
# 1. Generate synthetic datasets
bash scripts/prepare.sh --input-file reconstruction.npz --output-dir study_data --sim-images 10000

# 2. Run complete generalization study
./scripts/studies/run_complete_generalization_study.sh \
    --train-group-sizes "512 1024 2048" \
    --num-trials 3 \
    --output-dir generalization_study

# 3. Analyze results
python scripts/studies/aggregate_and_plot_results.py generalization_study --output plots/
```

### Pattern 3: Production Analysis Workflow

```bash
# 1. Train production model
ptycho_train --config configs/production_config.yaml --output_dir production_model

# 2. Inference sanity check on validation data
ptycho_inference --model_path production_model \
    --test_data datasets/validation_set.npz \
    --output_dir production_inference \
    --comparison_plot

# 3. Apply to new data
ptycho_inference --model_path production_model \
    --test_data datasets/new_experiment.npz \
    --output_dir production_results
```

## 📊 Understanding Output Structures

### Training Output (`ptycho_train`)
```
my_model/
├── logs/
│   └── debug.log           # Complete training log
├── wts.h5.zip             # Trained model weights
├── history.dill           # Training history
├── metrics.csv            # Training metrics (if ground truth available)
└── params.dill            # Configuration snapshot
```

### Inference Output (`ptycho_inference`)
```
my_model_infer/
├── logs/
│   └── debug.log           # Inference log
├── reconstructed_amplitude.png
├── reconstructed_phase.png
└── reconstruction_comparison.png  # Optional (with --comparison_plot when GT exists)
```

### Comparison Output (`compare_models.py`)
```
model_comparison/
├── logs/
│   └── debug.log           # Comparison log
├── comparison_results.csv  # Side-by-side metrics
├── side_by_side_comparison.png  # Visual comparison
├── metrics_comparison.png  # Metric bar charts
└── unified_results.npz     # All reconstructions for further analysis
```

## ⚡ Advanced Workflows

### Multi-Trial Statistical Analysis

```bash
# Run multiple trials for statistical robustness
for trial in {1..5}; do
    ptycho_train --train_data_file datasets/train.npz \
        --output_dir "trial_${trial}" \
        --subsample_seed $((42 + trial))
    
    ptycho_inference --model_path "trial_${trial}" \
        --test_data datasets/test.npz \
        --output_dir "trial_${trial}_infer"
done

# Aggregate results across trials
python scripts/studies/aggregate_and_plot_results.py . --output multi_trial_analysis/
```

### Cross-Validation Workflow

```bash
# Create multiple train/test splits
for fold in {1..5}; do
    python scripts/tools/split_dataset_tool.py \
        datasets/full_dataset.npz \
        "fold_${fold}/" \
        --split-fraction 0.8 \
        --seed $((100 + fold))
done

# Train and run inference on each fold
for fold in {1..5}; do
    ptycho_train --train_data_file "fold_${fold}/train.npz" \
        --test_data_file "fold_${fold}/test.npz" \
        --output_dir "model_fold_${fold}"
    
    ptycho_inference --model_path "model_fold_${fold}" \
        --test_data "fold_${fold}/test.npz" \
        --output_dir "infer_fold_${fold}"
done
```

## 🛠️ Troubleshooting Common Issues

### Issue: "Model directory not found"
**Solution**: Ensure you're using the correct path to the training output directory containing `wts.h5.zip`

### Issue: "No ground truth found for comparison plot"
**Solution**: This is expected for many production datasets. Omit `--comparison_plot` or provide data with `objectGuess`.

### Issue: "Memory errors during inference"
**Solution**: Use sampling parameters to reduce memory usage:
```bash
ptycho_inference --model_path model/ --test_data test.npz \
    --n_subsample 1000 --n_images 500 --output_dir infer/
```

### Issue: "Registration artifacts in comparison"
**Solution**: Try different alignment methods or skip registration for debugging:
```bash
python scripts/compare_models.py \
    --pinn_dir model/ \
    --baseline_dir baseline_model/ \
    --test_data test.npz \
    --skip-registration \
    --output_dir debug_compare/
```

## 📚 Related Documentation

- **Command Reference**: [docs/COMMANDS_REFERENCE.md](COMMANDS_REFERENCE.md) - Complete command-line options
- **Model Comparison Guide**: [docs/MODEL_COMPARISON_GUIDE.md](MODEL_COMPARISON_GUIDE.md) - Choosing comparison workflows
- **Configuration**: [docs/CONFIGURATION.md](CONFIGURATION.md) - Parameter tuning
- **Developer Guide**: [docs/DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) - Architecture and internals
