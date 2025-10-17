# PtychoPINN Workflow Guide

This guide documents the complete workflows for using PtychoPINN, from data preparation through model evaluation and comparison.

## Core Workflow: Train → Evaluate → Compare

The typical PtychoPINN workflow consists of three main stages:

1. **Training**: Create and train models using `ptycho_train`
2. **Evaluation**: Assess single model performance using `ptycho_evaluate` 
3. **Comparison**: Compare multiple models using `compare_models.py`

## 🎯 Quick Start: Complete Workflow

```bash
# Prepare your data (if needed)
python scripts/tools/transpose_rename_convert_tool.py raw_data.npz converted_data.npz

# Train a model
ptycho_train --train_data_file converted_data.npz --output_dir my_model --nepochs 50

# Evaluate the trained model
ptycho_evaluate --model-dir my_model --test-data converted_data.npz --output-dir my_model_eval

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

### Use `ptycho_evaluate` when:
- Analyzing performance of a single trained model
- Computing detailed metrics against ground truth
- Creating publication-ready visualizations
- Debugging model performance issues
- You need comprehensive quantitative analysis

### Use `compare_models.py` when:
- Comparing multiple models head-to-head
- Benchmarking PtychoPINN vs Baseline vs Tike
- Running systematic model comparisons
- You want side-by-side performance analysis

### Use `ptycho_inference` when:
- Applying trained models to new datasets without ground truth
- Processing production data
- You just need reconstructions without evaluation metrics

## 🔄 Common Workflow Patterns

### Pattern 1: Model Development Workflow

```bash
# 1. Train initial model
ptycho_train --train_data_file datasets/train.npz --output_dir model_v1 --nepochs 25

# 2. Quick evaluation to check performance
ptycho_evaluate --model-dir model_v1 --test-data datasets/test.npz --output-dir model_v1_eval

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

# 2. Comprehensive evaluation
ptycho_evaluate --model-dir production_model \
    --test-data datasets/validation_set.npz \
    --output-dir production_evaluation \
    --save-individual-images

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

### Evaluation Output (`ptycho_evaluate`)
```
my_model_eval/
├── logs/
│   └── debug.log           # Evaluation log
├── results.csv            # Quantitative metrics (MAE, PSNR, SSIM, etc.)
├── reconstruction_comparison.png  # Visual comparison plot
├── error_analysis.png     # Error maps and histograms
└── individual_images/     # Individual reconstruction plots (if requested)
    ├── amplitude.png
    ├── phase.png
    └── error_maps.png
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
    
    ptycho_evaluate --model-dir "trial_${trial}" \
        --test-data datasets/test.npz \
        --output-dir "trial_${trial}_eval"
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

# Train and evaluate on each fold
for fold in {1..5}; do
    ptycho_train --train_data_file "fold_${fold}/train.npz" \
        --test_data_file "fold_${fold}/test.npz" \
        --output_dir "model_fold_${fold}"
    
    ptycho_evaluate --model-dir "model_fold_${fold}" \
        --test-data "fold_${fold}/test.npz" \
        --output-dir "eval_fold_${fold}"
done
```

## 🛠️ Troubleshooting Common Issues

### Issue: "Model directory not found"
**Solution**: Ensure you're using the correct path to the training output directory containing `wts.h5.zip`

### Issue: "No ground truth found for evaluation"
**Solution**: Verify your test data contains `objectGuess` key for ground truth comparison

### Issue: "Memory errors during evaluation"
**Solution**: Use sampling parameters to reduce memory usage:
```bash
ptycho_evaluate --model-dir model/ --test-data test.npz --n-test-subsample 1000 --n-test-groups 500 --output-dir eval/
```

### Issue: "Registration artifacts in comparison"
**Solution**: Try different alignment methods or skip registration for debugging:
```bash
ptycho_evaluate --model-dir model/ --test-data test.npz --skip-registration --output-dir debug_eval/
```

## 📚 Related Documentation

- **Command Reference**: [docs/COMMANDS_REFERENCE.md](COMMANDS_REFERENCE.md) - Complete command-line options
- **Tool Selection**: [docs/TOOL_SELECTION_GUIDE.md](TOOL_SELECTION_GUIDE.md) - Choosing the right tool
- **Configuration**: [docs/CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md) - Parameter tuning
- **Developer Guide**: [docs/DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) - Architecture and internals