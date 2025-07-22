# Tool Selection Guide

This guide helps you choose the right tools and scripts for different PtychoPINN workflows. Understanding which tool to use is critical for efficient development and avoiding common mistakes.

**📚 Companion Guide:** For specific command syntax and examples, see the **<doc-ref type="guide">docs/COMMANDS_REFERENCE.md</doc-ref>**

## Workflow Tool Hierarchy

### For comprehensive model evaluation studies:
- **Primary**: `scripts/studies/run_complete_generalization_study.sh` 
- **Controls**: Training sizes, number of trials, synthetic/experimental datasets, full pipeline
- **Use when**: Running complete studies, need statistical robustness, comparing synthetic vs experimental data
- **Modes**: Synthetic data generation (default) or experimental data analysis (`--skip-data-prep`)

### For comparing existing trained models:
- **Primary**: `scripts/compare_models.py`
- **Controls**: Post-training comparison only, uses full test dataset
- **Use when**: Models already exist, quick comparisons, metric calculation

### For visualization of study results:
- **Primary**: `scripts/studies/aggregate_and_plot_results.py`
- **Controls**: Plot generation, metric selection, statistical aggregation
- **Use when**: Generating publication plots, analyzing completed studies

## Common Tool Selection Mistakes

❌ **Wrong**: Using `compare_models.py` with training size parameters (not supported)
✅ **Correct**: Use `run_complete_generalization_study.sh --train-sizes "512 1024"`

❌ **Wrong**: Trying to control number of training images in comparison scripts
✅ **Correct**: Control training size in the generalization study script, then use comparison for analysis

❌ **Wrong**: Manual training then manual comparison for multiple sizes/trials
✅ **Correct**: Use the complete generalization study script for automated workflows

## Decision Matrix

| Need | Tool | Key Parameters |
|------|------|----------------|
| **Train models with specific sizes (synthetic data)** | `run_complete_generalization_study.sh` | `--train-sizes`, `--num-trials` |
| **Train models with experimental data** | `run_complete_generalization_study.sh` | `--train-data`, `--test-data`, `--skip-data-prep` |
| **Compare existing models** | `compare_models.py` | `--pinn_dir`, `--baseline_dir` |
| **Visualize study results** | `aggregate_and_plot_results.py` | `--metric`, `--part` |
| **Debug dataset issues** | `scripts/tools/visualize_dataset.py` | Input dataset path |
| **Prepare datasets** | `scripts/tools/split_dataset_tool.py` | `--split-fraction` |

## Training Workflows

### Single Model Training
```bash
# For single model training with specific parameters
ptycho_train --config configs/fly_config.yaml

# For direct parameter specification
ptycho_train --train_data_file <data.npz> --n_images 5000 --output_dir <output>
```

### Comparison Training
```bash
# Train both PtychoPINN and baseline models for comparison
./scripts/run_comparison.sh <train_data.npz> <test_data.npz> <output_dir>
```

### Generalization Studies

#### Synthetic Data Mode (Default)
```bash
# Run complete generalization study with auto-generated synthetic data
./scripts/studies/run_complete_generalization_study.sh \
    --train-sizes "512 1024 2048 4096" \
    --num-trials 3 \
    --output-dir synthetic_study
```

#### Experimental Data Mode
```bash
# Run generalization study with existing experimental datasets
./scripts/studies/run_complete_generalization_study.sh \
    --train-data "datasets/fly64/fly001_64_train_converted.npz" \
    --test-data "datasets/fly64/fly001_64_train_converted.npz" \
    --skip-data-prep \
    --train-sizes "512 1024 2048" \
    --num-trials 3 \
    --output-dir experimental_study
```

## Inference Workflows

### Single Model Inference
```bash
# Run inference on a test dataset
ptycho_inference --model_path <model_dir> --test_data <test.npz> --output_dir <output>
```

### Batch Inference
```bash
# Use study scripts for batch inference across multiple models
# (Part of generalization study workflow)
```

## Data Processing Workflows

### Dataset Visualization
```bash
# Visualize dataset contents and structure
python scripts/tools/visualize_dataset.py <dataset.npz>
```

### Dataset Preparation
```bash
# Split dataset into train/test portions
python scripts/tools/split_dataset_tool.py \
    --input <original.npz> \
    --output-train <train.npz> \
    --output-test <test.npz> \
    --split-fraction 0.8

# Downsample data while maintaining physical consistency
python scripts/tools/downsample_data_tool.py \
    --input <high_res.npz> \
    --output <low_res.npz> \
    --factor 2

# Prepare data with smoothing/apodization
python scripts/tools/prepare_data_tool.py \
    --input <raw.npz> \
    --output <prepared.npz> \
    --smooth-probe \
    --apodize-object
```

### Dataset Updates
```bash
# Update NPZ file with new reconstruction results
python scripts/tools/update_tool.py \
    --input <dataset.npz> \
    --output <updated.npz> \
    --reconstruction <reconstruction.npz>
```

## Simulation Workflows

### Basic Simulation
```bash
# Simulate data from existing object/probe
python scripts/simulation/simulate_and_save.py \
    --input-file <obj_probe.npz> \
    --output-file <sim_data.npz> \
    --n-images 2000 \
    --gridsize 1
```

### Complex Simulation
```bash
# High-level simulation with synthetic objects
python scripts/simulation/run_with_synthetic_lines.py \
    --output-dir <sim_output> \
    --n-images 2000
```

## Analysis Workflows

### Model Comparison
```bash
# Compare two trained models (see MODEL_COMPARISON_GUIDE.md for details)
python scripts/compare_models.py \
    --pinn_dir <pinn_model> \
    --baseline_dir <baseline_model> \
    --test_data <test.npz> \
    --output_dir <comparison_output>
```

### Study Result Analysis
```bash
# Aggregate and visualize study results
python scripts/studies/aggregate_and_plot_results.py \
    --study-dir <study_output> \
    --metric amplitude_ssim \
    --part amplitude \
    --output-dir <plot_output>
```

## Debugging Workflows

### Environment Verification
```bash
# Verify installation and environment
ptycho_train --train_data_file datasets/fly/fly001_transposed.npz \
    --n_images 512 \
    --output_dir verification_run
```

### Debug Image Generation
```bash
# Generate debug images for metric calculation verification
python scripts/compare_models.py \
    --pinn_dir <pinn_model> \
    --baseline_dir <baseline_model> \
    --test_data <test.npz> \
    --output_dir <debug_output> \
    --save-debug-images
```

### Test Suite
```bash
# Run unit tests
python -m unittest discover -s ptycho -p "test_*.py"
```

## Configuration Management

### Using YAML Configurations
```bash
# Preferred method for reproducible experiments
ptycho_train --config configs/fly_config.yaml
```

### Parameter Override
```bash
# Override specific parameters while using config file
ptycho_train --config configs/fly_config.yaml --n_images 1000 --nepochs 100
```

## Performance Considerations

### Memory Management
- Use `--n_images` parameter to control memory usage
- Consider dataset size when choosing batch sizes
- Monitor disk space for large study outputs

### Parallel Processing
- Generalization studies can run multiple trials in parallel
- Use `--num-trials` to control statistical robustness vs. computation time
- Consider cluster resources for large-scale studies

## Best Practices

1. **Start with verification**: Always run the verification workflow first
2. **Use configurations**: Prefer YAML configs over command-line parameters for reproducibility
3. **Incremental development**: Start with small datasets and short training runs
4. **Systematic studies**: Use the generalization study framework for rigorous evaluation
5. **Debug incrementally**: Use debug images and visualization tools to understand results

## Common Pitfalls to Avoid

1. **Mixing tool purposes**: Don't use comparison scripts for training parameter control
2. **Ignoring data format**: Always verify data contracts before processing
3. **Skipping verification**: Environment issues cause hard-to-debug errors later
4. **Manual workflows**: Use automated study scripts instead of manual repetition
5. **Inadequate testing**: Always verify results with known-good data first