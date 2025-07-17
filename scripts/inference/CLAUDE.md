# Inference Workflow Agent Guide

## Quick Context
- **Primary Tool**: `ptycho_inference` command
- **Input**: Trained model directory + test data
- **Output**: Reconstruction comparison images, metrics
- **Purpose**: Evaluate trained models on test datasets

## Essential Commands

### Basic Inference
```bash
# Standard inference run
ptycho_inference --model_path <model_dir> --test_data <test.npz> --output_dir <output>

# Example with specific paths
ptycho_inference --model_path training_outputs/my_model --test_data datasets/fly/test.npz --output_dir inference_results
```

### Advanced Options
```bash
# With probe visualization
ptycho_inference --model_path <model_dir> --test_data <test.npz> --output_dir <output> --visualize_probe

# Specify number of test images
ptycho_inference --model_path <model_dir> --test_data <test.npz> --output_dir <output> --n_test_images 100
```

## Input Requirements

### Model Directory Structure
Required files in `model_path/`:
- **`wts.h5.zip`** - Primary model file (from training)
- **`params.dill`** - Configuration used during training

### Test Data Format
- Must follow data contracts: `probeGuess`, `objectGuess`, `diffraction`, `xcoords`, `ycoords`
- **Critical**: Test data format must match training data format
- Use same preprocessing pipeline as training data

## Output Structure

After successful inference, `output_dir/` contains:
- **`reconstruction_comparison.png`** - Side-by-side visual comparison
- **`reconstruction_amplitude.png`** - Reconstructed amplitude
- **`reconstruction_phase.png`** - Reconstructed phase  
- **`metrics.txt`** - Quantitative evaluation metrics
- **`reconstructed_object.npz`** - Full reconstruction data

## Common Patterns

### Quick Evaluation
```bash
# 1. Train a model
ptycho_train --config configs/my_config.yaml

# 2. Run inference on test set
ptycho_inference --model_path my_training_run --test_data test_dataset.npz --output_dir eval_results

# 3. Check reconstruction quality
open eval_results/reconstruction_comparison.png
```

### Batch Evaluation
```bash
# Evaluate multiple trained models
for model in training_outputs/*/; do
    ptycho_inference --model_path "$model" --test_data common_test.npz --output_dir "eval_$(basename $model)"
done
```

## Troubleshooting

### Model Loading Errors
**Problem**: Cannot load model from directory  
**Solutions**:
- Verify `wts.h5.zip` exists in model directory
- Check `params.dill` exists in model directory
- Ensure model directory path is correct

### Shape/Format Errors
**Problem**: Test data incompatible with model  
**Solutions**:
- Verify test data has same `N` parameter as training
- Check test data follows data contracts
- Ensure same preprocessing as training data
- Use `python scripts/tools/visualize_dataset.py test.npz` to verify format

### Poor Reconstruction Quality
**Problem**: Low-quality or failed reconstruction  
**Diagnosis**:
- Check training convergence in `training_dir/history.dill`
- Verify test data quality and preprocessing
- Compare with known-good test datasets

### Missing Ground Truth
**Problem**: No quantitative metrics generated  
**Solutions**:
- Ensure test data contains `objectGuess` for ground truth
- Verify ground truth has correct shape and format
- Check that ground truth corresponds to test scan positions

## Evaluation Metrics

When ground truth available:
- **MAE**: Mean Absolute Error (lower = better)
- **MSE**: Mean Squared Error (lower = better)  
- **PSNR**: Peak Signal-to-Noise Ratio (higher = better)
- **FRC**: Fourier Ring Correlation (resolution metric)

## Model Comparison Workflow

For systematic model comparison:
```bash
# Use comparison script instead
python scripts/compare_models.py \
    --pinn_dir <pinn_model> \
    --baseline_dir <baseline_model> \
    --test_data <test.npz> \
    --output_dir comparison_results
```

## Cross-References

- **Training workflow**: <doc-ref type="workflow-guide">scripts/training/CLAUDE.md</doc-ref>
- **Model comparison**: <doc-ref type="workflow-guide">scripts/studies/CLAUDE.md</doc-ref>
- **Data format specs**: <doc-ref type="contract">docs/data_contracts.md</doc-ref>
- **Detailed documentation**: <doc-ref type="workflow-guide">scripts/inference/README.md</doc-ref>