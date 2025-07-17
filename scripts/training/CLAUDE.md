# Training Workflow Agent Guide

## Quick Context
- **Primary Tool**: `ptycho_train` command (modern system)
- **Legacy Warning**: Never use `ptycho/train.py` - deprecated
- **Configuration**: Use YAML configs or command-line args
- **Data Format**: Critical - verify before training

## Essential Commands

### Basic Training
```bash
# Using configuration file (recommended)
ptycho_train --config configs/fly_config.yaml

# Direct parameter specification
ptycho_train --train_data_file <path/to/train.npz> --test_data_file <path/to/test.npz> --output_dir <output_path> --n_images 5000

# Verification test (proves environment works)
ptycho_train --train_data_file datasets/fly/fly001_transposed.npz --n_images 512 --output_dir verification_run
```

### Parameter Override
```bash
# Override specific parameters while using config
ptycho_train --config configs/fly_config.yaml --n_images 1000 --nepochs 100
```

## Common Patterns

### Quick Training Session
```bash
# 1. Start with verification
ptycho_train --train_data_file datasets/fly/fly001_transposed.npz --n_images 512 --output_dir test_run

# 2. Check output structure
ls test_run/  # Should contain: wts.h5.zip, history.dill, params.dill

# 3. Scale up if successful  
ptycho_train --config configs/my_experiment.yaml
```

### Configuration Templates
```yaml
# Basic PINN config
model:
  N: 64
  model_type: pinn
  object_big: true

training:
  nepochs: 50
  batch_size: 16
  output_dir: "my_training_run"

data:
  train_data_file: "datasets/fly/fly001_transposed.npz"
  n_images: 2000

physics:
  nphotons: 1000.0
  nll_weight: 1.0
  probe_trainable: true
```

### Memory Management
| Parameter | Memory Impact | Recommendation |
|-----------|---------------|----------------|
| `batch_size` | High | Use 16 (default), reduce to 8 if OOM |
| `n_images` | High | Start with 512 for testing |
| `N` | Very High | Use 64 for development, 128+ for production |

## Output Structure

After successful training, `output_dir/` contains:
- **`wts.h5.zip`** - Primary model output (load with `ModelManager.load_multiple_models()`)
- **`history.dill`** - Training history for loss plots
- **`params.dill`** - Configuration snapshot for reproducibility
- **`reconstructed_*.png`** - Visualization of final reconstruction
- **`metrics.csv`** - Quantitative metrics if ground truth available

## Troubleshooting

### TensorFlow Errors
**Problem**: Low-level TensorFlow shape/type errors  
**Solution**: Verify data format first - check data contracts  
**Command**: `python scripts/tools/visualize_dataset.py <dataset.npz>`

### Out of Memory
**Problem**: GPU/CPU memory exhausted  
**Solutions**:
- Reduce `batch_size` (16 → 8 → 4)
- Reduce `n_images` for testing
- Use smaller `N` value (128 → 64)

### Poor Performance
**Problem**: Low reconstruction quality  
**Solutions**:
- Check `nll_weight: 1.0` for PINN models
- Enable `probe_trainable: true`
- Increase `nepochs` (50 → 100)
- Verify data preprocessing consistency

### Configuration Errors
**Problem**: Parameter validation failures  
**Solutions**:
- Use only supported `N` values: [64, 128, 256]
- Ensure `batch_size` is power of 2
- Check file paths are absolute and exist

## Model Types

### PINN (Physics-Informed) - Default
```yaml
model:
  model_type: pinn
physics:
  nll_weight: 1.0  # Poisson loss
  mae_weight: 0.0  # No MAE loss
```

### Supervised Baseline
```yaml
model:
  model_type: supervised
physics:
  nll_weight: 0.0  # No Poisson loss
  mae_weight: 1.0  # MAE loss
```

## Best Practices

1. **Always verify data format first** - Most errors stem from incorrect data
2. **Start small** - Use `n_images: 512` for initial testing
3. **Use configurations** - YAML files for reproducibility
4. **Check outputs** - Verify `wts.h5.zip` exists before considering success
5. **Monitor memory** - Watch for OOM errors, adjust batch_size accordingly

## Cross-References

- **Configuration details**: <doc-ref type="guide">docs/CONFIGURATION_GUIDE.md</doc-ref>
- **Data format specs**: <doc-ref type="contract">docs/data_contracts.md</doc-ref>
- **Tool selection**: <doc-ref type="guide">docs/TOOL_SELECTION_GUIDE.md</doc-ref>
- **Model comparison**: <doc-ref type="workflow-guide">scripts/studies/CLAUDE.md</doc-ref>
- **Detailed documentation**: <doc-ref type="workflow-guide">scripts/training/README.md</doc-ref>