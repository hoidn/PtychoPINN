# Configuration Guide

This guide provides comprehensive information about configuring PtychoPINN models, training parameters, and experimental settings.

## Overview

PtychoPINN uses a modern dataclass-based configuration system defined in `<code-ref type="config">ptycho/config/config.py</code-ref>`. Parameters are controlled via YAML files (see `configs/`) or command-line arguments.

## Configuration Architecture

- **Modern System**: Uses `dataclasses` (`ModelConfig`, `TrainingConfig`)
- **Legacy Compatibility**: A legacy `params.cfg` dictionary is maintained for backward compatibility
- **One-way Flow**: Modern configs update legacy params at workflow start
- **Source of Truth**: `<code-ref type="config">ptycho/config/config.py</code-ref>` contains all configuration definitions

## Model Architecture Parameters

| Parameter           | Type                            | Description                                                                                             |
| ------------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `N`                 | `Literal[64, 128, 256]`         | The dimension of the input diffraction patterns (e.g., 64x64 pixels). Critical for network shape.         |
| `n_filters_scale`   | `int`                           | A multiplier for the number of filters in convolutional layers. `> 0`. Default: `2`.                      |
| `model_type`        | `Literal['pinn', 'supervised']` | The type of model to use. `pinn` is the main physics-informed model.                                    |
| `amp_activation`    | `str`                           | The activation function for the amplitude output layer (e.g., 'sigmoid', 'relu').                       |
| `object_big`        | `bool`                          | If `true`, the model reconstructs a large area by stitching patches. If `false`, it reconstructs a single NxN patch. |
| `probe_big`         | `bool`                          | If `true`, the probe representation can vary across the solution region.                                  |
| `probe_mask`        | `bool`                          | If `true`, applies a circular mask to the probe to enforce a finite support.                            |
| `gaussian_smoothing_sigma` | `float` | Standard deviation for the Gaussian filter applied to the probe. `0.0` means no smoothing. |

## Training Parameters

| Parameter      | Type   | Description                                                              |
| -------------- | ------ | ------------------------------------------------------------------------ |
| `nepochs`      | `int`  | Number of training epochs. `> 0`. Default: `50`.                         |
| `batch_size`   | `int`  | The number of samples per batch. Must be a power of 2. Default: `16`.    |
| `output_dir`   | `Path` | The directory where training outputs (model, logs, images) will be saved. |

## Data & Simulation Parameters

| Parameter           | Type          | Description                                                                                                    |
| ------------------- | ------------- | -------------------------------------------------------------------------------------------------------------- |
| `train_data_file`   | `Path`        | **Required.** Path to the training dataset (`.npz` file).                                                       |
| `test_data_file`    | `Optional[Path]`| Path to the test dataset (`.npz` file).                                                                        |
| `n_images`          | `int`         | The number of diffraction patterns to use from the dataset. Default: `512`.                                    |
| `gridsize`          | `int`         | For PINN-style models, this defines the number of neighboring patches to process together (e.g., 1 for single-patch processing, 2 for 2x2 neighbors). For supervised models, it defines the input channel depth. |

### Critical Parameter Interactions

> **⚠️ Important: `n_images` × `gridsize` Interaction**
>
> When using `gridsize > 1`, the `n_images` parameter creates spatially biased training data because subsampling occurs **before** nearest-neighbor grouping:
> 
> - **Sequential subsampling**: Training uses only the first N images from a small spatial region
> - **Broken neighbor relationships**: Random subsampling destroys physical adjacency required for overlap constraints
> - **Recommendation**: For `gridsize > 1`, prepare complete smaller datasets rather than using `n_images` parameter
>
> See the Developer Guide for detailed architectural explanation: <doc-ref type="guide">docs/DEVELOPER_GUIDE.md</doc-ref>

## Physics & Loss Parameters

| Parameter                   | Type    | Description                                                                                                     |
| --------------------------- | ------- | --------------------------------------------------------------------------------------------------------------- |
| `nphotons`                  | `float` | The target average number of photons per diffraction pattern, used for the Poisson noise model. `> 0`.             |
| `nll_weight`                | `float` | Weight for the Negative Log-Likelihood (Poisson) loss. Recommended: `1.0`. Range: `[0, 1]`.                      |
| `mae_weight`                | `float` | Weight for the Mean Absolute Error loss in diffraction space. Typically `0.0`. Range: `[0, 1]`.                  |
| `probe_scale`               | `float` | A normalization factor for the probe's amplitude. `> 0`.                                                        |
| `probe_trainable`           | `bool`  | If `true`, allows the model to learn and update the probe function during training.                               |
| `intensity_scale_trainable` | `bool`  | If `true`, allows the model to learn the global intensity scaling factor.                                       |

## Configuration Methods

### Using YAML Files (Recommended)

Create configuration files in the `configs/` directory:

```yaml
# configs/my_experiment.yaml
model:
  N: 64
  model_type: pinn
  object_big: true
  probe_big: false
  n_filters_scale: 2

training:
  nepochs: 100
  batch_size: 16
  output_dir: "my_experiment_output"

data:
  train_data_file: "datasets/fly/fly001_transposed.npz"
  test_data_file: "datasets/fly/fly001_transposed.npz"
  n_images: 2000

physics:
  nphotons: 1000.0
  nll_weight: 1.0
  mae_weight: 0.0
  probe_trainable: true
```

Use the configuration:
```bash
ptycho_train --config configs/my_experiment.yaml
```

### Command-Line Parameters

Override specific parameters:
```bash
ptycho_train --config configs/base_config.yaml --nepochs 200 --n_images 1000
```

Direct parameter specification:
```bash
ptycho_train \
    --train_data_file datasets/fly/fly001_transposed.npz \
    --test_data_file datasets/fly/fly001_transposed.npz \
    --n_images 5000 \
    --nepochs 100 \
    --batch_size 32 \
    --output_dir my_training_run
```

## Standard Configuration Examples

### Basic PINN Training
```yaml
model:
  N: 64
  model_type: pinn
  object_big: true
  probe_big: false

training:
  nepochs: 50
  batch_size: 16

data:
  train_data_file: "datasets/fly/fly001_transposed.npz"
  n_images: 512

physics:
  nphotons: 1000.0
  nll_weight: 1.0
  probe_trainable: true
```

### Supervised Baseline Training
```yaml
model:
  N: 64
  model_type: supervised
  object_big: true
  probe_big: false

training:
  nepochs: 50
  batch_size: 16

data:
  train_data_file: "datasets/fly/fly001_transposed.npz"
  n_images: 512

physics:
  nphotons: 1000.0
  nll_weight: 0.0
  mae_weight: 1.0
  probe_trainable: false
```

### High-Resolution Configuration
```yaml
model:
  N: 128
  model_type: pinn
  object_big: true
  probe_big: true
  n_filters_scale: 3

training:
  nepochs: 100
  batch_size: 8  # Smaller batch for memory

data:
  train_data_file: "datasets/high_res/data.npz"
  n_images: 1000

physics:
  nphotons: 2000.0
  nll_weight: 1.0
  probe_trainable: true
  intensity_scale_trainable: true
```

### Quick Test Configuration
```yaml
model:
  N: 64
  model_type: pinn
  object_big: false  # Single patch for speed

training:
  nepochs: 10
  batch_size: 16

data:
  train_data_file: "datasets/fly/fly001_transposed.npz"
  n_images: 100

physics:
  nphotons: 1000.0
  nll_weight: 1.0
```

## Configuration Best Practices

### File Organization
- Store configurations in `configs/` directory
- Use descriptive names: `fly_pinn_standard.yaml`, `baseline_comparison.yaml`
- Keep a `default.yaml` for common settings

### Parameter Selection
- **Start with defaults**: Use proven configurations as starting points
- **Incremental changes**: Modify one parameter at a time for systematic studies
- **Document changes**: Include comments in YAML files explaining parameter choices

### Memory Management
- **Batch size**: Must be power of 2; reduce for high-resolution or limited memory
- **N parameter**: Larger values require exponentially more memory
- **n_images**: Control dataset size for memory constraints

### Performance Optimization
- **n_filters_scale**: Higher values improve capacity but slow training
- **nepochs**: Balance convergence with training time
- **gridsize**: Affects patch processing efficiency

## Troubleshooting Configuration Issues

### Common Parameter Errors

**Invalid N value:**
```
Error: N must be one of [64, 128, 256]
```
Solution: Use only supported diffraction pattern sizes.

**Batch size not power of 2:**
```
Error: batch_size must be a power of 2
```
Solution: Use 8, 16, 32, 64, etc.

**Missing required files:**
```
Error: train_data_file not found
```
Solution: Verify file paths are correct and files exist.

### Memory Issues
- Reduce `batch_size` if out of memory
- Reduce `n_images` for large datasets
- Consider using `N=64` instead of higher resolutions

### Training Stability
- Use `nll_weight=1.0` for PINN models
- Set `probe_trainable=true` for better reconstruction
- Start with `nepochs=50` and increase if needed

## Advanced Configuration

### Legacy Parameter Access
New code should avoid this, but legacy modules use:
```python
from ptycho.params import params
value = params.get('parameter_name')
```

### Programmatic Configuration
```python
from ptycho.config.config import TrainingConfig, ModelConfig

config = TrainingConfig(
    model=ModelConfig(N=64, model_type='pinn'),
    training=TrainingParams(nepochs=100, batch_size=16),
    # ... other parameters
)
```

### Environment Variables
Some parameters can be set via environment variables:
```bash
export PTYCHO_OUTPUT_DIR="my_experiments"
export PTYCHO_BATCH_SIZE=32
```

## Configuration Validation

The system validates configurations at startup:
- Type checking for all parameters
- Range validation for numeric values
- File existence checks for data paths
- Compatibility checks between parameters

Error messages provide specific guidance for fixing invalid configurations.

## Migration from Legacy Configuration

If you have old configuration files:
1. Convert `params.cfg` entries to YAML format
2. Update parameter names to match new schema
3. Use dataclass field names instead of legacy keys
4. Test with small datasets first