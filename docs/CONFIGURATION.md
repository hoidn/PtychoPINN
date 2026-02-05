# PtychoPINN Configuration Guide

This document is the canonical reference for all configuration parameters used in the PtychoPINN project. It details the modern dataclass-based configuration system and provides a comprehensive reference for all available parameters.

## The Configuration System

The project uses a modern, robust configuration system based on Python dataclasses, defined in `ptycho/config/config.py`. This provides type safety, default values, and clear structure.

There are three main configuration classes:
- **ModelConfig**: Defines the core model architecture.
- **TrainingConfig**: Defines parameters for the training process.
- **InferenceConfig**: Defines parameters for running inference.

### Legacy Compatibility

For backward compatibility, a legacy global dictionary `ptycho.params.cfg` still exists. The modern dataclass configuration is the single source of truth. At the start of any workflow, the `TrainingConfig` or `InferenceConfig` object is used to populate the legacy `params.cfg` dictionary.

This is a one-way data flow: **dataclass → legacy dict**. New code should always accept a configuration dataclass as an argument and should not rely on the global `params` object.

### Backends and Config Bridging

PtychoPINN uses the same canonical configuration dataclasses for both TensorFlow and PyTorch backends. When operating with the PyTorch stack, configs from `ptycho_torch/config_params.py` are translated to the TensorFlow dataclasses via the bridge adapter and then flowed into the legacy `params.cfg`:

```
PyTorch config_params → ptycho_torch/config_bridge.py → TF dataclasses → update_legacy_dict(params.cfg, config)
```

- See the normative mapping: <doc-ref type="spec">docs/specs/spec-ptycho-config-bridge.md</doc-ref>
- Critical rule (CONFIG‑001): always call `update_legacy_dict(params.cfg, config)` before data loading or legacy module usage.

## Usage

You can configure a run in two ways, with the following order of precedence:

1. **Command-Line Arguments** (Highest Priority): Any parameter can be overridden from the command line (e.g., `--nepochs 100`).
2. **YAML Configuration File**: A base configuration can be provided using the `--config` argument (e.g., `--config configs/my_config.yaml`).
3. **Default Values** (Lowest Priority): If a parameter is not specified, its default value from the dataclass definition is used.

## Parameter Reference

### Model Architecture (ModelConfig)

These parameters define the structure and physics of the neural network.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `N` | `Literal[64, 128, 256]` | `64` | The dimension of the input diffraction patterns (e.g., 64×64 pixels). This is a critical parameter that defines the network's input shape. |
| `gridsize` | `int` | `1` | For PINN models, the number of neighboring patches to process together (e.g., 2 for a 2×2 grid). For supervised models, this defines the input channel depth. |
| `n_filters_scale` | `int` | `2` | A multiplier for the number of filters in the U-Net's convolutional layers. |
| `model_type` | `Literal['pinn', 'supervised']` | `'pinn'` | The type of model to use. 'pinn' is the main physics-informed model. |
| `architecture` | `Literal['cnn', 'fno', 'hybrid', 'stable_hybrid', 'fno_vanilla', 'hybrid_resnet']` | `'cnn'` | The generator architecture for PINN models. Used by the generator registry to select the network backbone. 'cnn' is the default U-Net based generator. PyTorch-only options include: 'fno' (Cascaded FNO→CNN), 'hybrid' (Hybrid U‑NO), 'stable_hybrid' (InstanceNorm-stabilized Hybrid), 'fno_vanilla' (constant‑resolution FNO), and 'hybrid_resnet' (FNO encoder + CycleGAN ResNet‑6 decoder). See `docs/architecture_torch.md` §4.1. |
| `fno_modes` | `int` | `12` | Number of spectral modes retained in FNO/Hybrid spectral convolutions (PyTorch only). |
| `fno_width` | `int` | `32` | Hidden channel width for FNO/Hybrid blocks (PyTorch only). |
| `fno_blocks` | `int` | `4` | Number of spectral blocks in the FNO/Hybrid encoder (PyTorch only). |
| `fno_cnn_blocks` | `int` | `2` | Number of CNN refiner blocks for the Cascaded FNO generator (PyTorch only). |
| `fno_input_transform` | `Literal['none','sqrt','log1p','instancenorm']` | `'none'` | Optional input dynamic-range transform for FNO/Hybrid lifter (PyTorch only). |
| `resnet_width` | `Optional[int]` | `None` | Fixed bottleneck width for `hybrid_resnet`. Must be divisible by 4 when set (PyTorch only). |
| `amp_activation` | `str` | `'sigmoid'` | The activation function for the amplitude output layer. Choices: 'sigmoid', 'swish', 'softplus', 'relu'. |
| `object_big` | `bool` | `True` | If True, the model reconstructs a large area by stitching patches. If False, it reconstructs a single N×N patch. Some workflows (e.g., grid-lines Torch runner) explicitly set `object_big=False` for TF parity; treat this as a workflow-level config decision, not an implicit default. |
| `probe_big` | `bool` | `True` | If True, the probe representation can vary across the solution region. Some workflows (e.g., grid-lines Torch runner) explicitly set `probe_big=False` for TF parity; treat this as a workflow-level config decision. |
| `probe_mask` | `bool` | `False` | If True, applies a circular mask to the probe to enforce a finite support. |
| `pad_object` | `bool` | `True` | Controls padding behavior in the model. |
| `probe_scale` | `float` | `4.0` | A normalization factor for the probe's amplitude. |
| `gaussian_smoothing_sigma` | `float` | `0.0` | Standard deviation for the Gaussian filter applied to the probe. 0.0 means no smoothing. |

### Training Parameters (TrainingConfig)

These parameters control the training loop, data handling, and loss functions.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_data_file` | `Optional[Path]` | `None` | **Required.** Path to the training dataset (.npz file). |
| `test_data_file` | `Optional[Path]` | `None` | Path to the test dataset (.npz file). |
| `batch_size` | `int` | `16` | The number of samples per batch. Must be a power of 2. |
| `nepochs` | `int` | `50` | Number of training epochs. |
| `mae_weight` | `float` | `0.0` | Weight for the Mean Absolute Error loss in diffraction space. Range: [0, 1]. |
| `nll_weight` | `float` | `1.0` | Weight for the Negative Log-Likelihood (Poisson) loss. Recommended: 1.0. Range: [0, 1]. |
| `realspace_mae_weight` | `float` | `0.0` | Weight for the MAE loss in the object domain. |
| `realspace_weight` | `float` | `0.0` | General weight for all real-space losses. |
| `nphotons` | `float` | `1e9` | The target average number of photons per diffraction pattern, used for the Poisson noise model. |
| `n_groups` | `int` | `512` | Number of groups to use from the dataset. Each group contains 1 image for gridsize=1, or gridsize² images for gridsize>1. **Replaces deprecated `n_images` parameter.** |
| `n_images` | `int` | `None` | **[DEPRECATED]** Legacy parameter name for `n_groups`. Still supported for backward compatibility but will show deprecation warnings. New code should use `n_groups`. |
| `n_subsample` | `Optional[int]` | `None` | Number of images to subsample from the dataset before grouping (independent control). When provided, controls data selection separately from grouping. |
| `subsample_seed` | `Optional[int]` | `None` | Random seed for reproducible subsampling. Ensures consistent data selection across runs. |
| `positions_provided` | `bool` | `True` | If True, use the provided scan positions. |
| `probe_trainable` | `bool` | `False` | If True, allows the model to learn and update the probe function during training. |
| `intensity_scale_trainable` | `bool` | `True` | If True, allows the model to learn the global intensity scaling factor. |
| `output_dir` | `Path` | `"training_outputs"` | The directory where training outputs (model, logs, images) will be saved. |
| `scheduler` | `str` | `'Default'` | Learning rate scheduler type: `'Default'`, `'Exponential'`, `'WarmupCosine'`, `'ReduceLROnPlateau'`. |
| `lr_warmup_epochs` | `int` | `0` | Warmup epochs for the WarmupCosine scheduler. |
| `lr_min_ratio` | `float` | `0.1` | Minimum LR ratio for WarmupCosine (eta_min = base_lr × ratio). |
| `plateau_factor` | `float` | `0.5` | ReduceLROnPlateau factor (multiplier applied when plateau detected). |
| `plateau_patience` | `int` | `2` | ReduceLROnPlateau patience (epochs without improvement before reducing LR). |
| `plateau_min_lr` | `float` | `1e-4` | ReduceLROnPlateau minimum learning rate. |
| `plateau_threshold` | `float` | `0.0` | ReduceLROnPlateau threshold for measuring improvement. |

### Inference Parameters (InferenceConfig)

These parameters control inference and evaluation workflows.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `Path` | **Required** | Path to the trained model directory containing `wts.h5.zip`. |
| `test_data_file` | `Path` | **Required** | Path to the test dataset (.npz file) for inference. |
| `output_dir` | `Path` | `"inference_outputs"` | Directory where inference results will be saved. |
| `n_groups` | `Optional[int]` | `None` | Number of groups to use for inference. If None, uses all available. Each group contains 1 image for gridsize=1, or gridsize² images for gridsize>1. **Replaces deprecated `n_images` parameter.** |
| `n_images` | `Optional[int]` | `None` | **[DEPRECATED]** Legacy parameter name for `n_groups`. Still supported for backward compatibility but will show deprecation warnings. New code should use `n_groups`. |
| `n_subsample` | `Optional[int]` | `None` | Number of images to subsample from test data (independent control). When provided, controls data selection separately from grouping. |
| `subsample_seed` | `Optional[int]` | `None` | Random seed for reproducible subsampling during inference. |
| `debug` | `bool` | `False` | Enable debug mode for additional logging. |

## Understanding Sampling Parameters

The project supports two modes for controlling data sampling:

### Legacy Mode (Backward Compatible)
When only the deprecated `n_images` parameter is used, it behaves as `n_groups`:
- **gridsize=1**: `n_images` specifies how many groups of 1 image each to use
- **gridsize>1**: `n_images` specifies how many neighbor groups to create (total patterns = n_images × gridsize²)

**Note**: New code should use `n_groups` instead of the deprecated `n_images` parameter.

### Independent Control Mode (New)
When `n_subsample` is provided, you get independent control:
- **`n_subsample`**: Controls how many images to randomly select from the dataset
- **`n_groups`**: Controls how many groups to use for training/inference
- **`subsample_seed`**: Ensures reproducible random selection

**Note**: The deprecated `n_images` parameter can still be used in place of `n_groups` but will show warnings.

#### Example Scenarios:
```yaml
# Dense grouping: Use almost all subsampled data in groups
n_subsample: 1200
n_groups: 1000  # Creates 1000 groups of 4 images each (gridsize=2)
gridsize: 2

# Sparse grouping: Subsample large dataset, use subset for groups  
n_subsample: 10000
n_groups: 500   # Creates 500 groups of 4 images each (gridsize=2)
gridsize: 2

# Memory-constrained: Limit data loading
n_subsample: 5000
n_groups: 2000  # Creates 2000 groups of 1 image each (gridsize=1)
gridsize: 1
```

## Example YAML Configuration

You can create a `.yaml` file to specify a set of parameters for a run. This is useful for managing and reproducing experiments.

```yaml
# File: configs/my_experiment_config.yaml

# Model Architecture Parameters
N: 64
gridsize: 2
n_filters_scale: 2
model_type: 'pinn'
amp_activation: 'swish'
probe_trainable: true

# Training Parameters
train_data_file: 'datasets/fly/fly001_prepared_train.npz'
test_data_file: 'datasets/fly/fly001_prepared_test.npz'
output_dir: 'results/my_experiment_run_1'
nepochs: 100
batch_size: 32
n_groups: 4096  # Use 4096 groups for this training run

# Loss Function Weights
nll_weight: 1.0
mae_weight: 0.0

# Physics Parameters
nphotons: 1e9
probe_scale: 4.0
gaussian_smoothing_sigma: 0.0
```

To use this configuration, you would run:

```bash
ptycho_train --config configs/my_experiment_config.yaml
```

You can still override any parameter from the command line:

```bash
# Use the config file but run for only 10 epochs
ptycho_train --config configs/my_experiment_config.yaml --nepochs 10
```

## Configuration Best Practices

1. **Use YAML files** for reproducible experiments and parameter sets you want to reuse.
2. **Use `n_groups` instead of deprecated `n_images`** in new configurations.
3. **Override sparingly** from the command line - use it mainly for quick parameter tweaks.
4. **Document your configs** with comments explaining the experimental purpose.
5. **Version control** your configuration files alongside your code.
6. **Test configurations** with small datasets before running full experiments.

## Parameter Migration

For migrating existing configurations:

```yaml
# Old (deprecated but still works)
n_images: 1000

# New (recommended)
n_groups: 1000  # Always means "number of groups" regardless of gridsize
```
