# CLAUDE.md

This file provides guidance to Claude when working with the PtychoPINN repository.

## ⚠️ Core Project Directives

1.  **The Physics Model is Correct**: The core ptychography physics simulation and the TensorFlow model architecture are considered stable and correct. **Do not modify the core logic in `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py` unless explicitly asked.**
2.  **Data Format is Paramount**: Most errors in this project stem from incorrect input data formats, not bugs in the model code. Before debugging the model, **always verify the input data structure first.**
3.  **Use Existing Workflows**: The `scripts/` directory contains high-level, tested workflows. Use these as entry points for tasks like training and simulation. Prefer using these scripts over writing new, low-level logic.
4.  **Configuration over Code**: Changes to experimental parameters (e.g., learning rate, image size) should be made via configuration files (`.yaml`) or command-line arguments, not by hardcoding values in the Python source.

## Project Overview

PtychoPINN is a TensorFlow-based implementation of physics-informed neural networks (PINNs) for ptychographic reconstruction. It combines a U-Net-like deep learning model with a differentiable physics layer to achieve rapid, high-resolution reconstruction from scanning coherent diffraction data.

## 1. Getting Started: Environment & Verification

First, set up the environment and run a verification test to ensure the system is working correctly.

```bash
# 1. Create and activate conda environment
conda create -n ptycho python=3.10
conda activate ptycho

# 2. Install the package in editable mode
pip install -e .

# 3. Run a verification test with known-good data
# This proves the model and environment are set up correctly.
# It uses a small number of images for a quick test.
ptycho_train --train_data_file datasets/fly/fly001_transposed.npz --n_images 512 --output_dir verification_run
```

If the verification run completes without errors, the environment is correct.

## 2. Key Workflows & Commands

### Training a Model

```bash
# Train using a YAML configuration file (preferred method)
ptycho_train --config configs/fly_config.yaml

# Train by specifying files and parameters directly
ptycho_train --train_data_file <path/to/train.npz> --test_data_file <path/to/test.npz> --output_dir <output_path> --n_images 5000
```

### Running Inference

```bash
# Run inference on a test dataset using a trained model
ptycho_inference --model_path <path/to/model_dir> --test_data <path/to/test.npz> --output_dir <inference_output>
```

### Simulating a Dataset

```bash
# Simulate data from an existing object/probe file
# This is useful for creating training data with specific noise properties or scan patterns.
ptycho_simulate --input-file <path/to/obj_probe.npz> --output-file <path/to/new_sim_data.npz> --nimages 2000 --gridsize 1
```

### Running Tests

```bash
# Run all unit tests
python -m unittest discover -s ptycho -p "test_*.py"
```

## 3. Configuration Parameters

Parameters are controlled via YAML files (see `configs/`) or command-line arguments. The system uses modern `dataclasses` for configuration, which are defined in `ptycho/config/config.py`.

### Model Architecture

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

### Training Parameters

| Parameter      | Type   | Description                                                              |
| -------------- | ------ | ------------------------------------------------------------------------ |
| `nepochs`      | `int`  | Number of training epochs. `> 0`. Default: `50`.                         |
| `batch_size`   | `int`  | The number of samples per batch. Must be a power of 2. Default: `16`.    |
| `output_dir`   | `Path` | The directory where training outputs (model, logs, images) will be saved. |

### Data & Simulation Parameters

| Parameter           | Type          | Description                                                                                                    |
| ------------------- | ------------- | -------------------------------------------------------------------------------------------------------------- |
| `train_data_file`   | `Path`        | **Required.** Path to the training dataset (`.npz` file).                                                       |
| `test_data_file`    | `Optional[Path]`| Path to the test dataset (`.npz` file).                                                                        |
| `n_images`          | `int`         | The number of diffraction patterns to use from the dataset. Default: `512`.                                    |
| `gridsize`          | `int`         | For *simulated gridded data*, this defines the grid of patches (e.g., 2 for 2x2). For non-grid data, use `1`. |
| `offset`            | `int`         | For *simulated gridded data*, the pixel spacing between adjacent scan positions.                               |

### Physics & Loss Parameters

| Parameter                   | Type    | Description                                                                                                     |
| --------------------------- | ------- | --------------------------------------------------------------------------------------------------------------- |
| `nphotons`                  | `float` | The target average number of photons per diffraction pattern, used for the Poisson noise model. `> 0`.             |
| `nll_weight`                | `float` | Weight for the Negative Log-Likelihood (Poisson) loss. Recommended: `1.0`. Range: `[0, 1]`.                      |
| `mae_weight`                | `float` | Weight for the Mean Absolute Error loss in diffraction space. Typically `0.0`. Range: `[0, 1]`.                  |
| `probe_scale`               | `float` | A normalization factor for the probe's amplitude. `> 0`.                                                        |
| `probe_trainable`           | `bool`  | If `true`, allows the model to learn and update the probe function during training.                               |
| `intensity_scale_trainable` | `bool`  | If `true`, allows the model to learn the global intensity scaling factor.                                       |

## 4. Critical: Data Format Requirements

**This is the most common source of errors.** A mismatch here will cause low-level TensorFlow errors that are hard to debug.

-   **`probeGuess`**: The scanning beam. A complex `(N, N)` array.
-   **`objectGuess`**: The full sample being scanned. A complex `(M, M)` array, where `M` is typically 3-5 times `N`.
-   **`diffraction`**: The stack of measured diffraction patterns. A real `(n_images, N, N)` array.
-   **`xcoords`, `ycoords`**: 1D arrays of scan positions, shape `(n_images,)`.

**Reference Example (Known-Good Data):**
File: `datasets/fly/fly001_transposed.npz`
- `probeGuess`: `(64, 64)`
- `objectGuess`: `(232, 232)`  *(Note: much larger than probe)*
- `diffraction`: `(10304, 64, 64)`

**Common Pitfall:** Creating a synthetic `objectGuess` that is the same size as the `probeGuess`. This leaves no room for the probe to scan across the object and will fail.

## 5. High-Level Architecture

-   **Configuration (`ptycho/config/`)**: Dataclass-based system (`ModelConfig`, `TrainingConfig`). This is the modern way to control the model. A legacy `params.cfg` dictionary is still used for backward compatibility. **Crucially, this is a one-way street:** at the start of a workflow, the modern `TrainingConfig` object is used to update the legacy `params.cfg` dictionary. This allows older modules that still use `params.get('key')` to receive the correct values from a single, modern source of truth. New code should always accept a configuration dataclass as an argument and avoid using the legacy `params.get()` function.
-   **Workflows (`ptycho/workflows/`)**: High-level functions that orchestrate common tasks (e.g., `run_cdi_example`). The `scripts/` call these functions.
-   **Data Loading (`ptycho/loader.py`, `ptycho/raw_data.py`)**: Defines `RawData` (for raw files) and `PtychoDataContainer` (for model-ready data).
-   **Model (`ptycho/model.py`)**: Defines the U-Net architecture and the custom Keras layers that incorporate the physics.
-   **Simulation (`ptycho/diffsim.py`, `ptycho/nongrid_simulation.py`)**: Contains the functions for generating simulated diffraction data from an object and probe.
