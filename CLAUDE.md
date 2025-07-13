# CLAUDE.md

This file provides guidance to Claude when working with the PtychoPINN repository.

## ⚠️ Core Project Directives

1.  **Check Project Status First**: Before starting any new task, you **MUST** first read `docs/PROJECT_STATUS.md` to understand the project's current state and active initiative.
2.  **The Physics Model is Correct**: The core ptychography physics simulation and the TensorFlow model architecture are considered stable and correct. **Do not modify the core logic in `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py` unless explicitly asked.**
3.  **Data Format is Paramount**: Most errors in this project stem from incorrect input data formats, not bugs in the model code. Before debugging the model, **always verify the input data structure first.**
4.  **Use Existing Workflows**: The `scripts/` directory contains high-level, tested workflows. Use these as entry points for tasks like training and simulation. Prefer using these scripts over writing new, low-level logic.
5.  **Configuration over Code**: Changes to experimental parameters (e.g., learning rate, image size) should be made via configuration files (`.yaml`) or command-line arguments, not by hardcoding values in the Python source.

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

If the verification run completes and creates files in the `verification_run/` directory, the environment is correct.

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
# Direct simulation tool - simulate data from an existing object/probe file
python scripts/simulation/simulate_and_save.py \
    --input-file <path/to/obj_probe.npz> \
    --output-file <path/to/new_sim_data.npz> \
    --n-images 2000 \
    --gridsize 1

# Example with visualization
python scripts/simulation/simulate_and_save.py \
    --input-file datasets/fly/fly001_transposed.npz \
    --output-file sim_outputs/fly_simulation.npz \
    --n-images 1000 \
    --visualize

# High-level simulation workflow (recommended for complex scenarios)
python scripts/simulation/run_with_synthetic_lines.py \
    --output-dir simulation_outputs \
    --n-images 2000
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
| `gridsize`          | `int`         | For PINN-style models, this defines the number of neighboring patches to process together (e.g., 1 for single-patch processing, 2 for 2x2 neighbors). For supervised models, it defines the input channel depth. |

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

**Authoritative Source:** For all tasks involving the creation or modification of `.npz` datasets, you **MUST** consult and adhere to the specifications in the **[Data Contracts Document](./docs/data_contracts.md)**. This file defines the required key names, array shapes, and data types.

-   **`probeGuess`**: The scanning beam. A complex `(N, N)` array.
-   **`objectGuess`**: The full sample being scanned. A complex `(M, M)` array, where `M` is typically 3-5 times `N`.
-   **`diffraction`**: The stack of measured diffraction patterns. This must be a real `(n_images, N, N)` array representing **amplitude** (i.e., the square root of the measured intensity). The model's Poisson noise layer will square this value internally to simulate photon counts.

```python
# Example: Convert measured intensity to required amplitude format
measured_intensity = ... # Your (n_images, N, N) intensity data
diffraction_amplitude = np.sqrt(measured_intensity)
```
-   **`xcoords`, `ycoords`**: 1D arrays of scan positions, shape `(n_images,)`.

**Reference Example (Known-Good Data):**
File: `datasets/fly/fly001_transposed.npz`
- `probeGuess`: `(64, 64)`
- `objectGuess`: `(232, 232)`  *(Note: much larger than probe)*
- `diffraction`: `(10304, 64, 64)`

**Common Pitfall:** Creating a synthetic `objectGuess` that is the same size as the `probeGuess`. This leaves no room for the probe to scan across the object and will fail. Another common issue is storing intensity instead of amplitude in the `diffraction` array.

## 5. High-Level Architecture

-   **Configuration (`ptycho/config/`)**: Dataclass-based system (`ModelConfig`, `TrainingConfig`). This is the modern way to control the model. A legacy `params.cfg` dictionary is still used for backward compatibility. **Crucially, this is a one-way street:** at the start of a workflow, the modern `TrainingConfig` object is used to update the legacy `params.cfg` dictionary. This allows older modules that still use `params.get('key')` to receive the correct values from a single, modern source of truth. New code should always accept a configuration dataclass as an argument and avoid using the legacy `params.get()` function.
-   **Workflows (`ptycho/workflows/`)**: High-level functions that orchestrate common tasks (e.g., `run_cdi_example`). The `scripts/` call these functions.
-   **Data Loading (`ptycho/loader.py`, `ptycho/raw_data.py`)**: Defines `RawData` (for raw files) and `PtychoDataContainer` (for model-ready data).
-   **Model (`ptycho/model.py`)**: Defines the U-Net architecture and the custom Keras layers that incorporate the physics.
-   **Simulation (`ptycho/diffsim.py`, `ptycho/nongrid_simulation.py`)**: Contains the functions for generating simulated diffraction data from an object and probe.
-   **Image Processing (`ptycho/image/`)**: The modern, authoritative location for image processing tasks.
    -   `stitching.py`: Contains functions for grid-based patch reassembly.
    -   `cropping.py`: Contains the crucial `align_for_evaluation` function for robustly aligning a reconstruction with its ground truth for metric calculation.

## 6. Comparing Models: PtychoPINN vs Baseline

### Complete Training + Comparison Workflow

Use the `run_comparison.sh` script to train both models and compare them in one workflow:

```bash
# Complete workflow: train both models + compare
./scripts/run_comparison.sh <train_data.npz> <test_data.npz> <output_dir>

# Example:
./scripts/run_comparison.sh \
    datasets/fly/fly001_transposed.npz \
    datasets/fly/fly001_transposed.npz \
    comparison_results
```

This workflow:
1. Trains PtychoPINN model with identical hyperparameters (from `configs/comparison_config.yaml`)
2. Trains baseline model with the same configuration
3. Runs comparison analysis using `compare_models.py`

### Compare Pre-Trained Models Only

If you already have trained models, use `compare_models.py` directly:

```bash
# Compare two existing trained models
python scripts/compare_models.py \
    --pinn_dir <path/to/pinn/model/dir> \
    --baseline_dir <path/to/baseline/model/dir> \
    --test_data <path/to/test.npz> \
    --output_dir <comparison_output_dir>

# Example:
python scripts/compare_models.py \
    --pinn_dir training_outputs/pinn_run \
    --baseline_dir training_outputs/baseline_run \
    --test_data datasets/fly/fly001_transposed.npz \
    --output_dir comparison_results
```

**Requirements:**
- Both model directories must contain trained models (`wts.h5.zip` for PtychoPINN, `baseline_model.h5` for baseline)
- Test data must contain `objectGuess` for ground truth comparison

**Outputs:**
- `comparison_plot.png` - Side-by-side visual comparison showing PtychoPINN, Baseline, and Ground Truth
- `comparison_metrics.csv` - Quantitative metrics (MAE, MSE, PSNR, FRC) for both models
- **Unified NPZ files** (enabled by default):
  - `reconstructions.npz` - Single file with all raw reconstructions (amplitude, phase, complex for all models, before registration)
  - `reconstructions_aligned.npz` - Single file with all aligned reconstructions (amplitude, phase, complex, and offsets, after registration)
  - `reconstructions_metadata.txt` - Description of arrays in raw reconstructions NPZ
  - `reconstructions_aligned_metadata.txt` - Description of arrays in aligned reconstructions NPZ

### Automatic Image Registration

**IMPORTANT:** Model comparisons now include automatic image registration to ensure fair evaluation.

**What it does:**
- Automatically detects and corrects translational misalignments between reconstructions and ground truth
- Uses sub-pixel precision phase cross-correlation for accurate alignment
- Prevents spurious metric differences caused by small shifts in reconstruction position

**Key Features:**
- **Automatic activation**: Registration is applied by default in all `compare_models.py` runs
- **Sub-pixel precision**: Detects offsets with ~0.1 pixel accuracy using upsampled FFT correlation
- **Logged results**: Detected offsets are logged and saved to the metrics CSV
- **Physical correctness**: Direction verification ensures offsets are applied correctly

**Understanding the output:**
```bash
# Example log output:
INFO - PtychoPINN detected offset: (-1.060, -0.280)
INFO - Baseline detected offset: (47.000, -1.980)
```

This means:
- PtychoPINN reconstruction needed a 1.06 pixel correction (excellent alignment)
- Baseline reconstruction had a 47 pixel misalignment (significant shift)

**Output format in CSV:**
```csv
PtychoPINN,registration_offset_dy,,,-1.060000
PtychoPINN,registration_offset_dx,,,-0.280000
Baseline,registration_offset_dy,,,47.000000
Baseline,registration_offset_dx,,,-1.980000
```

**Control options:**
```bash
# Normal operation (registration and NPZ exports both enabled by default)
python scripts/compare_models.py [other args]

# Disable registration for debugging/comparison
python scripts/compare_models.py --skip-registration [other args]

# Disable NPZ exports to save disk space
python scripts/compare_models.py --no-save-npz --no-save-npz-aligned [other args]

# Disable only raw NPZ export (keep aligned NPZ files)
python scripts/compare_models.py --no-save-npz [other args]

# Disable only aligned NPZ export (keep raw NPZ files)
python scripts/compare_models.py --no-save-npz-aligned [other args]

# Legacy explicit enable flags (redundant since now default)
python scripts/compare_models.py --save-npz --save-npz-aligned [other args]
```

**When to use --skip-registration:**
- Debugging registration behavior
- Comparing results with/without alignment correction
- Working with datasets where misalignment is intentional
- Performance testing (registration adds ~1-2 seconds per comparison)

**When to disable NPZ exports (--no-save-npz / --no-save-npz-aligned):**
- Limited disk space (unified NPZ files are typically 20-100MB each)
- Only need visual comparison and CSV metrics
- Batch processing many comparisons where raw data isn't needed
- Quick performance testing or debugging runs

**Unified NPZ file contents:**

*reconstructions.npz (raw data):*
- `ptychopinn_amplitude`, `ptychopinn_phase`, `ptychopinn_complex`: PtychoPINN reconstruction data
- `baseline_amplitude`, `baseline_phase`, `baseline_complex`: Baseline reconstruction data  
- `ground_truth_amplitude`, `ground_truth_phase`, `ground_truth_complex`: Ground truth data (if available)

*reconstructions_aligned.npz (aligned data):*
- Same amplitude, phase, complex arrays but after registration correction applied
- `pinn_offset_dy`, `pinn_offset_dx`: PtychoPINN registration offsets in pixels (float values)
- `baseline_offset_dy`, `baseline_offset_dx`: Baseline registration offsets in pixels (float values)

**Important notes about unified NPZ data:**
- **Single file convenience**: All reconstruction data for a comparison is in one unified NPZ file
- **Raw NPZ**: Data saved BEFORE registration correction (full resolution ~192x192 for models, ~232x232 for ground truth)
- **Aligned NPZ**: Data saved AFTER registration correction and coordinate cropping (smaller, aligned size ~179x179)
- **Metadata files**: Text files describe all arrays and their purposes for easy reference
- **Complex data precision**: All complex-valued data preserves full precision for downstream analysis
- **Easy loading**: `data = np.load('reconstructions.npz'); pinn_amp = data['ptychopinn_amplitude']`

**Troubleshooting registration:**

*Large offsets (>20 pixels):*
- Usually indicates genuine misalignment between models
- Check training convergence and reconstruction quality
- Verify ground truth alignment is correct

*Very small offsets (<0.5 pixels):*
- Indicates excellent alignment, registration working correctly
- Models are already well-positioned relative to ground truth

*Registration failures:*
- Check that reconstructions contain sufficient feature content
- Verify images are not all zeros or uniform values
- Ensure complex-valued images have reasonable amplitude variation

### Standard Model Comparison Examples

For common evaluation workflows, use these tested command patterns:

```bash
# Compare models from generalization study (recommended test setup)
python scripts/compare_models.py \
    --pinn_dir large_generalization_study_tike_test/train_1024/pinn_run \
    --baseline_dir large_generalization_study_tike_test/train_1024/baseline_run \
    --test_data tike_outputs/fly001_final_downsampled/fly001_final_downsampled_data_transposed.npz \
    --output_dir comparison_results

# With debug images and custom MS-SSIM parameters
python scripts/compare_models.py \
    --pinn_dir <pinn_model_dir> \
    --baseline_dir <baseline_model_dir> \
    --test_data <test_data.npz> \
    --output_dir <output_dir> \
    --save-debug-images \
    --ms-ssim-sigma 2.0 \
    --phase-align-method plane
```

### Standard Test Datasets

**Primary test data:** `tike_outputs/fly001_final_downsampled/fly001_final_downsampled_data_transposed.npz`
- Large-scale, high-quality test dataset
- Used in generalization studies
- Contains ground truth for all metrics

**Training data:** `datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_train.npz`
- Corresponding training dataset
- Use for training models that will be tested on the above

### Generalization Study Model Structure

```
large_generalization_study_tike_test/
├── train_512/
│   ├── pinn_run/wts.h5.zip                              # PtychoPINN model (512 training images)
│   └── baseline_run/07-XX-XXXX-XX.XX.XX_baseline_gs1/baseline_model.h5  # Baseline model
├── train_1024/                                         # Models trained on 1024 images
├── train_2048/                                         # Models trained on 2048 images
└── train_4096/                                         # Models trained on 4096 images
```

## 6.1. Debug Image Workflow

### Generating Fresh Debug Images

Debug images show the exact preprocessing applied before metric calculations. Always regenerate for accurate analysis:

```bash
# Clean old debug images and generate fresh ones
rm -rf debug_images_*
python scripts/compare_models.py \
    --pinn_dir <model_dir> \
    --baseline_dir <baseline_dir> \
    --test_data <test_data> \
    --output_dir <output> \
    --save-debug-images
```

### Debug Image Output Structure

- **PtychoPINN debug images:** `debug_images_PtychoPINN/`
- **Baseline debug images:** `debug_images_Baseline/`

**Image types generated:**
- `*_amp_pred_for_ms-ssim.png`: Normalized amplitude prediction used in MS-SSIM
- `*_amp_target_for_ms-ssim.png`: Ground truth amplitude used in MS-SSIM
- `*_phase_pred_for_ms-ssim.png`: Scaled phase prediction ([0,1]) used in MS-SSIM
- `*_phase_target_for_ms-ssim.png`: Scaled phase ground truth used in MS-SSIM
- `*_amp_*_for_frc.png`: Same normalized amplitudes used in FRC calculation
- `*_phi_*_for_frc.png`: Plane-aligned phase data used in FRC calculation

**Key verification points:**
- Target images should be identical between PtychoPINN and Baseline (same ground truth)
- Prediction images show model-specific reconstructions after consistent preprocessing
- Color scaling (vmin/vmax) is consistent between pred/target pairs for fair comparison

## 6.2. compare_models.py Complete Reference

### Essential Command-Line Flags

**Debugging & Analysis:**
- `--save-debug-images`: Generate preprocessing visualization images
- `--ms-ssim-sigma N`: Gaussian smoothing sigma for MS-SSIM amplitude calculation (default: 1.0)
- `--phase-align-method {plane,mean}`: Phase alignment method (default: plane)
- `--frc-sigma N`: Gaussian smoothing for FRC calculation (default: 0.0)

**Registration Control:**
- `--skip-registration`: Disable automatic image registration alignment
- Default: Registration enabled for fair comparison

**Output Control:**
- `--save-npz` / `--no-save-npz`: Control raw reconstruction NPZ export (default: enabled)
- `--save-npz-aligned` / `--no-save-npz-aligned`: Control aligned NPZ export (default: enabled)

### Complete Output Files

**Essential outputs:**
- `comparison_metrics.csv`: Quantitative metrics (MAE, MSE, PSNR, SSIM, MS-SSIM, FRC50)
- `comparison_plot.png`: Side-by-side visual comparison
- `*_frc_curves.csv`: Full FRC curves for detailed analysis

**Optional outputs (controlled by flags):**
- `reconstructions.npz`: Raw reconstruction data before alignment
- `reconstructions_aligned.npz`: Aligned reconstruction data after registration
- `reconstructions*_metadata.txt`: Human-readable descriptions of NPZ contents
- `debug_images_*/`: Preprocessing visualization images

### Metric Interpretation Guide

**Amplitude Metrics** (higher = better, except MAE/MSE):
- **SSIM/MS-SSIM**: Structural similarity, range [0,1], >0.8 = good
- **PSNR**: Peak signal-to-noise ratio, >80dB = excellent
- **FRC50**: Spatial resolution in pixels, higher = better resolution

**Phase Metrics** (higher = better, except MAE/MSE):
- **SSIM/MS-SSIM**: After plane fitting and [0,1] scaling
- **MAE**: Mean absolute error in radians, <0.1 = good
- **PSNR**: After plane fitting, >65dB = good

**Registration Offsets** (smaller = better alignment):
- Values <2.0 pixels indicate excellent model-to-ground-truth alignment
- Values >20 pixels suggest significant misalignment issues

## 6.3. Advanced Evaluation Features

For detailed information on evaluation methodology and debugging:

**Current evaluation status:** `docs/refactor/eval_enhancements/implementation_eval_enhancements.md`
- Tracks evaluation pipeline enhancements (SSIM, MS-SSIM, etc.)
- Phase-by-phase implementation status
- Technical specifications for new metrics

**Phase implementation checklists:** `docs/refactor/eval_enhancements/phase_*_checklist.md`
- Detailed task breakdowns for evaluation improvements
- Implementation guidance for specific features

**Generalization studies:** `docs/studies/GENERALIZATION_STUDY_GUIDE.md`
- Complete guide for running training size studies
- Performance scaling analysis workflows
- Publication-ready result generation

## 7. Understanding the Output Directory

After a successful training run using `ptycho_train --output_dir <my_run>`, the output directory will contain several key files:

- **`wts.h5.zip`**: This is the primary output. It's a zip archive containing the trained model weights and architecture for both the main autoencoder and the inference-only `diffraction_to_obj` model. Use `ModelManager.load_multiple_models()` to load it.
- **`history.dill`**: A Python pickle file (using dill) containing the training history dictionary. You can load it to plot loss curves:
  ```python
  import dill
  with open('<my_run>/history.dill', 'rb') as f:
      history = dill.load(f)
  plt.plot(history['loss'])
  ```
- **`reconstructed_amplitude.png` / `reconstructed_phase.png`**: Visualizations of the final reconstructed object from the test set, if stitching was performed.
- **`metrics.csv`**: If a ground truth object was available, this file contains quantitative image quality metrics (MAE, PSNR, FRC) comparing the reconstruction to the ground truth.
- **`params.dill`**: A snapshot of the full configuration used for the run, for reproducibility.

## 8. Advanced & Undocumented Features

### 8.1. Caching Decorators (`ptycho/misc.py`)

- **`@memoize_disk_and_memory`**: Caches the results of expensive functions to disk to speed up subsequent runs with the same parameters.
- **`@memoize_simulated_data`**: Specifically designed for caching simulated data generation, avoiding redundant computation.

### 8.2. Data Utility Tools (`scripts/tools/`)

- **`downsample_data_tool.py`**: For cropping k-space and binning real-space arrays to maintain physical consistency.
- **`prepare_data_tool.py`**: For apodizing, smoothing, or interpolating probes/objects before simulation.
- **`update_tool.py`**: For updating an NPZ file with a new reconstruction result.
- **`visualize_dataset.py`**: For generating a comprehensive visualization plot of an NPZ dataset.

### 8.3. Automated Testing Framework (`ptycho/autotest/`)

- This internal framework provides testing utilities for the project.
- The `@debug` decorator (imported from `ptycho.autotest.debug`) is used to serialize function inputs and outputs during development for creating regression tests.
- This is a developer-facing feature primarily used for debugging and test creation.

## 9. Legacy Code & Deprecation Warnings

- **Legacy Training Script (`ptycho/train.py`):** The file `ptycho/train.py` is a legacy script that uses an older configuration system. **Do not use it.** Always use the `ptycho_train` command-line tool (which points to `scripts/training/train.py`) for all training workflows, as it uses the modern, correct configuration system.