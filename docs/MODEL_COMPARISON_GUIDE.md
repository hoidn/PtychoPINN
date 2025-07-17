# Model Comparison Guide

This guide provides comprehensive information on comparing PtychoPINN models against baseline models, including training workflows, evaluation metrics, and analysis tools.

## Overview

PtychoPINN model comparison involves training both a physics-informed neural network (PINN) model and a supervised baseline model, then evaluating their performance using various metrics including traditional image quality measures and advanced perceptual metrics.

## Complete Training + Comparison Workflow

Use the `<workflow-entrypoint>scripts/run_comparison.sh</workflow-entrypoint>` script to train both models and compare them in one workflow:

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

## Compare Pre-Trained Models Only

If you already have trained models, use `<code-ref type="script">scripts/compare_models.py</code-ref>` directly:

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

## Automatic Image Registration

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

## Enhanced Evaluation Metrics

**New Metrics Available:**

The evaluation pipeline now includes advanced perceptual metrics beyond traditional MAE/MSE/PSNR:

1. **SSIM (Structural Similarity Index)**
   - Measures structural similarity between images
   - Range: [0, 1] where 1 = perfect match
   - More aligned with human perception than pixel-wise metrics
   - Calculated for both amplitude and phase

2. **MS-SSIM (Multi-Scale SSIM)**
   - Evaluates structural similarity at multiple scales
   - Range: [0, 1] where 1 = perfect match
   - Better captures features at different resolutions
   - Configurable sigma parameter for Gaussian weighting

**Phase Preprocessing Options:**

Phase comparison now supports two alignment methods:

```bash
# Plane fitting (default) - removes linear phase ramps
python scripts/compare_models.py --phase-align-method plane [other args]

# Mean subtraction - simpler centering approach
python scripts/compare_models.py --phase-align-method mean [other args]
```

**Debug Visualization:**

Generate preprocessing visualization images to verify metric calculations:

```bash
# Enable debug image generation
python scripts/compare_models.py --save-debug-images [other args]

# This creates debug directories with:
# - *_amp_*_for_ms-ssim.png   - Amplitude after preprocessing
# - *_phase_*_for_ms-ssim.png - Phase after preprocessing  
# - *_for_frc.png             - Images used for FRC calculation
```

**Advanced FRC Options:**

Enhanced Fourier Ring Correlation with smoothing:

```bash
# Default (no smoothing)
python scripts/compare_models.py --frc-sigma 0.0 [other args]

# Smooth FRC curves (reduces noise)
python scripts/compare_models.py --frc-sigma 2.0 [other args]
```

**MS-SSIM Configuration:**

Control multi-scale behavior:

```bash
# Custom sigma for MS-SSIM Gaussian weights
python scripts/compare_models.py --ms-ssim-sigma 1.5 [other args]
```

**CSV Output Format:**

The `comparison_metrics.csv` now includes:
- `mae` - Mean Absolute Error (amplitude, phase)
- `mse` - Mean Squared Error (amplitude, phase)
- `psnr` - Peak Signal-to-Noise Ratio (amplitude, phase)
- `ssim` - Structural Similarity (amplitude, phase) **[NEW]**
- `ms_ssim` - Multi-Scale SSIM (amplitude, phase) **[NEW]**
- `frc50` - Fourier Ring Correlation at 0.5 threshold
- Registration offsets (if applicable)

**Phase Preprocessing Transparency:**

When running comparisons, phase preprocessing steps are now logged:
```
Phase preprocessing: plane-fitted range [-3.142, 3.142] -> scaled range [0.000, 1.000]
```

This shows the exact transformations applied before metric calculation, ensuring transparency and reproducibility.

## Standard Model Comparison Examples

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

## Standard Test Datasets

**Primary test data:** `tike_outputs/fly001_final_downsampled/fly001_final_downsampled_data_transposed.npz`
- Large-scale, high-quality test dataset
- Used in generalization studies
- Contains ground truth for all metrics

**Training data:** `datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_train.npz`
- Corresponding training dataset
- Use for training models that will be tested on the above

## Generalization Study Model Structure

```
large_generalization_study_tike_test/
├── train_512/
│   ├── pinn_run/wts.h5.zip                              # PtychoPINN model (512 training images)
│   └── baseline_run/07-XX-XXXX-XX.XX.XX_baseline_gs1/baseline_model.h5  # Baseline model
├── train_1024/                                         # Models trained on 1024 images
├── train_2048/                                         # Models trained on 2048 images
└── train_4096/                                         # Models trained on 4096 images
```

## Debug Image Workflow

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

## compare_models.py Complete Reference

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

## Advanced Evaluation Features

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