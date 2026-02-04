### `docs/COMMANDS_REFERENCE.md` 

```markdown
# PtychoPINN Commands Reference

**Purpose:** A quick reference for essential PtychoPINN command-line workflows. This guide provides the "what"; for the "why," please consult the linked detailed guides.

## 📋 Quick Navigation
- [Data Preparation Golden Paths](#data-preparation-golden-paths)
- [Training](#training) 
- [Inference](#inference)
- [Model Evaluation](#model-evaluation)
- [Model Comparison](#model-comparison)
- [Studies](#studies)
- [Best Practices & Key Guidelines](#best-practices--key-guidelines)
- [Quick Troubleshooting](#quick-troubleshooting)

---

## Data Preparation Golden Paths

Choose the path that matches your starting point and goal.

### Golden Path 1: Preparing an *Existing* Experimental Dataset

**Use this path when:** You have an existing `.npz` file (like `fly64`) with thousands of diffraction patterns and you want to prepare it for training.

**Goal:** To canonicalize, randomize, and split an existing dataset.

```bash
# 1. Canonicalize raw data (REQUIRED FIRST STEP for experimental data)
#    Why: Converts uint16 intensity to float32 amplitude and renames keys.
python scripts/tools/transpose_rename_convert_tool.py raw_data.npz converted_data.npz

# 2. Shuffle the dataset (OPTIONAL - useful for creating canonical benchmark datasets)
#    Note: No longer required for gridsize=1 training as of the unified sampling update.
#    Still useful for creating reproducible, pre-randomized datasets for benchmarking.
python scripts/tools/shuffle_dataset_tool.py converted_data.npz shuffled_data.npz --seed 42

# 3. Split into train/test sets (optional, but good practice)
#    Why: Creates dedicated, non-overlapping sets for training and validation.
python scripts/tools/split_dataset_tool.py shuffled_data.npz output_dir/ --split-fraction 0.8

# 4. Always visualize your final dataset to verify its integrity
#    Why: A quick visual check can catch many common data format errors.
python scripts/tools/visualize_dataset.py output_dir/train.npz train_set_visualization.png
```

### Golden Path 2: Creating a *New* Synthetic Dataset from a Reconstruction

**Use this path when:** You have a single, high-quality reconstructed `objectGuess` (e.g., from the Tike algorithm) and you want to generate a new, large, clean dataset for robust studies.

**Goal:** To simulate a new, large-scale dataset from a single high-quality object.

```bash
# Basic usage with defaults (backward compatible)
bash scripts/prepare.sh

# Custom input and output
bash scripts/prepare.sh --input-file path/to/reconstruction.npz --output-dir experiments/my_study

# Low-photon dataset generation
bash scripts/prepare.sh --input-file synthetic.npz --output-dir studies/photons_1e4 --sim-photons 1e4 --sim-images 10000

# See all options
bash scripts/prepare.sh --help
```

**New Parameters (as of latest update):**
- `--input-file PATH`: Specify input NPZ file (default: tike_outputs/fly001/fly001_reconstructed.npz)
- `--output-dir DIR`: Organize all outputs in a single directory (default: uses traditional structure)
- `--sim-images N`: Number of images to simulate (default: 35000)
- `--sim-photons P`: Photons per image (default: 1e9)

**Output Structure with `--output-dir`:**
```
DIR/
├── stages/          # All intermediate processing stages
│   ├── 01_padded/
│   ├── 02_transposed/
│   └── ...
└── dataset/         # Final train/test splits
    ├── train.npz
    └── test.npz
```

**What `prepare.sh` does internally:**
1.  **Cleans & Upsamples:** Takes the input `objectGuess` and `probeGuess`, pads them, and interpolates them to a higher resolution.
2.  **Simulates New Data:** Uses `scripts/simulation/simulate_and_save.py` to generate thousands of **new** diffraction patterns from the upsampled object. This is the key step.
3.  **Downsamples:** Processes the new high-resolution synthetic data back down to the target resolution, ensuring physical consistency.
4.  **Splits:** Creates final train and test sets from the new synthetic data.

---

## Training

```bash
# Basic training
ptycho_train --train_data_file dataset.npz --n_groups 2000 --nepochs 50 --output_dir my_run

# Legacy compatibility (deprecated but still works)
ptycho_train --train_data_file dataset.npz --n_images 2000 --nepochs 50 --output_dir my_run

# With configuration file (recommended)
ptycho_train --config configs/my_config.yaml

# Independent sampling control (NEW)
ptycho_train --train_data_file dataset.npz --n_subsample 5000 --n_groups 1000 --nepochs 50 --output_dir my_run

# Reproducible sampling (NEW)
ptycho_train --train_data_file dataset.npz --n_subsample 3000 --n_groups 500 --subsample_seed 42 --output_dir my_run

# K choose C oversampling (NEW - Phase 6 with explicit opt-in)
ptycho_train --train_data_file dataset.npz \
    --n_subsample 500 --n_groups 2000 \
    --gridsize 2 --neighbor_count 7 \
    --enable_oversampling --neighbor_pool_size 7 \
    --output_dir oversampled_run
```

### 📊 NEW: Independent Sampling Control

The project now supports **independent control** of data subsampling and neighbor grouping:

- **`--n-subsample`**: Controls how many images to randomly select from the dataset
- **`--n-groups`**: Controls how many groups to use for training/inference (regardless of gridsize)
- **`--subsample-seed`**: Ensures reproducible random selection

**Note**: `--n-images` is deprecated but still supported for backward compatibility.

**Example Use Cases:**
```bash
# Dense grouping: Use most subsampled data
ptycho_train --n-subsample 1200 --n-groups 1000 --gridsize 2 ...

# Sparse grouping: Large subsample, fewer groups
ptycho_train --n-subsample 10000 --n-groups 500 --gridsize 2 ...

# Memory-constrained: Limit data loading
ptycho_train --n-subsample 5000 --n-groups 2000 --gridsize 1 ...
```

### ⚠️ CRITICAL: Understanding `gridsize` and `--n-groups`

The `--n-groups` parameter **always** refers to the number of groups to use, regardless of the `gridsize` parameter. This provides consistent behavior and eliminates confusion.

| GridSize | `--n-groups` Refers To... | Total Patterns Used | Subsampling Method |
|----------|---------------------------|---------------------|--------------------|
| 1        | **Groups (each with 1 image)**    | `n_groups` × 1      | **Unified Random Sampling.** Each group contains 1 image. |
| > 1      | **Groups (neighbor groups)**       | `n_groups` × `gridsize`² | **Unified Random Sampling.** Each group contains gridsize² images. |

**Key Insight**: With `--n-groups`, the parameter always means "number of groups" regardless of gridsize. For gridsize=1, each group contains 1 image. For gridsize>1, each group contains multiple neighboring images.

**Log Message Examples to Watch For:**
```
# GridSize=1 (Unified group interpretation)
INFO - Parameter interpretation: --n-groups=1000 refers to 1000 groups of 1 image each (gridsize=1)

# GridSize=2 (Unified group interpretation)
INFO - Parameter interpretation: --n-groups=250 refers to 250 groups of 4 images each (gridsize=2, total patterns=1000)
INFO - Using grouping-aware subsampling strategy for gridsize=2
```

**Backward Compatibility**: The deprecated `--n-images` parameter still works but will show a deprecation warning.

### 🔄 NEW: K Choose C Oversampling (Phase 6)

**Use case:** When you want to create more training groups than available seed points by sampling multiple combinations from each seed's neighbors.

**Prerequisites (OVERSAMPLING-001):**
- `gridsize > 1` (so C = gridsize² > 1)
- `--enable-oversampling` flag (explicit opt-in)
- `--neighbor-pool-size >= C` (pool size must be at least gridsize²)

**Example:**
```bash
# Create 2000 groups from only 500 seed points
ptycho_train --train_data_file dataset.npz \
    --n_subsample 500 \
    --n_groups 2000 \
    --gridsize 2 \
    --neighbor_count 7 \
    --enable_oversampling \
    --neighbor_pool_size 7 \
    --output_dir oversampled_run
```

**Important Notes:**
- **Overfitting risk:** Oversampling reuses local neighborhoods; monitor using spatial validation splits
- **Debug logs:** Look for `[OVERSAMPLING DEBUG]` messages showing which branch was taken
- **Error handling:** Clear error messages guide you if prerequisites aren't met (see `docs/SAMPLING_USER_GUIDE.md`)

---

## Inference

```bash
# Basic inference (uses all test data)
ptycho_inference --model_path trained_model/ --test_data test.npz --output_dir inference_out

# With specific number of test groups
ptycho_inference --model_path trained_model/ --test_data test.npz --n_groups 500 --output_dir inference_out

# Independent sampling control (NEW)
ptycho_inference --model_path trained_model/ --test_data test.npz --n_subsample 2000 --n_groups 500 --output_dir inference_out

# GridSize=2 inference (must match the gridsize used for training)
ptycho_inference --model_path gs2_model/ --test_data test.npz --n_groups 125 --gridsize 2 --output_dir gs2_inference
```

---

## Reconstruction

```bash
# Tike iterative reconstruction
python scripts/reconstruction/run_tike_reconstruction.py \
    input_data.npz \
    tike_output/ \
    --iterations 1000 \
    --extra-padding 32

# Quick reconstruction (fewer iterations)
python scripts/reconstruction/run_tike_reconstruction.py \
    input_data.npz \
    tike_output/ \
    --iterations 100
```

### Pty-Chi Reconstruction

```bash
# Basic pty-chi reconstruction (DM algorithm, 200 epochs)
python scripts/reconstruction/ptychi_reconstruct_tike.py

# High-quality reconstruction with extended convergence
# Note: Parameters are currently hardcoded in script main() function
# Modify tike_dataset, algorithm, num_epochs, n_images as needed

# Available algorithms: 'DM', 'LSQML', 'PIE'
# Default: DM with 200 epochs on 2000 images
```

---

## Model Evaluation

```bash
# Basic model evaluation (single model against ground truth)
ptycho_evaluate --model-dir trained_model/ --test-data test.npz --output-dir eval_results

# Evaluation with sampling control (NEW)
ptycho_evaluate --model-dir trained_model/ --test-data test.npz --n-test-subsample 2000 --n-test-groups 500 --output-dir eval_results

# Custom visualization settings
ptycho_evaluate --model-dir model/ --test-data test.npz --output-dir eval_vis \
    --phase-align-method plane --save-individual-images --phase-colormap viridis

# Skip registration for debugging
ptycho_evaluate --model-dir model/ --test-data test.npz --output-dir eval_debug --skip-registration

# Quiet mode for automation
ptycho_evaluate --model-dir model/ --test-data test.npz --output-dir eval_auto --quiet
```

### 📊 Key Features

- **Comprehensive Metrics**: Computes MAE, MSE, PSNR, SSIM, MS-SSIM, and FRC against ground truth
- **Automatic Registration**: Aligns reconstructions with ground truth for fair comparison
- **Visual Outputs**: Generates amplitude/phase plots, error maps, and comparison figures
- **CSV Export**: Saves all metrics to `results.csv` for downstream analysis
- **Independent Sampling**: Control test data subsampling with `--n-test-subsample` and `--n-test-groups`

### 📋 When to Use Model Evaluation vs Comparison

- **Use `ptycho_evaluate`** when:
  - Evaluating a single trained model against ground truth
  - Computing detailed metrics for model analysis
  - Creating publication-ready visualizations
  - Debugging model performance issues

- **Use `compare_models.py`** when:
  - Comparing multiple models head-to-head
  - Benchmarking PtychoPINN vs Baseline vs Tike
  - Running systematic model comparisons

---

## Model Comparison

```bash
# Two-way comparison (PtychoPINN vs Baseline)
python scripts/compare_models.py \
    --pinn_dir pinn_model/ \
    --baseline_dir baseline_model/ \
    --test_data test.npz \
    --output_dir comparison_out

# With independent sampling control (NEW)
python scripts/compare_models.py \
    --pinn_dir pinn_model/ \
    --baseline_dir baseline_model/ \
    --test_data test.npz \
    --n-test-subsample 3000 \
    --n-test-groups 500 \
    --output_dir comparison_out

# Three-way comparison (PtychoPINN vs Baseline vs Tike)
python scripts/compare_models.py \
    --pinn_dir pinn_model/ \
    --baseline_dir baseline_model/ \
    --test_data test.npz \
    --output_dir comparison_out \
    --tike_recon_path tike_output/tike_reconstruction.npz

# Optional metrics table (LaTeX/PDF/PNG when available)
python scripts/compare_models.py \
    --pinn_dir pinn_model/ \
    --baseline_dir baseline_model/ \
    --test_data test.npz \
    --output_dir comparison_out \
    --metrics-table

# Complete training + comparison workflow
./scripts/run_comparison.sh train.npz test.npz output_dir

# With specific training/test sizes and independent control
./scripts/run_comparison.sh train.npz test.npz output_dir \
    --n-train-groups 2000 \
    --n-train-subsample 5000 \
    --n-test-groups 500 \
    --n-test-subsample 2000 \
    --skip-training
```

---

## Studies

> **Parameter Migration Notice**: The generalization study script now uses `--train-group-sizes` instead of the deprecated `--train-sizes`. The old parameter is still supported but will show deprecation warnings.

```bash
# Synthetic data generalization study (auto-generates datasets)
./scripts/studies/run_complete_generalization_study.sh \
    --train-group-sizes "512 1024 2048 4096" \
    --num-trials 3 \
    --output-dir synthetic_study

# Independent training/test control with new parameters  
./scripts/studies/run_complete_generalization_study.sh \
    --train-group-sizes "512 1024" \
    --train-subsample-sizes "1024 2048" \
    --test-groups 500 \
    --test-subsample 1500 \
    --num-trials 2 \
    --output-dir independent_control_study

# Experimental data generalization study (uses existing datasets)
./scripts/studies/run_complete_generalization_study.sh \
    --train-data "datasets/fly64/fly001_64_train_converted.npz" \
    --test-data "datasets/fly64/fly001_64_train_converted.npz" \
    --skip-data-prep \
    --train-group-sizes "512 1024 2048" \
    --num-trials 3 \
    --output-dir experimental_study

# Spatial bias analysis study (specialized dataset)
./scripts/studies/run_complete_generalization_study.sh \
    --train-data "datasets/fly64/fly64_top_half_shuffled.npz" \
    --test-data "datasets/fly64/fly001_64_train_converted.npz" \
    --skip-data-prep \
    --train-group-sizes "512 1024 2048" \
    --test-groups 1000 \
    --test-subsample 2048 \
    --output-dir spatial_bias_study

# Plot results from a completed study
python scripts/studies/aggregate_and_plot_results.py study_results --output plots/
```

### Grid-Lines (TF + Torch) Comparison Harness

```bash
# Run the grid-lines harness (TF cnn+baseline + Torch FNO/Hybrid)
python scripts/studies/grid_lines_compare_wrapper.py \
    --N 64 \
    --gridsize 1 \
    --output-dir outputs/grid_lines_gs1_n64 \
    --architectures cnn,baseline,fno,hybrid \
    --set-phi

# Use cubic interpolation instead of the default pad+phase-extrapolate probe scaling
python scripts/studies/grid_lines_compare_wrapper.py \
    --N 64 \
    --gridsize 1 \
    --output-dir outputs/grid_lines_gs1_n64_interp \
    --architectures cnn,baseline \
    --probe-scale-mode interpolate

# Apply a centered disk probe mask during dataset generation
python scripts/studies/grid_lines_compare_wrapper.py \
    --N 64 \
    --gridsize 1 \
    --output-dir outputs/grid_lines_gs1_n64_mask64 \
    --architectures cnn,baseline \
    --probe-mask-diameter 64
```

---

## Best Practices & Key Guidelines

-   **Always specify `--output_dir`** to avoid accidentally overwriting previous results.
-   **Match `gridsize`** between training and inference. A model trained with `gridsize=1` cannot be used for inference with `gridsize=2`.
-   **Verify your data format** before starting a long training run. Use `scripts/tools/visualize_dataset.py`.
-   **Unified sampling for all gridsize values:** As of the latest update, the system uses the same efficient random sampling strategy for all gridsize values. Manual shuffling is no longer required.
-   **Use `--sequential_sampling` flag** if you need the old sequential behavior (first N images) for debugging or specific scan region analysis.
-   **Monitor training logs** for parameter interpretation messages to confirm the script is behaving as you expect.

---

## Quick Troubleshooting

```bash
# Check dataset format and key info
python -c "import numpy as np; data=np.load('dataset.npz'); print('Keys:', list(data.keys())); print({k: data[k].shape for k in data.keys()})"

# Verify environment and see all CLI options
ptycho_train --help

# Monitor training progress in real-time
tail -f output_dir/logs/debug.log

# Check GPU usage for bottlenecks or OOM errors
nvidia-smi
```

For detailed explanations, see the <doc-ref type="guide">docs/DEVELOPER_GUIDE.md</doc-ref> and <doc-ref type="guide">docs/TOOL_SELECTION_GUIDE.md</doc-ref>.
```
