### `docs/COMMANDS_REFERENCE.md` 

```markdown
# PtychoPINN Commands Reference

**Purpose:** A quick reference for essential PtychoPINN command-line workflows. This guide provides the "what"; for the "why," please consult the linked detailed guides.

## 📋 Quick Navigation
- [Data Preparation Golden Paths](#data-preparation-golden-paths)
- [Training](#training) 
- [Inference](#inference)
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

# 2. Shuffle the dataset (CRITICAL for random subsampling with gridsize=1)
#    Why: Ensures that taking the first N images results in a random, representative sample.
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
# The `prepare.sh` script is the high-level orchestrator for this entire workflow.
bash scripts/prepare.sh
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
ptycho_train --train_data_file dataset.npz --n_images 2000 --nepochs 50 --output_dir my_run

# With configuration file (recommended)
ptycho_train --config configs/my_config.yaml
```

### ⚠️ CRITICAL: Understanding `gridsize` and `--n_images`

The interpretation of the `--n_images` flag and the correct subsampling method **change fundamentally** based on the `gridsize` parameter. Using the wrong method will lead to invalid training results.

| GridSize | `--n_images` Refers To... | Total Patterns Used | Subsampling Method |
|----------|---------------------------|---------------------|--------------------|
| 1        | Individual images         | `n_images`          | **Manual Shuffle Required.** You MUST shuffle the dataset with `shuffle_dataset_tool.py` first to get a random subset. |
| > 1      | **Neighbor groups**       | `n_images` × `gridsize`² | **Safe & Recommended.** The script automatically uses a robust "group-then-sample" strategy. It finds all valid neighbor groups in the entire dataset and then randomly samples from them. **Do not shuffle the dataset beforehand.** |

**Log Message Examples to Watch For:**
```
# GridSize=1 (Safe Subsampling after manual shuffle)
INFO - Parameter interpretation: --n-images=1000 refers to individual images (gridsize=1)

# GridSize=2 (Safe, Automatic Group-Aware Subsampling)
INFO - Parameter interpretation: --n-images=250 refers to neighbor groups (gridsize=2, total patterns=1000)
INFO - Using grouping-aware subsampling strategy for gridsize=2
```

---

## Inference

```bash
# Basic inference
ptycho_inference --model_path trained_model/ --test_data test.npz --output_dir inference_out

# With specific number of test patterns
ptycho_inference --model_path trained_model/ --test_data test.npz --n_images 500 --output_dir inference_out

# GridSize=2 inference (must match the gridsize used for training)
ptycho_inference --model_path gs2_model/ --test_data test.npz --n_images 125 --gridsize 2 --output_dir gs2_inference
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

---

## Model Comparison

```bash
# Two-way comparison (PtychoPINN vs Baseline)
python scripts/compare_models.py \
    --pinn_dir pinn_model/ \
    --baseline_dir baseline_model/ \
    --test_data test.npz \
    --output_dir comparison_out

# Three-way comparison (PtychoPINN vs Baseline vs Tike)
python scripts/compare_models.py \
    --pinn_dir pinn_model/ \
    --baseline_dir baseline_model/ \
    --test_data test.npz \
    --output_dir comparison_out \
    --tike_recon_path tike_output/tike_reconstruction.npz

# Complete training + comparison workflow
./scripts/run_comparison.sh train.npz test.npz output_dir

# With specific training/test sizes (see gridsize warning in Training section)
./scripts/run_comparison.sh train.npz test.npz output_dir --n-train-images 2000 --n-test-images 500

# With custom stitch crop size
python scripts/compare_models.py \
    --pinn_dir pinn_model/ \
    --baseline_dir baseline_model/ \
    --test_data test.npz \
    --output_dir comparison_out \
    --stitch-crop-size 30
```

---

## Studies

```bash
# Synthetic data generalization study (auto-generates datasets)
./scripts/studies/run_complete_generalization_study.sh \
    --train-sizes "512 1024 2048 4096" \
    --num-trials 3 \
    --output-dir synthetic_study

# Experimental data generalization study (uses existing datasets)
./scripts/studies/run_complete_generalization_study.sh \
    --train-data "datasets/fly64/fly001_64_train_converted.npz" \
    --test-data "datasets/fly64/fly001_64_train_converted.npz" \
    --skip-data-prep \
    --train-sizes "512 1024 2048" \
    --num-trials 3 \
    --output-dir experimental_study

# Spatial bias analysis study (specialized dataset)
./scripts/studies/run_complete_generalization_study.sh \
    --train-data "datasets/fly64/fly64_top_half_shuffled.npz" \
    --test-data "datasets/fly64/fly001_64_train_converted.npz" \
    --skip-data-prep \
    --train-sizes "512 1024 2048" \
    --output-dir spatial_bias_study

# Study with custom stitch crop size
./scripts/studies/run_complete_generalization_study.sh \
    --train-sizes "512 1024" \
    --num-trials 3 \
    --stitch-crop-size 30 \
    --output-dir custom_stitch_study

# Plot results from a completed study
python scripts/studies/aggregate_and_plot_results.py study_results --output plots/
```

---

## Best Practices & Key Guidelines

-   **Always specify `--output_dir`** to avoid accidentally overwriting previous results.
-   **Match `gridsize`** between training and inference. A model trained with `gridsize=1` cannot be used for inference with `gridsize=2`.
-   **Verify your data format** before starting a long training run. Use `scripts/tools/visualize_dataset.py`.
-   **For `gridsize=1` subsampling, always shuffle the dataset first** to ensure a random, representative sample.
-   **For `gridsize > 1` subsampling, use the `--n_images` flag directly** on the original, ordered dataset. The system handles randomization automatically.
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
