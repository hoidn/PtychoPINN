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
# Basic training with type defaults
ptycho_train --data.train_data_file dataset.npz --output_dir my_run

# Numeric, Boolean, model, and sampling values belong in nested YAML
ptycho_train --config configs/my_config.yaml
```

On the current `refactor` tip, generated numeric and Boolean CLI values are not
decoded before strict Pydantic validation. Use YAML for those types until the
CLI decoder is fixed. The nested dotted spellings below are the registered
public field names, but they are not currently reliable runnable overrides for
numeric/Boolean values.

### Native Torch count-intensity profile

The native Torch CLI owns the CI profile and rectangular-gauge flag. Omitting
the flag authors no override, so the profile resolves to `dose_closure`; an
explicit `ones` wins:

```bash
# Fixed representative dose closure (the ci profile default)
python -m ptycho_torch.train \
  --train_data_file counts_train.npz \
  --output_dir ci_run \
  --profile ci

# Keep exact unit initialization instead
python -m ptycho_torch.train \
  --train_data_file counts_train.npz \
  --output_dir ci_unit_init \
  --profile ci --rect-s1s2-init ones
```

Do not pass `--rect-s1s2-init` to the unified `ptycho_train` command; it does
not expose that flag. The
[configuration guide](CONFIGURATION.md#dose-closure-initialization) defines
dose-closure sampling and failure behavior.

### Independent sampling control

The unified training CLI mirrors the nested public configuration:

- `--sampling.n_subsample` selects raw rows before grouping.
- `--sampling.n_groups` selects the number of grouped samples, independent of
  grid size.
- `--sampling.subsample_seed` makes raw-row selection reproducible.
- `--sampling.n_images` is a deprecated alias for
  `--sampling.n_groups`; conflicting alias/canonical values fail validation.

Model fields belong under `model` in YAML. For example:

```yaml
model:
  gridsize: 2
sampling:
  n_subsample: 10000
  n_groups: 500
  subsample_seed: 3
```

With `gridsize=1`, each group contains one image. With `gridsize>1`, each
group contains `gridsize²` neighboring images. `n_groups` always counts groups;
it never changes meaning based on grid size.

---

## Inference

```bash
# Basic inference (uses all test data)
ptycho_inference --model_path trained_model/ --test_data test.npz --output_dir inference_out

# With specific number of test groups
ptycho_inference --model_path trained_model/ --test_data test.npz --n_groups 500 --output_dir inference_out

# Independent sampling control (NEW)
ptycho_inference --model_path trained_model/ --test_data test.npz --n_subsample 2000 --n_groups 500 --output_dir inference_out

# GridSize is restored from the saved model; the test data must match it
ptycho_inference --model_path gs2_model/ --test_data test.npz --n_groups 125 --output_dir gs2_inference
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

Model evaluation is currently performed via the comparison/study scripts (see `scripts/studies/README.md`); there is no `ptycho_evaluate` console command.

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

### Parameterized Study Composition

`ptycho_study` composes each arm and delegates it to the configured public
runner; it is not a separate trainer.

```bash
ptycho_study --help
```

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

---

## Best Practices & Key Guidelines

-   **Always specify `--output_dir`** to avoid accidentally overwriting previous results.
-   **Match `gridsize`** between training and inference. A model trained with `gridsize=1` cannot be used for inference with `gridsize=2`.
-   **Verify your data format** before starting a long training run. Use `scripts/tools/visualize_dataset.py`.
-   **Unified sampling for all gridsize values:** As of the latest update, the system uses the same efficient random sampling strategy for all gridsize values. Manual shuffling is no longer required.
-   **Set `sampling.sequential_sampling: true` in unified training YAML** to
    use the first grouping anchors within the already selected raw-row pool.
    Raw-row subsampling remains random; set `sampling.subsample_seed` to make
    that pool reproducible.
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

For detailed explanations, see the [Configuration Guide](CONFIGURATION.md) and
[Troubleshooting Guide](TROUBLESHOOTING.md).
