# PtychoPINN Commands Reference

**Purpose:** A quick reference for essential PtychoPINN command-line workflows. This guide provides the "what"; for the "why," please consult the linked detailed guides.

## 📋 Quick Navigation
- [Data Preparation Golden Paths](#data-preparation-golden-paths)
- [Generic Synthetic PyTorch Workflow](#generic-synthetic-pytorch-workflow)
- [Training](#training) 
- [Inference](#inference)
- [Single-Model Inference](#single-model-inference)
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
#    Why: Converts uint16 diffraction/intensity arrays to float32 and renames keys.
python scripts/tools/transpose_rename_convert_tool.py raw_data.npz converted_data.npz

# 2. Shuffle the dataset (OPTIONAL - useful for creating canonical benchmark datasets)
#    Note: No longer required for gridsize=1 training as of the unified sampling update.
#    Still useful for creating reproducible, pre-randomized datasets for benchmarking.
python scripts/tools/shuffle_dataset_tool.py --input-file converted_data.npz --output-file shuffled_data.npz --seed 42

# 3. Split into train/test sets (optional, but good practice)
#    Why: Creates dedicated, non-overlapping sets for training and validation.
python scripts/tools/split_dataset_tool.py shuffled_data.npz output_dir/ --split-fraction 0.8

# 4. Always visualize your final dataset to verify its integrity
#    Why: A quick visual check can catch many common data format errors.
python scripts/tools/visualize_dataset.py output_dir/shuffled_data_train.npz train_set_visualization.png
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

**Parameters:**
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
    ├── <input_stem>_train.npz
    └── <input_stem>_test.npz
```

**What `prepare.sh` does internally:**
1.  **Cleans & Upsamples:** Takes the input `objectGuess` and `probeGuess`, pads them, and interpolates them to a higher resolution.
2.  **Simulates New Data:** Uses `scripts/simulation/simulate_and_save.py` to generate thousands of **new** diffraction patterns from the upsampled object. This is the key step.
3.  **Downsamples:** Processes the new high-resolution synthetic data back down to the target resolution, ensuring physical consistency.
4.  **Splits:** Creates final train and test sets from the new synthetic data.

---

## Generic Synthetic PyTorch Workflow

Use `ptycho_synthetic` for new synthetic PyTorch work. With no `--stages`
override it simulates flat train/test acquisitions, trains through the shared
generic workflow, strictly reloads the saved Torch bundle, reconstructs the
held-out scan through mmap-backed barycentric assembly, and evaluates raw
complex arrays.

```bash
# Complete default profile: N=128, GS1 CNN, ideal probe, 50 epochs
ptycho_synthetic \
  --profile synthetic-lines \
  --output-root outputs/synthetic_hybrid_resnet_gs1

# Sealed five-epoch CNN GS1/C1 quality recipe
ptycho_synthetic \
  --profile synthetic-lines \
  --output-root outputs/synthetic_hybrid_resnet_gs1_5ep_quality \
  --gridsize 1 \
  --epochs 5 \
  --batch-size 16 \
  --seed 3 \
  --probe-source custom \
  --probe-path datasets/Run1084_recon3_postPC_shrunk_3.npz \
  --probe-transform 'pad_extrapolate:128|smooth:0.5' \
  --train-patterns 4489 \
  --test-patterns 729 \
  --train-raw-selection 4489 \
  --training-groups 4489 \
  --validation-groups 729 \
  --neighbor-count 1 \
  --groups-per-center 1 \
  --accelerator cuda \
  --devices 1 \
  --precision 32-true \
  --workers 0 \
  --logger csv \
  --deterministic

# CI count-intensity profile with pre-fit rectangular gauge initialization
ptycho_synthetic \
  --profile cnn-lines-ci \
  --rect-s1s2-init dose_closure \
  --output-root outputs/synthetic_ci
```

The structured config equivalent accepts JSON, TOML, or YAML. For example,
`configs/synthetic_gs1.yaml` may contain:

```yaml
profile: synthetic-lines
simulation:
  gridsize: 1
  train_patterns: 4489
  test_patterns: 729
  probe:
    source: custom
    source_path: datasets/Run1084_recon3_postPC_shrunk_3.npz
    transform_pipeline: "pad_extrapolate:128|smooth:0.5"
training:
  epochs: 5
  train_raw_selection: 4489
  training_groups: 4489
  validation_groups: 729
  neighbor_count: 1
inference:
  groups_per_center: 1
workflow:
  output_root: outputs/synthetic_hybrid_resnet_gs1_5ep_quality
  accelerator: cuda
  devices: 1
  precision: 32-true
  num_workers: 0
  logger_backend: csv
```

CLI values override this file; file values override the profile. To split one
run across invocations while preserving its identity:

```bash
ptycho_synthetic --config configs/synthetic_gs1.yaml \
  --stages simulate,train
ptycho_synthetic --config configs/synthetic_gs1.yaml \
  --stages reconstruct,evaluate
```

The second command requires the first command's complete, matching
`resolved_workflow.json`, `stage_manifest.json`, datasets, and bundle under the
same output root. Configuration mismatches and partial stage artifacts fail
before expensive work.

### Synthetic Sampling Names

The generated NPZ stays flat regardless of gridsize; the shared loader forms
`C = gridsize ** 2` channels later.

| Flag | Meaning |
| --- | --- |
| `--train-patterns` / `--test-patterns` | Raw frames generated in each split |
| `--train-raw-selection` | Train frames selected before grouping; persisted as training `DataConfig.n_raw_frames_selected` |
| `--training-groups` | Exact grouped train samples |
| `--validation-groups` | Exact grouped validation samples, independently chosen from the complete test acquisition |
| `--groups-per-center` | Reconstruction-only repeated neighbor groups per valid center |

`--training-groups` and `--validation-groups` are independent. Reconstruction
starts from the strictly loaded persisted `DataConfig` and threads
`groups_per_center` to the dataset constructor as an explicit runtime argument
(no dataclass field round-trip); it never overwrites the saved training
selection.

The runner deliberately trains with `do_stitching=False`: generic stitching
reduces grouped predictions at their centers and cannot validate all GS2/C4
channels. Reconstruction instead starts from strict bundle reload and uses a
fresh mmap workspace, all channel coordinates, probe-weighted barycentric
assembly, and the profile's VarPro policy.

Key outputs are `datasets/{train,test}.npz`, `training/wts.h5.zip`,
`training/training_summary.json`,
`reconstruction/reconstruction.npz`, `reconstruction/metrics.json`,
`reconstruction/diagnostics.json`, and `reconstruction/comparison.png`, with
invocation, resolved-workflow, dataset, and stage manifests at the run root.
Fresh summaries record a strict `rect-s1s2-initialization-v2` startup result;
strict v1 reading remains for prefix-era history. The record is not the final
learned `s1`/`s2` or inference VarPro solve. Current
`synthetic-stage-manifest-v2` roots validate it on reuse; version-1 manifest
roots need a new output root or retraining. See the
[core contract](specs/spec-ptycho-core.md#ci-rectangular-gauge-initialization-normative)
for record and sampling semantics.

The one-epoch Hybrid GS1, GS2, and C4-CI preflights must pass before
calibration. Each quality contract has its own immutable fixture,
two-fit/one-holdout envelope, raw-array metrics, manual image reviews, pytest
ID, and output variable. Tests never rewrite a sealed fixture and skip outside
its recorded CUDA/software fingerprint. See the
[PyTorch workflow guide](workflows/pytorch.md#41-synthetic-generation-through-evaluation-recommended)
and `docs/TESTING_GUIDE.md` for policy and selectors. C4-CI has a sealed
five-epoch quality gate in the shared module using
`PTYCHO_C4_CI_OUTPUT_BASE`; only the CNN one-epoch C4 path remains operational
smoke coverage.

The live preflight nodes are
`test_synthetic_hybrid_resnet_gs1_one_epoch_preflight`,
`test_synthetic_hybrid_resnet_gs2_one_epoch_preflight`, and
`test_synthetic_hybrid_resnet_c4_ci_one_epoch_preflight`; the sealed C4-CI
node is `test_synthetic_hybrid_resnet_c4_ci_five_epoch_quality`.

---

## Training

```bash
# Basic training
ptycho_train --train_data_file dataset.npz --training_groups 2000 --nepochs 50 --output_dir my_run

# With configuration file (recommended)
ptycho_train --config configs/my_config.yaml

# Independent sampling control
ptycho_train --train_data_file dataset.npz --train_raw_selection 5000 --training_groups 1000 --nepochs 50 --output_dir my_run

# Reproducible sampling
ptycho_train --train_data_file dataset.npz --train_raw_selection 3000 --training_groups 500 --subsample_seed 42 --output_dir my_run

# Centered-nearest grouping: 2000 candidate rows, 500 exact groups
ptycho_train --train_data_file dataset.npz \
    --train_raw_selection 2000 --training_groups 500 \
    --gridsize 2 --neighbor_count 7 \
    --subsample_seed 42 --output_dir centered_run
```

Python and Jupyter callers use the Torch resolver's canonical setting names;
the output directory can have any name:

```python
from ptycho_torch.inference import reconstruct
from ptycho_torch.train import train

model = train("dataset.npz", "my_run", {
    "architecture": "ffno",
    "training_groups": 256,
    "gridsize": 2,  # C = gridsize**2 = 4
    "nphotons": 1e9,
    "epochs": 1,
})
result = reconstruct(model, "dataset.npz")
```

Use `help(train)` for common fields and
[`docs/CONFIGURATION.md`](CONFIGURATION.md#canonical-programmatic-torch-training-settings)
for the exact resolver-derived table. Unknown fields fail before output is
created and suggest the closest canonical spelling. `training_groups` is the
exact group count; `n_raw_frames_selected` is an optional candidate-frame cap,
and omission uses the full acquisition. The CI profile is the default. For a
metadata-free normalized-amplitude NPZ, author `nphotons`; omit it when the
metadata-free NPZ is already correctly count-scaled.

The native Torch entry point delegates to the same `train` function and
exposes the rectangular startup override:

```bash
# Omission keeps the ci profile's dose_closure default
python -m ptycho_torch.train \
  --profile ci \
  --train_data_file dataset.npz \
  --output_dir ci_run

# Explicit unit startup overrides the profile default
python -m ptycho_torch.train \
  --profile ci \
  --rect-s1s2-init ones \
  --train_data_file dataset.npz \
  --output_dir ci_ones_run
```

`python -m ptycho_torch.train` accepts
`--rect-s1s2-init {ones,dose_closure}` but not `--config`; omission does not
author an override. The installed `ptycho_train` command shown above is a
separate unified/legacy entry point and does accept its YAML `--config`.

### 📊 Independent Sampling Control

The project supports **independent control** of the candidate pool and the exact group count:

- **`--train_raw_selection`**: How many scan rows are loaded from the dataset as the candidate pool
- **`--training_groups`**: The exact number of groups to train on. Each group has one designated center drawn from the pool (first member, column zero), so this is also the number of unique centers and can never exceed the pool size
- **`--neighbor_count`**: K, the nearest non-center candidates considered per group; must be at least `C - 1` where `C = gridsize²`
- **`--subsample_seed`**: Ensures reproducible random selection

**Note:** `--n_images` / `--n_groups` (map to `--training_groups`) and `--n_subsample` (maps to `--train_raw_selection`) are deprecated aliases that still parse.

**Example Use Cases:**
```bash
# Dense grouping: every candidate row becomes a group center
ptycho_train --train_raw_selection 1200 --training_groups 1200 --gridsize 2 ...

# Sparse grouping: large candidate pool, fewer exact groups
ptycho_train --train_raw_selection 10000 --training_groups 500 --gridsize 2 ...

# Memory-constrained: bound the loaded pool
ptycho_train --train_raw_selection 5000 --training_groups 2000 --gridsize 1 ...
```

### ⚠️ CRITICAL: Understanding `gridsize` and `--training_groups`

The `--training_groups` parameter **always** refers to the number of groups (and therefore the number of unique centers), regardless of the `gridsize` parameter. This provides consistent behavior and eliminates confusion.

| GridSize | `--training_groups` Refers To... | Group-member slots / distinct rows | Grouping |
|----------|---------------------------|---------------------|--------------------|
| 1 | **Groups (each with 1 row)** | Exactly `training_groups` slots and distinct rows | Each group is its center alone. |
| > 1 | **Groups (neighbor groups)** | Exactly `training_groups × gridsize²` slots; at most the candidate-pool size in distinct rows | Each group is its designated center plus `gridsize² - 1` non-center rows selected from its K nearest candidates; groups may overlap. |

**Key Insight**: With `--training_groups`, the parameter always means "number of groups" regardless of gridsize. Training counts unique centers, not `groups × C` distinct images.

**Log Message Examples to Watch For:**
```
# GridSize=1 (independent control)
INFO - Independent sampling control: subsampling 2000 images, using 500 groups for training

# GridSize=2 (independent control)
INFO - Independent sampling control: subsampling 2000 images, creating 500 groups (approx 2000 patterns from groups)
```

**Backward Compatibility**: The deprecated `--n_images` / `--n_groups` / `--n_subsample` parameters still work but emit deprecation warnings.

### 🎯 Centered-Nearest Grouping (Exact Centers, No Oversampling)

The grouping policy is the shared backend-neutral centered-nearest planner (`centered-nearest-v1`): every group contains its designated center in column zero, K is the nearest non-center candidate pool (K ≥ C − 1), groups are object-partitioned, and the group count is the exact number of unique centers drawn from the candidate pool.

**Use case:** `training_groups` can never exceed the candidate pool size — more groups require a larger pool, not oversampling.

**Example:**
```bash
# 500 candidate rows -> exactly 500 groups max; use 400 groups here
ptycho_train --train_data_file dataset.npz \
    --train_raw_selection 500 \
    --training_groups 400 \
    --gridsize 2 \
    --neighbor_count 7 \
    --subsample_seed 42 \
    --output_dir centered_run
```

**Important Notes:**
- Requesting more groups than candidate rows fails with a clear error (`requested N unique centers from only M candidates`)
- `group_padding_step` sizes only the Torch canvas — it does not change grouping membership
- Inference repeats are contiguous per bounded center; see the [grouping guide](GRIDSIZE_N_GROUPS_GUIDE.md)
- The retired K-choose-C oversampling policy (`--enable_oversampling`, `--neighbor_pool_size`) is removed; migration diagnostics name the retired fields

---

## Inference

```bash
# Basic inference (uses all test data)
ptycho_inference --model_path trained_model/ --test_data test.npz --output_dir inference_out

# With specific number of test groups
ptycho_inference --model_path trained_model/ --test_data test.npz --inference_groups 500 --output_dir inference_out

# Independent sampling control
ptycho_inference --model_path trained_model/ --test_data test.npz --inference_raw_selection 2000 --inference_groups 500 --output_dir inference_out

# Inference for a model trained with GridSize=2
ptycho_inference --model_path gs2_model/ --test_data test.npz --inference_groups 125 --output_dir gs2_inference
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
python scripts/reconstruction/ptychi_reconstruct_tike.py \
    --input-npz input_data.npz \
    --output-dir ptychi_output/

# High-quality reconstruction with extended convergence
python scripts/reconstruction/ptychi_reconstruct_tike.py \
    --input-npz input_data.npz \
    --output-dir ptychi_lsqml_output/ \
    --algorithm LSQML \
    --num-epochs 500 \
    --n-images 2000

# Available algorithms include 'DM', 'LSQML', and 'PIE'.
```

---

## Single-Model Inference

```bash
# Basic single-model inference
ptycho_inference --model_path trained_model/ --test_data test.npz --output_dir infer_results

# Inference with sampling control
ptycho_inference --model_path trained_model/ --test_data test.npz \
    --inference_raw_selection 2000 --inference_groups 500 --output_dir infer_results

# Include comparison plot when ground truth is available
ptycho_inference --model_path model/ --test_data test.npz \
    --output_dir infer_with_plot --comparison_plot

# Select backend explicitly
ptycho_inference --model_path model/ --test_data test.npz \
    --output_dir infer_torch --backend pytorch
```

### 📊 Key Features

- **Reconstruction Outputs**: Generates amplitude/phase reconstructions and debug artifacts
- **Optional GT Plotting**: `--comparison_plot` renders ground-truth comparisons when available
- **Backend Selection**: Supports TensorFlow and PyTorch backends from the same entrypoint
- **Independent Sampling**: Select candidate test rows with `--inference_raw_selection` and bound reconstructed centers with `--inference_groups`

### 📋 When to Use Inference vs Comparison

- **Use `ptycho_inference`** when:
  - Running a single trained model to produce reconstructions
  - Doing backend-specific smoke checks
  - Creating per-model visuals quickly

- **Use `compare_models.py`** when:
  - Computing quantitative cross-model metrics (MAE/SSIM/MS-SSIM/FRC)
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
python scripts/studies/aggregate_and_plot_results.py study_results --output plots/generalization_plot.png
```

### Parameterized Study Composition

Use `ptycho_study` for multi-arm studies. Each arm supplies configuration to the
same public training service as `ptycho_synthetic` or `ptycho_train`; the study
layer owns comparison and collation, not an alternate trainer.

```bash
ptycho_study --help
```

### CNN Schematic Generator (TikZ + DOT)

Generate architecture schematics for `hybrid_resnet` directly from module execution with
shape capture. This writes source artifacts that are easy to diff and regenerate.

```bash
python scripts/studies/render_hybrid_resnet_schematics.py \
    --output-dir .artifacts/hybrid_resnet_schematics/latest \
    --N 128 \
    --gridsize 2 \
    --fno-width 32 \
    --fno-blocks 4 \
    --fno-modes 12
```

Expected artifacts:
- `.artifacts/hybrid_resnet_schematics/latest/hybrid_resnet_manifest.json`
- `.artifacts/hybrid_resnet_schematics/latest/hybrid_resnet_high_level.tex`
- `.artifacts/hybrid_resnet_schematics/latest/hybrid_resnet_module_flow.dot`

Optional rendering (if tools are installed):

```bash
pdflatex -interaction=nonstopmode \
    -output-directory .artifacts/hybrid_resnet_schematics/latest \
    .artifacts/hybrid_resnet_schematics/latest/hybrid_resnet_high_level.tex

dot -Tsvg .artifacts/hybrid_resnet_schematics/latest/hybrid_resnet_module_flow.dot \
    -o .artifacts/hybrid_resnet_schematics/latest/hybrid_resnet_module_flow.svg
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

For detailed explanations, see the <doc-ref type="guide">docs/DEVELOPER_GUIDE.md</doc-ref> and <doc-ref type="guide">docs/WORKFLOW_GUIDE.md</doc-ref>.
