# Physics constrained machine learning for rapid, high resolution diffractive imaging

This repository contains the codebase for the methods presented in the paper "[Physics Constrained Unsupervised Deep Learning for Rapid, High Resolution Scanning Coherent Diffraction Reconstruction](https://www.nature.com/articles/s41598-023-48351-7)". 

## Overview
PtychoPINN is an unsupervised, physics-informed neural network method for scanning CDI reconstruction. It combines the diffraction forward model with real-space overlap constraints.

## Documentation

- [Commands reference](./docs/COMMANDS_REFERENCE.md)
- [Configuration guide](./docs/CONFIGURATION.md)
- [Data contracts](./docs/data_contracts.md)
- [PyTorch workflow](./docs/workflows/pytorch.md)

## Features
- **Unsupervised / self-supervised learning**: Does not require large labeled datasets.
- **Resolution**: Reported gains include about 10 dB PSNR and 3x to 6x improvement in linear resolution.
- **Speed**: Runs much faster than iterative scanning CDI reconstruction.

### Dual-Backend Architecture

PtychoPINN supports both TensorFlow and PyTorch backends:

- **Default Backend**: TensorFlow remains the default for backward compatibility.
- **PyTorch Backend**: PyTorch implementation is available via Lightning orchestration (`ptycho_torch/workflows/components.py`) with training, checkpointing, inference, and stitching.
- **Backend Selection**: Configure backend choice through `TrainingConfig.backend` or `InferenceConfig.backend` fields (`'tensorflow'` or `'pytorch'`). See the [PyTorch Workflow Guide](./docs/workflows/pytorch.md) for configuration details.

Both backends share public configuration contracts and acquisition schemas.
Backend- and entry-point-specific adapters select the actual in-memory, NPZ, or
mmap data route.

![Architecture diagram](diagram/lett.png)

## Installation
`conda create -n ptycho python=3.10`

`conda activate ptycho`

`pip install .`

**Note:** This installs the declared PyTorch dependency; supported PyTorch
workflows require PyTorch >= 2.2.
For GPU acceleration with a specific CUDA version, install PyTorch first using
the [official installation guide](https://pytorch.org/get-started/locally/),
then run `pip install .`. Public training configuration and CLI generation use
the declared `pydantic-settings` dependency.

## Usage

### Synthetic PyTorch workflow

For new synthetic PyTorch work, use the installed generic runner. The default
`synthetic-lines` profile retains the legacy normalized-amplitude workflow;
select `cnn-lines-ci` for the count-intensity CNN workflow:

```bash
ptycho_synthetic --output-root outputs/synthetic-cnn
ptycho_synthetic --profile cnn-lines-ci --output-root outputs/synthetic-cnn-ci
```

See the [simulation workflow guide](./scripts/simulation/README.md) for the
profile contracts, structured `--config` files, partial-stage replay, and
output artifacts. For existing count-intensity NPZs, the separate Torch
training-only `ci` profile is documented in the
[configuration guide](./docs/CONFIGURATION.md#torch-training-only-ci-profile).

### Training
`ptycho_train --data.train_data_file <train_path.npz> --data.test_data_file <test_path.npz> --output_dir <my_run>`

Use a nested YAML file for numeric and Boolean training settings on the current
`refactor` tip. The generated dotted flags are registered, but their numeric
and Boolean values are not yet decoded before strict Pydantic validation.

### Evaluation
Use the comparison and study scripts described in
[`scripts/studies/README.md`](./scripts/studies/README.md); there is no
`ptycho_evaluate` console command.

### Inference 
`ptycho_inference --model_path <my_run> --test_data <test_path.npz> --output_dir <inference_out>`

### Workflow Status

#### Use These by Default
- Generate synthetic data and run the complete PyTorch workflow with
  `ptycho_synthetic`.
- Train with `scripts/training/train.py` (or `ptycho_train`).
- Run inference with `scripts/inference/inference.py` (or `ptycho_inference`).
- Pick backend with `--backend tensorflow` or `--backend pytorch`.
- In unified training configuration, use `sampling.n_groups` for sample count
  and `sampling.n_subsample` for separate raw-row subsampling. Author numeric
  values in YAML for now. Native Torch CLIs retain their own flat flags.
- For PyTorch execution flags:
  - Unified scripts: use `--torch-accelerator` and `--torch-logger`
  - PyTorch-native CLIs: use `--accelerator` and `--logger`

#### Also Supported
- Grid-lines multi-model runs:
  - `scripts/studies/grid_lines_compare_wrapper.py`
- Grid-lines Torch runner:
  - `scripts/studies/grid_lines_torch_runner.py`
  - Architectures: `fno`, `fno_vanilla`

#### Older Flags and Modes
- Unified training's `sampling.n_images` field is deprecated; use
  `sampling.n_groups`.
- PyTorch `--device` and `--disable_mlflow` are older; use `--accelerator` and `--logger none`.
- MLflow-only inference mode in `ptycho_torch/inference.py` (`--run_id`, `--infer_dir`) is still available, but not the default path.

See examples and READMEs under scripts/.

For an example of interactive (Jupyter) usage, see notebooks/nongrid_simulations.ipynb. If you don't have inputs in the right .npz format you can simulate data.

non_grid_CDI_example.ipynb shows interactive usage using a dataset that is provided with the repo.

### Model Evaluation & Generalization Studies

Run generalization studies:
```bash
# Multi-trial study with uncertainty quantification
./scripts/studies/run_complete_generalization_study.sh \
    --train-sizes "512 1024 2048" \
    --num-trials 3 \
    --output-dir robust_study
```

See the [study README](scripts/studies/README.md) for detailed usage and options.
