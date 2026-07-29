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

Both backends share the same data pipeline and configuration system, ensuring consistent behavior across workflows.

![Architecture diagram](diagram/lett.png)

## Installation
`conda create -n ptycho python=3.10`

`conda activate ptycho`

`pip install .`

**Note:** This will automatically install PyTorch >= 2.2 as a required dependency. For GPU acceleration with specific CUDA versions, you may want to install PyTorch manually first following the [official PyTorch installation guide](https://pytorch.org/get-started/locally/), then run `pip install .`

## Usage
### Training
`ptycho_train --train_data_file <train_path.npz> --test_data_file <test_path.npz> --output_dir <my_run>`

### Evaluation
`ptycho_evaluate --model-dir <my_run> --test-data <test_path.npz> --output-dir <eval_results>`

### Inference 
`ptycho_inference --model_path <my_run> --test_data <test_path.npz> --output_dir <inference_out>`

### Workflow Status

#### Use These by Default
- Train with `scripts/training/train.py` (or `ptycho_train`).
- Run inference with `scripts/inference/inference.py` (or `ptycho_inference`).
- Pick backend with `--backend tensorflow` or `--backend pytorch`.
- Use `--n_groups` for sample count. Add `--n_subsample` only when you want separate subsampling control.
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
- `--n_images` in training is older; use `--n_groups`.
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
