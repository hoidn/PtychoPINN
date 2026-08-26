# Physics constrained machine learning for rapid, high resolution diffractive imaging

This repository contains the codebase for the methods presented in the paper "[Physics Constrained Unsupervised Deep Learning for Rapid, High Resolution Scanning Coherent Diffraction Reconstruction](https://www.nature.com/articles/s41598-023-48351-7)".

## Overview
PtychoPINN is an unsupervised, physics-informed neural network method for scanning CDI reconstruction. It combines the diffraction forward model with real-space overlap constraints.

## Features
- **Unsupervised / self-supervised learning**: Does not require large labeled datasets.
- **Resolution**: Reported gains include about 10 dB PSNR and 3x to 6x improvement in linear resolution.
- **Speed**: Runs much faster than iterative scanning CDI reconstruction.

### Dual-Backend Architecture

PtychoPINN supports both TensorFlow and PyTorch backends:

- **Default Backend**: TensorFlow remains the default for backward compatibility.
- **PyTorch Backend**: Lightning-based implementation under `ptycho_torch/` with training, checkpointing, inference, and stitching. Model architectures (CNN, FNO, FFNO, hybrid variants, and more) are selected through a generator registry.
- **Backend Selection**: Configure through `TrainingConfig.backend` / `InferenceConfig.backend` (`'tensorflow'` or `'pytorch'`), or use the PyTorch-native CLIs directly (`python -m ptycho_torch.train`, `python -m ptycho_torch.inference`). See the [PyTorch Workflow Guide](./docs/workflows/pytorch.md) for configuration details.

Both backends share core configuration and data contracts; backend- and
measurement-specific paths validate their resolved contract explicitly.

![Architecture diagram](diagram/lett.png)

## Documentation

- **[Documentation hub](./docs/index.md)** — complete map of guides, specs, and workflows.
- **[Developer Guide](./docs/DEVELOPER_GUIDE.md)** — architecture, data flow, and development conventions.
- **[Commands Reference](./docs/COMMANDS_REFERENCE.md)** — CLI recipes for training, inference, evaluation, and tests.

## Installation

```bash
conda create -n ptycho python=3.11
conda activate ptycho
pip install .
```

**Note:** This will automatically install PyTorch >= 2.2 as a required dependency. For GPU acceleration with specific CUDA versions, you may want to install PyTorch manually first following the [official PyTorch installation guide](https://pytorch.org/get-started/locally/), then run `pip install .`

## Usage

### Synthetic PyTorch workflow

For new synthetic PyTorch work, use the installed generic runner. With no
scientific overrides it resolves the legacy normalized-amplitude
`hybrid-resnet-lines` profile and runs simulation, training, strict
reload/reconstruction, and evaluation. Select
`--profile hybrid-resnet-lines-ci` for the coherent count-intensity Poisson
Hybrid ResNet workflow.

```bash
ptycho_synthetic --output-root outputs/synthetic-hybrid-resnet
```

The default is a full 50-epoch run. A grid-size-2 custom-probe run can select
the simulation and grouping controls explicitly:

```bash
ptycho_synthetic \
  --output-root outputs/synthetic-hybrid-resnet-gs2 \
  --gridsize 2 \
  --probe-source custom \
  --probe-path datasets/custom_probe.npz \
  --probe-transform 'pad_extrapolate:128|smooth:0.5' \
  --train-patterns 4096 \
  --test-patterns 1024 \
  --train-raw-selection 4096 \
  --training-groups 4096 \
  --validation-groups 1024
```

See the [simulation workflow guide](./scripts/simulation/README.md) for profile
semantics and defaults, structured `--config` files, partial-stage replay,
output artifacts, and the distinction between flat raw acquisitions and grouped
model samples. For existing count-intensity NPZs, the separate Torch
training-only `ci` profile is documented in the
[configuration guide](./docs/CONFIGURATION.md#torch-training-only-ci-profile).

### Training
`ptycho_train --train_data_file <train_path.npz> --test_data_file <test_path.npz> --output_dir <my_run>`

### Inference
`ptycho_inference --model_path <my_run> --test_data <test_path.npz> --output_dir <inference_out>`

### Evaluation & model comparison
Compare a trained PINN against the supervised baseline with
`python scripts/compare_models.py` or the end-to-end wrapper
`./scripts/run_comparison.sh train.npz test.npz output_dir`.
See the [Commands Reference](./docs/COMMANDS_REFERENCE.md) for full recipes.

### Data
Input datasets are NPZ files following the format defined in
[docs/specs/spec-ptycho-core.md](./docs/specs/spec-ptycho-core.md). If you don't have
data in that format, you can simulate it — see the
[Data Generation Guide](./docs/DATA_GENERATION_GUIDE.md).

### Workflow Status

#### Use These by Default
- Generate synthetic data and run the complete PyTorch workflow with
  `ptycho_synthetic`.
- Train with `scripts/training/train.py` (or `ptycho_train`).
- Run inference with `scripts/inference/inference.py` (or `ptycho_inference`) —
  TensorFlow-only. For PyTorch inference, use `python -m ptycho_torch.inference`.
- Pick the training backend with `--backend tensorflow` or `--backend pytorch`.
- Use `--training_groups` for the group count (training) / `--inference_groups` (inference). Add `--train_raw_selection` only when you want separate raw-frame selection. The older `--n_groups` / `--n_subsample` flags still parse as deprecated aliases.
- For PyTorch execution flags:
  - Unified scripts: use `--torch-accelerator` and `--torch-logger`
  - PyTorch-native CLIs: use `--accelerator` and `--logger`

#### Multi-Run Studies

- Compose parameterized study arms with `ptycho_study`; each arm delegates to
  the same public training service used by `ptycho_synthetic` and `ptycho_train`.
- Keep historical result tables and completed plans as provenance, not as
  executable runner APIs.

#### Older Flags and Modes
- `--n_images` in training is older; use `--training_groups` (deprecated `--n_groups` also parses).
- PyTorch `--device` and `--disable_mlflow` are older; use `--accelerator` and `--logger none`.

See examples and READMEs under `scripts/`.

### Notebooks
For interactive (Jupyter) usage, see `notebooks/nongrid_simulations.ipynb` (simulating
data) and `notebooks/non_grid_CDI_example.ipynb` (reconstruction on a dataset provided
with the repo).

### Model Evaluation & Generalization Studies

Run generalization studies:
```bash
# Multi-trial study with uncertainty quantification
./scripts/studies/run_complete_generalization_study.sh \
    --train-sizes "512 1024 2048" \
    --num-trials 3 \
    --output-dir robust_study
```

See `scripts/studies/README.md` for detailed usage and options.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{Hoidn2023,
  author  = {Hoidn, Oliver and Mishra, Aashwin Ananda and Mehta, Apurva},
  title   = {Physics constrained unsupervised deep learning for rapid, high resolution scanning coherent diffraction reconstruction},
  journal = {Scientific Reports},
  volume  = {13},
  pages   = {22789},
  year    = {2023},
  doi     = {10.1038/s41598-023-48351-7}
}
```

## License

This project is licensed under the GNU General Public License v3.0 — see the
[LICENSE](./LICENSE) file for details.
