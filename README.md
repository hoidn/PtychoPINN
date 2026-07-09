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

Both backends share the same data pipeline and configuration system, ensuring consistent behavior across workflows.

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
  - Architectures come from the torch generator registry, including `cnn`, `fno`, `ffno`, `fno_vanilla`, `hybrid`, `stable_hybrid`, and `neuralop_uno`.

#### Older Flags and Modes
- `--n_images` in training is older; use `--n_groups`.
- PyTorch `--device` and `--disable_mlflow` are older; use `--accelerator` and `--logger none`.
- MLflow-only inference mode in `ptycho_torch/inference.py` (`--run_id`, `--infer_dir`) is still available, but not the default path.

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
