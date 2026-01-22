# CLI Flags Quick Reference

## Training (`scripts/training/train.py`)

- `--train_data_file`: path to training NPZ (required)
- `--test_data_file`: optional test NPZ
- `--config`: YAML config path

## Inference (`scripts/inference/inference.py`)

- `--model_prefix`: model directory prefix (required)
- `--test_data`: test NPZ (required)
- `--output_path`: output directory prefix
- `--visualize_probe`: save probe visualization
- `--K`, `--nsamples`: grouping parameters

## Simulation (`scripts/simulation/simulation.py`)

- `--nimages`, `--seed`, `--nepochs`
- `--N`, `--gridsize`, `--nphotons`
- `--mae_weight`, `--nll_weight`
- `--train_data_file_path`, `--test_data_file_path`
