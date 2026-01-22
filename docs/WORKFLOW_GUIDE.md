# Workflow Guide

## Typical Flow

1. Prepare NPZ data (`ptycho/raw_data.py`, `scripts/simulation/simulation.py`).
2. Train a model (`scripts/training/train.py` or `ptycho/train.py`).
3. Run inference and inspect reconstructions (`scripts/inference/inference.py`).

## Training

- `python scripts/training/train.py --train_data_file <train.npz> [--config config.yaml]`
- Outputs: model artifacts under `training_outputs/` and logs.

## Inference

- `python scripts/inference/inference.py --model_prefix <model_dir> --test_data <test.npz>`
- Outputs: comparison image and logs.

## Simulation

- `python scripts/simulation/simulation.py <probe_object.npz> <output_dir> [options]`
