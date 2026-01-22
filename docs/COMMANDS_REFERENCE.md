# Commands Reference

## Install

`python -m pip install .`

## Training

`python scripts/training/train.py --train_data_file <train.npz> [--config config.yaml]`

## Inference

`python scripts/inference/inference.py --model_prefix <model_dir> --test_data <test.npz>`

## Simulation

`python scripts/simulation/simulation.py <probe_object.npz> <output_dir>`

## Tests

`pytest tests/test_generate_data.py`
