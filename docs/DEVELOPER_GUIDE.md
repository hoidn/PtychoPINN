# Developer Guide

## Repository Layout

- `ptycho/`: Core TensorFlow implementation.
- `scripts/`: CLI workflows for training, inference, simulation.
- `tests/`: Test suite organized by topic.
- `datasets/`, `outputs/`, `training_outputs/`: data and results (large, usually untracked).

## Configuration

- Legacy config: `ptycho/params.py` (`params.cfg`).
- Dataclass config: `ptycho/config/config.py`.
- Bridge: `update_legacy_dict()` syncs dataclass values into `params.cfg`.

## Data Contracts

- Training and inference rely on NPZ files with defined keys.
- See `specs/data_contracts.md` for required keys and shapes.

## Conventions

- Keep TensorFlow dtype and device handling explicit in model code.
- Avoid writing large outputs into tracked directories.
- Prefer `scripts/` for CLI entrypoints and `ptycho/` for reusable logic.
