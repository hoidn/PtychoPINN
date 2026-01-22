# Architecture Overview

This repository is centered on a TensorFlow implementation of PtychoPINN.
Core components live under `ptycho/` with scripts in `scripts/`.

## Core Flow

1. Load or generate raw data (`ptycho/raw_data.py`, `ptycho/generate_data.py`).
2. Group and normalize diffraction patterns (`ptycho/loader.py`).
3. Build the model and physics losses (`ptycho/model.py`, `ptycho/losses.py`, `ptycho/physics.py`).
4. Train via the CLI or workflow helpers (`ptycho/train.py`, `scripts/training/train.py`).
5. Run inference and reconstruction (`ptycho/inference.py`, `scripts/inference/inference.py`).

## Key Modules

- `ptycho/raw_data.py`: RawData container and NPZ IO.
- `ptycho/loader.py`: Grouping and `PtychoDataContainer`.
- `ptycho/model.py`: Model definition and reconstruction heads.
- `ptycho/train_pinn.py`: PINN training loop and losses.
- `ptycho/model_manager.py`: Save/load models and params.
- `ptycho/workflows/components.py`: CLI workflow helpers used by scripts.

## Data Shapes

- Diffraction data is typically `(num_patterns, N, N)`.
- Grouped data encodes local and global offsets for stitching.
- Probe and object guesses are 2D arrays.

See `ptycho/loader_structure.md` for a diagram of the loader flow.
