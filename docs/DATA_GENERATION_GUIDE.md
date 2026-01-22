# Data Generation Guide

## Sources

- Simulation helpers: `ptycho/diffsim.py`, `ptycho/generate_data.py`.
- CLI simulation: `scripts/simulation/simulation.py`.

## RawData

`ptycho/raw_data.py` provides `RawData` and `RawData.from_file()` for NPZ datasets.
Key fields are defined in `specs/data_contracts.md`.

## Typical Flow

1. Create or load probe and object guesses (NPZ).
2. Use simulation to generate diffraction patterns.
3. Save as NPZ with required keys.
