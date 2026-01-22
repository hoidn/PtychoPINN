# Model Comparison Guide

This branch includes a simple comparison path in the simulation workflow.

- `scripts/simulation/simulation.py` runs a baseline reconstruction and compares
  against PtychoPINN output.
- `ptycho/nongrid_simulation.py` and `ptycho/nbutils.py` hold comparison helpers.

For more advanced comparisons, extend these helpers and record results under `results/`.
