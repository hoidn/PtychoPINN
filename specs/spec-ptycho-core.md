# Core Model Spec

## Inputs

- Diffraction patterns of size `N x N`.
- Scan coordinates and offsets for grouping.
- Probe and object guesses for simulation and evaluation.

## Outputs

- Reconstructed object amplitude and phase.
- Optional stitched reconstruction for visualization.

## References

- `ptycho/model.py`
- `ptycho/train_pinn.py`
