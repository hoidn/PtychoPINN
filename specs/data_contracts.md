# Data Contracts

This spec defines the NPZ keys used by the TensorFlow pipeline.

## RawData NPZ

Required keys:
- `xcoords`, `ycoords`
- `xcoords_start`, `ycoords_start` (deprecated but present)
- `diff3d` (shape: num_patterns x N x N)
- `probeGuess`
- `scan_index`
Optional:
- `objectGuess`

## PtychoDataContainer NPZ

Produced by `PtychoDataContainer.to_npz()`:
- `X`, `Y_I`, `Y_phi`, `norm_Y_I`, `YY_full`
- `coords_nominal`, `coords_true`
- `nn_indices`, `global_offsets`, `local_offsets`
- `probe`
