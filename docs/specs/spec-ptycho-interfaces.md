# spec-ptycho-interfaces.md — Public API and Data Interfaces (Normative)

Overview (Normative)
- Purpose: Define primary programmatic entry points, tensor/data shape contracts, and precedence rules for configuration.

API Surface (Normative)
- Data Ingestion:
  - `RawData.from_file(path) -> RawData` — Load NPZ to RawData with validation.
  - `RawData.generate_grouped_data(N, K, nsamples, gridsize=..., ...) -> dict` — Group and normalize diffraction and coordinates.
- Tensor Conversion:
  - `loader.load(cb, probeGuess, which, create_split) -> PtychoDataContainer`
    • `cb()` returns the grouped dict; `which` ∈ {`'train'`, `'test'`}; `probeGuess` complex (N,N).
  - `PtychoDataContainer` fields:
    • `X (B,N,N,C) float32`, `Y_I (B,N,N,C) float32`, `Y_phi (B,N,N,C) float32`, `Y (B,N,N,C) complex64?`, `coords_nominal/true (B,1,2,C) float32`, `probe (N,N) complex64`.
- Model and Training:
  - `ptycho.model.autoencoder` — training model with outputs `[object, amplitude, intensity]`.
  - `ptycho.model.diffraction_to_obj` — inference model `[object]`.
  - `train_pinn.train(train_data, intensity_scale=None, model_instance=None)` — Setup probe, scale, compile and fit.
  - `train_pinn.eval(test_data, history=None, trained_model=None, model_path=None)` — Inference + optional stitching.
- Image Post‑processing:
  - `image.stitching.reassemble_patches(patches, config, part='amp'|'phase'|'complex', norm_Y_I=..., norm=...)` — CPU numpy reassembly.

Data Formats (Normative)
- NPZ:
  - Required: `xcoords (M,)`, `ycoords (M,)`, `diff3d (M,N,N)` amplitude, `probeGuess (N,N)` complex, `scan_index (M,)`.
  - Optional: `objectGuess (H,W)` complex, `xcoords_start (M,)`, `ycoords_start (M,)`.
- Grouped dict keys:
  - `diffraction (B,N,N,C)`, `X_full (B,N,N,C)`, `coords_offsets (B,1,2,1)`, `coords_relative (B,1,2,C)`, `nn_indices (B,C)`, optional `Y (B,N,N,C) complex`.
- Model I/O:
  - Inputs: `[diffraction (B,N,N,C) float32 · s, positions (B,1,2,C) float32]`.
  - Outputs: `[object (B,N,N,1) complex64, amplitude (B,N,N,C) float32, intensity (B,N,N,C) float32]`.

Precedence and Configuration (Normative)
- `ptycho.params.cfg` (legacy) provides global defaults:
  - Keys: N, gridsize, offset, nphotons, batch_size, n_filters_scale, loss weights, jitter, probe mask, object/probe.big, etc.
  - Values set via `params.set(key, value)`; some models are created at import time and thus read these values on import.
- Environment overrides:
  - `USE_XLA_TRANSLATE`, `USE_XLA_COMPILE` as described in runtime spec.
- CLI (Legacy):
  - `ptycho/train.py` is deprecated and SHALL NOT be used for new development.

Errors (Normative)
- Missing required NPZ keys SHALL error.
- Shape mismatches (channels, coordinates, N) SHALL error.
- Unsupported N SHALL error.

Notes (Informative)
- Prefer workflow orchestrations when available; legacy global config remains to support existing modules.

