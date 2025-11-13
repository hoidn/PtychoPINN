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
- NPZ (keys, dtypes, semantics):
  - Required:
    • `xcoords (M,) float64` pixels, `ycoords (M,) float64` pixels
    • `diff3d (M,N,N) float32` amplitude (sqrt of photon counts)
    • `probeGuess (N,N) complex64`
  - Optional:
    • `scan_index (M,) int64` (defaults to zeros if missing)
    • `objectGuess (H,W) complex64`
    • `xcoords_start (M,)`, `ycoords_start (M,)` (default to `xcoords`, `ycoords` if missing)
- Grouped dict keys:
  - Canonical keys:
    • `diffraction (B,N,N,C) float32` amplitude
    • `X_full (B,N,N,C) float32` normalized amplitude
    • `coords_offsets (B,1,2,1) float32`, `coords_relative (B,1,2,C) float32`
    • `nn_indices (B,C) int32`
    • Optional `Y (B,N,N,C) complex64`
  - Coordinate axis order and mapping:
    • The third axis is `[x, y]` in that order.
    • Channel index c maps to `(row, col)` via row‑major: `row = c // gridsize`, `col = c % gridsize`.
  - Aliases:
    • Implementations MAY provide `coords_start_offsets`/`coords_start_relative` as legacy synonyms; loaders MUST accept either the canonical or legacy pair.
- Model I/O:
  - Inputs: `[diffraction (B,N,N,C) float32 · s, positions (B,1,2,C) float32]`.
  - Outputs: `[object (B,N,N,1) complex64, amplitude (B,N,N,C) float32, intensity (B,N,N,C) float32]`.

Reassembly Contract (Normative)
- Inference pipelines MUST stitch predicted patches with global offsets (pixel units). Offsets MUST be zero‑centered (e.g., relative to center‑of‑mass) prior to placement on the canvas.
- The stitched canvas MUST be large enough to accommodate all translated patches without clipping: `M ≥ N + 2·max(|dx|, |dy|)` for the used subset. Odd total padding MUST be distributed across borders without shifting the reconstruction’s center.
- Averaging patches in place without offset‑aware translation is prohibited.
- Forward‑path reassembly semantics MUST match across backends when `gridsize > 1`. The default is `object.big=True` (reassemble before diffraction), and the configuration bridge MUST carry this behavior between TensorFlow and PyTorch.

Loader Behavior (Normative)
- When ground truth `Y` is absent, loaders MAY emit a shape‑compatible placeholder `Y (complex64)`; training integrations MUST disable any MAE terms that require real ground truth. NLL and real‑space terms remain valid without `Y`.
- `coords_nominal` and `coords_true` SHALL be equal when jitter/true positions are not provided by the dataset.

Precedence and Configuration (Normative)
- `ptycho.params.cfg` (legacy) provides global defaults:
  - Keys: N, gridsize, offset, nphotons, batch_size, n_filters_scale, loss weights, jitter, probe mask, object/probe.big, etc.
  - Values set via `params.set(key, value)`; some models are created at import time and thus read these values on import.
- Environment overrides:
  - `USE_XLA_TRANSLATE`, `USE_XLA_COMPILE` as described in runtime spec.
- CLI (Legacy):
  - `ptycho/train.py` is deprecated and SHALL NOT be used for new development.

Errors (Normative)
- Missing required NPZ keys (listed above) SHALL error. Optional keys SHALL fall back as specified.
- Shape mismatches (channels, coordinates, N) SHALL error.
- Unsupported N SHALL error.

Notes (Informative)
- Prefer workflow orchestrations when available; legacy global config remains to support existing modules.
