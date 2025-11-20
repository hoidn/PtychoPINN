# spec-ptycho-core.md — Core Physics and Data Contracts (Normative)

Overview (Normative)
- Purpose: Define the precise ptychographic forward model, all mathematical operations implemented, normalization rules, coordinate semantics, and strict data contracts for PtychoPINN.
- Scope: Normative definition of the ptychographic physics, coordinate systems, and data contracts that every backend (TensorFlow, PyTorch, or future engines) MUST satisfy. TensorFlow helpers cited throughout serve as the reference implementation for the math described here.

Units, Conventions, and Frames (Normative)
- Pixels and coordinates: All image‑plane arrays are sampled on square pixel grids; offsets and scan coordinates are in pixel units on the object sampling grid.
- Complex fields: Complex64 tensors represent object/probe fields; amplitude and phase are float32 tensors.
- Diffraction domain: Far‑field amplitude is sqrt of intensity; intensity represents expected photon counts per pixel.
- Phase range and wrapping: Phase values SHALL be represented on [-π, π] with standard wrapping; evaluation MAY apply mean or plane alignment but SHALL NOT change the underlying contract.
- Tensor ordering:
  - Full images and patches: `(batch, H, W, C)`; channel dimension C = gridsize² for grouped patches.
  - Grid format: `(batch, gridsize, gridsize, N, N, 1)`.
  - Flat format: `(batch × gridsize², N, N, 1)`.
  - Channel index mapping: Channel index c in [0..C−1] SHALL map to grid coordinates `(row, col)` via row‑major order: `row = c // gridsize`, `col = c % gridsize`. Implementations MUST preserve this mapping across all format transformations.

Ptychographic Forward Model (Normative)
Let the object patch be represented by amplitude `Y_I(x,y)` and phase `Y_φ(x,y)`, probe `P(x,y)`, and object field `O(x,y) = Y_I(x,y) · exp(i · Y_φ(x,y))`. Probe illumination yields `Ψ(x,y) = P(x,y) · O(x,y)`.

- Fourier transform and amplitude:
  - Implementation uses a symmetric FFT with energy normalization:
    - `F(u,v) = FFT2{Ψ}(u,v)`
    - `I(u,v) = |F(u,v)|² / (N·N)`
    - `A(u,v) = sqrt(fftshift(I(u,v)))`
  - Code: `ptycho/tf_helper.py:340`, `ptycho/diffsim.py:64`.
  - The returned “amplitude” is sqrt of the centered intensity (after `fftshift`).

- Photon observation model:
  - Observed amplitude is sampled as `A_obs(u,v) = sqrt(Poisson(A(u,v)²))`.
  - Code: `ptycho/diffsim.py:32`.

- Intensity scale (dataset‑level normalization):
  - Let `S = E_batch[Σ_xy |Ψ|²]`. Given target expected photon count `nphotons > 0`, the dataset‑level scale factor is `s = sqrt(nphotons / S)`.
  - In simulation, both input diffraction (`X`) and target amplitude (`Y_I`) SHALL be divided by `s` to keep symmetry; phase is scale‑invariant. The implementation asserts this symmetry.
  - Code: `ptycho/diffsim.py:44–60`, `ptycho/diffsim.py:82–140`.

Coordinate Semantics and Patch Extraction (Normative)
- Grouping constructs `C = gridsize²` overlapping patches per solution region.
- Offsets:
  - `coords_offsets`: global center per group, shape `(B,1,2,1)`.
  - `coords_relative`: local offsets relative to the global center, shape `(B,1,2,C)`.
  - Combined offsets used for sampling: `offsets_xy = coords_offsets + coords_relative` (channel format).
  - Sign convention for local offsets is enforced by `local_offset_sign = −1`.
- Extraction (forward sampling for ground truth patches):
  - From a single padded object canvas (padding ≥ N/2), extract per‑position patches by bilinear translation followed by cropping to `N×N`.
  - Code: `ptycho/raw_data.py:873`, `ptycho/tf_helper.py:600`.
- Reassembly (inverse, for stitching or losses):
  - Shift‑and‑sum with normalization by overlap count (sum of ones passed through the same transform) to reconstruct the canvas region; supports a batched vectorized path and a streaming fallback to avoid OOM.
  - Code: `ptycho/tf_helper.py:1040–1184`, `ptycho/tf_helper.py:1200–1258`.

Tensor Format Transformations (Normative)
- Flat ↔ Grid:
  - `_togrid`: `(B×C, N, N, 1) → (B, gr, gr, N, N, 1)`
  - `_fromgrid`: inverse.
- Grid ↔ Channel:
  - `_grid_to_channel`: `(B, gr, gr, N, N, 1) → (B, N, N, C)`
  - `_channel_to_flat`: `(B, N, N, C) → (B×C, N, N, 1)`
- Transformations SHALL preserve pixel ordering and spatial adjacencies. In particular, the row‑major channel→(row,col) mapping above is normative and MUST be preserved by all transforms.

Probe, Masking, and Smoothing (Normative)
- Probe `P(x,y)`:
  - Default disk‑like probe from low‑pass mask; optionally learned (trainable) and masked.
  - Mask `M_probe` is a circular support of radius `N/4`; if enabled (`probe.mask`=True), effective probe is `P·M_probe`.
  - Code: `ptycho/probe.py`; `ptycho/model.py` (ProbeIllumination layer).
- Optional complex Gaussian smoothing of illuminated field with configurable `sigma` (0 disables; default 0).
- In `probe.big` mode, per‑channel probe MAY be used; otherwise a shared probe.

Model Sizes and Valid Inputs (Normative)
- `N` (patch size) SHALL be one of {64, 128, 256}. Encoder/decoder filter ladders are defined per `N`.
- `gridsize (≥1)` and `offset (>0)` SHALL be integers; `C = gridsize²`.
- Coordinates SHALL be float32 tensors with shapes:
  - `coords_offsets`: `(B,1,2,1)`; `coords_relative`: `(B,1,2,C)`.
  - Combined offsets in channel format `(B,1,2,C)`; flattened to `(B×C, 2)` for translation.

Losses and Regularization (Normative)
- Poisson negative log‑likelihood (primary):
  - Labels: `Y_true = (s · X)^2` (counts); Prediction: `Y_pred = (s · Â)^2` with `Â` predicted amplitude.
  - Loss per pixel: `L_poisson = Y_pred − Y_true · log(Y_pred)` (TensorFlow log‑Poisson form). Implementations SHALL ensure `Y_pred > 0` prior to `log`. Compliance MAY be achieved by (a) using a strictly‑positive amplitude activation (e.g., sigmoid/softplus) or (b) adding a small epsilon before log with `ε ≥ 1e−12`.
  - Code: `ptycho/model.py` (negloglik).
- Real‑space terms (optional):
  - Total variation on complex object (sum of squared finite differences over real+imag components).
  - Complex MAE on object amplitude/phase (optionally masked in center for non‑big probe configurations).
  - Code: `ptycho/tf_helper.py` (total_variation_complex, realspace_loss, complex_mae).

Normalization Invariants (Normative)
- Dataset‑level `intensity_scale` `s` is a learned or fixed parameter used symmetrically. Two compliant calculation modes are allowed, with the following precedence:
  1) Dataset‑derived mode (preferred): `s = sqrt(nphotons / E_batch[Σ_xy |Ψ|²])` computed from illuminated objects over the dataset.
  2) Closed‑form fallback: `s ≈ sqrt(nphotons) / (N/2)` when dataset statistics are unavailable at runtime.
  In both modes symmetry SHALL hold:
  - Training inputs: `X_scaled = s · X`.
  - Labels: `Y_amp_scaled = s · X` (amplitude), `Y_int = (s · X)^2` (intensity).
  - Model output amplitude `Â` is appropriately inverse‑scaled for intensity; symmetry SHALL be preserved (asserted in simulation paths).

Data Contracts (Normative)
- Raw NPZ (RawData.from_file):
  - Required keys and dtypes:
    • `xcoords (M,) float64` pixels, `ycoords (M,) float64` pixels
    • `diff3d (M, N, N) float32` amplitude (sqrt of counts)
    • `probeGuess (N, N) complex64`
  - Optional keys and semantics:
    • `scan_index (M,) int64` (defaults to zeros if missing)
    • `objectGuess (H, W) complex64`
    • `xcoords_start (M,)`, `ycoords_start (M,)` (default to `xcoords`, `ycoords` if missing)
- Grouped dataset (RawData.generate_grouped_data → dict):
  - `diffraction (B, N, N, C)` amplitude, `X_full` normalized amplitude (same shape).
  - `coords_offsets (B,1,2,1)`, `coords_relative (B,1,2,C)`, `nn_indices (B,C)`.
  - `Y (optional)` ground‑truth patches, same shape as diffraction but complex.
- Loader (PtychoDataContainer):
  - `X: tf.float32 (B,N,N,C)`, `Y_I: tf.float32`, `Y_φ: tf.float32`, `Y: tf.complex64 (optional)`.
  - `coords_nominal/true: tf.float32 (B,1,2,C)`. `probe: complex64 (N,N)`.
  - Intensity scale recorded in params; used in I/O preparation.

Outputs (Normative)
- Training model (`autoencoder`) yields:
  - `trimmed_obj (B,N,N,1)` complex64 — reconstructed object (center crop).
  - `pred_amp_scaled (B,N,N,C)` float32 — predicted amplitude (scaled domain, pre‑squaring).
  - `pred_intensity (B,N,N,C)` float32 — squared amplitude (counts).
- Inference model (`diffraction_to_obj`) yields:
  - `object (B,N,N,1)` complex64 only.
- Post‑stitching (numpy):
  - Reassembled object images via `image.stitching.reassemble_patches`.

Error Tolerances (Normative)
- Equality vs tolerance: forward amplitude equivalence assessed with max relative error ≤ 1e−6; shapes and index mapping MUST match exactly; counts/intensity positivity MUST hold strictly (after epsilon, if used).

Non‑Goals (Informative)
- Multi‑slice propagation, partial coherence, and rectangular pixels are out of scope for this version.
- Advanced priors beyond TV/MAE are not modeled in the current architecture.
