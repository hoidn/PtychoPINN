# spec-ptycho-core.md — Core Physics and Data Contracts (Normative)

Overview (Normative)
- Purpose: Define the precise ptychographic forward model, all mathematical operations implemented, normalization rules, coordinate semantics, and strict data contracts for PtychoPINN.
- Scope: Normative definition of the ptychographic physics, coordinate systems, and data contracts that every backend (TensorFlow, PyTorch, or future engines) MUST satisfy. TensorFlow helpers cited throughout serve as the reference implementation for the math described here.

Units, Conventions, and Frames (Normative)
- Pixels and coordinates: All image‑plane arrays are sampled on square pixel grids; offsets and scan coordinates are in pixel units on the object sampling grid.
- Complex fields: Complex64 tensors represent object/probe fields; amplitude and phase are float32 tensors.
- Diffraction domain: Far‑field amplitude is sqrt of intensity; intensity represents expected photon counts per pixel.
- Scaling profile: PyTorch rectangular workflows SHALL identify their measurement units with the inseparable pair `scale_contract_version` and `measurement_domain`. The supported pairs are `ci_intensity_v2`/`count_intensity` and `legacy_v1`/`normalized_amplitude`.
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
  - Reference implementation: `ptycho.tf_helper.combine_complex` and `ptycho.diffsim` diffraction simulation helpers.
  - The returned “amplitude” is sqrt of the centered intensity (after `fftshift`).

- Photon observation model:
  - Observed amplitude is sampled as `A_obs(u,v) = sqrt(Poisson(A(u,v)²))`.
  - Reference implementation: `ptycho.diffsim`.

- Legacy intensity scale (dataset‑level normalization):
  - Let `S = E_batch[Σ_xy |Ψ|²]`. Given target expected photon count `nphotons > 0`, the dataset‑level scale factor is `s = sqrt(nphotons / S)`.
  - In simulation, both input diffraction (`X`) and target amplitude (`Y_I`) SHALL be divided by `s` to keep symmetry; phase is scale‑invariant. The implementation asserts this symmetry.
  - Reference implementation: `ptycho.diffsim`.

- CI count-intensity profile (`ci_intensity_v2`/`count_intensity`):
  - The measured tensor is physical detector count intensity `I_meas >= 0`, not diffraction amplitude.
  - A calibrated physical probe `P_physical` fixes the probe/object gauge. Training MAY use `P_training = q * P_physical` for conditioning, but the forward field SHALL compensate exactly by `1/q`.
  - For incoherent modes `p`, predicted detector intensity is `I_pred = sum_p |FFT2(M * P_physical,p * O)|^2` with the same orthonormal, shifted Fourier convention and probe mask `M` used by training and inference.
  - The rectangular object is `O = s1 * a_tilde + i * s2 * b_tilde`; `a_tilde` and `b_tilde` are normalized network textures and `s1`, `s2` carry object contrast in the fixed physical-probe gauge.
  - Missing profile fields default to this CI pair for new rectangular workflows. Historical behavior requires both explicit legacy fields.

Coordinate Semantics and Patch Extraction (Normative)
- Grouping constructs `C = gridsize²` overlapping patches per solution region.
- Offsets:
  - `coords_offsets`: global center per group, shape `(B,1,2,1)`.
  - `coords_relative`: local offsets relative to the global center, shape `(B,1,2,C)`.
  - Combined offsets used for sampling: `offsets_xy = coords_offsets + coords_relative` (channel format).
  - Sign convention for local offsets is enforced by `local_offset_sign = −1`.
- Extraction (forward sampling for ground truth patches):
  - From a single padded object canvas (padding ≥ N/2), extract per‑position patches by bilinear translation followed by cropping to `N×N`.
  - Reference implementation: `ptycho.raw_data.RawData` patch-generation methods and `ptycho.tf_helper`.
- Reassembly (inverse, for stitching or losses):
  - Shift‑and‑sum with normalization by overlap count (sum of ones passed through the same transform) to reconstruct the canvas region; supports a batched vectorized path and a streaming fallback to avoid OOM.
  - Reference implementation: `ptycho.tf_helper` reassembly helpers.

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
- Legacy amplitude-profile Poisson negative log‑likelihood:
  - Labels: `Y_true = (s · X)^2` (counts); Prediction: `Y_pred = (s · Â)^2` with `Â` predicted amplitude.
  - Loss per pixel: `L_poisson = Y_pred − Y_true · log(Y_pred)` (TensorFlow log‑Poisson form). Implementations SHALL ensure `Y_pred > 0` prior to `log`. Compliance MAY be achieved by (a) using a strictly‑positive amplitude activation (e.g., sigmoid/softplus) or (b) adding a small epsilon before log with `ε ≥ 1e−12`.
  - Code: `ptycho/model.py` (negloglik).
- CI count-intensity Poisson negative log-likelihood:
  - CI scaling is valid only for unsupervised `rectangular_scaled` training with `torch_loss_mode='poisson'`. CI with MAE SHALL fail before data or model construction.
  - The rate is clamped to at least `1e-8`; observations SHALL be finite and nonnegative. The raw count NLL is accumulated stably and exposed as a metric.
  - The optimization data term is the per-sample raw count NLL divided by the frozen training-set mean measured intensity. Auxiliary real-space or overlap regularizers are added after this normalization and SHALL NOT be divided by the count mean.
- Real‑space terms (optional):
  - Total variation on complex object (sum of squared finite differences over real+imag components).
  - Complex MAE on object amplitude/phase (optionally masked in center for non‑big probe configurations).
  - Code: `ptycho/tf_helper.py` (total_variation_complex, realspace_loss, complex_mae).

Normalization Invariants (Normative)
- The following invariant applies only to TensorFlow and explicit `legacy_v1`/`normalized_amplitude` workflows. Dataset‑level `intensity_scale` `s` is a learned or fixed parameter used symmetrically. Two compliant calculation modes are allowed, with the following precedence:
  1) Dataset‑derived mode (preferred): `s = sqrt(nphotons / E_batch[Σ_xy |Ψ|²])` computed from illuminated objects over the dataset.
  2) Closed‑form fallback: `s ≈ sqrt(nphotons) / (N/2)` when dataset statistics are unavailable at runtime.
  In both modes symmetry SHALL hold:
  - Training inputs: `X_scaled = s · X`.
  - Labels: `Y_amp_scaled = s · X` (amplitude), `Y_int = (s · X)^2` (intensity).
  - Model output amplitude `Â` is appropriately inverse‑scaled for intensity; symmetry SHALL be preserved (asserted in simulation paths).
- For `ci_intensity_v2`:
  - `rms_input_scale = sqrt((N/2)^2 / mean_BC(sum_HW(I_meas^2)))`, derived only from the finalized training split.
  - `mean_measured_intensity = mean_BCHW(I_meas)`, also derived only from the finalized training split.
  - Both statistics are one immutable scalar per experiment, persisted in checkpoints/bundles and reused unchanged by validation and inference.
  - The raw calibrated probe is used for inference and VarPro. Neither `physics_scaling_constant` nor a training `output_scale` participates in CI inference.
  - Across dose changes where counts and the calibrated probe scale consistently, recovered object contrast SHALL remain invariant. Holding the probe fixed is a different gauge and produces square-root dose scaling.

Data Contracts (Normative)
- Scaling metadata for new rectangular datasets:
  - `scale_contract_version='ci_intensity_v2'`, `measurement_domain='count_intensity'` for CI.
  - `scale_contract_version='legacy_v1'`, `measurement_domain='normalized_amplitude'` for historical reproduction.
  - Partial, contradictory, or unknown pairs SHALL error. Metadata-free provenance-known legacy checkpoints SHALL require both explicit legacy overrides.
- Raw NPZ (RawData.from_file):
  - Required keys and dtypes:
    • `xcoords (M,) float64` pixels, `ycoords (M,) float64` pixels
    • `diff3d (M, N, N) float32` amplitude (sqrt of counts)
    • `probeGuess (N, N) complex64`
  - Optional keys and semantics:
    • `scan_index (M,) int64` (defaults to zeros if missing)
    • `objectGuess (H, W) complex64`
    • `xcoords_start (M,)`, `ycoords_start (M,)` (default to `xcoords`, `ycoords` if missing)
  - CI PyTorch ingestion additionally accepts measured count intensity under `diff3d` or `diffraction`, plus a physical probe in `(N,N)`, `(P,N,N)`, or legacy `(N,N,1)` layout. Probe layouts SHALL be canonicalized to an explicit mode axis before channel or batch expansion.
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
- Multi‑slice propagation, continuous partial-coherence models beyond a finite incoherent probe-mode sum, and rectangular pixels are out of scope for this version.
- Advanced priors beyond TV/MAE are not modeled in the current architecture.
