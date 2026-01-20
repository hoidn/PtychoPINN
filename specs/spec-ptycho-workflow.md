# spec-ptycho-workflow.md — End‑to‑End Workflow (Normative)

Overview (Normative)
- Purpose: Define the end‑to‑end workflow from data ingestion through grouping, normalization, physics‑informed model training, inference, stitching, and evaluation.

Pipeline (Normative)
1) Ingest Raw Data (NPZ → RawData)
   - Load NPZ fields: `xcoords, ycoords, diff3d, probeGuess, [objectGuess], [xcoords_start], [ycoords_start], [scan_index]`.
   - Construct `RawData` with shape validation (vectors for coords, 2D for probe/object; `diff3d` is (M,N,N) amplitude).
   - Optional: metadata‑validated parameter checks.

2) Grouping & Sampling (Nearest‑Neighbor Solution Regions)
   - Given `gridsize` (`C = gridsize²`), form groups by sample‑then‑group strategy:
     • Sample seed indices (random or sequential) then build KD‑tree to find K neighbors.
     • For `C>1`, select `C` indices from neighbors; for `C=1`, use seeds directly.
   - Produce grouped dict:
     • `diffraction` (B,N,N,C) amplitude; `coords_nn`; `nn_indices (B,C)`.
     • Compute `coords_offsets (B,1,2,1)`, `coords_relative (B,1,2,C)` with `local_offset_sign = −1`.
     • Optional `Y` patches from `objectGuess` by translating and cropping via `get_image_patches`.

3) Normalization (Amplitude Domain)
   - Normalize `X_full = α · diffraction` where `α = sqrt(((N/2)²) / mean(sum(diffraction²)))` over the batch.
   - Persist `X_full` as normalized amplitude input; do not square at this stage.

4) Loader → Model‑Ready Tensors
   - Convert arrays to tensors: X float32, coords float32, Y complex64 (optional); enforce shape/dtype contracts.
   - Train/test split SHALL apply consistently across X, coords, and optional Y.
   - Compute or set dataset‑level `intensity_scale` `s` for model I/O. Precedence: (1) dataset‑derived `s = sqrt(nphotons / E_batch[Σ_xy |Ψ|²])` when available; (2) closed‑form fallback `s ≈ sqrt(nphotons)/(N/2)` otherwise.

5) Model Construction (U‑Net + Physics)
   - Inputs: `[diffraction (B,N,N,C) float32 · s, positions (B,1,2,C) float32]`.
   - Physics layers:
     • IntensityScaler → Encoder/Decoder → CombineComplex → (Reassemble or Pad) → Trim.
     • ExtractPatchesPosition (object→positions) → ProbeIllumination (·probe, optional smoothing/mask) → PadAndDiffract (FFT→amplitude) → FlatToChannel → IntensityScaler_inv → Square.
   - Outputs: `[object complex64, amplitude float32, intensity float32]`.

6) Loss and Optimization
   - Loss = `realspace_weight · realspace_loss(object)` + `mae_weight · MAE(pred_amp_scaled, target_amp)` + `nll_weight · PoissonNLL((s·X)², (s·Â)²)`.
   - Default emphasis: `nll_weight = 1`; TV/MAE optional; optimizer: Adam(1e‑3).
   - Positivity guard: `Y_pred` (intensity) fed to `log` SHALL be strictly positive, achieved either via strictly‑positive amplitude activation or by adding an epsilon `ε ≥ 1e−12` before log.
   - Ground truth absence: When `Y` is not provided, `mae_weight` MUST be 0 (or otherwise disabled). NLL and real‑space terms remain valid.

7) Inference
   - Use `diffraction_to_obj` for object prediction given `[diffraction, positions]`; ensure probe is consistent with training.

Reassembly Requirements (Normative)
- Implementations MUST reassemble predicted patches using global offsets (pixel units). Offsets MUST be relative to the center-of-mass (or an equivalent normalization that zero-centers the scan trajectory) prior to canvas placement.
- The stitched canvas size MUST satisfy: `M ≥ N + 2·max(|dx|, |dy|)` over the used patches. Implementations SHOULD compute `M` dynamically from offsets to avoid clipping. Odd total padding MUST be split across sides without changing the object’s center.
- Averaging patches without offset-aware placement is prohibited.
- Forward reassembly semantics MUST match across backends when `gridsize > 1`. The default is `object.big=True` (reassemble inside the forward path before diffraction), and the configuration bridge MUST enforce parity across TensorFlow and PyTorch.

8) Stitching & Evaluation
   - Stitch predicted patches (numpy) into full object using `image.stitching.reassemble_patches`.
   - Stitching config contract (required keys): `N (int)`, `gridsize (int)`, `offset (int)`, `nimgs_test (int)`; optional `outer_offset_test (int)`.
   - Border math (normative): Let `outer_offset = outer_offset_test if set else offset`. Then `bordersize = (N − outer_offset/2)/2`, `borderleft = ceil(bordersize)`, `borderright = floor(bordersize)`, `clipsize = bordersize + ((gridsize − 1)·offset)/2`, `clipleft = ceil(clipsize)`, `clipright = floor(clipsize)`.
   - Evaluation metrics: MS‑SSIM, FRC(0.5), MAE on amplitude/phase with appropriate phase alignment.

Staging and Options (Normative)
- `object.big=True`: reassemble patches into padded canvas before trimming.
- `probe.big=True`: per-channel probe; else shared probe.
- Gaussian smoothing sigma configurable (0 disables; default 0).
- Extraction jitter MAY be enabled for augmentation.

### Resource Constraints (Informative)
Current implementations allocate GPU tensors eagerly inside `PtychoDataContainer` (see finding PINN-CHUNKED-001 in `docs/findings.md`). Large datasets that exceed VRAM therefore require manual chunk-size reductions until the planned lazy-loading/chunking refactor lands. Future revisions will make lazy loading a normative requirement once the architecture supports it.

Outputs (Normative)
- Training run yields Keras history, saved weights, and optional stitched object; metrics if GT available.

Failure Modes and Guards (Normative)
- Mismatched channels between X and Y SHALL raise.
- NaNs/negatives in predicted intensity SHALL be guarded before log in Poisson NLL.
- Unsupported N SHALL raise in encoder/decoder factories.
- Missing required stitching config keys SHALL raise.
- Coordinate channel→(row,col) mapping violations (non row‑major) SHALL raise.
