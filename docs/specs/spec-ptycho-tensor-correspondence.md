# spec-ptycho-tensor-correspondence.md — TF↔PyTorch Tensor Correspondence

Status legend used below:
- **Documented** — the fact already exists in an authoritative doc/spec/code comment; this shard consolidates the citation(s) and/or adds the cross-backend pairing.
- **Divergence** — TensorFlow and PyTorch genuinely differ (by design or by accident); both sides are cited and the reason for the difference is stated.
- **Previously-undocumented** — no prior doc/spec stated this fact; this shard is the first record of it.

## 1. Scope

Canonical map of tensor shapes, axis conventions, and representational choices across the TensorFlow (`ptycho/`) and PyTorch (`ptycho_torch/`) backends, from raw grouped data through model I/O and reassembly. This shard does not restate the external inference contract — see `specs/ptychodus_api_spec.md` §4.4.1 for the authoritative `diffraction_to_obj` I/O shapes — and does not restate `docs/architecture_torch.md` §3.1's reassembly walkthrough; both are linked, not duplicated, below.

Non-goals: config-bridge field mappings (see `docs/specs/spec-ptycho-config-bridge.md`), NPZ/grouped-dict key contracts (see `docs/specs/spec-ptycho-core.md` and `docs/specs/spec-ptycho-interfaces.md`), and Ptychodus HDF5 product contracts (see `specs/data_contracts.md`).

## 2. Tensor/Shape Correspondence Table

### 2.1 Grouped diffraction `X`
- TF: `(B,N,N,C)` channel-last — `ptycho.loader.PtychoDataContainer` docstring.
- Torch container: `(B,N,N,C)`, identical to TF (no permute) — `ptycho_torch.data_container_bridge.PtychoDataContainerTorch`, assignment from `X_full`.
- Torch model input: `(B,C,N,N)` after the Dataset's channel-last→channel-first permute — `ptycho_torch.workflows.components.PtychoLightningDataset`, via `images_indexed.permute(0, 3, 1, 2)` for batched samples and `permute(2, 0, 1)` for the unbatched case.
- **Status: Previously-undocumented.** Each stage's shape is individually documented in its own docstring; the three-stage chain (TF loader → torch container → torch model input) had not been written down together.

### 2.2 Positions / offsets
- TF: `(B,1,2,C)`, axis order `[x, y]` — `ptycho.loader.PtychoDataContainer` docstring.
- Torch: `(B,C,1,2)` — permuted from the TF-shaped `coords_relative` in `ptycho_torch.workflows.components.PtychoLightningDataset` (`coords_rel.permute(0, 3, 1, 2)` batched / `permute(2, 0, 1)` unbatched), per the inline rationale that the PyTorch translation helper expects `(..., C, 1, 2)`.
- **Status: Divergence (by design).** The axis reordering is a Torch-side implementation requirement of the `Translation` helper, not a semantic difference in the coordinates; previously stated only as an inline code comment, not in any spec.

### 2.3 Reassembly
- Torch `reassemble_patches_position_real` takes complex patches `(B,C,N,N)` and offsets `(B,C,1,2)` and returns a canvas `(B,M,M)` — `ptycho_torch.helper.reassemble_patches_position_real`.
- Offsets are negated before translation (`offsets_flat = -offsets_flat`) to match TF's `Translation([imgs, -offsets_flat])` convention — both the unweighted `reassemble_patches_position_real` path and the probe-weighted helper path in `ptycho_torch.helper`.
- **Status: Documented.** Shape contract: `docs/architecture_torch.md` §3.1 Reassembly Contract. Sign convention: `docs/findings.md` TORCH-REASSEMBLY-SIGN-001.

### 2.4 Generator raw real/imag → complex conversion; output modes
- Generator (FNO/Hybrid) heads produce raw real/imag tensors `(B,H,W,C,2)`, converted to complex — `ptycho_torch.model._real_imag_to_complex_channel_first` and the `real_imag` branch of `_predict_complex_patches`. Matches `specs/ptychodus_api_spec.md` §4.4.1.
- Converted to complex `(B,C,N,N)` before `forward_predict` returns — `ptycho_torch.model.PtychoPINN.forward_predict` and `Ptycho_Supervised.forward_predict`.
- Three `generator_output` modes, dispatched in `ptycho_torch.model._predict_complex_patches`:
  - `amp_phase`: autoencoder returns `(amp, phase)` directly.
  - `amp_phase_logits`: autoencoder returns stacked logits `(...,2)`; bounded via `amp = sigmoid(amp_logits)`, `phase = π·tanh(phase_logits)`.
  - `real_imag`: either a CNN `(real, imag)` tuple or the FNO/Hybrid `(B,H,W,C,2)` tensor, converted via §2.4's first bullet.
- **Status: Documented** for the raw→complex conversion (now stated in `specs/ptychodus_api_spec.md` §4.4.1, corrected). **Previously-undocumented** for the three `generator_output` branches and their (non-)bounded activations — code-only until this shard.

### 2.5 Probe layouts
- Container (parity with TF): `(N,N)` — `ptycho_torch.data_container_bridge.PtychoDataContainerTorch` docstring and TF analog `ptycho.loader.PtychoDataContainer`.
- Grid-lines dict-container path: raw per-sample probe `(H,W)` or `(P,H,W)`, collated by `DataLoader` to `(B,H,W)` / `(B,P,H,W)` — `ptycho_torch.workflows.components.PtychoLightningDataset`.
- Mmap loader path: `(B,C,P,N,N)` (probe duplicated across channels via `unsqueeze(1).expand(...)`) — `ptycho_torch.dataloader.PtychoDataset`.
- Rectangular-scaled variant: `(C,P,H,W)` — `ptycho_torch.workflows.components.PtychoLightningDataset`.
- Dispatch on `probe.ndim` with no silent fallback (5 = modes layout, 2 = plain, 3 = batch-broadcast) — `ptycho_torch.helper.reassemble_patches_position_real_probe`.
- **Status: Previously-undocumented.** No prior doc enumerated the four Torch probe layouts or the dispatch rule.

### 2.6 `pad_and_diffract`
- TF: flat single-channel input asserted `(B·C,N,N,1)` — `ptycho.tf_helper.pad_and_diffract` uses `tf.ensure_shape(input, (None, h, w, 1))` and asserts the final axis is `1`. (Protected file; cited read-only.)
- Torch: `(N,C,P,H,W)` with an explicit mode-sum over the probe-mode axis before the FFT — `ptycho_torch.helper.pad_and_diffract`.
- **Status: Previously-undocumented.** Each side's shape assertion exists in its own code; the cross-backend comparison (TF's single-channel assert vs. Torch's mode-summed multi-probe-mode input) had not been written down.

### 2.7 Padded-size formulas (DIVERGENCE)
- TF: `bigN = N + (gridsize-1)·offset`; `padded_size = bigN + max_position_jitter` — `ptycho.params.get_bigN` and `get_padded_size`.
- Torch: `bigN = N + (gridsize[0]-1)·ceil(max_neighbor_distance, rounded up to even)`; `padded_size = bigN + 0` (no jitter buffer) — `ptycho_torch.helper.get_bigN` and `get_padded_size`.
- **Status: Divergence (by design, tracked).** Torch's padded size omits the TF jitter buffer entirely. Tracked and resolved as an accepted divergence in `docs/findings.md` TORCH-PADDED-SIZE-001.

### 2.8 Amplitude, not intensity (both backends)
- TF: `ptycho.loader.load` documents images as amplitude, not intensity.
- Torch: `ptycho_torch.data_container_bridge.PtychoDataContainerTorch` documents `X` as diffraction-pattern amplitude, not intensity.
- Canonical NPZ contract: `diff3d (M,N,N) float32` amplitude (sqrt of counts) — `docs/specs/spec-ptycho-core.md` §Raw NPZ.
- **Status: Documented.** Each backend already states this individually; this row consolidates the citations rather than adding a new fact.

### 2.9 Complex representation
- TF: a single `tf.complex64` tensor throughout (`ptycho.tf_helper.combine_complex`).
- Torch: mode-dependent intermediates — `amp`/`phase` float32 pair, or a raw real/imag tensor, converted to `torch.complex64` only at the `ptycho_torch.model._predict_complex_patches` boundary; which intermediate exists at any given point depends on `generator_output` (§2.4).
- **Status: Previously-undocumented.**

### 2.10 `local_offset_sign = -1` (shared exemplar convention)
- Normative in three existing shards: `docs/specs/spec-ptycho-core.md`, `docs/specs/spec-ptycho-workflow.md`, and `docs/specs/spec-ptycho-conformance.md`.
- **Status: Documented.** Cited here as the exemplar of a convention shared byte-for-byte across backends (no divergence); no new fact added.

### 2.11 Channel ↔ (row, col) mapping
- Row-major: `row = c // gridsize`, `col = c % gridsize` — `ptycho.loader.PtychoDataContainer` docstring, `docs/architecture.md`, and `docs/specs/spec-ptycho-interfaces.md`.
- Applies identically to both backends (channel semantics are backend-agnostic; only the tensor axis carrying `C` differs, per §2.1/§2.2).
- **Status: Documented.** Consolidates existing citations; no new fact.

### 2.12 Single-patch inference reshape
- When a Torch inference pass has exactly one patch per sample (no grouping), batch is collapsed into channels before reassembly: `patch_complex.reshape(1, -1, N, N)` / `offsets.reshape(1, -1, 1, 2)` — `docs/architecture_torch.md` §3.1.
- **Status: Documented.** Restated here only as a pointer; full walkthrough stays in `architecture_torch.md` §3.1.

## 3. Cross-references

- External inference I/O contract (authoritative): `specs/ptychodus_api_spec.md` §4.4.1.
- Reassembly contract and single-patch inference reshape: `docs/architecture_torch.md` §3.1.
- Offset-sign convention finding: `docs/findings.md` TORCH-REASSEMBLY-SIGN-001.
- Padded-size divergence finding: `docs/findings.md` TORCH-PADDED-SIZE-001.
- Grouped-dict and dict-container key contracts: `docs/specs/spec-ptycho-interfaces.md`.
