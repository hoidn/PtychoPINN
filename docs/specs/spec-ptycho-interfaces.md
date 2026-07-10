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
- Canonical NPZ contract: spec-ptycho-core.md §Raw NPZ (RawData.from_file); this section defers to it.
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
  - Torch grid-lines dict-container contract:
    • The grid-lines Torch runner builds a plain dict container (not `PtychoDataContainer`), consumed by `ptycho_torch.workflows.components.PtychoLightningDataset`. Keys:
    • `X (B,N,N,C) float32` — conditioned model input; may carry appended non-physical channels depending on `--input-conditioning-mode`.
    • `observed_images (B,N,N,C) float32` — loss-side raw diffraction; equals `train_data['diffraction']` in every input-conditioning mode (unconditioned, unlike `X`).
    • `coords_relative`/`coords_nominal` — plays the `positions` role from Model I/O below; `coords_relative` preferred, `coords_nominal` accepted as fallback by `PtychoLightningDataset`.
    • `probe` — probe array bridged from `train_data['probeGuess']` (or an all-ones array in `diffraction_only` conditioning mode).
    • CI rectangular containers carry `measured_intensity`, `rms_input_scale`, `mean_measured_intensity`, `probe_training`, `probe_physical`, and `probe_normalization`. They do not carry `physics_scaling_constant`.
    • Explicit legacy and amplitude containers retain `physics_scaling_constant` and `rms_scaling_constant` compatibility fields. `--count-scale-mode` and `derive_dict_physics_scale` are legacy/amplitude mechanisms, not CI photon calibration.
    • `label_amp`/`label_phase` — supervised-mode ground truth, bridged from `train_data['Y_I']`/`train_data['Y_phi']`; required when `training_procedure='supervised'`.
    • Producing code: `scripts.studies.grid_lines_torch_runner.run_torch_training`. Shapes cross-reference the TF↔Torch tensor-correspondence shard where applicable.
- Model I/O:
  - Inputs: `[diffraction (B,N,N,C) float32 · s, positions (B,1,2,C) float32]`.
  - Outputs: `[object (B,N,N,1) complex64, amplitude (B,N,N,C) float32, intensity (B,N,N,C) float32]`.

Torch Data Loader and Batch Contract (Normative)
- Torch training loaders SHALL yield batches consumable by `ptycho_torch.model.PtychoPINN_Lightning.compute_loss` as:
  - `(tensor_dict, probe, probe_scaling)`.
- Native Torch mmap path:
  - Producer: `ptycho_torch.dataloader.PtychoDataset`.
  - Diffraction source: accepts standalone-NPZ `diff3d` and the compatibility alias `diffraction`; legacy `(H,W,M)` arrays are transposed to canonical `(M,H,W)`.
  - CI `tensor_dict` includes `images`, `measured_intensity`, coordinates, `nn_indices`, `experiment_id`, `rms_input_scale`, and `mean_measured_intensity`. CI statistics are attached from one immutable per-experiment authority and are not duplicated as writable mmap arrays.
  - CI named probes are `probe_training`, `probe_physical`, and `probe_normalization`, each expanded as `(B,C,P,N,N)`. `probe_training = q * probe_physical`; `q` is computed jointly over modes before channel expansion.
  - The tuple aliases remain `(tensor_dict, probe, probe_scaling)` for compatibility. In CI, the named fields own the physical meaning; explicit legacy retains the historical tuple interpretation and scaling fields.
  - Reused mmap stores SHALL contain a matching schema/profile manifest. Missing or incompatible manifests fail with a rebuild instruction.
- Grid-lines dict-container path:
  - Producer: `scripts.studies.grid_lines_torch_runner.run_torch_training`.
  - Consumer: `ptycho_torch.workflows.components.PtychoLightningDataset`.
  - Input containers are channel-last `(B,N,N,C)` and are permuted to channel-first `(B,C,N,N)` for `tensor_dict['images']` and `tensor_dict['observed_images']`.
  - `coords_relative` is preferred; `coords_nominal` is accepted as fallback when `object_big=False`. Coordinates are permuted from `(B,1,2,C)` to `(B,C,1,2)` for Torch translation helpers.
  - CI containers use the same named CI fields as the mmap path and always collate physical/training probes to `(B,C,P,H,W)`. Training-derived statistics are attached unchanged to validation/test containers.
  - Amplitude and explicit legacy modes preserve `observed_images`, `rms_scaling_constant`, `physics_scaling_constant`, and their existing absent-default behavior.
  - `(N,N)`, `(P,N,N)`, and legacy `(N,N,1)` probes SHALL canonicalize to an explicit mode-first layout before collation. CI diffraction sums per-mode intensities incoherently and SHALL be invariant to batch size.
- Shape/layout details are owned by `docs/specs/spec-ptycho-tensor-correspondence.md`; standalone-NPZ and grouped-dict keys are owned by `docs/specs/spec-ptycho-core.md` and the Data Formats section above.

Reassembly Contract (Normative)
- Inference pipelines MUST stitch predicted patches with global offsets (pixel units). Offsets MUST be zero‑centered (e.g., relative to center‑of‑mass) prior to placement on the canvas.
- The stitched canvas MUST be large enough to accommodate all translated patches without clipping: `M ≥ N + 2·max(|dx|, |dy|)` for the used subset. Odd total padding MUST be distributed across borders without shifting the reconstruction’s center.
- Averaging patches in place without offset‑aware translation is prohibited.
- Forward‑path reassembly semantics MUST match across backends when `gridsize > 1`. The default is `object.big=True` (reassemble before diffraction), and the configuration bridge MUST carry this behavior between TensorFlow and PyTorch.

Loader Behavior (Normative)
- When ground truth `Y` is absent, loaders MAY emit a shape‑compatible placeholder `Y (complex64)`; training integrations MUST disable any MAE terms that require real ground truth. NLL and real‑space terms remain valid without `Y`.
- `coords_nominal` and `coords_true` SHALL be equal when jitter/true positions are not provided by the dataset.

> Known gap: the current TF loader (`ptycho.loader.load`) sources both `coords_nominal` and `coords_true` from the same `coords_start_relative` key, so they never diverge even when true/jittered positions ARE provided; the `coords_relative`/`coords_offsets` pair computed by `ptycho.raw_data.RawData.generate_grouped_data` is not read by the loader (finding TF-LOADER-COORDS-001, Active). This requirement stands and the implementation is the acknowledged deviation.

Precedence and Configuration (Normative)
- PyTorch rectangular scaling profile:
  - Missing profile fields select `ci_intensity_v2`/`count_intensity`.
  - Explicit overrides are an inseparable pair. A one-sided or contradictory override SHALL error.
  - Known metadata-free legacy checkpoints/bundles require `legacy_v1` and `normalized_amplitude` together. CI checkpoints/bundles require persisted frozen training statistics.
  - Architecture-era config fields MAY be removed only through an explicit versioned compatibility list. Model and physics weights SHALL load strictly; incompatible artifacts fail with deterministic regeneration guidance.
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
