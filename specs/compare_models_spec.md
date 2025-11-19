# Compare Models Spec

Status: draft for refactor alignment  
Scope: `scripts/compare_models.py` CLI and its thin wrapper API target

## Overview
Compare PtychoPINN vs Baseline (2-way) or vs Tike/PtyChi (3-way). Load trained runs + test NPZ, run inference (with optional chunking), registration, stitching, metrics, and plots. Future state: thin CLI delegating to a modular Python API with the same behavior.

## Inputs / Interface
- Required:
  - `--pinn_dir <path>`: trained PINN run (contains `wts.h5.zip`).
  - `--baseline_dir <path>`: trained Baseline run (contains `wts.h5.zip` or baseline_model.h5).
  - `--test_data <npz>`: NPZ with diffraction + coords (+ optional ground truth).
  - `--output_dir <path>`: destination for metrics/plots/NPZ.
- Optional reconstruction arm:
  - `--tike_recon_path <npz>`: Tike or PtyChi NPZ. Must include `reconstructed_object`, `reconstructed_probe`, `algorithm` keys; supports (N,H,W) or (H,W,N) diffraction metadata for detection.
- Sampling:
  - `--n-test-groups <int>`: limit groups/images loaded.
  - `--n-test-subsample <int>`: independent subsample size.
  - `--test-subsample-seed <int>`: seed for reproducible subsample (must be shared with recon generation to keep subsets aligned).
- Registration / stitching:
  - `--skip-registration`: disable pixel-level registration.
  - `--register-ptychi-only`: apply registration only to the recon arm.
  - `--stitch-crop-size M`: crop size for stitching (default 20; must satisfy 0 < M <= N).
- Chunking / batching:
  - `--pinn-chunk-size`, `--baseline-chunk-size`, `--pinn-predict-batch-size`, `--baseline-predict-batch-size`.
- Logging / output:
  - `--[no-]save-npz`, `--[no-]save-npz-aligned`, `--quiet/--verbose`, `--console-level`.

## Data Contracts
- Test NPZ: diffraction as `diff3d` or `diffraction` (H,W,N or N,H,W), `probeGuess`, `xcoords`, `ycoords`, optional `objectGuess`. Gridsize inferred from channel count; forced to match models.
- Recon NPZ (tike/ptychi): `reconstructed_object`, `reconstructed_probe`, `algorithm` string (used for labeling).
- Model dirs: PINN/Baseline contain weights and params to restore gridsize/N and preprocessing (e.g., intensity scaling).

## Behavior
- Gridsize enforcement: match diffraction channels; override params.cfg gridsize if needed before inference.
- Subsampling: random unless seed provided; if `--n-test-subsample` present, it selects that many indices (sorted) from the dataset size. Seed must be shared with external recon generation for subset alignment.
- Registration: two stages (coordinate crop + pixel cross-correlation) unless `--skip-registration`. `--register-ptychi-only` limits registration to the recon arm.
- Stitching: uses `--stitch-crop-size` M for cropping patches before assembly.
- Chunking: optional chunked inference per model to control memory.
- Outputs:
  - `comparison_metrics.csv` (per-model metrics: MAE, MSE, PSNR, SSIM, MS-SSIM, FRC/FRC50, timings).
  - `comparison_plot.png` (2x3 for 2-way, 2x4 for 3-way).
  - `reconstructions.npz` (raw) and `reconstructions_aligned.npz` (post-registration) with amplitude/phase/complex per model + ground truth if present.
  - `*_frc_curves.csv`, logs under `logs/`.
- Error handling: fail fast on missing files/keys, invalid stitch size, shape mismatches; log gridsize mismatches with auto-fix; raise if recon NPZ lacks required keys.

## Compatibility / Performance
- Backends: TensorFlow for PINN/Baseline; optional PtyChi/Tike recon NPZ ingestion (treated as data).
- Supported gridsize: inferred from channels; requires matching trained model.
- Performance knobs: chunk size, batch size, stitching M, registration toggle, subsample size/seed, XLA/CPU overrides (env flags).

## Cross-References
- Data formats: `specs/data_contracts.md`.
- Metrics definitions: `specs/overlap_metrics.md`.
caption: `docs/MODEL_COMPARISON_GUIDE.md`, `docs/COMMANDS_REFERENCE.md`.
- PtyChi runner: `scripts/reconstruction/ptychi_reconstruct_tike.py` (supports `--subsample-seed` to align with `--test-subsample-seed`).

## Refactor Target Notes
- Extract a Python API: e.g., `compare_models.compare(pinn_dir, baseline_dir, test_npz, *, recon_npz=None, subsample=None, seed=None, registration=True, stitch_crop_size=20, chunking=..., save_npz=True, save_aligned=True, logger=...)` returning a results dataclass and writing artifacts optionally.
- Keep CLI as a thin wrapper over the API; CLI stays stable per default behavior and outputs.
