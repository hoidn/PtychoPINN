# Model Comparison Guide

Use this guide to pick the right entry point, reuse existing runs, and apply sampling/registration/stitching consistently for 2‑way or 3‑way comparisons.

## Entry Points
- **Direct compare:** `scripts/compare_models.py` (2‑way, or 3‑way with `--tike_recon_path` for PtyChi/Tike NPZ).
- **Train+compare wrapper:** `scripts/run_comparison.sh` (can reuse models with `--skip-training --pinn-model --baseline-model`).
- **Study orchestration:** `scripts/studies/run_complete_generalization_study.sh` (multi‑trial sweeps; more overhead).

## Common Flags
- Sampling: `--n-test-groups`, `--n-test-subsample`, `--test-subsample-seed` (set seed to keep subsets reproducible; share it with external recon).
- Registration: `--skip-registration`; selective: `--register-ptychi-only`.
- Stitching: `--stitch-crop-size M` (default 20); `--fixed-canvas` to keep plots consistent across subsets by using full test-data coords.
- Chunking/batching: `--pinn-chunk-size`, `--baseline-chunk-size`, `--pinn-predict-batch-size`, `--baseline-predict-batch-size`.
- 3‑way: `--tike_recon_path <ptychi_or_tike_reconstruction.npz>`; auto-detected as PtyChi/Tike via metadata.

## PtyChi Recon Recipe (example)
```
python scripts/reconstruction/ptychi_reconstruct_tike.py \
  --input-npz datasets/fly64/fly64_bottom_half_shuffled.npz \
  --output-dir tmp/ptychi_out \
  --algorithm LSQML --num-epochs 200 --n-images 64
```
Then pass `tmp/ptychi_out/ptychi_reconstruction.npz` to `compare_models.py --tike_recon_path ...`.

## Subsampling Guidance
- Dataset generation subsampling (prepare/split) permanently shrinks the NPZ.
- Eval-time subsampling: `--n-test-groups`/`--n-test-subsample` trims what gets loaded; use `--test-subsample-seed` for determinism.
- For matching external recon subsets, use the same seed on both sides (or supply a pre-sliced NPZ to the recon).

## Artifact Conventions
- Model dirs: `pinn_run/`, `baseline_run/` with `wts.h5.zip` and logs.
- Recon arms: `ptychi_run/` or `tike_run/` with `*reconstruction.npz`.
- Compare outputs: `comparison_metrics.csv`, `comparison_plot.png`, `reconstructions.npz` and `reconstructions_aligned.npz`, `*_frc_curves.csv`, logs under `output_dir/logs/`.

## Special Case: Run1084 Axis Fix for PtyChi
Some Run1084 NPZs store diffraction as H×W×N. Transpose to N×H×W first:
```
python - <<'PY'
import numpy as np
src = np.load("datasets/Run1084_recon3_postPC_shrunk_3.npz")
np.savez("tmp/Run1084_reordered.npz",
         diffraction=np.transpose(src["diffraction"], (2,0,1)),
         probeGuess=src["probeGuess"],
         objectGuess=src["objectGuess"],
         xcoords=src["xcoords"], ycoords=src["ycoords"],
         xcoords_start=src["xcoords_start"], ycoords_start=src["ycoords_start"])
print("Wrote tmp/Run1084_reordered.npz")
PY
```
Then run PtyChi on the reordered NPZ and point `compare_models.py --tike_recon_path` to it. Use `--fixed-canvas` if you want consistent plot crops across different subset sizes/seeds.

## Reference
For full interface/behavior contract, see `specs/compare_models_spec.md`. Quick CLI examples remain in `docs/COMMANDS_REFERENCE.md`.
