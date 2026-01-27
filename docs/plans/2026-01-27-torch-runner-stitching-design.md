# Torch Runner Stitching Design (Grid-Lines)

Date: 2026-01-27

## Summary
Fix Torch grid-lines evaluation by stitching predictions before `eval_reconstruction`. The Torch runner currently computes metrics on patch-level predictions, which fails shape assertions because ground truth is a stitched global object. The solution adds a native PyTorch stitching path (preferred) using `ptycho_torch.workflows.components._reassemble_cdi_image_torch_mmap`, and computes metrics on the stitched reconstruction. Patch-level metrics are still computed for debugging.

## Root Cause
Torch runner feeds unstitched patch predictions into `eval_reconstruction`, but `eval_reconstruction` asserts that prediction and ground truth shapes match. For grid-lines, predictions are `(n_patches, N, N[, C])` while ground truth is `(H, W, 1)`, so the assertion fails.

## Goals
- Use native PyTorch stitching when possible (preferred).
- Preserve patch-level metrics for debug/analysis.
- Keep gridsize=1 path reliable; fall back to TF-style stitching if native path cannot run.
- Keep output metrics JSON structure explicit (stitched vs patch).

## Non-Goals
- Rewrite reassembly or physics layers.
- Change training or forward prediction semantics.
- Support new data formats beyond the cached grid-lines NPZs.

## Proposed Approach (Preferred)
### Native PyTorch Stitching via `_reassemble_cdi_image_torch_mmap`
Implement a stitching helper in `scripts/studies/grid_lines_torch_runner.py` that:
1. Creates a temporary reassembly NPZ file (single-file dataset) from cached grid-lines test NPZ.
2. Builds a `PtychoDataset` from that directory.
3. Constructs a minimal `InferencePayload` and calls `_reassemble_cdi_image_torch_mmap`.
4. Returns a stitched complex object suitable for `eval_reconstruction`.

### Temporary Reassembly NPZ Schema
Create `output_dir/tmp/torch_reassembly/reassembly_test.npz` with:
- `diffraction`: cached `diffraction` squeezed to `(N_images, N, N)` (for DATA-001 compliance in PyTorch dataloader)
- `xcoords`, `ycoords`: derived from `coords_nominal` (gridsize=1 only):
  - `xcoords = coords_nominal[:, 0, 0, 0]`
  - `ycoords = coords_nominal[:, 0, 1, 0]`
- `probeGuess`: cached `probeGuess`
- `objectGuess`: `YY_full` (preferred, squeeze to 2D); fallback to `YY_ground_truth`

### Config / Payload Construction
- Build `PTDataConfig` and `PTInferenceConfig` directly (no `create_inference_payload`, which requires `wts.h5.zip`).
- Assemble `InferencePayload` manually:
  - `pt_data_config`, `pt_inference_config`
  - `execution_config` (with `inference_batch_size=cfg.infer_batch_size`)
  - `tf_inference_config`: minimal `InferenceConfig` for legacy bridge compatibility
- Call `_reassemble_cdi_image_torch_mmap(test_data, payload, execution_config, train_results)` and convert to NumPy complex.

## Fallback Path
If native stitching prerequisites are missing (e.g., gridsize != 1, missing coords), fall back to TF-style stitch helper (mirroring `ptycho/workflows/grid_lines_workflow.py::stitch_predictions`). Log a warning and proceed.

## Metrics
- **Stitched metrics** (primary): compute `eval_reconstruction(stitched_obj, YY_ground_truth)`.
- **Patch metrics** (debug): compute amplitude/phase MAE and MSE directly against `Y_I`/`Y_phi`.

Example metrics JSON:
```json
{
  "stitched": { "mse": [..], "ssim": [..], ... },
  "patch": { "mae_amp": 0.0, "mse_amp": 0.0, "mae_phase": 0.0, "mse_phase": 0.0 }
}
```

## Output Artifacts
- Keep existing model checkpoints and per-run outputs.
- Add stitched object arrays/PNGs only if needed (optional).
- Store temporary reassembly NPZ under `output_dir/tmp/torch_reassembly/` (kept for debug unless explicitly cleaned).

## Testing
- Unit test: mocked `_reassemble_cdi_image_torch_mmap` returns a stitched tensor with correct shape; verify metrics JSON has `stitched` and `patch` entries.
- Unit test: forcing fallback (gridsize > 1 or missing coords) uses TF-style stitching path.

## Open Questions
- If future gridsize > 1 runs require native stitching, we will need a mapping from grouped coords to raw coords, or a separate raw NPZ generator.
