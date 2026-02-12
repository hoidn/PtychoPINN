# Bug Report: TF Bundle Loader Fails on Keras 3 (`inputs` not connected)

## Summary
`load_inference_bundle()` can fail while loading TensorFlow `wts.h5.zip` bundles with:

```
ValueError: `inputs` not connected to `outputs`
```

The failure occurs in the Keras 3 load path inside `ModelManager.load_model()` when it calls `tf.keras.models.load_model(..., compile=False)` on `model.keras` extracted from the bundle.

## Impact
- Blocks post-training inference reuse for affected TF bundles.
- Blocks recovery workflows that depend on loading `pinn/wts.h5.zip` without retraining.
- Causes study scripts to fail even when training completed successfully and bundle artifacts exist.

## Reproduction Context
- Bundle: `outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed13_pinn_only_retry1/pinn/wts.h5.zip`
- Failing path:
  - `ptycho/workflows/components.py:load_inference_bundle`
  - `ptycho/model_manager.py:load_multiple_models -> load_model`
  - `ptycho/model_manager.py` Keras 3 branch using `tf.keras.models.load_model`
- Error observed in:
  - `.artifacts/studies/grid_lines_n64_pinn_hybrid_resnet_e20_retry1/logs/finalize_retry1.log`

## Expected Behavior
If `wts.h5.zip` is present and structurally valid, `load_inference_bundle()` should return the `diffraction_to_obj` model and restored config without requiring retraining.

## Actual Behavior
The Keras 3 model load call can fail with graph-connectivity deserialization error (`inputs` not connected to `outputs`) on valid bundles.

## Current Workaround (Verified)
Bypass `tf.keras.models.load_model(model.keras)` and instead:
1. Extract `diffraction_to_obj/params.dill` and update `params.cfg`.
2. Extract `model.weights.h5` from `diffraction_to_obj/model.keras`.
3. Recreate architecture with `create_model_with_gridsize(gridsize, N)`.
4. Call `model.load_weights("model.weights.h5")`.
5. Run inference normally.

This workaround successfully produced `recons/pinn/recon.npz` and downstream visuals/metrics for the affected run.

## Suspected Root Cause
Keras 3 full-model deserialization of this specific saved graph (custom layers + wiring) is brittle in some bundle states/environments, while raw weight loading remains robust.

## Files Involved
- `ptycho/model_manager.py`
- `ptycho/workflows/components.py`

## Status
Open

