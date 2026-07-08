# Backlog: Harden TF Bundle Loading Against Keras 3 Deserialization Failures

**Created:** 2026-02-11  
**Status:** Open  
**Priority:** High  
**Related Bug:** `docs/bugs/2026-02-11-tf-loader-keras3-graph-disconnect.md`  
**Impacts:** `ptycho/model_manager.py`, `ptycho/workflows/components.py`, TF inference reuse workflows

## Problem
`load_inference_bundle()` can fail on valid TF bundles with:

```
ValueError: `inputs` not connected to `outputs`
```

The failure occurs in `ModelManager.load_model()` when loading `model.keras` via Keras 3 deserialization.

## Goal
Make TF bundle loading reliable without retraining by adding a robust fallback path that loads weights directly when full-model deserialization fails.

## Proposed Implementation
1. Keep the current primary path: `tf.keras.models.load_model(model.keras, custom_objects=..., compile=False)`.
2. On this specific failure class (and closely related Keras deserialization graph errors), fallback to:
   - read `params.dill`,
   - recreate architecture with `create_model_with_gridsize(gridsize, N)`,
   - extract `model.weights.h5` from `model.keras`,
   - call `model.load_weights(...)`.
3. Keep existing behavior for non-recoverable errors.
4. Add structured logging identifying primary-path failure and fallback success.

## Acceptance Criteria
1. `load_inference_bundle()` succeeds for the known failing artifact shape (or equivalent fixture) without manual intervention.
2. Existing passing bundle loads remain passing (no regression).
3. Add regression coverage proving fallback is used when full-model deserialize fails.
4. Document behavior in loader-facing workflow docs (brief note, not deep internals).

## Suggested Tests
1. Unit test around `ModelManager.load_model()` with mocked Keras deserialize failure and successful weight fallback.
2. Integration-style test for `load_inference_bundle()` returning a callable model after fallback.
3. Negative test confirming unrelated errors still propagate.

