# Phase C1d TF Scaled Run Blocker

**Date:** 2025-11-14T15:37:00Z  
**Focus:** FIX-TF-C1D-SCALED-RERUN-001  
**Selector:** tests/tf_helper/test_translation_shape_guard.py::test_non_xla_translation_guard  

## Summary

The scaled TF training CLI failed during the eval phase with a reshape error in `_translate_images_simple` (ptycho/tf_helper.py:199). The guard selector passed GREEN, indicating the guarded path works correctly in isolation, but the full training run still hits the shape error during inference/stitching.

## Error Signature

```
tensorflow.python.framework.errors_impl.InvalidArgumentError: Graph execution error:
Input to reshape is a tensor with 0 values, but the requested shape has 4
[[{{node functional_1/padded_objs_with_offsets_1/translation_36_1/Reshape_4}}]]
```

**Location:** `ptycho/tf_helper.py:199` in `_translate_images_simple`  
**Phase:** Eval/inference after epoch 0 training completion  
**Context:** Translation operation during patch extraction with gridsize=2, batch_size=4

## Environment

- `TF_XLA_FLAGS="--tf_xla_auto_jit=0"`
- `USE_XLA_TRANSLATE=0`
- Backend: tensorflow
- Gridsize: 2
- n_images: 64
- n_groups: 32
- batch_size: 4
- neighbor_count: 7

## Artifacts

- Guard log: `$HUB/green/pytest_tf_translation_guard.log` (PASSED)
- CLI log: `$TF_BASE/cli/train_tf_phase_c1_scaled.log` (55K, contains full stack trace)
- Env capture: `$TF_BASE/cli/env_capture_scaled.txt`

## Analysis

The guard test (`test_non_xla_translation_guard`) exercises the fallback path with mismatched batch dimensions and passes, but the real training run with gridsize=2 still fails during eval. This suggests:

1. The guard test may not fully replicate the actual batch/shape conditions during stitching
2. The eval phase may be using different tensor shapes than the guard anticipated
3. There may be a configuration mismatch between training and eval phases

## Next Actions

1. Inspect the guard test parameters (tests/tf_helper/test_translation_shape_guard.py:46-87) to verify they match the actual eval batch shapes
2. Add instrumentation to log the actual tensor shapes in `_translate_images_simple` before the reshape
3. Compare the guard's mocked shapes against the real eval shapes to identify the gap
4. Either fix the guard to catch this case, or fix the eval code to handle the shape correctly

## Return Condition

This blocker is resolved when either:
- The scaled TF CLI completes successfully and produces `$TF_BASE/analysis/forward_parity_debug_tf/{stats.json,offsets.json,pngs}`
- A refined guard test is added that catches this specific failure mode, demonstrating we understand the root cause
