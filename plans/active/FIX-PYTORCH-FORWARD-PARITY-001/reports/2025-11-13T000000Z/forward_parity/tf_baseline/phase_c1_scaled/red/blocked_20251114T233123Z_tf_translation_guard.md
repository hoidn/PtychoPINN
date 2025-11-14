# TensorFlow Phase C1d Scaled Run Blocker

**Date**: 2025-11-14T23:31:23Z
**Focus**: FIX-TF-C1D-SCALED-RERUN-001
**Guard Selector**: PASSED (tests/tf_helper/test_translation_shape_guard.py::test_non_xla_translation_guard)
**CLI Status**: FAILED during inference/stitching

## Error Signature

```
Input to reshape is a tensor with 0 values, but the requested shape has 4
[[{{node functional_1/padded_objs_with_offsets_1/translation_36_1/Reshape_4}}]]
[Op:__inference_one_step_on_data_distributed_36867]
```

**Location**: `ptycho/tf_helper.py:199` in `_translate_images_simple`
**Context**: Error occurs during reshape of `dx` or `dy` tensors:
```python
dx_expanded = tf.reshape(dx, [batch_size, 1, 1])  # Line 199
```

## Environment

- `TF_XLA_FLAGS="--tf_xla_auto_jit=0"`
- `USE_XLA_TRANSLATE=0`
- Backend: tensorflow
- Dataset: fly001_reconstructed_prepared (train/test split)
- Parameters:
  - `--n_images 64`
  - `--n_groups 32`
  - `--batch_size 4`
  - `--gridsize 2`
  - `--neighbor_count 7`
  - `--nepochs 1`
  - `--do_stitching`

## Execution Timeline

1. **Training phase**: Completed successfully (8/8 batches, 1 epoch)
   - Log shows normal training progress with loss metrics
   - Checkpoint saved

2. **Inference/Stitching phase**: Failed during reassembly
   - Error occurred in `_reassemble_patches_position_real` → `translate` → `translate_core` → `_translate_images_simple`
   - The failure happened after training completed but during the stitching operation

## Root Cause Analysis

The error "Input to reshape is a tensor with 0 values" indicates that either `dx` or `dy` has an empty tensor (shape with 0 elements).

### Code Path

1. `translate_core` (tf_helper.py:702-791) extracts dx/dy from translations
2. When batch dimensions mismatch, it uses `tf.cond` to broadcast translations (line 776-780)
3. The `broadcast_translations` function computes `repeat_factor = images_batch // trans_batch` (line 768)
4. If `trans_batch` is 0, this creates an invalid graph even if the branch isn't executed
5. Later, `_translate_images_simple` tries to reshape the (possibly empty) dx/dy tensors (line 199)

### Why This Wasn't Caught by the Guard

The guard test `test_non_xla_translation_guard` creates fake tensors with non-zero batch sizes:
- images: shape `(4, 64, 64, 1)` (gridsize^2 = 4)
- translations: shape `(1, 2)`

The guard verifies the fallback path works when batches mismatch, but doesn't test the edge case where either batch dimension could be 0 during graph construction.

## Artifacts

- Guard log (PASSED): `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/green/pytest_tf_translation_guard.log`
- CLI log (FAILED): `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/tf_baseline/phase_c1_scaled/cli/train_tf_phase_c1_scaled.log`

## Related

- Guard implementation: `ptycho/tf_helper.py:702-791` (translate_core with batch dimension guard)
- Guard tests: `tests/tf_helper/test_translation_shape_guard.py:46-87`
- Finding: TF-NON-XLA-SHAPE-001 (docs/findings.md)
- Original blocker: `tf_baseline/phase_c1/red/blocked_20251114T074039Z_tf_non_xla_shape_error.md`

## Next Steps

1. **Option A - Fix the broadcast logic**: Add guards to handle zero-batch scenarios in the `tf.cond` branches
   - Check for `trans_batch > 0` before computing `repeat_factor`
   - May require additional graph-mode safety checks

2. **Option B - Investigate why stitching produces zero-batch tensors**:
   - Review the stitching/reassembly logic that calls translate_core
   - Determine if empty batches are expected or indicate upstream issues

3. **Option C - Update test coverage**: Extend the guard test to include zero-batch edge cases

## Status

**BLOCKED** - TensorFlow Phase C1d scaled run cannot proceed until the zero-batch reshape issue is resolved.

Per the Return Condition in docs/fix_plan.md, this blocker documents:
- ✅ Guard selector log exists and shows GREEN
- ❌ CLI failed during stitching (log captured)
- ❌ No forward_parity_debug_tf artifacts generated
- ✅ Environment captures documented above
