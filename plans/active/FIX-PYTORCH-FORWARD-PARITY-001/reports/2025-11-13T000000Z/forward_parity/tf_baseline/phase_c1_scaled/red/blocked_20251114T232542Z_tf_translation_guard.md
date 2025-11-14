# TensorFlow Translation Guard Blocker

**Timestamp:** 2025-11-14T23:25:42Z
**Focus:** FIX-TF-C1D-SCALED-RERUN-001
**Phase:** C1d TensorFlow scaled baseline rerun

## Summary

The scaled TensorFlow training CLI failed during inference/evaluation with a reshape error in `_translate_images_simple`, despite the guard selector passing GREEN. Training completed epoch 1 successfully, but evaluation crashed with the same inference blocker previously documented.

## Environment

```bash
TF_XLA_FLAGS="--tf_xla_auto_jit=0"
USE_XLA_TRANSLATE=0
```

## Guard Selector: GREEN

**Selector:** `pytest tests/tf_helper/test_translation_shape_guard.py::test_non_xla_translation_guard -vv`
**Result:** PASSED (1 passed in 5.03s)
**Log:** `$HUB/green/pytest_tf_translation_guard.log`

The guard test passed, confirming that the fix for training-time translation is working correctly.

## CLI Command

```bash
python scripts/training/train.py \
  --backend tensorflow \
  --train_data_file datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_train.npz \
  --test_data_file datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_test.npz \
  --output_dir plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/tf_baseline/phase_c1_scaled/run_scaled \
  --n_images 64 \
  --n_groups 32 \
  --batch_size 4 \
  --gridsize 2 \
  --neighbor_count 7 \
  --nepochs 1 \
  --do_stitching \
  --quiet
```

## Failure Stack Trace

**Error:** `tensorflow.python.framework.errors_impl.InvalidArgumentError: Graph execution error`

**Root Cause:** Reshape error in `_translate_images_simple` during inference/evaluation phase:

```
File "/home/ollie/Documents/PtychoPINN/ptycho/tf_helper.py", line 199, in _translate_images_simple
Input to reshape is a tensor with 0 values, but the requested shape has 4
[[{{node functional_1/padded_objs_with_offsets_1/translation_36_1/Reshape_4}}]]
[Op:__inference_one_step_on_data_distributed_36867]
```

## Observations

1. **Training phase succeeded:** Epoch 1 completed with 8/8 batches processed
2. **Inference phase failed:** The error occurred during evaluation/stitching, not during training
3. **Guard test passed:** The shape guard correctly handles training-time translation with mismatched batch dimensions
4. **Inference-specific issue:** The reshape error `0 values → shape has 4` suggests a different code path during eval/inference that is not covered by the guard test

## Status

**BLOCKED** - The scaled TF rerun did not produce `analysis/forward_parity_debug_tf/` artifacts because the run crashed during evaluation before stitching/analysis artifacts could be generated.

## Next Actions

1. The guard test only covers training-time translation (`_channel_to_flat` → `translate_core`)
2. The inference/eval code path appears to have a different shape issue not covered by the guard
3. Need to investigate why inference/eval has a 0-element tensor during translation
4. May need a separate guard test for the eval/inference translation path

## References

- Guard log: `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/green/pytest_tf_translation_guard.log`
- CLI log: `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/tf_baseline/phase_c1_scaled/cli/train_tf_phase_c1_scaled.log`
- Previous blocker mention: `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/summary.md` (inference blocker)
- Finding: TF-NON-XLA-SHAPE-001 (covers training path only)
