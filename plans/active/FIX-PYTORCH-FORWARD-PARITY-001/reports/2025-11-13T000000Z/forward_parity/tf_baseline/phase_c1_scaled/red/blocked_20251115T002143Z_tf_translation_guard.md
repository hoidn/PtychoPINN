# Blocker: TensorFlow Scaled Rerun Failed (Fifth Attempt)

**Timestamp:** 2025-11-15T002143Z
**Focus:** FIX-TF-C1D-SCALED-RERUN-001
**Attempt:** Fifth consecutive failure with identical error signature

## Summary

The scaled TensorFlow training command succeeded through epoch 1 training (8/8 batches complete) but failed during the evaluation/inference/stitching phase with a reshape error in `_translate_images_simple`.

## Environment

```
TF_XLA_FLAGS=--tf_xla_auto_jit=0
USE_XLA_TRANSLATE=0
```

Captured in: `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/tf_baseline/phase_c1_scaled/cli/env_capture_scaled.txt`

## Guard Test Result

**Status:** GREEN (1 passed in 5.26s)

```bash
pytest tests/tf_helper/test_translation_shape_guard.py::test_non_xla_translation_guard -vv
```

Log: `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/green/pytest_tf_translation_guard.log`

**Interpretation:** The guard test confirms that the training-time translation layer fix is working correctly. The guard exercises the `translate_core` function with batch dimension mismatches during forward pass training, which now succeeds.

## Training CLI Result

**Status:** FAILED during evaluation/inference

```bash
python scripts/training/train.py \
  --backend tensorflow \
  --train_data_file datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_train.npz \
  --test_data_file datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_test.npz \
  --output_dir "$TF_BASE/run_scaled" \
  --n_images 64 \
  --n_groups 32 \
  --batch_size 4 \
  --gridsize 2 \
  --neighbor_count 7 \
  --nepochs 1 \
  --do_stitching \
  --quiet
```

Log: `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/tf_baseline/phase_c1_scaled/cli/train_tf_phase_c1_scaled.log` (701 lines, 54KB)

### Training Success

Epoch 1 completed successfully:
- 8/8 batches processed
- Training loss: -2624654.5
- Validation loss: -2793478.0

### Error During Evaluation

The failure occurred during the post-training evaluation/stitching phase:

```
Input to reshape is a tensor with 0 values, but the requested shape has 4
[[{{node functional_1/padded_objs_with_offsets_1/translation_36_1/Reshape_4}}]]
```

**Error location:** `ptycho/tf_helper.py:199` in `_translate_images_simple`

**Stack trace context:**
```
File "ptycho/tf_helper.py", line 789, in translate_core
File "ptycho/tf_helper.py", line 199, in _translate_images_simple
```

## Root Cause Analysis

The guard test exercises the **training forward path** where batch dimensions are controlled and predictable (batch_size=4, groups=32). This path now works correctly after the non-XLA translation fix.

However, the **evaluation/inference/stitching path** uses different batching logic:
- Training dataset: 32 groups → 4 batches of 8 groups
- Test dataset: 7597 images → dynamic batching during stitching

The reshape error (`0-element tensor → shape 4`) suggests that during evaluation, the batching or translation logic produces empty tensors, likely due to:
1. Different batch size handling in eval mode
2. Stitching attempting to process edge cases with no valid patches
3. Mismatch between training's controlled batching and eval's dynamic batching

## Gap in Test Coverage

The current guard test (`test_non_xla_translation_guard`) only covers:
- Training-time forward pass
- Controlled batch dimensions (4 images, gridsize=2)
- Single translation call with known inputs

The guard does **NOT** cover:
- Evaluation/inference code path
- Stitching reassembly logic
- Dynamic batching with large test datasets
- Edge cases during patch extraction/reassembly

## Artifacts Generated

### Success
- ✅ Guard test log: `green/pytest_tf_translation_guard.log` (GREEN)
- ✅ Environment capture: `tf_baseline/phase_c1_scaled/cli/env_capture_scaled.txt`
- ✅ Training CLI log: `tf_baseline/phase_c1_scaled/cli/train_tf_phase_c1_scaled.log` (701 lines)

### Failure
- ❌ Forward parity debug artifacts: `tf_baseline/phase_c1_scaled/analysis/forward_parity_debug_tf/` **NOT GENERATED**
  - No `stats.json`
  - No `offsets.json`
  - No normalized PNG grids

## Next Actions (Supervisor Decision Required)

### Option 1: Extend Guard Test to Cover Eval Path
Add a new guard test `test_non_xla_translation_guard_eval` that:
- Simulates evaluation/stitching batching
- Tests translation layer with edge cases (0 patches, empty batches)
- Exercises the reassembly code path

### Option 2: Investigate Eval Batch Shapes
Add instrumentation at `ptycho/tf_helper.py:789` to log:
- Batch shapes entering `translate_core` during eval
- Translation tensor shapes
- Reshape target dimensions

### Option 3: Defer TensorFlow Parity (POLICY-001)
Given five consecutive failures with the same error signature, proceed PyTorch-only per POLICY-001:
- Document TensorFlow eval/inference reshape bug as known limitation
- Complete Phase C forward parity using PyTorch evidence only
- Open separate initiative for TensorFlow eval/inference fix

## Cross-References

- Finding: `TF-NON-XLA-SHAPE-001` (docs/findings.md:98-100)
- Policy: `POLICY-001` (PyTorch mandatory, TensorFlow optional)
- Guard implementation: `ptycho/tf_helper.py:702-790`
- Guard regression tests: `tests/tf_helper/test_translation_shape_guard.py:46-87`
- Prior blocker attempts:
  - 2025-11-14T232542Z
  - 2025-11-14T233123Z
  - 2025-11-14T153747Z
  - 2025-11-14T155500Z

## Conclusion

The guard test is GREEN, confirming the training-time fix works. However, TensorFlow evaluation/inference/stitching continues to fail with reshape errors not covered by the current guard. This is the **fifth consecutive failure** with identical symptoms, indicating a systematic gap between the training code path (which the guard covers) and the eval code path (which triggers the actual error).

Recommendation: Supervisor should decide whether to invest in eval path debugging or proceed PyTorch-only per POLICY-001.
