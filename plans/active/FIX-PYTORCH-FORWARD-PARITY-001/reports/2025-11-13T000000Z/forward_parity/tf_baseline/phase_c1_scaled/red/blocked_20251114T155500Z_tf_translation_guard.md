# Blocker: TensorFlow Inference Translation Reshape Failure (Fourth Attempt)

**Date:** 2025-11-14T15:55:00Z
**Focus:** FIX-TF-C1D-SCALED-RERUN-001
**Status:** BLOCKED — Guard selector GREEN but scaled TF CLI failed during eval/inference

## Executive Summary

Fourth consecutive attempt to run Phase C1d scaled TensorFlow baseline has failed with identical error signature: `InvalidArgumentError` during inference at `ptycho/tf_helper.py:199` in `_translate_images_simple` when trying to reshape a 0-element tensor to shape `(4,)`.

**Critical Finding:** The guard test `test_non_xla_translation_guard` covers the **training-time** translation path and passes GREEN (1/1 in 5.00s), but does NOT exercise the **eval/inference** code path where the failure occurs.

## Evidence Files

- **Guard log:** `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/green/pytest_tf_translation_guard.log` (GREEN: 1 passed in 5.00s)
- **CLI log:** `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/tf_baseline/phase_c1_scaled/cli/train_tf_phase_c1_scaled.log` (705 lines, exit code 1)
- **Env capture:** `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/tf_baseline/phase_c1_scaled/cli/env_capture_scaled.txt`

## Execution Details

### Environment Configuration
```bash
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"
export USE_XLA_TRANSLATE=0
```

### CLI Command
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

## Error Signature

```
tensorflow.python.framework.errors_impl.InvalidArgumentError: Graph execution error:

Detected at node functional_1/padded_objs_with_offsets_1/translation_36_1/Reshape_4 defined at (most recent call last):
  <stack trace omitted for brevity>
  File "/home/ollie/Documents/PtychoPINN/ptycho/tf_helper.py", line 199, in _translate_images_simple

Input to reshape is a tensor with 0 values, but the requested shape has 4
	 [[{{node functional_1/padded_objs_with_offsets_1/translation_36_1/Reshape_4}}]]
```

**Location:** `ptycho/tf_helper.py:199` in `_translate_images_simple`

**Call Path:**
```
scripts/training/train.py:440
  → backend_selector.py:141 run_cdi_example_with_backend
  → components.py:846 run_cdi_example
  → components.py:726 train_cdi_model
  → train_pinn.py:96 train_eval
  → train_pinn.py:142 eval (model.predict)
  → custom_layers.py:82 ExtractPatchesPositionLayer.call
  → tf_helper.py:642 extract_patches_position
  → tf_helper.py:836 Translation.__call__
  → tf_helper.py:789 translate_core
  → tf_helper.py:199 _translate_images_simple
```

## Execution Timeline

1. **Training Phase:** Completed successfully (epoch 1/1)
   - 8/8 batches processed
   - Final loss: -2,700,843.25
   - No errors during training

2. **Evaluation Phase:** Failed during `model.predict()`
   - Error occurred in inference/eval path
   - Triggered during stitching operation
   - 0-element tensor produced where batch dimension of 4 was expected

## Root Cause Analysis

The guard test `tests/tf_helper/test_translation_shape_guard.py::test_non_xla_translation_guard` only validates the **training-time** forward path where batch dimensions are controlled by the training loop. It does not exercise the **inference/eval** path where:

1. Different batching logic applies
2. The stitching operation processes test data differently
3. Batch dimensions can become 0-element tensors

The training path succeeds because:
- Controlled batch_size=4
- Data shapes match expectations
- `translate_core` fallback works correctly

The eval path fails because:
- Test dataset processing produces mismatched batch dimensions
- 0-element intermediate tensors are created
- `_translate_images_simple` cannot reshape 0-element tensor to shape (4,)

## Prior Attempts

This is the **fourth consecutive failure** with identical error signature:

1. **2025-11-14T15:37:47Z** — First blocker documented
2. **2025-11-14T23:31:23Z** — Second attempt, same error
3. **2025-11-14T23:25:42Z** — Third attempt, same error
4. **2025-11-14T15:55:00Z** — This attempt (fourth)

All failures share:
- Same error location: `tf_helper.py:199`
- Same reshape request: 0-element tensor → shape (4,)
- Same call path through eval/inference
- Training succeeds, eval fails

## Next Actions (Recommendations)

### Option 1: Extend Guard Test (Recommended)
Add eval/inference coverage to the guard test:
```python
def test_eval_inference_translation_guard():
    """Test translate_core during eval/inference path with test dataset."""
    # Setup test data similar to train.py eval path
    # Force non-XLA via mock
    # Verify no 0-element tensor reshapes
```

### Option 2: Investigate Eval Batch Shape Logic
Debug why `_translate_images_simple` receives 0-element tensors during eval:
- Add tracing to `tf_helper.py:789` (translate_core)
- Log batch shapes before fallback
- Identify where 0-element tensor originates

### Option 3: Defer TensorFlow Parity
Accept PyTorch-only evidence for Phase C1d and document TensorFlow inference blocker in:
- `docs/findings.md` (update TF-NON-XLA-SHAPE-001)
- Phase C1 plan
- Return to TensorFlow parity after PyTorch forward-parity complete

## Blocking Conditions

This blocker prevents:
- ✗ Phase C1d TensorFlow scaled baseline artifacts (`forward_parity_debug_tf/`)
- ✗ TensorFlow reassembly evidence for parity validation
- ✗ Full backend parity demonstration
- ✗ Resumption of FIX-PYTORCH-FORWARD-PARITY-001 parent focus

## Return Condition

This blocker is resolved when EITHER:

1. **Guard + CLI both GREEN:**
   - Guard test extended to cover eval path → GREEN
   - Scaled TF CLI completes without error
   - Artifacts exist: `$TF_BASE/analysis/forward_parity_debug_tf/{stats.json,offsets.json,*.png}`
   - Inventory updated with sha1 and env capture

2. **Documented deferral:**
   - Decision recorded to proceed PyTorch-only
   - TF-NON-XLA-SHAPE-001 updated in `docs/findings.md`
   - Phase C1d marked TensorFlow-deferred in plan
   - FIX-PYTORCH-FORWARD-PARITY-001 resumes with PyTorch evidence only

---

**Blocker Owner:** Ralph
**Escalation:** After fifth consecutive identical failure, escalate to supervisor for scope decision
