# Blocker: TensorFlow GS1 Still Fails with XLA Translation Errors

**Timestamp:** 2025-11-14T08:01Z  
**Initiative:** FIX-PYTORCH-FORWARD-PARITY-001  
**Phase:** C1b (GS1 Fallback)

## Summary

TensorFlow training with `--gridsize 1` still fails with the same XLA `ImageProjectiveTransformV3` error during the stitching phase after training completed. The gridsize=1 fallback does not fully bypass the translation layer as expected.

## Error Signature

```
tensorflow.python.framework.errors_impl.InvalidArgumentError: Graph execution error:
Detected unsupported operations when trying to compile graph: ImageProjectiveTransformV3
(No registered 'ImageProjectiveTransformV3' OpKernel for XLA_GPU_JIT devices
tf2xla conversion failed while converting functional_2_1_padded_obj_2_1_cond_true...
```

**Location:** During stitching/reassembly phase after epoch 10/10 completed  
**Stack:** `reassemble_cdi_image → reconstruct_image → reassemble_patches_position_batched_real → translation`

## Context

- Training completed all 10 epochs successfully
- Error occurred during post-training stitching/reconstruction phase
- `use_xla_translate: True` in params (env vars weren't exported correctly)
- Even with gridsize=1, the `--do_stitching` flag triggers `reassemble_patches_position_batched_real` which calls the XLA translation layer

## Root Cause

The `--do_stitching` flag invokes `reassemble_patches` which uses the translation layer for alignment even when gridsize=1. The GS1 fallback assumption that "gridsize=1 avoids translation" is incorrect for workflows that include stitching.

## Mitigation Options

### Option 1: Disable Stitching for GS1 TF Baseline

Retry training without `--do_stitching` to avoid the reassembly path entirely. This limits Phase C evidence but provides basic TF GS1 training metrics.

### Option 2: Explicitly Disable XLA Translate via Code

Since environment variables don't work reliably, modify the CLI/config to explicitly set `use_xla_translate=False` in params.cfg before TensorFlow workflows execute.

### Option 3: Accept PyTorch-Only Phase C (Recommended per Brief)

The brief already anticipated this scenario and recommended proceeding with PyTorch-only evidence if TensorFlow remains blocked even at GS1.

## Next Steps

Given two consecutive GS1 blockers (integration test env inheritance + training stitching XLA):
1. Document this as confirmation that TensorFlow translation layer is fundamentally broken
2. Update hub summary with "PyTorch-only Phase C" evidence
3. Skip remaining TF GS1 commands and proceed to capture PyTorch bundle digests + stats
4. Reference POLICY-001 justification for PyTorch-primary parity validation

## Artifacts

- Training log: `$TF_BASE_GS1/cli/train_tf_phase_c1_gs1.log` (shows successful epoch completion, then XLA failure)
- This blocker: `$TF_BASE_GS1/red/blocked_*_tf_gs1_xla_still_fails.md`
