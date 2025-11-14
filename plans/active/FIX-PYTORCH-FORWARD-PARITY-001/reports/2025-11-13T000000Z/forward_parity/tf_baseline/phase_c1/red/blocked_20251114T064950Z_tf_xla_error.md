# Blocker: TensorFlow XLA Compilation Error (Phase C1)

**Date:** 2025-11-14T06:49:50Z
**Focus:** FIX-PYTORCH-FORWARD-PARITY-001 Phase C1
**Severity:** Critical

## Error Summary

TensorFlow training baseline failed during model compilation with XLA JIT error:

```
tensorflow.python.framework.errors_impl.InvalidArgumentError: Exception while converting functional_1/padded_objs_with_offsets_1/translation_36_1/PartitionedCall_1 to XLA:
XlaRuntimeError: INTERNAL: RET_CHECK failure (third_party/tensorflow/compiler/xla/service/dynamic_padder.cc:1009) ChooseSimpleReductionOperandToBroadcast(reduce)->has_value()
```

## Root Cause

The fly64 dataset with identity coordinate transformation triggers an XLA compilation error in the projective warp / translation layer during reassembly. This appears related to:
- Finding: XLA-DYN-DOT-001 (XLA dynamic shape handling in projective warp)
- Location: `ptycho/projective_warp_xla.py:182` → `__inference_projective_warp_xla_jit`

## Evidence

- Full log: `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/tf_baseline/phase_c1/cli/train_tf_phase_c1.log` (partial, tee failed on path)
- Error occurs during first training step after model compilation
- Stack trace shows: `ptycho/tf_helper.py:1318` → `reassemble_patches_position_batched_real` → `projective_warp_xla.py:145,182`

## Command Attempted

```bash
python scripts/training/train.py \
  --backend tensorflow \
  --train_data_file datasets/fly64_coord_variants/fly001_64_train_converted_identity.npz \
  --test_data_file datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_test.npz \
  --output_dir outputs/tf_forward_parity_baseline \
  --n_groups 256 \
  --gridsize 2 \
  --neighbor_count 7 \
  --batch_size 4 \
  --nepochs 10 \
  --do_stitching
```

## Relation to POLICY-001 / CONFIG-001

- **POLICY-001**: PyTorch is mandatory; TensorFlow issues should not block PyTorch work
- **CONFIG-001**: Legacy `params.cfg` was synchronized correctly per logs
- This blocker is specific to TensorFlow XLA with fly64 identity coords

## Proposed Mitigations

1. **Switch dataset**: Use the fly001_reconstructed_prepared train split instead of fly64_coord_variants/identity
   - This is the dataset used in PyTorch Phase B3 (`scaling_alignment/phase_b3/cli/train_patch_stats_scaling.log`)
   - Would provide better parity comparison basis

2. **Disable XLA**: Set `TF_XLA_FLAGS="--tf_xla_auto_jit=0"` environment variable
   - May significantly slow training
   - Not a long-term solution

3. **Use older TF baseline artifacts**: If prior successful TF runs exist with similar params, reference those instead

## Recommendation

Given that:
- Phase B3 PyTorch baseline used `fly001_64_train_converted.npz` (not `_identity`)
- The goal is TF vs Torch parity proof
- The XLA error is dataset-specific

**Action**: Retry Phase C1 with the standard fly64 coordinate dataset (non-identity transform) OR use the fly001_reconstructed dataset that PyTorch B3 used.

## References

- Finding: XLA-DYN-DOT-001 in `docs/findings.md:8`
- Phase B3 PyTorch command: `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/implementation.md:134-147`
- Spec: `docs/specs/spec-ptycho-runtime.md` (XLA guardrails)
