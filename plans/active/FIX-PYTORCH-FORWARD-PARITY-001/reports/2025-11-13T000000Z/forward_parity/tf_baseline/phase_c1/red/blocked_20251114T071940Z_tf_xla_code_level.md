# Blocker: TensorFlow XLA Still Active Despite TF_XLA_FLAGS

**Timestamp:** 2025-11-14T07:19:40Z
**Initiative:** FIX-PYTORCH-FORWARD-PARITY-001
**Phase:** C1 (TensorFlow Baseline Capture)
**Finding Reference:** XLA-DYN-DOT-001

## Summary

TensorFlow training failed with XLA compilation error even after setting `TF_XLA_FLAGS="--tf_xla_auto_jit=0"`. The environment flag only disables auto-JIT, but the codebase explicitly calls `translate_xla` functions (`ptycho/projective_warp_xla.py`), bypassing the environment setting.

## Error Signature

```
tf2xla conversion failed while converting __inference_projective_warp_xla_jit_66185[]
[[functional_1/padded_objs_with_offsets_1/translation_36_1/PartitionedCall_1]]
[Op:__inference_one_step_on_data_distributed_102449]
```

**Location:** `ptycho/projective_warp_xla.py:121,145,182` → `projective_warp_xla_jit` → `projective_warp_xla`

## Dataset Used

- **Train:** `datasets/fly64/fly001_64_train_converted.npz` (non-identity variant)
- **Test:** `datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_test.npz`
- **Configuration:** n_groups=256, gridsize=2, neighbor_count=7, batch_size=4, nepochs=10

## Environment

- `TF_XLA_FLAGS="--tf_xla_auto_jit=0"` was exported but ineffective
- Integration pytest passed (34.86s GREEN)
- Training failed during first epoch at first training step

## Root Cause

The `TF_XLA_FLAGS` environment variable controls TensorFlow's automatic XLA compilation of arbitrary ops, but does NOT disable explicit XLA usage in the code. The translation layer (`ptycho/tf_helper.py:798`) calls `translate_xla()` which invokes `projective_warp_xla_jit()`, forcing XLA compilation regardless of the environment flag.

## Code Path

```
ptycho/workflows/components.py:846 run_cdi_example()
→ ptycho/train_pinn.py:90 train_eval()
→ ptycho/model.py:593 train()
→ ptycho/custom_layers.py:155 ReassemblePatches.call()
→ ptycho/tf_helper.py:1070 reassemble_patches()
→ ptycho/tf_helper.py:1318 reassemble_patches_position_batched_real()
→ ptycho/tf_helper.py:834 Translation.call()
→ ptycho/tf_helper.py:798 translate()
→ ptycho/projective_warp_xla.py:294 translate_xla()
→ ptycho/projective_warp_xla.py:209 projective_warp_xla_jit()
→ XLA compilation failure
```

## Mitigation Options

### Option 1: Disable XLA at params.cfg level (Recommended for Phase C1)

Set `params.cfg['use_xla_translate'] = False` before training to force the non-XLA translation path. This requires:

1. Update `scripts/training/train.py` or `ptycho/workflows/components.py` to set `params.cfg['use_xla_translate'] = False` after `update_legacy_dict()` call
2. Or pass `--use-xla-translate=false` CLI flag if exposed
3. Rerun training with the same dataset and configuration

### Option 2: Fix XLA dynamic shape handling (Deferred)

Address the underlying XLA dynamic shape issue in `projective_warp_xla.py` per Finding XLA-DYN-DOT-001. This is a larger refactor and not required for Phase C1 parity proof.

### Option 3: Use PyTorch-only parity (Fallback)

If TensorFlow backend cannot be unblocked for this dataset, document the limitation and proceed with PyTorch-only validation per POLICY-001. This reduces Phase C scope but maintains forward progress.

## Dataset Parity Note

The brief specified `datasets/fly64_coord_variants/fly001_64_train_converted.npz`, but that file does not exist. Used `datasets/fly64/fly001_64_train_converted.npz` instead (non-identity variant from the same preparation pipeline). This diverges from Phase B3 PyTorch baseline which used `datasets/fly64_coord_variants/fly001_64_train_converted_identity.npz`.

**Consequence:** Even if TF baseline succeeds, the datasets will not match PyTorch Phase B3, requiring either:
- A matching PyTorch rerun with `fly001_64_train_converted.npz`, OR
- Documentation that Phase C2 comparison uses different train splits

## Next Steps

1. **Immediate:** Set `params.cfg['use_xla_translate'] = False` and retry training
2. **If successful:** Continue with inference and debug dump capture
3. **If blocked:** Document in `docs/fix_plan.md` and propose Option 3 (PyTorch-only) to supervisor

## Artifacts

- **Integration pytest:** `$TF_BASE/green/pytest_tf_integration.log` (PASSED, 34.86s)
- **Training log:** `$TF_BASE/cli/train_tf_phase_c1.log` (FAILED at first epoch)
- **This blocker:** `$TF_BASE/red/blocked_20251114T071940Z_tf_xla_code_level.md`

## References

- **Finding:** XLA-DYN-DOT-001 (docs/findings.md:8)
- **Config policy:** CONFIG-001 (docs/findings.md:13)
- **PyTorch policy:** POLICY-001 (docs/findings.md:11)
- **Brief:** plans/active/FIX-PYTORCH-FORWARD-PARITY-001/input.md:3-5
