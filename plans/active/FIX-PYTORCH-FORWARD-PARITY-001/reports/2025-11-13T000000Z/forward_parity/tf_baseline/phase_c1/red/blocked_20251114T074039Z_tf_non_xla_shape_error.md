# Blocker: Non-XLA Translation Path Shape Mismatch

**Timestamp:** 2025-11-14T07:40Z  
**Initiative:** FIX-PYTORCH-FORWARD-PARITY-001  
**Phase:** C1 (TensorFlow Baseline Capture)  
**Finding Reference:** New issue (non-XLA translate_core shape error)

## Summary

After successfully disabling XLA via `USE_XLA_TRANSLATE=0`, TensorFlow training now fails with a shape mismatch in the non-XLA translation path during the first epoch. The environment mitigation worked (XLA is disabled), but revealed a latent bug in the fallback implementation.

## Error Signature

```
Shapes of all inputs must match: values[0].shape = [4] != values[2].shape = [128]
	 [[{{node functional_1/padded_objs_with_offsets_1/translation_36_1/stack_1}}]]
[Op:__inference_one_step_on_data_distributed_109105]
```

**Location:** `ptycho/tf_helper.py:798` → `translate` → line 749 `translate_core`  
**Stack:** `ReassemblePatchesLayer.call` → `extract_patches_position` → `Translation.call` → `translate` → `translate_core`

## Environment

- `TF_XLA_FLAGS="--tf_xla_auto_jit=0"` ✅ exported
- `USE_XLA_TRANSLATE=0` ✅ exported and working (no XLA compilation attempts)
- **Dataset:** `datasets/fly64/fly001_64_train_converted.npz` (non-identity variant)
- **Configuration:** n_groups=256, gridsize=2, neighbor_count=7, batch_size=4, nepochs=10

## Evidence

1. **XLA successfully disabled:** No `XlaRuntimeError` or `tf2xla conversion failed` messages
2. **Training progressed to first epoch:** Model built successfully, first batch forward pass started
3. **New failure mode:** Shape error in `translate_core` at `stack_1` node

The log shows training reached epoch 1/10 and started processing batches before failing, which is further than the XLA blocker.

## Root Cause Analysis

The non-XLA `translate_core` function has a shape incompatibility when stacking tensors:
- `values[0].shape = [4]` (likely batch size or gridsize-related)
- `values[2].shape = [128]` (likely a flattened offset or coordinate array)

This suggests the non-XLA translation path may not be fully tested or compatible with gridsize=2 workflows.

## Mitigation Options

### Option 1: Switch to gridsize=1 for Phase C1 (Quick Unblock)

Run the TF baseline with `--gridsize 1` to avoid the translation layer entirely (single patches don't require reassembly). This provides Phase C evidence but limits parity comparison to gridsize=1 only.

**Pros:**  
- Immediate unblock for Phase C1
- Still provides TF vs Torch comparison data
- Avoids both XLA and non-XLA translation bugs

**Cons:**  
- Not apples-to-apples with Phase B3 PyTorch baseline (which used gridsize=2)
- Would require a matching PyTorch gridsize=1 rerun for fair comparison

### Option 2: Debug and Fix translate_core (Deferred)

Investigate the shape mismatch in `ptycho/tf_helper.py:749` and fix the non-XLA path. This is a larger debugging effort that would benefit the entire TensorFlow backend but is out of scope for Phase C1 parity proof.

### Option 3: Proceed with PyTorch-only Phase C (Recommended)

Given two consecutive TensorFlow blockers (XLA failure, now non-XLA shape error), document this as a TensorFlow backend limitation per POLICY-001 and proceed with PyTorch-only validation for forward parity.

**Rationale:**
- Phase B3 already captured healthy PyTorch baseline with gridsize=2
- Two distinct TF bugs (XLA + non-XLA) indicate broader stability issues
- PyTorch backend is primary per POLICY-001; TF parity is nice-to-have

## Next Steps

1. **Immediate:** Document this blocker in `docs/fix_plan.md` with both error signatures (XLA + non-XLA)
2. **Decision:** Escalate to supervisor for Phase C direction:
   - Accept gridsize=1 TF baseline (Option 1)
   - Defer to PyTorch-only (Option 3, recommended)
   - Debug TF translate_core (Option 2, out of scope)

## Artifacts

- **Training log:** `$TF_BASE/cli/train_tf_phase_c1.log` (shows XLA disabled, then shape error)
- **Environment capture:** Env vars logged at top of training log
- **This blocker:** `$TF_BASE/red/blocked_*_tf_non_xla_shape_error.md`

## References

- **PyTorch policy:** POLICY-001 (docs/findings.md:11)
- **XLA blocker:** `$TF_BASE/red/blocked_20251114T071940Z_tf_xla_code_level.md`
- **Brief:** plans/active/FIX-PYTORCH-FORWARD-PARITY-001/input.md:3-6
