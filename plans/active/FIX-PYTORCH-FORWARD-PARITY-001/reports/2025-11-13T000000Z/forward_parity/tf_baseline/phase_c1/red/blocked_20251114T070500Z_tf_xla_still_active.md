# Phase C1 TensorFlow Baseline Blocker — XLA Compilation Errors Despite TF_XLA_FLAGS

**Timestamp:** 2025-11-14T070500Z
**Focus:** FIX-PYTORCH-FORWARD-PARITY-001 Phase C1
**Brief Step:** 3 (TensorFlow training with identity dataset)

## Problem Statement

TensorFlow training with `fly64_coord_variants/fly001_64_train_converted_identity.npz` failed during first training step with XLA compilation error in `projective_warp_xla_jit`, even though `TF_XLA_FLAGS="--tf_xla_auto_jit=0"` was exported to disable dynamic JIT compilation.

## Error Signature

```
tensorflow.python.framework.errors_impl.InvalidArgumentError: Exception while executing TFLite code.
...
tf2xla conversion failed while converting __inference_projective_warp_xla_jit_2335[].
Run with TF_DUMP_GRAPH_PREFIX=/path/to/dump/dir and --vmodule=xla_compiler=2 to obtain a dump of the compiled functions.
[[functional_1/padded_objs_with_offsets_1/translation_36_1/PartitionedCall_1]] [Op:__inference_one_step_on_data_distributed_38575]
```

**Error Location:** `ptycho/projective_warp_xla.py:182` → `projective_warp_xla` → `projective_warp_xla_jit` → dynamic_padder RET_CHECK

## Root Cause Analysis

1. **Environment flag insufficient:** Setting `TF_XLA_FLAGS="--tf_xla_auto_jit=0"` disables TensorFlow's automatic XLA compilation for eligible operations
2. **Explicit XLA call path:** The code explicitly calls `translate_xla` functions which are decorated with `@tf.function(jit_compile=True)`, bypassing the auto_jit flag
3. **params.cfg hardcoded:** Output shows `use_xla_translate: True` in params.cfg, forcing XLA path regardless of environment flags
4. **Identity coordinate issue:** This specific dataset (identity-transformed coordinates) triggers XLA compilation failure in projective transform path

## Evidence

- **Integration test:** PASSED (34.75s) at `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/tf_baseline/phase_c1/green/pytest_tf_integration.log`
- **Training log:** Failed during first epoch at `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/tf_baseline/phase_c1/cli/train_tf_phase_c1.log`
- **TF_XLA_FLAGS:** Captured at `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/tf_baseline/phase_c1/green/env_capture.txt`

## Related Findings

- **XLA-DYN-DOT-001:** TF XLA JIT failures with dynamic shapes in einsum/dot operations (docs/findings.md:8)
- **Finding pattern match:** Exact same RET_CHECK failure location reported in previous Phase C1 attempt (docs/fix_plan.md:68)

## Mitigation Options

Per brief step 7 (Fallback path):

### Option A: Switch to non-identity dataset (RECOMMENDED)
- Use `datasets/fly64_coord_variants/fly001_64_train_converted.npz` (non-identity coordinates)
- This dataset successfully trained PyTorch Phase B3, suggesting better XLA compatibility
- **Impact:** Requires documenting in `$HUB/summary.md` whether PyTorch Phase B3 needs matching rerun before Phase C2 comparisons

### Option B: Disable XLA at params.cfg level
- Modify training workflow to set `use_xla_translate=False` before model creation
- **Risk:** May affect TF/PyTorch parity if XLA vs non-XLA paths produce different numerical results
- **Scope:** Requires code changes beyond environment configuration

### Option C: Escalate as TF-specific limitation
- Document per POLICY-001 that identity-coordinate datasets are incompatible with TensorFlow XLA compilation
- Proceed with PyTorch-only parity validation for Phase C
- **Impact:** Phase C TF baseline would remain blocked; initiative proceeds without cross-backend comparison

## Recommended Next Action

Execute Option A: Retry Phase C1 training with `datasets/fly64_coord_variants/fly001_64_train_converted.npz`, keeping all other parameters identical (256 groups, gridsize=2, neighbor_count=7, batch_size=4, 10 epochs). Document dataset change in artifact inventory and note whether PyTorch Phase B3 rerun is required for apples-to-apples comparison.

## Cross-References

- **Brief:** plans/active/FIX-PYTORCH-FORWARD-PARITY-001/implementation.md Phase C1 Action Plan step 7
- **Prior blocker:** docs/fix_plan.md:68 (2025-11-14T064950Z attempt with same error)
- **Knowledge base:** docs/findings.md XLA-DYN-DOT-001
- **Initiative summary:** plans/active/FIX-PYTORCH-FORWARD-PARITY-001/summary.md
