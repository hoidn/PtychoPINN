# Environment Mitigation Summary — Phase C1

**Date:** 2025-11-14
**Initiative:** FIX-PYTORCH-FORWARD-PARITY-001
**Phase:** C1 TensorFlow Baseline

## Environment Configuration Attempts

### Attempt 1: TF_XLA_FLAGS only
- **Env:** `TF_XLA_FLAGS="--tf_xla_auto_jit=0"`
- **Result:** FAILED — XLA still active via explicit `translate_xla()` calls
- **Evidence:** `blocked_20251114T071940Z_tf_xla_code_level.md`

### Attempt 2: TF_XLA_FLAGS + USE_XLA_TRANSLATE
- **Env:** `TF_XLA_FLAGS="--tf_xla_auto_jit=0"` + `USE_XLA_TRANSLATE=0`
- **Result:** XLA disabled ✓, but non-XLA path has shape mismatch bug
- **Error:** `Shapes of all inputs must match: values[0].shape = [4] != values[2].shape = [128]`
- **Location:** `ptycho/tf_helper.py:749` (`translate_core` stack operation)
- **Evidence:** `blocked_20251114T074039Z_tf_non_xla_shape_error.md`

## Key Findings

1. **`USE_XLA_TRANSLATE=0` is effective:** No XLA compilation errors in Attempt 2
2. **Non-XLA translation path is broken:** Shape mismatch in `translate_core` for gridsize=2
3. **Progress vs Attempt 1:** Training reached first epoch before failing (vs immediate XLA crash)

## TensorFlow Baseline Status

**BLOCKED** — Two distinct failure modes:
- XLA path: RET_CHECK dynamic shape error (XLA-DYN-DOT-001)
- Non-XLA path: Shape mismatch in translate_core (new finding)

## Recommendations

Per brief step 7 and POLICY-001:

1. **Document both env vars and both blockers** in initiative summary and fix_plan
2. **Propose PyTorch-only Phase C evidence** to supervisor (TF backend has multiple translation bugs)
3. **Alternative:** Retry with `--gridsize 1` to avoid translation layer entirely (requires matching PyTorch rerun)

## Artifacts

- **Integration pytest:** `green/pytest_tf_integration.log` (also failed due to subprocess env inheritance)
- **Training log:** `cli/train_tf_phase_c1.log` (env vars exported, non-XLA shape error)
- **Blockers:** `red/blocked_*_{tf_xla_code_level,tf_non_xla_shape_error}.md`
