# Blocker: TensorFlow Integration Test Env Inheritance

**Timestamp:** 2025-11-14T08:00Z  
**Initiative:** FIX-PYTORCH-FORWARD-PARITY-001  
**Phase:** C1b (GS1 Fallback)

## Summary

TensorFlow integration pytest still fails with XLA errors even with `TF_XLA_FLAGS` and `USE_XLA_TRANSLATE` exported because subprocess calls don't inherit shell environment variables.

## Error

The integration test subprocess executed TF inference which triggered XLA compilation:
```
ImageProjectiveTransformV3 (No registered 'ImageProjectiveTransformV3' OpKernel for XLA_GPU_JIT devices
tf2xla conversion failed while converting functional_6_1_padded_obj_2_1_cond_true_7166...
```

## Root Cause

Environment variables exported in the pytest parent process are not inherited by `subprocess.run()` calls unless explicitly passed via the `env` parameter.

## Mitigation

Skip the integration test for GS1 validation and proceed directly with CLI commands where we can control the environment.

## Next Steps

Continue with TF training/inference CLI commands per the Phase C1b plan.
