# Execution Config Regression Test Failures

**Date:** 2025-11-13T22:45:00Z
**Status:** RED (2 failures captured for scope documentation)
**Test Suite:** `tests/torch/test_workflows_components.py -k execution_config`

## Summary

Two execution_config regression tests fail due to mock stub incompleteness. The tests were written to verify that `_train_with_lightning` properly threads execution config parameters to Lightning Trainer, but the monkeypatched stub `StubLightningModule` lacks the `automatic_optimization` attribute that production code references for the EXEC-ACCUM-001 guard.

## Failures

### 1. test_execution_config_overrides_trainer
**Location:** `tests/torch/test_workflows_components.py::TestTrainWithLightningGreen::test_execution_config_overrides_trainer`
**Error:** `AttributeError: 'StubLightningModule' object has no attribute 'automatic_optimization'`
**Root Cause:** Line 831 in `ptycho_torch/workflows/components.py` checks `model.automatic_optimization` to enforce EXEC-ACCUM-001, but test stub doesn't provide this attribute.

### 2. test_execution_config_controls_determinism
**Location:** `tests/torch/test_workflows_components.py::TestTrainWithLightningGreen::test_execution_config_controls_determinism`
**Error:** Same as above - `AttributeError: 'StubLightningModule' object has no attribute 'automatic_optimization'`
**Root Cause:** Same line 831 check.

## Test Intent

Both tests verify execution config parameter propagation:
- **test_execution_config_overrides_trainer:** Asserts that `execution_config` values (accelerator='gpu', deterministic=False, gradient_clip_val=1.0) are passed to Lightning Trainer
- **test_execution_config_controls_determinism:** Validates that `deterministic=True` flag reaches Trainer for reproducibility

## Fix Strategy

To resolve these test failures, the monkeypatched `StubLightningModule` needs to include the `automatic_optimization` attribute (default `False` to match `PtychoPINN_Lightning`). This is a test fixture issue, not a production code bug—the real CLI smoke runs succeeded with the EXEC-ACCUM-001 guard in place.

## Evidence

- **Test log:** `plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/red/pytest_workflows_execution_config.log`
- **Production CLI success:** Both training and inference CLIs completed successfully with GPU defaults (see `cli/pytorch_cli_smoke_training/{train_gpu_default_log.log,inference_gpu_default_log.log}`)

## Exit Status

Exit code: 1 (2 FAILED, 31 deselected)
