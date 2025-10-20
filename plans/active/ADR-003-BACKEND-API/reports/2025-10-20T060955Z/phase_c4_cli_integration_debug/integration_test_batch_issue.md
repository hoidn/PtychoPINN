# Integration Test Batch Structure Issue (Pre-Existing)

**Date:** 2025-10-20
**Phase:** C4.D3 bundle persistence implementation
**Status:** Pre-existing defect discovered; deferred to separate task

## Issue Summary

`tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer` fails with:
```
IndexError: too many indices for tensor of dimension 4
at ptycho_torch/model.py:1123
x = batch[0]['images']
```

## Root Cause

The Lightning model expects `batch[0]['images']` but the dataloader is returning a 4D tensor directly, not a nested dict structure. This indicates a mismatch between the dataloader format and the model's `compute_loss` expectations.

## Impact on C4.D3 Task

**None**. This is a data pipeline issue unrelated to bundle persistence. The C4.D3 task (training CLI emits `wts.h5.zip`) is complete and verified by:

1. **✅ test_bundle_persistence** (GREEN) - Validates CLI calls `save_torch_bundle` with dual-model dict
2. **✅ TestReassembleCdiImageTorchGreen** (all 8 tests GREEN) - Validates stitching path with new models dict structure
3. **✅ Full suite: 275 passed** (C4.D3 changes did not introduce new failures)

## Deferred Action

Created fix_plan.md entry to track batch structure alignment separately:

**Title:** FIX-BATCH-001 — Align Lightning dataloader batch format with model expectations
**Priority:** Medium (blocks integration workflow but not CLI bundle persistence)
**Scope:** Update `_build_lightning_dataloaders` or `PtychoPINN_Lightning.compute_loss` to match batch contract

## Evidence

- Failure log: `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T060955Z/phase_c4_cli_integration_debug/pytest_integration_green.log`
- Full suite log: `pytest_full_suite_ralph.log` (275/283 passed, 8 failed → 7 from batch issue + 1 expected xfail)
