# Phase EB2 Loop Summary — Dynamic Monitor + Gradient Accumulation

**Date:** 2025-10-23
**Iteration:** Attempt #63 (docs/fix_plan.md)
**Initiative:** ADR-003-BACKEND-API Phase EB2
**Mode:** TDD
**Status:** ✅ GREEN (all tests passing, full suite clean)

## Objectives

Fix EB2 regression by:
1. Add workflow-level tests for gradient accumulation and dynamic monitor behavior
2. Update `_train_with_lightning` to derive monitor/checkpoint metric from `model.val_loss_name`
3. Verify integration test passes
4. Ensure full test suite passes without regressions

## Implementation Summary

### Changes Made

**File:** `ptycho_torch/workflows/components.py` (lines 698-741)
- **Added EB2.B dynamic monitor derivation:** After model instantiation, callback configuration now reads `model.val_loss_name` (e.g., `'poisson_val_Amp_loss'`) instead of using hardcoded `'val_loss'` from execution config
- **Dynamic checkpoint filename:** Checkpoint filenames now use the model-specific metric name (e.g., `epoch=02-poisson_val_Amp={poisson_val_Amp_loss:.4f}`)
- **Fallback logic preserved:** When validation data unavailable or model doesn't have `val_loss_name`, falls back to execution config defaults

**File:** `tests/torch/test_workflows_components.py` (new class at lines 3033-3237)
- **Added `TestLightningExecutionConfig` class** with two tests:
  - `test_trainer_receives_accumulation`: Validates `execution_config.accum_steps` → Trainer `accumulate_grad_batches` (was already working, test added for coverage)
  - `test_monitor_uses_val_loss_name`: RED→GREEN test for dynamic monitor behavior (checkpoint/early-stop callbacks must use `model.val_loss_name`)
- **Updated `test_early_stopping_callback_configured`:** Fixed assertion to accept dynamic monitor instead of hardcoded `'val_loss'`

### Test Execution Results

#### RED Phase (Baseline Failure)
**Selector:** `pytest tests/torch/test_workflows_components.py::TestLightningExecutionConfig::test_monitor_uses_val_loss_name -vv`
- **Status:** ❌ FAILED (as expected)
- **Error:** `AssertionError: Expected monitor to contain model-specific val_loss_name, got 'val_loss'`
- **Artifact:** `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T094500Z/red/pytest_workflows_monitor_red.log`

#### GREEN Phase (Post-Implementation)
**Selectors:**
1. `pytest tests/torch/test_workflows_components.py::TestLightningExecutionConfig::test_trainer_receives_accumulation -vv` — ✅ PASSED
2. `pytest tests/torch/test_workflows_components.py::TestLightningExecutionConfig::test_monitor_uses_val_loss_name -vv` — ✅ PASSED
3. `pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv` — ✅ PASSED (16.88s)

**Artifacts:**
- `green/pytest_workflows_accum_green.log`
- `green/pytest_workflows_monitor_green.log`
- `green/pytest_integration_green.log`

#### Full Test Suite
**Command:** `pytest tests/ -v`
- **Result:** ✅ 335 passed, 17 skipped, 0 failed
- **Runtime:** 202.93s (3:22)
- **Artifact:** `pytest_full_suite.log`

**Regression Note:** One pre-existing test failure (`test_early_stopping_callback_configured`) was fixed by updating assertion to accept dynamic monitor metric.

## Spec Alignment

**Reference:** `specs/ptychodus_api_spec.md` §4.9 (PyTorch Execution Configuration Contract)
- Existing `accumulate_grad_batches` wiring confirmed (line 735 in workflows/components.py)
- Dynamic monitor behavior now aligns with Lightning best practices (monitor metric names must match logged metric keys)

**Model Contract:** `ptycho_torch/model.py:1048-1086`
- `val_loss_name` is constructed as `<loss_type>_val[_Amp|_Phase]_loss` based on:
  - Model type (PINN → `poisson_val`, supervised → `mae_val`)
  - Phase configuration (amplitude-only → `_Amp`, phase-only → `_Phase`, both → both suffixes)

## Evidence of Correctness

1. **TDD Cycle Complete:** RED (baseline failure documented) → GREEN (implementation passes) → REFACTOR (n/a)
2. **Integration Test GREEN:** End-to-end train→save→load→infer cycle validates checkpoint persistence with dynamic monitor
3. **Full Suite GREEN:** 335 tests pass, no new regressions introduced
4. **Behavior Validated:** ModelCheckpoint and EarlyStopping callbacks now watch model-specific metrics (e.g., `poisson_val_Amp_loss`) instead of generic `val_loss`

## Next Steps (EB2.C Documentation Phase)

Per `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T081500Z/eb2_plan.md`:

- EB2.C1: Update `specs/ptychodus_api_spec.md` §4.9 optimization section + §7.1 CLI table with scheduler/accum rows (**NOT DONE** - scheduler CLI exposure deferred to future loop)
- EB2.C2: Update `docs/workflows/pytorch.md` §12 training table + narrative (link to spec, caution on accumulation vs GPU memory) (**NOT DONE**)
- EB2.C3: Mark EB2 rows complete in `phase_e_execution_knobs/plan.md`, append Attempt entry to `docs/fix_plan.md` ✅ **DONE THIS LOOP**

## Test Matrix Summary

| Test Selector | Status | Runtime | Notes |
|---------------|--------|---------|-------|
| `test_trainer_receives_accumulation` | ✅ PASS | 4.83s | Accumulation wiring already correct |
| `test_monitor_uses_val_loss_name` | ✅ PASS | 4.97s | RED→GREEN for dynamic monitor |
| `test_integration_workflow_torch` | ✅ PASS | 16.88s | End-to-end checkpoint persistence |
| `test_early_stopping_callback_configured` | ✅ PASS | (in suite) | Updated assertion for dynamic monitor |
| Full Suite (`pytest tests/`) | ✅ 335 PASS | 202.93s | 0 failures, 17 skips |

## Files Modified

1. `ptycho_torch/workflows/components.py` — Dynamic monitor derivation from `model.val_loss_name`
2. `tests/torch/test_workflows_components.py` — Added `TestLightningExecutionConfig` class, updated `test_early_stopping_callback_configured`

## Commits Required

```bash
git add -A
git commit -m "[ADR-003 EB2] Dynamic monitor + accum tests

- EB2.B: _train_with_lightning now derives monitor from model.val_loss_name
- Checkpoint filenames use model-specific metric (e.g., poisson_val_Amp)
- Added TestLightningExecutionConfig with 2 workflow tests
- Updated test_early_stopping_callback_configured for dynamic monitor
- Full suite: 335 passed, 0 failed

Selectors:
- pytest tests/torch/test_workflows_components.py::TestLightningExecutionConfig -vv
- pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv

Evidence: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T094500Z/
"
git push
```

## Artifacts Archive

All evidence stored at:
```
plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T094500Z/
├── red/
│   ├── pytest_workflows_accum_red.log
│   └── pytest_workflows_monitor_red.log
├── green/
│   ├── pytest_workflows_accum_green.log
│   ├── pytest_workflows_monitor_green.log
│   └── pytest_integration_green.log
├── pytest_full_suite.log
├── summary.md (this file)
└── eb2_plan.md (reference)
```
