# Phase EB1 GREEN Summary (2025-10-20T160900Z)

## Status: SUCCESS ✅

All three Lightning checkpoint callback tests now passing after fixing test mocks.

## Changes Made

### Test Fixes (`tests/torch/test_workflows_components.py`)
**Root Cause:** Tests were patching `lightning.Trainer` instead of `lightning.pytorch.Trainer`, causing mocks to fail. Additionally, factory path validation prevented tests from running with dummy `/tmp/dummy_train.npz` paths.

**Solutions Applied:**
1. Updated all three test methods in `TestLightningCheckpointCallbacks`:
   - `test_model_checkpoint_callback_configured`
   - `test_early_stopping_callback_configured`
   - `test_disable_checkpointing_skips_callbacks`

2. For the first two tests (which validate callback wiring):
   - Added `tmp_path` fixture parameter
   - Created actual NPZ files with minimal valid data
   - Updated config paths to point to real files
   - Preserved mock patch targets for callbacks and Trainer

3. Created `tests/torch/conftest.py` with session-scoped `create_dummy_npz_files` fixture:
   - Auto-creates `/tmp/dummy_train.npz` and `/tmp/dummy_test.npz` at session start
   - Fixes all other tests using dummy paths without individual edits
   - Cleanup on session end

### Test Results

**Targeted selector (checkpoint tests):**
```bash
CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py -k checkpoint -vv
```
Result: **3 passed, 27 deselected, 4 warnings**

**Full test suite regression:**
```bash
CUDA_VISIBLE_DEVICES="" pytest tests/ -v
```
Result: **326 passed, 2 failed, 17 skipped, 85 warnings**

**Pre-existing failures** (not introduced by this work):
- `tests/torch/test_integration_workflow_torch.py::test_bundle_loader_returns_modules`
- `tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer`

## Validation Evidence

**RED logs:** `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-20T160900Z/red/pytest_workflows_checkpoint_red.log`

**GREEN logs:**
- CLI tests: `green/pytest_cli_checkpoint_green.log` (4 passed)
- Factory tests: `green/pytest_factory_checkpoint_green.log` (3 passed)
- Workflows tests (final): `green/pytest_workflows_checkpoint_final.log` (3 passed)

## Production Code Status
**No changes to production code.** The callback wiring in `ptycho_torch/workflows/components.py` lines 690-740 was already implemented correctly in a previous loop. This loop only fixed test scaffolding to observe the existing behavior.

## Next Steps (EB1.F)
1. Update spec §4.9 / §7.2 tables with `checkpoint_mode` field (EB1.A backlog)
2. Update `docs/workflows/pytorch.md` §12 CLI tables
3. Record Attempt #58 in `docs/fix_plan.md`
4. Mark EB1.D and EB1.E as `[x]` in phase plan checklist
