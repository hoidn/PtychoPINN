# Phase EB1 Blockers (2025-10-20T160900Z)

## Status: RESOLVED (2025-10-20T160900Z)

### Original Blocker: Mock Patching Failures
**Issue:** Tests in `tests/torch/test_workflows_components.py::TestLightningCheckpointCallbacks` were failing because:
1. Tests patched `lightning.Trainer` instead of `lightning.pytorch.Trainer`
2. `_train_with_lightning` calls `create_training_payload` which validates file paths
3. Fixture uses dummy paths `/tmp/dummy_train.npz` that don't exist
4. Function exits early with `FileNotFoundError` before reaching callback instantiation code

### Resolution Strategy
Fixed test mocks to:
1. ✅ Patch `lightning.pytorch.Trainer` correctly (lines 2848, 2914, 2989)
2. ✅ Move imports outside patch context to avoid AttributeError
3. ✅ Add `tmp_path` fixture to create real NPZ files for path validation

### Follow-Up
All three checkpoint callback tests now GREEN (see `green/pytest_workflows_checkpoint_final.log`). Remaining EB1 work is documentation/ledger cleanup tracked in `phase_e_execution_knobs/plan.md` row EB1.F.
