# Phase EB1 Blockers (2025-10-20T160900Z)

## Status: PARTIALLY RESOLVED

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

### Remaining Work
Need to complete test updates for all three test methods:
- `test_model_checkpoint_callback_configured`: IN PROGRESS (updated)
- `test_early_stopping_callback_configured`: PENDING (needs same NPZ file approach)
- `test_disable_checkpointing_skips_callbacks`: PENDING (needs same NPZ file approach)

### Next Steps
1. Apply tmp_path + NPZ file pattern to remaining two tests
2. Rerun full selector: `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py -k checkpoint -vv`
3. Capture GREEN logs if all pass
4. Update spec/workflow docs per EB1.A+EB1.F

### Time Spent
~90 minutes debugging mock strategies before identifying core issue (path validation)
