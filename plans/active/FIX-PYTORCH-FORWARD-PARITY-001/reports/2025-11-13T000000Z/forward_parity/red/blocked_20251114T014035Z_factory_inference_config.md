# Blocker: Patch Stats Flags Not Applied to InferenceConfig

**Date:** 2025-11-14
**Focus:** FIX-PYTORCH-FORWARD-PARITY-001 Phase A
**Status:** blocked

## Issue
Patch stats instrumentation flags (`--log-patch-stats`, `--patch-stats-limit`) are parsed by CLI and passed to overrides dict, but they are not being applied to the PT InferenceConfig instance that the Lightning model receives.

## Root Cause
The `create_training_payload()` function in `ptycho_torch/config_factory.py` does not create a PT InferenceConfig at all. The training workflow uses an InferenceConfig created through a different path (legacy config loading), which doesn't receive the CLI override values.

## Evidence
1. Training completed without patch stats output: `outputs/torch_forward_parity_baseline/analysis/` does not exist
2. No "Torch patch stats" messages in training log (see `cli/train_patch_stats.log`)
3. Factory instantiation at `ptycho_torch/config_factory.py:201-236` only creates PTDataConfig, PTModelConfig, PTTrainingConfig - no PTInferenceConfig

## Required Fix
Update `ptycho_torch/config_factory.py:create_training_payload()` to:
1. Create a PT InferenceConfig instance
2. Apply `log_patch_stats` and `patch_stats_limit` from overrides
3. Add it to the TrainingPayload dataclass
4. Ensure the main() function uses this instance when creating the model

## Code Already Landed (Phase A Nucleus)
- PatchStatsLogger class (`ptycho_torch/patch_stats_instrumentation.py`)
- CLI flags in `ptycho_torch/train.py` (lines 501-522)
- Model integration (`ptycho_torch/model.py`):
  - Import at line 19
  - Instantiation in `__init__` at lines 1041-1045
  - Updated `_log_patch_stats` method at lines 1157-1180
  - `on_train_end` hook at lines 1335-1344
- Test coverage (`tests/torch/test_patch_stats_cli.py`) - PASSED

## Next Steps
1. Update factory to create PT InferenceConfig with override fields
2. Rerun training command with instrumentation flags
3. Verify JSON/PNG artifacts appear in analysis/

## Commands for Retry
```bash
python -m ptycho_torch.train \
  --train_data_file datasets/fly64/fly001_64_prepared_final_train.npz \
  --test_data_file datasets/fly64/fly001_64_prepared_final_test.npz \
  --output_dir outputs/torch_forward_parity_baseline \
  --max_epochs 10 --n_images 256 --gridsize 2 --batch_size 4 \
  --torch-loss-mode poisson --accelerator gpu --deterministic \
  --log-patch-stats --patch-stats-limit 2 --quiet
```
