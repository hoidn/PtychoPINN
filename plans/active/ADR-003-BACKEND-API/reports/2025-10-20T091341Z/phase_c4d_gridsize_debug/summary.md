# Phase C4.D Gridsize Debugging Notes (2025-10-20T091341Z)

## Context
- Focus: ADR-003-BACKEND-API Phase C4.D — Lightning training parity after factory refactor.
- Trigger: Bundled integration selector now fails despite `_train_with_lightning` factory integration.

## Key Observations
- Targeted test `test_lightning_training_respects_gridsize` remains GREEN (channel count fix verified).
- Both `test_bundle_loader_returns_modules` and CLI smoke commands abort with `RuntimeError: shape '[4, 2, 1]' is invalid for input of size 16` during Lightning validation (see `pytest_bundle_loader_failure.log`).
- Failure occurs inside `ptycho_torch/helper.py::Translation`, implying grouped offset tensor shape no longer matches expected `(n, 2)` pairs for `gridsize=1`.

## Hypothesis
- CLI now constructs configs via `create_training_payload` without overriding `neighbor_count`.
- Factory default (`neighbor_count=6`) propagates through canonical `TrainingConfig` so `_train_with_lightning` reuses this value when calling the factory again.
- `_build_lightning_dataloaders` therefore assembles six-neighbor groups even when `gridsize=1`, yielding offset tensors with 16 elements per sample (8 coordinate pairs) and tripping Translation’s reshape.

## Evidence
- Reproduction command (stored in `pytest_bundle_loader_failure.log`):
  ```bash
  CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_bundle_loader_returns_modules -vv
  ```
- Scripted inspection shows `payload.tf_training_config.neighbor_count == 6` unless overrides supply `neighbor_count` explicitly.

## Next Steps
1. Ensure CLI overrides pass `neighbor_count` (default 4) into `create_training_payload`.
2. Re-run targeted selector + integration workflow after override fix.
3. Capture GREEN logs + CLI output, then update plan row B3 and docs/fix_plan Attempts.

