# FNO_INPUT_TRANSFORM_IGNORED

**Date:** 2026-01-29  
**Status:** Open  
**Category:** PyTorch config propagation / FNO-Hybrid input preprocessing  
**Systems:** PyTorch (`ptycho_torch/`), studies (`scripts/studies/`)

## Summary
`fno_input_transform` (e.g., `log1p`, `sqrt`) is **not applied** in PyTorch FNO/Hybrid runs when invoked through the Lightning workflow. The transform is set on `TrainingConfig.model`, but is dropped when `create_training_payload()` builds the PyTorch `ModelConfig`, so the generator always uses the default `"none"` transform. As a result, hyperparameter sweeps comparing `none` vs `log1p` vs `sqrt` are effectively identical.

## Evidence
Minimal repro using the same config path as the sweep:

1) Build a `TorchRunnerConfig` with `fno_input_transform='log1p'`.
2) `setup_torch_configs()` sets `TrainingConfig.model.fno_input_transform = log1p`.
3) `_train_with_lightning()` builds `factory_overrides` for `create_training_payload()` **without** `fno_input_transform`.
4) `payload.pt_model_config.fno_input_transform` remains `"none"`.

Observed with:
```
TrainingConfig model fno_input_transform: log1p
Payload pt_model_config fno_input_transform: none
```

## Root Cause
`ptycho_torch/workflows/components.py` builds `factory_overrides` but does not include `fno_input_transform`, so `create_training_payload()` falls back to its default.

Relevant code:
- `ptycho_torch/workflows/components.py` — `_train_with_lightning()` override block only forwards `fno_modes`, `fno_width`, `fno_blocks`, `fno_cnn_blocks`, `generator_output_mode`.

## Impact
- FNO/Hybrid sweeps comparing `none` vs `log1p` vs `sqrt` are invalid.
- Any analysis of input-transform effects is misleading; metrics will match the `"none"` path even when a transform is specified.

## Suggested Follow-ups
- Add `fno_input_transform` to `factory_overrides` in `_train_with_lightning()` so it propagates into the PyTorch `ModelConfig`.
- Add a small regression test that asserts `pt_model_config.fno_input_transform` matches the `TrainingConfig` value when set via the runner or factory overrides.
