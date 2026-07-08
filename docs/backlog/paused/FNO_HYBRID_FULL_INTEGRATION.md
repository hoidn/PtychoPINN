# FNO/Hybrid Full Pipeline Integration

**Created:** 2026-01-27
**Status:** Completed (2026-01-27)
**Priority:** High
**Related:** `ptycho_torch/generators/fno.py`, `ptycho_torch/model.py`, `ptycho_torch/workflows/components.py`
**Depends on:** FNO_HYBRID_TESTING_GAPS.md (completed 2026-01-27)
**Plan:** `docs/plans/2026-01-27-fno-hybrid-full-integration-plan.md`

## Summary

The FNO and Hybrid U-NO generators are **now fully integrated** with the `PtychoPINN_Lightning` training pipeline. This enables performance comparison against PINN/CNN baseline and production use for reconstruction.

## Completion Notes (2026-01-27)

The full integration was implemented following the plan in `docs/plans/2026-01-27-fno-hybrid-full-integration-plan.md`:

### What Was Done

1. **Real/Imag Adapter Path** (`ptycho_torch/model.py`)
   - Added `_real_imag_to_complex_channel_first()` helper function
   - Extended `PtychoPINN` with `generator` and `generator_output` parameters
   - Added `_predict_complex()` method for handling both amp_phase and real_imag modes
   - Extended `PtychoPINN_Lightning` with `generator_module` and `generator_output` parameters

2. **Generator Registry Wiring** (`ptycho_torch/workflows/components.py`)
   - `_train_with_lightning` now uses `resolve_generator()` to build models
   - FNO/Hybrid generators return `PtychoPINN_Lightning` instances with physics pipeline

3. **FNO/Hybrid Generators** (`ptycho_torch/generators/fno.py`)
   - Updated `FnoGenerator.build_model()` to use dict-style `pt_configs`
   - Updated `HybridGenerator.build_model()` to use dict-style `pt_configs`
   - Both generators now return `PtychoPINN_Lightning` with `generator_output="real_imag"`
   - Fixed `_FallbackSpectralConv2d` to handle varying spatial dimensions

4. **Inference Path Alignment** (`ptycho_torch/workflows/components.py`)
   - Fixed `_build_inference_dataloader` to convert channel-last to channel-first format
   - Fixed `_reassemble_cdi_image_torch` to use `forward_predict` with proper arguments
   - Fixed scale factor shape for `IntensityScalerModule` broadcasting

## Current State (after integration)

| Component | Status |
|-----------|--------|
| FNO/Hybrid generators (`fno.py`) | ✅ Unit tested, Lightning integrated |
| `HAS_NEURALOPERATOR` flag | ✅ Added, tests skip gracefully |
| `_LossHistoryCallback` | ✅ Added, collects per-epoch loss |
| Synthetic fixture (`synthetic_ptycho_npz`) | ✅ Added to conftest.py |
| Lightning training with CNN | ✅ Working |
| Lightning training with FNO/Hybrid | ✅ **Working** |
| Inference/stitching with FNO/Hybrid | ✅ **Working** |

## Acceptance Criteria

- [x] `TrainingConfig(model=ModelConfig(architecture='fno'))` trains via Lightning
- [x] `TrainingConfig(model=ModelConfig(architecture='hybrid'))` trains via Lightning
- [x] Loss history collected for all architectures
- [ ] Reconstruction quality metrics (SSIM, MSE) computed (future work)
- [ ] Performance comparison documented (FNO vs Hybrid vs CNN) (future work)

## Test Coverage

The following tests verify the integration:

| Test File | Test Name | Status |
|-----------|-----------|--------|
| `tests/torch/test_generator_adapter.py` | `test_real_imag_to_complex_channel_first` | ✅ Pass |
| `tests/torch/test_generator_adapter.py` | `test_ptychopinn_with_custom_generator` | ✅ Pass |
| `tests/torch/test_generator_adapter.py` | `test_ptychopinn_predict_complex_real_imag` | ✅ Pass |
| `tests/torch/test_fno_lightning_integration.py` | `test_train_history_collects_epochs_for_fno_hybrid[fno]` | ✅ Pass |
| `tests/torch/test_fno_lightning_integration.py` | `test_train_history_collects_epochs_for_fno_hybrid[hybrid]` | ✅ Pass |
| `tests/torch/test_fno_lightning_integration.py` | `test_reassemble_cdi_image_torch_handles_real_imag_outputs` | ✅ Pass |

## Related Files

- `ptycho_torch/model.py` - `PtychoPINN_Lightning` class (modified)
- `ptycho_torch/generators/fno.py` - FNO/Hybrid generator implementations (modified)
- `ptycho_torch/generators/registry.py` - Generator registry
- `ptycho_torch/workflows/components.py` - Training orchestration (modified)
- `tests/torch/test_generator_adapter.py` - Unit tests (created)
- `tests/torch/test_fno_lightning_integration.py` - Integration tests (extended)

## Future Work

1. **Performance Comparison** - Train FNO/Hybrid on same datasets as CNN and compare SSIM/MSE metrics
2. **Documentation** - Document reconstruction quality differences
3. **Probe Handling** - Extract real probe from container instead of using dummy probe for inference
