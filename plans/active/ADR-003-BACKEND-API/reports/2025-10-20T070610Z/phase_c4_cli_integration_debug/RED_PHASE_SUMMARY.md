# RED Phase Test Success Summary

**Test**: `test_lightning_poisson_count_contract`
**Date**: 2025-10-20
**Status**: ✅ PASSING (RED phase complete)

## Objective

Verify that the current PyTorch Poisson loss implementation correctly raises a ValueError when given amplitude floats instead of integer photon counts, demonstrating the bug exists before fixing it (TDD RED phase).

## Issues Fixed

### Issue 1: Config instantiation errors
**Problem**: Test was using TensorFlow config classes but needed PyTorch config classes.

**Solution**: Import and instantiate PyTorch config classes from `ptycho_torch.config_params`:
- `DataConfig`
- `ModelConfig` (aliased as `PTModelConfig`)
- `TrainingConfig` (aliased as `PTTrainingConfig`)
- `InferenceConfig`

### Issue 2: Channel mismatch with neighbor sampling
**Problem**: Initial configuration used `gridsize=2` which creates 4-channel inputs, but this caused:
1. Conv layer expecting 1 channel got 4 channels
2. Position coordinates had 4 values instead of 2 (one per neighbor)
3. Complex neighbor sampling logic incompatible with simple test

**Solution**: Simplified test configuration to use `gridsize=1` (no neighbor sampling):
```python
model_config = ModelConfig(
    N=64,
    gridsize=1,  # No neighbor sampling
    model_type='pinn',
)

training_config = TrainingConfig(
    neighbor_count=1,  # No neighbors
    ...
)
```

This creates single-channel inputs and simple (x, y) position coordinates.

## Final Test Configuration

**TensorFlow Config** (for data loading):
- N=64
- gridsize=1
- neighbor_count=1
- nphotons=1e9

**PyTorch Config** (for model):
```python
pt_model_config = PTModelConfig(
    mode='Unsupervised',
    n_filters_scale=2,
)

pt_training_config = PTTrainingConfig(
    nll=True,  # Enable Poisson loss - KEY REQUIREMENT
)
```

## Test Results

**Test Status**: PASSED ✅

**ValueError Raised**: Yes, as expected
**Error Message**:
```
Expected value argument (Tensor of shape (10, 1, 64, 64)) to be within the support
(IntegerGreaterThan(lower_bound=0)) of the distribution Poisson(rate: torch.Size([10, 1, 64, 64])),
but found invalid values:
tensor([[[[0.7650, 0.6111, 0.6404, ...
```

**Validation**:
- ✓ Error message contains 'support'
- ✓ Error message references 'IntegerGreaterThan'
- ✓ Error shows raw amplitude floats (0.0-1.0 range) were passed to Poisson

## Root Cause Confirmed

The PyTorch Poisson loss implementation in `ptycho_torch/model.py` (PoissonIntensityLayer) receives:
- **pred**: model predictions (amplitudes)
- **raw**: raw data (amplitudes)

Both are in the range [0.0, 1.0] (normalized amplitudes), NOT integer photon counts.

The Poisson distribution requires:
- **rate (λ)**: Expected number of events (photon counts)
- **value (k)**: Observed number of events (photon counts, integers ≥ 0)

Current implementation passes amplitude floats directly, violating the integer support constraint.

## Next Steps (GREEN Phase)

Fix the PoissonIntensityLayer to:
1. Square amplitudes to get intensities: `pred²`, `raw²`
2. Apply nphotons scaling using `rms_scaling_constant` from batch
3. Pass photon counts to `Poisson.log_prob()`, not amplitudes

Reference: TensorFlow implementation in `ptycho/model.py:541-561`

## Files Modified

- `/home/ollie/Documents/PtychoPINN2/tests/torch/test_workflows_components.py`
  - Line 211-227: Updated minimal_training_config fixture (gridsize=1, neighbor_count=1)
  - Line 518-539: Added PyTorch config instantiation
  - Line 446-564: Complete test implementation

## Test Command

```bash
CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_lightning_poisson_count_contract -vv
```

**Result**: 1 passed in 5.18s ✅

## Evidence

Full pytest output saved to:
- `/home/ollie/Documents/PtychoPINN2/plans/active/ADR-003-BACKEND-API/reports/2025-10-20T070610Z/phase_c4_cli_integration_debug/pytest_poisson_red.log`
