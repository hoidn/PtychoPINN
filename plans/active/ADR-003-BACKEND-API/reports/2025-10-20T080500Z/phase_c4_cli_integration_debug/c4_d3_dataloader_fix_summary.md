# C4.D3 Lightning Dataloader Fix - Summary Report

## Task
Fix PyTorch Lightning dataloader TensorDict-style batch structure to match `PtychoPINN_Lightning.compute_loss` expectations.

## Problem Statement
Integration test `test_run_pytorch_train_save_load_infer` failing with:
```
IndexError: too many indices for tensor of dimension 4
```
at `ptycho_torch/model.py:1123` (`x = batch[0]['images']`).

Root cause: `_build_lightning_dataloaders()` wrapped tensors in `TensorDataset(train_X, train_coords)`, yielding `(Tensor, Tensor)` tuples instead of the expected `(tensor_dict, probe, scaling)` structure.

## Solution Implemented

### 1. Created Custom Dataset Class
**File**: `ptycho_torch/workflows/components.py:332-402`

Implemented `PtychoLightningDataset` class that yields proper batch structure:
- `batch[0]`: dict with keys `['images', 'coords_relative', 'rms_scaling_constant', 'physics_scaling_constant']`
- `batch[1]`: probe tensor
- `batch[2]`: scaling constant tensor

### 2. Fixed Channel Ordering (TensorFlow → PyTorch)
**File**: `ptycho_torch/workflows/components.py:374-383`

Added channel-last to channel-first permutation:
```python
if images_indexed.ndim == 4:
    # DataLoader batched case: (batch, H, W, C) → (batch, C, H, W)
    images_indexed = images_indexed.permute(0, 3, 1, 2)
elif images_indexed.ndim == 3:
    # Single sample case: (H, W, C) → (C, H, W)
    images_indexed = images_indexed.permute(2, 0, 1)
```

**Rationale**: TensorFlow `RawData.generate_grouped_data()` returns `X_full` in channel-last format `(N, H, W, C)`, but PyTorch conv2d expects channel-first `(N, C, H, W)`.

### 3. Fixed Scaling Constant Broadcasting
**File**: `ptycho_torch/workflows/components.py:396-399`

Flattened scaling constants to scalars before DataLoader collation:
```python
if rms_scale is not None:
    rms_scale = rms_scale.flatten()[0] if rms_scale.numel() == 1 else rms_scale.squeeze()
```

**Rationale**: Container stores scaling with shape `(nsamples, 1, 1, 1, 1)`. After indexing → `(1, 1, 1, 1)`. Default collate would stack 4 samples → `(4, 1, 1, 1, 1)` (5D!). Flattening to scalar gives `(batch,)` after collation, which model can reshape to `(batch, 1, 1, 1)` for proper broadcasting.

### 4. Added Model-Side Reshaping
**File**: `ptycho_torch/model.py:859-862`

Ensured 1D scale factors are reshaped for broadcasting:
```python
if input_scale_factor.ndim == 1:
    input_scale_factor = input_scale_factor.view(-1, 1, 1, 1)
```

### 5. Added Regression Test
**File**: `tests/torch/test_workflows_components.py:344-444`

Created `test_lightning_dataloader_tensor_dict_structure` to validate batch structure compliance.

## Test Results

### Targeted Test (GREEN)
```bash
pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_lightning_dataloader_tensor_dict_structure -vv
```
**Status**: ✅ PASSED

### Integration Test Status
The original dimension error is RESOLVED. Training now progresses past the batch structure issue to a separate Poisson distribution domain error (out of scope for C4.D3).

## Files Modified
1. `ptycho_torch/workflows/components.py` - Dataloader implementation
2. `ptycho_torch/model.py` - Scale factor reshaping
3. `tests/torch/test_workflows_components.py` - Regression test

## Next Steps (Out of Scope for C4.D3)
The Poisson loss invalid values error indicates a data/scaling issue in the forward model, not a dataloader structure problem. This is tracked separately in the fix_plan.

## Artifacts
- RED test log: `pytest_dataloader_red.log`
- GREEN test log: `pytest_dataloader_green.log`
- Integration test logs: `pytest_integration.log`, `pytest_integration_v2.log`
- This summary: `c4_d3_dataloader_fix_summary.md`
