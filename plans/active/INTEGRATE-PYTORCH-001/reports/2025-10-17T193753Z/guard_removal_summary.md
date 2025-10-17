# Phase F3.2 Guard Removal Summary

**Initiative:** INTEGRATE-PYTORCH-001  
**Phase:** F3.2 — Remove guarded imports & flags  
**Date:** 2025-10-17  
**Status:** ✅ COMPLETE

## Objective

Remove `TORCH_AVAILABLE` guards and NumPy fallback logic from PyTorch production modules, enforcing PyTorch >= 2.2 as a mandatory dependency per Phase F3.1 gate.

## Modules Modified

### 1. `ptycho_torch/config_params.py` (lines 4-13)
- **Change:** Removed optional import guard + `TORCH_AVAILABLE` flag
- **Result:** Unconditional `import torch` with actionable RuntimeError on failure
- **Type alias:** Kept `TensorType = torch.Tensor` (no `Any` fallback)

### 2. `ptycho_torch/config_bridge.py` (lines 70-76, 146-148)
- **Change:** Removed `TORCH_AVAILABLE` import from config_params
- **Result:** Simplified probe_mask translation logic (removed torch availability check)
- **Behavior:** Non-None probe_mask → `True` (no guard needed)

### 3. `ptycho_torch/data_container_bridge.py` (lines 98-106, 208-233, 256-260)
- **Change:** Removed torch-optional guard + NumPy fallback branch
- **Result:** Always convert to torch tensors (no dual-path logic)
- **__repr__:** Simplified dtype detection (removed `TORCH_AVAILABLE` check)

### 4. `ptycho_torch/memmap_bridge.py` (lines 32-40)
- **Change:** Removed `TORCH_AVAILABLE` flag (module doesn't use torch directly)
- **Result:** Import torch with RuntimeError on failure (downstream dependency)

### 5. `ptycho_torch/model_manager.py` (lines 49-57, 158-171, 221)
- **Change:** Removed `TORCH_AVAILABLE` flag + sentinel dict fallback
- **Result:** Require `nn.Module` instances (no dict fallback)
- **load_torch_bundle:** Removed availability check (PyTorch now mandatory)

### 6. `ptycho_torch/workflows/components.py` (lines 66-77, 178-188, 221, 457)
- **Change:** Removed `TORCH_AVAILABLE` flag + conditional imports
- **Result:** Unconditional imports of RawDataTorch, PtychoDataContainerTorch, save/load helpers
- **Fallback removal:** Deleted `if xyz is None` guards (imports at module scope)

### 7. `ptycho_torch/__init__.py` (lines 16-32)
- **Change:** Removed `TORCH_AVAILABLE` import from config_bridge
- **Result:** Preserved `TORCH_AVAILABLE=True` for backward compatibility
- **Error handling:** Raise RuntimeError on torch import failure

## Test Results

### Targeted Tests (Phase F3.2 selectors)
```bash
pytest tests/torch/test_config_bridge.py -q         # 46 passed, 17 warnings in 3.74s
pytest tests/torch/test_data_pipeline.py -q        # 5 passed in 6.72s
pytest tests/torch/test_model_manager.py -q        # 4 passed, 1 xfailed in 12.56s
```

### Full Regression Suite
```bash
pytest tests/ --ignore=tests/test_benchmark_throughput.py --ignore=tests/test_run_baseline.py
# Result: 203 passed, 13 skipped, 1 xfailed, 17 warnings in 212.48s
```

**No new failures introduced.** Pre-existing import errors in `test_benchmark_throughput.py` and `test_run_baseline.py` unrelated to guard removal.

## Code Touchpoints Summary

| Module | Lines Changed | Guard Instances Removed | Fallback Branches Deleted |
|--------|--------------|------------------------|---------------------------|
| config_params.py | 4-13 | 1 | 1 (Any fallback) |
| config_bridge.py | 70-76, 146-148 | 1 | 1 (probe_mask) |
| data_container_bridge.py | 98-106, 208-233, 256-260 | 1 | 2 (tensor creation + __repr__) |
| memmap_bridge.py | 32-40 | 1 | 0 |
| model_manager.py | 49-57, 158-171, 221 | 1 | 2 (save fallback + availability check) |
| workflows/components.py | 66-77, 178-188, 221, 457 | 1 | 4 (conditional imports + None guards) |
| __init__.py | 16-32 | 1 | 0 |
| **Total** | **~50 lines** | **7 instances** | **10 fallback branches** |

## RuntimeError Messaging

All modules now raise actionable RuntimeError on torch import failure:
```python
raise RuntimeError(
    "PyTorch is required for ptycho_torch modules. "
    "Install PyTorch >= 2.2 with: pip install torch>=2.2"
) from e
```

## Exit Criteria

- [x] All 6 production modules updated (config_params, config_bridge, data_container_bridge, memmap_bridge, model_manager, workflows/components)
- [x] Targeted pytest selectors pass (test_config_bridge, test_data_pipeline, test_model_manager)
- [x] Full regression suite green (203 passed, 13 skipped, 1 xfailed)
- [x] RuntimeError messaging provides install guidance
- [x] No NumPy fallback logic remaining in production paths

## Next Phase

Phase F3.3: Rewrite pytest skip logic in `tests/conftest.py` to remove torch-optional whitelist and update test expectations.
