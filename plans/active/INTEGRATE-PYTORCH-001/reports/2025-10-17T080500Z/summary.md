# Phase C.C2 Implementation Summary — PtychoDataContainerTorch

**Date:** 2025-10-17
**Loop:** Attempt #35 (Ralph engineer loop)
**Phase:** INTEGRATE-PYTORCH-001 Phase C.C2
**Deliverable:** Torch-optional PtychoDataContainerTorch adapter

---

## Implementation Overview

Successfully implemented `PtychoDataContainerTorch` in `ptycho_torch/data_container_bridge.py` (280 lines) with full TensorFlow `PtychoDataContainer` API parity. Container accepts grouped data dictionary from `RawDataTorch.generate_grouped_data()` and exposes model-ready tensors using PyTorch when available or NumPy fallback when unavailable.

## Key Design Decisions

1. **Torch-Optional Architecture**: Module importable without PyTorch; uses `TORCH_AVAILABLE` flag with guarded imports
2. **TensorFlow Tensor Conversion**: Added `.numpy()` conversion for TensorFlow tensors in grouped data dict (Y patches may come from TensorFlow loader)
3. **DATA-001 Enforcement**: Explicit complex64 dtype validation with ValueError on mismatch
4. **Dtype Fidelity**: Preserved float64 for offsets per TensorFlow baseline; float32 for diffraction/coords; complex64 for Y/probe
5. **Decomposition Pattern**: Rebuilt Y_I/Y_phi from Y using torch.abs/torch.angle (PyTorch) or np.abs/np.angle (NumPy fallback)

## Files Created/Modified

### New Files
- `ptycho_torch/data_container_bridge.py` (280 lines)
  - `PtychoDataContainerTorch` class with 10+ tensor attributes
  - Torch-optional import guards
  - Comprehensive docstrings with contract references

### Modified Files
- `ptycho_torch/__init__.py`: Added `PtychoDataContainerTorch` export
- `tests/torch/test_data_pipeline.py`: Replaced pytest.fail() calls with full test implementation (lines 248-325, 368-403)

## Test Results

### Targeted Selectors (Phase C.C2 acceptance)
```bash
pytest tests/torch/test_data_pipeline.py -k "data_container" -vv
# Result: 1 PASSED in 5.62s

pytest tests/torch/test_data_pipeline.py -k "y_patches_are_complex64" -vv
# Result: 1 PASSED in 5.68s
```

### Full Regression Check
```bash
pytest tests/ --ignore=tests/test_benchmark_throughput.py --ignore=tests/test_run_baseline.py -v
# Result: 186 passed, 13 skipped, 17 warnings in 203.31s (0:03:23)
# No new failures introduced
```

## Attributes Implemented (API Parity)

| Attribute | Shape (N=64, gridsize=2, nsamples=10) | Dtype (PyTorch) | Dtype (NumPy) | Notes |
|-----------|---------------------------------------|-----------------|---------------|-------|
| `X` | (10, 64, 64, 4) | torch.float32 | float32 | Diffraction amplitude |
| `Y` | (10, 64, 64, 4) | torch.complex64 | complex64 | Combined ground truth (DATA-001) |
| `Y_I` | (10, 64, 64, 4) | torch.float32 | float32 | Amplitude from torch.abs(Y) |
| `Y_phi` | (10, 64, 64, 4) | torch.float32 | float32 | Phase from torch.angle(Y) |
| `coords_nominal` | (10, 1, 2, 4) | torch.float32 | float32 | Scan coordinates |
| `coords_true` | (10, 1, 2, 4) | torch.float32 | float32 | Alias for coords_nominal |
| `coords` | (10, 1, 2, 4) | torch.float32 | float32 | Convenience alias |
| `probe` | (64, 64) | torch.complex64 | complex64 | Probe function |
| `nn_indices` | (10, 4) | torch.int32 | int32 | Nearest neighbor indices |
| `global_offsets` | (10, 1, 2, 1) | torch.float64 | float64 | Global coordinate offsets |
| `local_offsets` | (10, 1, 2, 4) | torch.float64 | float64 | Local offsets per channel |
| `norm_Y_I` | Optional | — | — | Normalization factor (preserved) |
| `YY_full` | Optional | — | — | Full object reconstruction (preserved) |

## Critical Fixes

### Issue 1: TensorFlow Tensor in Grouped Data
**Problem:** When using TensorFlow RawData baseline in tests, grouped_data['Y'] contains `tf.EagerTensor` instead of NumPy array.  
**Solution:** Added `.numpy()` conversion check in constructor:
```python
Y_raw = grouped_data['Y']
if hasattr(Y_raw, 'numpy'):  # TensorFlow tensor
    Y_np = Y_raw.numpy()
else:
    Y_np = np.asarray(Y_raw)
```
**Location:** `ptycho_torch/data_container_bridge.py:181-186`

### Issue 2: DATA-001 Dtype Validation
**Enforcement:** Explicit dtype validation with actionable error messages:
```python
if Y_np.dtype != np.complex64:
    raise ValueError(
        f"DATA-001 violation: Y patches MUST be complex64, got {Y_np.dtype}. "
        f"Historical bug: silent float64 conversion caused major training failure. "
        f"See docs/findings.md:DATA-001 and specs/data_contracts.md:19"
    )
```
**Location:** `ptycho_torch/data_container_bridge.py:189-194`

## Artifacts Captured

All artifacts saved to `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T080500Z/`:
- `pytest_data_container_green.log` — Data container parity test output (1 PASSED)
- `pytest_y_dtype_green.log` — Y dtype validation test output (1 PASSED)
- `summary.md` — This document

## Contract Compliance

- ✅ specs/data_contracts.md §3 (PtychoDataContainer attributes table)
- ✅ specs/ptychodus_api_spec.md §4.4 (tensor exposure requirements)
- ✅ docs/findings.md:DATA-001 (complex64 dtype enforcement)
- ✅ plans/active/INTEGRATE-PYTORCH-001/phase_c_data_pipeline.md C.C2 checklist

## Deferred Work

1. **Memory-mapped dataset integration** (Phase C.C3): Current implementation consumes NumPy arrays from grouped data; future work will integrate with `ptycho_torch/dset_loader_pt_mmap.py` memory-mapped tensors
2. **Train/test splitting** (Phase C.C3): Container currently handles unsplit datasets (`create_split=False`); split support deferred to Phase C.C3 or Phase D
3. **Probe Tensor Handling** (Phase C.D): Probe currently converted via `torch.from_numpy()`; downstream consumer expectations in Phase D workflows may require adjustments

## Next Steps (Phase C.C3)

Per `phase_c_data_pipeline.md` C.C3 checklist:
- Bridge memory-mapped dataset usage (translate MemoryMappedTensor outputs into RawDataTorch inputs or refactor dataset to delegate to RawData)
- Document cache reuse semantics for .groups_cache.npz files
- Integrate config bridge touchpoints to ensure pipeline pulls configuration from dataclasses

## Status

**Phase C.C2:** ✅ COMPLETE  
**Next Phase:** C.C3 (Memory-mapped dataset integration)  
**Blocking Issues:** None
