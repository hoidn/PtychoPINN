# Phase C.C3 Implementation Strategy

**Initiative:** INTEGRATE-PYTORCH-001
**Phase:** C.C3 — Memory-mapped Dataset Bridge
**Date:** 2025-10-17T084500Z
**Loop:** Attempt #37

---

## Problem Analysis

The existing `ptycho_torch/dset_loader_pt_mmap.py` is a complex 429-line implementation that:
- Reimplements grouping logic (lines 242-257)
- Uses singleton configs instead of dataclass bridge
- Outputs TensorDict objects, not grouped-data dicts
- Hard-codes torch imports

Meanwhile, Phase C.C3 requires:
1. Delegation to RawDataTorch for grouping
2. Config bridge integration (dataclass configs)
3. Outputs compatible with PtychoDataContainerTorch
4. Torch-optional design

## Implementation Decision

**Approach:** Create a new lightweight bridge adapter (`ptycho_torch/memmap_bridge.py`) rather than refactoring the entire existing dataset.

**Rationale:**
1. **Pragmatic scope**: Existing `dset_loader_pt_mmap.py` serves Lightning training workflows with TensorDict outputs. Refactoring it risks breaking those workflows.
2. **Test alignment**: Tests expect a simple `MemmapDatasetBridge` that returns grouped-data dicts.
3. **Phase separation**: Phase C focuses on data pipeline parity; Phase D will integrate with training workflows.
4. **Reversible**: New bridge can coexist with existing implementation, allowing gradual migration.

## Implementation Plan

### Step 1: Create `ptycho_torch/memmap_bridge.py`

New module providing:
```python
class MemmapDatasetBridge:
    """
    Lightweight bridge connecting memory-mapped NPZ files to RawDataTorch.

    Phase C.C3 Goal: Delegate grouping to RawDataTorch while preserving
    memory-mapping benefits for large datasets.
    """

    def __init__(self, npz_path, config, memmap_dir="data/memmap"):
        # Load NPZ data (potentially memory-mapped)
        # Instantiate RawDataTorch with config
        # Ensure update_legacy_dict called (CONFIG-001)

    def get_grouped_data(self, N, K, nsamples, gridsize, seed=None):
        # Delegate to self.raw_data_torch.generate_grouped_data(...)
        # Return grouped-data dict compatible with PtychoDataContainerTorch
```

### Step 2: Implementation Details

1. **NPZ Loading**:
   - Use `np.load(npz_path, mmap_mode='r')` for read-only memory mapping
   - Extract required keys: diff3d, xcoords, ycoords, probeGuess, objectGuess, scan_index

2. **RawDataTorch Integration**:
   - Pass config to RawDataTorch constructor (automatic `update_legacy_dict`)
   - Delegate `generate_grouped_data()` calls to underlying adapter

3. **Cache Reuse**:
   - RawDataTorch already delegates to TensorFlow RawData, which manages `.groups_cache.npz`
   - No additional cache logic needed (inherits TensorFlow caching)

4. **Torch-Optional**:
   - Guard torch imports with try/except
   - Use `TORCH_AVAILABLE` flag from config_params.py pattern

### Step 3: Test Integration

Update tests to use `MemmapDatasetBridge`:
1. Remove `pytest.skip()` from `test_memmap_loader_matches_raw_data_torch`
2. Remove `pytest.skip()` from `test_cache_reuse_validation`
3. Implement fixture logic as documented in test comments

### Step 4: Validation

- Run `pytest tests/torch/test_data_pipeline.py -k "memmap or cache_reuse" -vv`
- Capture GREEN logs to `pytest_memmap_green.log`
- Document cache reuse evidence (timestamps/hashes)

## Deferred Work

- **Existing dset_loader_pt_mmap.py**: Leave unchanged for now
- **TensorDict compatibility**: Phase D can add wrapper converting grouped-data dict → TensorDict
- **Lightning integration**: Phase D orchestration task

## Risk Mitigation

- **Minimal changes**: New module doesn't modify existing code
- **Test-driven**: Implementation guided by explicit test contracts
- **Reversible**: Can delete bridge if approach doesn't work

---

**Next Steps:**
1. Implement `ptycho_torch/memmap_bridge.py`
2. Update tests to use new bridge
3. Run tests and capture GREEN logs
4. Update fix_plan.md
