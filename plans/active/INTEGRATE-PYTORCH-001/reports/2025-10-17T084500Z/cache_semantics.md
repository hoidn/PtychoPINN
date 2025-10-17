# Phase C.C3 Cache Semantics Documentation

**Initiative:** INTEGRATE-PYTORCH-001
**Focus:** Memory-Mapped Dataset Bridge Cache Behavior
**Date:** 2025-10-17T084500Z
**Loop:** Attempt #37

---

## 1. Cache Architecture Discovery

During implementation of the memmap bridge, we discovered that the current TensorFlow RawData implementation **does not use cache files**.

**Evidence:**
- Source: `ptycho/raw_data.py:408`
- Quote: "The new efficient implementation eliminates the need for caching. Performance is fast enough that first-run and subsequent runs have similar execution times."
- Implementation: Sample-then-group strategy with O(nsamples * K) complexity instead of O(n_points * K)

## 2. Cache Semantics for MemmapDatasetBridge

### 2.1 Current Behavior

The `MemmapDatasetBridge` adapter inherits the cache-free design from TensorFlow RawData:

1. **No disk cache files**: No `.groups_cache.npz` files are created
2. **Memory-mapped NPZ loading**: NPZ files are loaded with `mmap_mode='r'` for read-only access
3. **Deterministic generation**: Same seed → identical grouped data (validated by tests)
4. **Fast recomputation**: Sample-then-group is efficient enough that caching provides minimal benefit

### 2.2 Validation Evidence

**Test:** `test_deterministic_generation_validation`
- **Location:** `tests/torch/test_data_pipeline.py:514`
- **Result:** PASSED
- **Evidence:**
  - First instantiation (seed=42) produces grouped data G1
  - Second instantiation (seed=42) produces grouped data G2
  - Assertion: G1 == G2 (byte-identical)
  - Different seed (seed=123) produces G3 != G1 (non-trivial generation)

**Conclusion:** Delegation to RawDataTorch ensures deterministic, reproducible grouped data without requiring cache files.

## 3. Memory-Mapping Strategy

### 3.1 NPZ Memory Mapping

```python
self._npz_data = np.load(self.npz_path, mmap_mode='r')
```

- **Benefit:** Large NPZ arrays are not loaded into RAM until accessed
- **Trade-off:** Must materialize arrays when casting dtypes or performing operations
- **Use case:** Suitable for datasets larger than RAM

### 3.2 Current Limitations

1. **Dtype casting forces materialization:**
   ```python
   arr = self._npz_data[key].astype(dtype)  # Materializes full array
   ```
   - Impact: For 100GB dataset, casting loads all 100GB into RAM
   - Future optimization: Chunked processing or lazy casting wrapper

2. **RawDataTorch delegation requires in-memory arrays:**
   - RawDataTorch constructor expects NumPy arrays
   - Cannot pass memory-mapped proxies directly
   - Current design prioritizes correctness over memory optimization

## 4. Performance Characteristics

### 4.1 Observed Performance (from test run)

- **Test dataset:** 20 points, 64x64 diffraction patterns
- **Grouped data generation:** ~5.5 seconds total for 2 test cases
- **No cache overhead:** No disk I/O for cache creation/loading

### 4.2 Scalability Considerations

For production datasets (10,000+ points), the current strategy:
- ✅ Fast: O(nsamples * K) instead of O(n_points * K)
- ✅ Deterministic: Seed-based reproducibility
- ✅ Memory-efficient for read-only NPZ access
- ⚠️ Memory-heavy for dtype casting (future improvement area)

## 5. Comparison with Legacy TensorDict Implementation

| Aspect | `dset_loader_pt_mmap.py` (Legacy) | `memmap_bridge.py` (Phase C.C3) |
|--------|-----------------------------------|----------------------------------|
| **Grouping Logic** | Reimplemented (duplicate) | Delegated to RawDataTorch |
| **Config Access** | Singleton `ModelConfig().get()` | Dataclass via config_bridge |
| **Cache Files** | Custom memmap tensors in `data/memmap/` | None (inherits RawData behavior) |
| **Output Format** | TensorDict | grouped-data dict |
| **Torch Dependency** | Hard import | Torch-optional |
| **Parity** | Diverges from TensorFlow | Exact parity (delegation) |

## 6. Recommendations for Phase D

### 6.1 Immediate (Phase D Orchestration)

1. **TensorDict Wrapper:** Create lightweight adapter to wrap grouped-data dict → TensorDict for Lightning compatibility
2. **Memory-mapping optimization:** Investigate chunked processing for large datasets
3. **Documentation:** Update user-facing docs to explain cache-free architecture

### 6.2 Future (Post-Phase E)

1. **Lazy dtype casting:** Implement wrapper that defers materialization until actual use
2. **Streaming support:** For datasets >1TB, implement true streaming pipelines
3. **Profiling hooks:** Add optional timing/memory tracking for production optimization

## 7. Test Coverage Summary

| Test | Status | Coverage |
|------|--------|----------|
| `test_memmap_loader_matches_raw_data_torch` | PASSED | Delegation correctness |
| `test_deterministic_generation_validation` | PASSED | Deterministic generation |

**Regression:** Full suite: 188 passed, 13 skipped, 0 failed

---

**Artifacts:**
- RED state log: `pytest_memmap_red.log`
- GREEN state log: `pytest_memmap_green_final.log`
- Implementation: `ptycho_torch/memmap_bridge.py`
- Tests: `tests/torch/test_data_pipeline.py:406-572`

**Conclusion:** Phase C.C3 complete. Memory-mapped dataset bridge successfully delegates to RawDataTorch while maintaining deterministic grouped data generation without cache files.
