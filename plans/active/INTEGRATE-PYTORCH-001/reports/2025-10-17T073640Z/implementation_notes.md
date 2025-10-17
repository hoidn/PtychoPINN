# Phase C.C1 Implementation Notes — RawDataTorch Adapter

**Initiative:** INTEGRATE-PYTORCH-001
**Phase:** C.C1 — Implement RawDataTorch wrapper
**Date:** 2025-10-17
**Loop:** Ralph Attempt #33

---

## Summary

Successfully implemented torch-optional `RawDataTorch` adapter that delegates to `ptycho.raw_data.RawData`, achieving 100% parity with TensorFlow baseline. Implementation completed in single loop with green test coverage.

---

## Implementation Details

### Module Created

- **Location:** `ptycho_torch/raw_data_bridge.py` (324 lines)
- **Exports:** `RawDataTorch` class exported from `ptycho_torch/__init__.py`
- **Torch-Optional:** Module importable without PyTorch installed (per CLAUDE.md:57-59)

### Core Design Decisions

1. **Delegation Strategy (NOT Reimplementation)**
   - Wraps `ptycho.raw_data.RawData.from_coords_without_pc()` factory method
   - Delegates `generate_grouped_data()` calls directly to TensorFlow implementation
   - Zero code duplication of grouping logic (maintains single source of truth)

2. **Configuration Bridge Integration**
   - Constructor accepts optional `TrainingConfig`/`InferenceConfig` parameter
   - Automatically calls `update_legacy_dict(params.cfg, config)` when provided
   - Prevents CONFIG-001 shape mismatch bugs (documented in findings.md)

3. **NumPy-First Return Types**
   - Returns NumPy arrays (matching TensorFlow RawData behavior)
   - No PyTorch tensor conversion in this phase (deferred to DataContainer layer)
   - Simplifies testing and maintains compatibility with existing fixtures

4. **Property Access**
   - Exposes underlying TensorFlow RawData properties: `probeGuess`, `objectGuess`, `xcoords`, `ycoords`, `diff3d`
   - Enables transparent delegation without breaking encapsulation

### API Surface

```python
class RawDataTorch:
    def __init__(
        xcoords, ycoords, diff3d, probeGuess, scan_index,
        objectGuess=None, config=None
    )

    def generate_grouped_data(
        N, K=4, nsamples=1, seed=None,
        sequential_sampling=False, gridsize=None
    ) -> Dict[str, np.ndarray]

    # Properties
    @property probeGuess -> np.ndarray
    @property objectGuess -> Optional[np.ndarray]
    @property xcoords -> np.ndarray
    @property ycoords -> np.ndarray
    @property diff3d -> np.ndarray
```

### Test Coverage

**Test:** `tests/torch/test_data_pipeline.py::TestRawDataTorchAdapter::test_raw_data_torch_matches_tensorflow`

**Coverage:**
- ✅ Shape parity: diffraction, X_full, coords_offsets, coords_relative, nn_indices
- ✅ Dtype parity: float32 for diffraction, int32 for nn_indices
- ✅ Exact data parity: `np.testing.assert_array_equal()` for nn_indices
- ✅ Numerical parity: `np.testing.assert_allclose()` for diffraction/coords (rtol=1e-6)

**Test Status:** PASSED (1/1 in 5.22s)

**Selector:** `pytest tests/torch/test_data_pipeline.py -k raw_data -vv`

---

## Dtype Handling (DATA-001 Compliance)

Per `docs/findings.md:DATA-001`, Y patches MUST be complex64 to avoid historical silent conversion bug.

**Current Implementation:**
- RawData baseline preserves complex64 for Y patches
- RawDataTorch delegates to RawData without modification
- Dtype preservation verified in TensorFlow baseline (test line 267)
- PyTorch container dtype validation deferred to Phase C.C2

**Validation Evidence:**
```python
# From test fixture
obj = np.ones((128, 128), dtype=np.complex64)  # Input
Y4d_nn = get_image_patches(...)  # TensorFlow function
assert Y4d_nn.dtype == np.complex64  # Preserved
```

---

## Configuration Bridge Touchpoints

**Input:** `TrainingConfig` or `InferenceConfig` with populated fields

**Bridge Flow:**
1. Adapter constructor receives config parameter
2. Calls `update_legacy_dict(params.cfg, config)` if config provided
3. TensorFlow RawData reads `params.cfg['gridsize']`, `params.cfg['N']`
4. Grouping logic executes with correct parameters

**Critical Fields:**
- `ModelConfig.N` → `params.cfg['N']`
- `ModelConfig.gridsize` → `params.cfg['gridsize']`
- `TrainingConfig.neighbor_count` → used by caller (not passed to RawData directly)
- `TrainingConfig.n_groups` → nsamples parameter

**Documented Warning (from ptycho/raw_data.py:412-425):**
```
⚠️ CRITICAL DEPENDENCY WARNING ⚠️
This method requires params.cfg['gridsize'] to be initialized.
Common failure scenario:
- Symptom: Getting shape (*, 64, 64, 1) instead of (*, 64, 64, 4)
- Cause: params.cfg['gridsize'] not set, defaults to 1
- Fix: Ensure update_legacy_dict() called before this method
```

**Adapter Solution:** Handles initialization automatically in constructor when config provided.

---

## Test Determinism Fix

**Issue Encountered:** Initial test failure due to random sampling producing different nn_indices on each call.

**Root Cause:** Two sequential calls to `generate_grouped_data()` without seed parameter:
1. TensorFlow baseline call (no seed) → random indices A
2. PyTorch adapter call (no seed) → random indices B
3. Assertion: A == B ❌ FAILED

**Resolution:** Added `seed=42` to both baseline and adapter calls:
```python
tf_grouped = minimal_raw_data.generate_grouped_data(..., seed=42)
pt_grouped = pt_raw.generate_grouped_data(..., seed=42)
```

**Artifact:** Test updated in commit (lines 117, 153 of test_data_pipeline.py)

---

## Deferred Work (Not MVP-Blocking)

1. **PyTorch Tensor Conversion** (Phase C.C2)
   - Current: Returns NumPy arrays
   - Future: Optional torch.Tensor conversion when `TORCH_AVAILABLE=True`
   - Rationale: DataContainer layer is better suited for tensor type decisions

2. **Memory-Mapped Dataset Integration** (Phase C.C3)
   - Current: Accepts NumPy arrays in constructor
   - Future: Thin layer translating `MemoryMappedTensor` outputs to RawDataTorch inputs
   - Design Note: Prefer delegation to RawData over reimplementing memmap logic

3. **Cache Lifecycle Documentation** (Phase C.D2)
   - Current TensorFlow implementation: NO cache files (eliminated per performance update)
   - PyTorch behavior: Inherits zero-cache design via delegation
   - Validation: No `.groups_cache.npz` files created during tests

---

## Follow-Up Tasks (Next Loops)

**Phase C.C2:** Implement `PtychoDataContainerTorch` with attributes:
- X, Y, Y_I, Y_phi, coords_nominal, probe, nn_indices, global_offsets
- Torch-optional tensor types (torch.Tensor when available, np.ndarray fallback)

**Phase C.C3:** Bridge memory-mapped dataset usage

**Phase C.D1:** Run full parity test suite (`pytest tests/torch/test_data_pipeline.py -k "raw_data or data_container" -vv`)

**Phase C.D2:** Verify cache reuse semantics (document zero-cache behavior)

**Phase C.D3:** Update parity ledger & docs

---

## Artifacts Generated

1. **Implementation:**
   - `ptycho_torch/raw_data_bridge.py` (324 lines, torch-optional)
   - `ptycho_torch/__init__.py` (updated exports)

2. **Tests:**
   - `tests/torch/test_data_pipeline.py` (updated test_raw_data_torch_matches_tensorflow)

3. **Logs:**
   - `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T073640Z/pytest_raw_data_green.log`
   - `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T073640Z/implementation_notes.md` (this file)

4. **Regression:**
   - Full test suite: 184 passed, 13 skipped, 2 xfail (Phase C.C2/C.C3), 0 new failures

---

## References

**Normative Contracts:**
- specs/data_contracts.md:58-176 (RawData.generate_grouped_data() contract)
- specs/ptychodus_api_spec.md:§4.3 (data ingestion requirements)

**Implementation Guidance:**
- plans/active/INTEGRATE-PYTORCH-001/phase_c_data_pipeline.md (Phase C plan)
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T070200Z/data_contract.md (TF baseline)
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T071836Z/notes.md (TDD guidance)

**Findings Applied:**
- CONFIG-001: Mandatory update_legacy_dict() before data operations
- DATA-001: Preserve complex64 dtype for Y patches
- NORMALIZATION-001: Do not apply photon scaling to data

**Test Blueprint:**
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T070200Z/test_blueprint.md

---

## Lessons Learned

1. **Delegation > Reimplementation:** Wrapping TensorFlow RawData instead of duplicating logic saved ~300 lines and guaranteed parity.

2. **Config Bridge Critical:** Automatic `update_legacy_dict()` in constructor prevents CONFIG-001 bugs without requiring caller changes.

3. **Test Determinism:** Random sampling requires explicit seed parameter for reproducible parity tests.

4. **Torch-Optional Design:** Import guards at module scope enable testing without PyTorch runtime dependency.

5. **NumPy-First Strategy:** Deferring tensor conversion to DataContainer layer simplifies adapter and improves test portability.

---

**Status:** Phase C.C1 ✅ COMPLETE
**Next:** Phase C.C2 (PtychoDataContainerTorch implementation)
