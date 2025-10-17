# Phase C.B2+C.B3 Notes — TDD Red Phase for Data Pipeline Parity

**Date:** 2025-10-17
**Loop:** INTEGRATE-PYTORCH-001 Attempt #31 (Phase C.B2/C.B3)
**Mode:** TDD (Red Phase)
**Objective:** Author failing pytest cases documenting expected RawDataTorch and PtychoDataContainerTorch behavior

---

## Summary

Successfully authored 3 failing tests documenting data pipeline parity requirements:

1. **`test_raw_data_torch_matches_tensorflow`** — RawData wrapper delegation
2. **`test_data_container_shapes_and_dtypes`** — DataContainer API surface
3. **`test_y_patches_are_complex64`** — Ground truth dtype validation (DATA-001 finding)

All tests execute successfully and fail with expected messages, providing clear guidance for Phase C.C implementation.

---

## Test Execution Results

### C.B2: RawData Parity Test
**Selector:** `pytest tests/torch/test_data_pipeline.py -k raw_data -vv`
**Status:** ✅ FAILED (as expected)
**Log:** `pytest_raw_data_red.log`

**Failure Message:**
```
RawDataTorch adapter not yet implemented (Phase C.C1).
Expected module: ptycho_torch/raw_data_bridge.py.
Expected delegation: wrapper calls ptycho.raw_data.RawData.generate_grouped_data().
See plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T070200Z/data_contract.md §2
for required output dict structure.
```

**Baseline Validation (Passed):**
- ✅ TensorFlow baseline shape: `(10, 64, 64, 4)` (correct for gridsize=2)
- ✅ TensorFlow baseline dtype: `float32`
- ✅ nn_indices shape: `(10, 4)`

**Key Observations:**
- RawData fixture creation successful using `from_coords_without_pc()` factory method
- params.cfg initialization via `update_legacy_dict()` working correctly
- TensorFlow grouping algorithm produces expected channel format (`gridsize² = 4` channels)

### C.B3: DataContainer Parity Test
**Selector:** `pytest tests/torch/test_data_pipeline.py -k data_container -vv`
**Status:** ✅ FAILED (as expected)
**Log:** `pytest_data_container_red.log`

**Failure Message:**
```
PtychoDataContainerTorch not yet implemented (Phase C.C2).
Expected module: ptycho_torch/data_container.py or loader_bridge.py.
Required attributes per data_contract.md §3:
X, Y, Y_I, Y_phi, coords_nominal, probe, nn_indices, global_offsets.
See plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T070200Z/data_contract.md §3
for complete attribute table with shapes and dtypes.
```

**Baseline Validation (Passed):**
- ✅ TensorFlow container has X attribute
- ✅ TensorFlow container has Y attribute
- ✅ TensorFlow container has coords_nominal attribute
- ✅ X shape: `(10, 64, 64, 4)`
- ✅ Y dtype: `tf.complex64`

**Container String Representation (Captured Output):**
```
<PtychoDataContainer
    X=(10, 64, 64, 4)
    Y_I=(10, 64, 64, 4)
    Y_phi=(10, 64, 64, 4)
    norm_Y_I=()
    coords_nominal=(10, 1, 2, 4)
    coords_true=(10, 1, 2, 4)
    nn_indices=(10, 4)
    mean=59.900
    global_offsets=(10, 1, 2, 1)
    mean=5.778
    local_offsets=(10, 1, 2, 4)
    mean=0.000
    probe=(64, 64)>
```

**Key Observations:**
- `loader.load()` successfully created TensorFlow baseline container
- Ground truth patches generated from `objectGuess` (Y array not provided in fixture)
- All expected attributes present in TensorFlow container

---

## Implementation Guidance Captured

### For Phase C.C1 (RawDataTorch)

**Expected Module:** `ptycho_torch/raw_data_bridge.py`

**Expected API:**
```python
class RawDataTorch:
    def __init__(self, xcoords, ycoords, diff3d, probeGuess, scan_index, objectGuess=None):
        # Delegate to TensorFlow RawData
        self._tf_raw_data = RawData.from_coords_without_pc(
            xcoords, ycoords, diff3d, probeGuess, scan_index, objectGuess
        )

    def generate_grouped_data(self, N, K=4, nsamples=1, dataset_path=None,
                              seed=None, sequential_sampling=False, gridsize=None):
        # Delegate to TensorFlow implementation
        return self._tf_raw_data.generate_grouped_data(
            N, K, nsamples, dataset_path, seed, sequential_sampling, gridsize
        )
```

**Required Output Dict Keys (from test baseline):**
- `diffraction`: `(nsamples, N, N, gridsize²)` float32
- `X_full`: `(nsamples, N, N, gridsize²)` float32 (normalized)
- `coords_offsets`: `(nsamples, 1, 2, 1)` float32
- `coords_relative`: `(nsamples, 1, 2, gridsize²)` float32
- `nn_indices`: `(nsamples, gridsize²)` int32
- `Y`: `(nsamples, N, N, gridsize²)` complex64 (if objectGuess provided)

### For Phase C.C2 (PtychoDataContainerTorch)

**Expected Module:** `ptycho_torch/data_container.py` or `ptycho_torch/loader_bridge.py`

**Required Attributes (from TensorFlow baseline):**
| Attribute | Shape | Dtype | Notes |
|-----------|-------|-------|-------|
| `X` | `(n_images, N, N, gridsize²)` | float32 | Diffraction patterns |
| `Y` | `(n_images, N, N, gridsize²)` | complex64 | Combined ground truth |
| `Y_I` | `(n_images, N, N, gridsize²)` | float32 | Amplitude patches |
| `Y_phi` | `(n_images, N, N, gridsize²)` | float32 | Phase patches |
| `coords_nominal` | `(n_images, 1, 2, gridsize²)` | float32 | Scan coordinates |
| `coords_true` | `(n_images, 1, 2, gridsize²)` | float32 | True coordinates |
| `probe` | `(N, N)` | complex64 | Probe function |
| `nn_indices` | `(n_images, gridsize²)` | int32 | Neighbor indices |
| `global_offsets` | `(n_images, 1, 2, 1)` | float32 | Global position offsets |
| `local_offsets` | `(n_images, 1, 2, gridsize²)` | float32 | Local position offsets |

**Torch-Optional Behavior:**
- When PyTorch available: attributes should be `torch.Tensor`
- When PyTorch unavailable: attributes should be `np.ndarray`
- Container creation must work in both modes

---

## Issues Encountered & Resolutions

### Issue 1: RawData Constructor Signature
**Problem:** Initial fixture used incorrect RawData constructor signature
```python
# ❌ WRONG
return RawData(xcoords, ycoords, diff3d, probe, obj, scan_index=None)

# TypeError: RawData.__init__() missing 1 required positional argument: 'probeGuess'
```

**Root Cause:** RawData requires `xcoords_start` and `ycoords_start` parameters

**Resolution:** Used `RawData.from_coords_without_pc()` factory method
```python
# ✅ CORRECT
return RawData.from_coords_without_pc(
    xcoords, ycoords, diff3d, probe, scan_index, objectGuess=obj
)
```

**Source:** `ptycho/raw_data.py:172-191`

### Issue 2: Probe Attribute Name
**Problem:** Test referenced `minimal_raw_data.probe` (attribute does not exist)
```python
# ❌ WRONG
probe_tf = tf.convert_to_tensor(minimal_raw_data.probe, dtype=tf.complex64)

# AttributeError: 'RawData' object has no attribute 'probe'
```

**Resolution:** Corrected to `probeGuess` (canonical attribute name)
```python
# ✅ CORRECT
probe_tf = tf.convert_to_tensor(minimal_raw_data.probeGuess, dtype=tf.complex64)
```

**Source:** `ptycho/raw_data.py:128-166` (RawData.__init__ stores as `self.probeGuess`)

---

## Torch-Optional Test Pattern Validation

### Whitelist Registration
**File:** `tests/conftest.py:38-47`
```python
TORCH_OPTIONAL_MODULES = ["test_config_bridge", "test_data_pipeline"]
is_torch_optional = any(module in str(item.fspath) for module in TORCH_OPTIONAL_MODULES)
```

**Status:** ✅ Working correctly
- Tests in `tests/torch/test_data_pipeline.py` execute even when PyTorch unavailable
- Auto-skip logic exempts whitelisted modules

### Fixture Strategy
**Fixture:** `params_cfg_snapshot` (from test_config_bridge.py pattern)
**Status:** ✅ Reused successfully
- Saves/restores `params.cfg` state
- Prevents test pollution
- Critical for data pipeline tests per CONFIG-001 finding

**Fixture:** `minimal_raw_data` (synthetic data pattern)
**Status:** ✅ Working correctly
- Deterministic (seed=42)
- Fast (<100ms creation time)
- No I/O dependencies
- Respects data contracts (dtype, normalization)

---

## Next Steps (Phase C.C — Green Phase)

### Immediate Blocking Work
1. **Implement RawDataTorch adapter** (Phase C.C1)
   - Create `ptycho_torch/raw_data_bridge.py`
   - Wrapper delegates to `ptycho.raw_data.RawData`
   - Ensure torch-optional imports (guard with try/except)

2. **Implement PtychoDataContainerTorch** (Phase C.C2)
   - Create `ptycho_torch/data_container.py` or `loader_bridge.py`
   - Expose all required attributes per table above
   - Support both torch.Tensor and np.ndarray backends

3. **Ground Truth Loading** (Phase C.C3)
   - Validate Y patches are complex64 (not float64)
   - Implement patch extraction using nn_indices
   - Ensure DATA-001 finding compliance

### Testing Checklist
- [ ] Remove `pytest.fail()` calls from tests once adapters implemented
- [ ] Run targeted selectors to verify green state
- [ ] Capture green logs under new timestamped directory
- [ ] Run full pytest suite to validate no regressions

### Documentation Updates
- [ ] Update `phase_c_data_pipeline.md` to mark C.B2/C.B3 complete
- [ ] Update `implementation.md` Phase C row with artifact paths
- [ ] Append to `docs/fix_plan.md` Attempts History

---

## Artifacts Generated

**Directory:** `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T071836Z/`

| Artifact | Description |
|----------|-------------|
| `pytest_raw_data_red.log` | RawData parity test failure log (5.26s runtime) |
| `pytest_data_container_red.log` | DataContainer parity test failure log (5.64s runtime) |
| `notes.md` | This summary document |

---

## Spec & Finding Cross-References

**Data Contracts:**
- `specs/data_contracts.md:7-70` — NPZ schema, normalization requirements
- `plans/.../data_contract.md:110-176` — RawData.generate_grouped_data() contract
- `plans/.../data_contract.md:179-200` — PtychoDataContainer attributes table

**Architecture:**
- `docs/architecture.md:75-105` — Data loading pipeline flow
- `ptycho/raw_data.py:365-486` — generate_grouped_data() implementation
- `ptycho/loader.py:93-138` — PtychoDataContainer class definition

**Findings:**
- `docs/findings.md:CONFIG-001` — params.cfg initialization requirement
- `docs/findings.md:DATA-001` — Y patches must be complex64 (historical bug)
- `CLAUDE.md:76-93` — Parameter initialization golden rule

---

**Status:** Red phase complete ✅
**Next Loop:** Implement adapters (Phase C.C1/C.C2)
