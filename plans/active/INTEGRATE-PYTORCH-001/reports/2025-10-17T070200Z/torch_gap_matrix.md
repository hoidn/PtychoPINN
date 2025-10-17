# PyTorch Data Pipeline Gap Analysis (Phase C.A2)

**Initiative:** INTEGRATE-PYTORCH-001
**Phase:** C.A — Canonical Data Contract Alignment
**Date:** 2025-10-17
**Purpose:** Inventory PyTorch dataset implementation and identify gaps vs TensorFlow contract.

**Cross-Reference:** Gap #2 from `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T020000Z/parity_map.md`

---

## 1. Current PyTorch Implementation Overview

### 1.1 Main Dataset Class

**Location:** `ptycho_torch/dset_loader_pt_mmap.py:46-429`

```python
class PtychoDataset(Dataset):
    """Memory-mapped PyTorch dataset for ptychography data."""

    def __init__(self, ptycho_dir, probe_dir, data_dir='data/memmap', remake_map=True):
        # Creates memory-mapped TensorDict for efficient large-dataset handling
        # Memory-maps: diffraction images
        # In-memory: probes, scaling constants, coordinates
```

**Key Characteristics:**
- Uses `tensordict.MemoryMappedTensor` for diffraction data storage
- Generates grouped patches via `ptycho_torch.patch_generator.group_coords()`
- Singleton configuration via `ptycho_torch.config_params.py`
- Returns `TensorDict` + probe + scaling constant from `__getitem__()`

### 1.2 Coordinate Grouping

**Location:** `ptycho_torch/patch_generator.py:10-65`

```python
def group_coords(xcoords_full, ycoords_full, xcoords_bounded, ycoords_bounded,
                 neighbor_function, valid_mask, data_config, C=None):
    """
    Assemble flat dataset into solution regions using K-nearest neighbors.
    Returns:
        nn_indices: shape (M, C)
        coords_nn: shape (M, C, 1, 2)
    """
```

**Algorithm:** K-nearest neighbor search with optional quadrant enforcement

---

## 2. PyTorch vs TensorFlow Contract Gaps

### 2.1 Data Structures

| Aspect | TensorFlow | PyTorch | Status |
|--------|------------|---------|--------|
| **Input Class** | `RawData` (ptycho/raw_data.py:100-485) | `PtychoDataset` (dset_loader_pt_mmap.py:46) | ❌ Different |
| **Initialization** | `RawData.from_file(npz_path)` | `PtychoDataset(ptycho_dir, probe_dir)` | ❌ Different API |
| **Grouped Data** | dict with keys (diffraction, coords_offsets, coords_relative, X_full, Y, nn_indices) | TensorDict with keys (images, coords_center, coords_relative, nn_indices, experiment_id) | ⚠️ Different keys |
| **Output Container** | `PtychoDataContainer` (ptycho/loader.py:93-138) | No named container | ❌ Missing |
| **Tensor Type** | TensorFlow (`tf.Tensor`) | PyTorch (`torch.Tensor`) | ✅ Expected difference |

#### Key Naming Discrepancies

| TensorFlow Key | PyTorch Key | Shape Match? | Notes |
|----------------|-------------|--------------|-------|
| `diffraction` | `images` | ⚠️ | TF: `(N, N, N, C)`, PT: `(N, C, H, W)` — **dimension order differs** |
| `coords_offsets` | `coords_center` | ✅ | Both: `(M, 1, 2, 1)` → `(M, 1, 1, 2)` (minor reshape) |
| `coords_relative` | `coords_relative` | ✅ | Both: `(M, 1, 2, C)` → `(M, C, 1, 2)` (minor reshape) |
| `X_full` | (computed inline) | ❌ | TF: normalized diffraction array; PT: no separate normalization key |
| `Y` (ground truth) | (not present) | ❌ | **CRITICAL GAP** — PyTorch dataset doesn't load Y patches |
| `nn_indices` | `nn_indices` | ✅ | Both: `(M, C)` |
| N/A | `experiment_id` | N/A | PyTorch-specific for multi-experiment batching |
| N/A | `scaling_constant` | N/A | PyTorch-specific normalization factor |

### 2.2 Configuration Interface

| Aspect | TensorFlow | PyTorch | Gap |
|--------|------------|---------|-----|
| **Config Source** | `ptycho.params.cfg` (global dict) | `ptycho_torch.config_params` (singletons) | ⚠️ Isolated systems |
| **Initialization** | `update_legacy_dict(params.cfg, config)` | `ModelConfig().set_settings(dict)` | ❌ No bridge |
| **Consumed Fields** | gridsize, N, n_groups, neighbor_count | grid_size, N, C, K, n_subsample | ⚠️ Name mismatches |
| **Dataclass Support** | `TrainingConfig(model=ModelConfig(...))` | No dataclass ingestion | ❌ Missing |

**Critical Field Mappings:**

| TensorFlow (params.cfg) | PyTorch (config singletons) | Notes |
|-------------------------|----------------------------|-------|
| `gridsize` (int) | `grid_size` (tuple) | **TYPE MISMATCH** — TF: `2`, PT: `(2, 2)` → C=4 |
| `n_groups` (int) | `n_subsample` (int) | **SEMANTIC COLLISION** — different meanings |
| `neighbor_count` (int) | `K` (int) | **NAME MISMATCH** — same semantics |
| `n_subsample` (int) | `n_subsample` (int) | **SEMANTIC COLLISION** — TF: pre-group sampling, PT: post-group sampling |

### 2.3 Grouping Algorithm Differences

| Feature | TensorFlow (`raw_data.py`) | PyTorch (`patch_generator.py`) | Impact |
|---------|---------------------------|---------------------------------|--------|
| **Algorithm** | Sample-then-group | Group-all-then-sample | Different computational costs |
| **Cache Files** | None (eliminated in current version) | None (memory-mapped) | ✅ Both cache-free |
| **Oversampling** | K choose C combinations when `nsamples > n_points` | Not implemented | ⚠️ Feature gap |
| **Sequential Sampling** | `sequential_sampling` flag | Not implemented | ⚠️ Feature gap |
| **Seed Control** | `seed` parameter for reproducibility | `np.random.choice()` without seed control | ❌ Non-deterministic |

### 2.4 Data Types & Device Handling

| Aspect | TensorFlow | PyTorch | Gap |
|--------|------------|---------|-----|
| **Diffraction Dtype** | `tf.float32` | `torch.float32` | ✅ Equivalent |
| **Coords Dtype** | `tf.float32` | `torch.float32` | ✅ Equivalent |
| **Ground Truth (Y)** | `tf.complex64` | Not loaded | ❌ **CRITICAL** |
| **Indices Dtype** | `int32` | `torch.int64` | ⚠️ Size difference |
| **Device Placement** | CPU/GPU auto-managed by TF | Explicit `.to(device)` required | ℹ️ Framework difference |

---

## 3. Missing Components (vs TensorFlow Contract)

### 3.1 RawData-Equivalent Wrapper

**TensorFlow Implementation:** `ptycho/raw_data.py:100-485`

**Required Functionality:**
1. **from_file() Constructor:**
   ```python
   @classmethod
   def from_file(cls, npz_path, ignore_patches=False):
       """Load NPZ and return RawData container."""
   ```

2. **generate_grouped_data() Method:**
   ```python
   def generate_grouped_data(
       self, N, K=4, nsamples=1, seed=None,
       sequential_sampling=False, gridsize=None
   ) -> dict:
       """Return grouped data dictionary matching TF contract."""
   ```

**PyTorch Status:** ❌ **MISSING**
- `PtychoDataset` is a `torch.utils.data.Dataset`, not a data container
- No factory method accepting single NPZ path
- No `generate_grouped_data()` method returning dict

**Impact:**
- Ptychodus `create_raw_data()` workflow cannot delegate to PyTorch
- Test fixtures cannot share RawData instances across backends
- NPZ files prepared for TensorFlow are incompatible

### 3.2 PtychoDataContainer-Equivalent

**TensorFlow Implementation:** `ptycho/loader.py:93-138`

**Required Attributes:**

| Attribute | Type | Shape | PyTorch Status |
|-----------|------|-------|----------------|
| `X` | Tensor | `(n_images, N, N, C)` | ⚠️ Exists as `mmap_ptycho['images']` but wrong dim order |
| `Y` | Tensor | `(n_images, N, N, C)` complex | ❌ Not loaded |
| `Y_I` | Tensor | `(n_images, N, N, C)` | ❌ Not computed |
| `Y_phi` | Tensor | `(n_images, N, N, C)` | ❌ Not computed |
| `coords_nominal` | Tensor | `(n_images, 2)` | ⚠️ Exists but 4D: `(n_images, 1, 1, 2)` |
| `coords_true` | Tensor | `(n_images, 2)` | ❌ Not present |
| `probe` | Tensor | `(N, N)` complex | ✅ Exists in `dataset.data_dict['probes']` |
| `norm_Y_I` | NumPy/scalar | Varies | ⚠️ Exists as `scaling_constant` but different semantics |
| `nn_indices` | NumPy | `(n_images, C)` | ✅ Exists in `mmap_ptycho['nn_indices']` |
| `global_offsets` | NumPy | `(n_images, 1, 2, 1)` | ⚠️ Exists as `coords_center` with shape `(n_images, 1, 1, 2)` |
| `local_offsets` | NumPy | `(n_images, 1, 2, C)` | ⚠️ Exists as `coords_relative` with shape `(n_images, C, 1, 2)` |

**PyTorch Status:** ❌ **MISSING NAMED CONTAINER**
- TensorDict is returned by `__getitem__()` but not a persistent class attribute
- No `.X`, `.Y`, `.probe` attribute access pattern
- Missing `from_raw_data_without_pc()` factory method

**Impact:**
- Cannot replace `ptycho.loader.load()` calls with PyTorch equivalent
- Model code expecting `container.X` won't work with TensorDict
- Missing ground truth (`Y`) blocks supervised training

### 3.3 Ground Truth (Y) Patch Loading

**TensorFlow Implementation:** `ptycho/raw_data.py:517-527`

```python
if self.Y is not None:
    Y_nn = self.Y[nn_indices]  # Extract patches via nn_indices
    Y4d_nn = np.transpose(Y_nn, [0, 2, 3, 1])  # Reshape to channel format
    dset['Y'] = Y4d_nn.astype(np.complex64)  # MUST be complex64
```

**PyTorch Status:** ❌ **CRITICAL GAP**
- `PtychoDataset` does not load `objectGuess` or `Y` from NPZ files
- No `Y` key in memory-mapped TensorDict
- Cannot support supervised training modes

**Required Work:**
1. Add `objectGuess` loading in `PtychoDataset.__init__()`
2. Extract Y patches using `nn_indices` during memory mapping
3. Store in `mmap_ptycho['Y']` as `torch.complex64`
4. Validate dtype (avoid silent `float64` conversion bug from DATA-001)

### 3.4 NPZ Round-Trip Compatibility

**TensorFlow Workflow:** NPZ file → `RawData.from_file()` → `generate_grouped_data()` → dict

**PyTorch Workflow:** NPZ directory → `PtychoDataset(ptycho_dir, ...)` → memory mapping → TensorDict

**Gaps:**
1. **Single-file vs directory:** TF accepts one NPZ path; PT expects directory of NPZ files
2. **NPZ schema:** PT hardcodes key names (`diff3d`, `xcoords`, `ycoords`) without validation vs `specs/data_contracts.md`
3. **Normalization state:** PT applies normalization during loading; TF expects pre-normalized data per spec
4. **Amplitude validation:** No check that `diffraction` is amplitude (not intensity) per `specs/data_contracts.md:23-37`

**Impact:**
- NPZ files exported by Ptychodus (single-file format) are incompatible
- Mixed TensorFlow/PyTorch workflows break on data format assumptions
- Risk of violating canonical normalization requirements

---

## 4. Reuse Opportunities

### 4.1 Delegate to TensorFlow RawData

**Proposal:** Create thin PyTorch wrapper that delegates grouping to existing `ptycho/raw_data.py`

```python
# Proposed: ptycho_torch/raw_data_bridge.py
from ptycho.raw_data import RawData as TFRawData
from ptycho.config.config import update_legacy_dict
from ptycho import params as p

class RawDataTorch:
    """Torch-agnostic wrapper delegating to TensorFlow RawData."""

    def __init__(self, npz_path, config):
        # 1. Update params.cfg from config
        update_legacy_dict(p.cfg, config)

        # 2. Delegate to TensorFlow RawData
        self._tf_raw_data = TFRawData.from_file(npz_path)

    def generate_grouped_data(self, N, K, nsamples, seed, sequential_sampling, gridsize):
        # Delegate to TensorFlow implementation
        return self._tf_raw_data.generate_grouped_data(
            N=N, K=K, nsamples=nsamples, seed=seed,
            sequential_sampling=sequential_sampling, gridsize=gridsize
        )
```

**Advantages:**
- ✅ Reuses battle-tested TensorFlow grouping algorithm
- ✅ Shares `.groups_cache.npz` files (if re-enabled)
- ✅ Ensures identical `nn_indices` for parity tests
- ✅ Maintains `params.cfg` initialization contract

**Disadvantages:**
- ⚠️ Requires TensorFlow dependency in PyTorch-only environments
- ⚠️ NumPy→PyTorch tensor conversion overhead

### 4.2 Shared Cache Files

**Current TensorFlow:** No cache files (sample-then-group eliminated them per `ptycho/raw_data.py:29-30`)

**Current PyTorch:** Memory-mapped TensorDict under `data/memmap/` directory

**Compatibility:** ❌ Incompatible storage formats
- TensorFlow: Temporary NumPy dict (no disk persistence)
- PyTorch: `tensordict.MemoryMappedTensor` (custom binary format)

**Recommendation:** Do NOT attempt cache unification; delegate grouping instead.

### 4.3 Torch-Agnostic Code Paths

**Fully Reusable (NumPy-only):**
- `ptycho/raw_data.py` — All functions operate on NumPy arrays
- `ptycho/config/config.py` — Dataclass definitions and `update_legacy_dict()`
- `specs/data_contracts.md` — NPZ schema validation logic

**Framework-Specific:**
- `ptycho/loader.py` — TensorFlow tensor conversion (`tf.convert_to_tensor()`)
- `ptycho_torch/dset_loader_pt_mmap.py` — PyTorch tensor conversion (`torch.from_numpy()`)

**Design Pattern:** Shared NumPy pipeline → Framework-specific tensor conversion

---

## 5. Type Mismatches & Conversions

### 5.1 Configuration Field Mapping

| TensorFlow Field | PyTorch Field | Conversion Rule | Blocker? |
|------------------|---------------|-----------------|----------|
| `gridsize: int` | `grid_size: tuple` | `gridsize=2` → `grid_size=(2, 2)` | ❌ Yes |
| `n_groups: int` | `n_subsample: int` | **SEMANTIC COLLISION** — cannot map | ❌ Yes |
| `neighbor_count: int` | `K: int` | Direct 1:1 mapping | ✅ No |
| `N: int` | `N: int` | Direct 1:1 mapping | ✅ No |
| `sequential_sampling: bool` | (not implemented) | Must add flag support | ⚠️ Feature gap |

**Required Config Bridge (from Phase B):**
```python
# ptycho_torch/config_bridge.py (already implemented in Attempt #9)
from ptycho_torch.config_params import ModelConfig as PTModelConfig
from ptycho.config.config import ModelConfig as TFModelConfig

def to_model_config(pt_config, overrides=None):
    """Convert PyTorch singleton → TensorFlow dataclass."""
    grid_size = pt_config.get('grid_size')  # e.g., (2, 2)
    gridsize = validate_square_grid(grid_size)  # → 2

    return TFModelConfig(
        N=pt_config.get('N'),
        gridsize=gridsize,
        # ... other fields
    )
```

### 5.2 Tensor Dimension Ordering

**TensorFlow Convention:** `(batch, height, width, channels)` (NHWC)

**PyTorch Convention:** `(batch, channels, height, width)` (NCHW)

**Memory-Mapped Diffraction:**
- TensorFlow: `diffraction.shape == (nsamples, N, N, gridsize²)`
- PyTorch: `mmap_ptycho['images'].shape == (nsamples, gridsize², N, N)`

**Conversion Required:**
```python
# NumPy (from generate_grouped_data) → TensorFlow
tf_diffraction = np.transpose(diff4d_nn, [0, 2, 3, 1])  # (N, C, H, W) → (N, H, W, C)

# NumPy → PyTorch
pt_diffraction = torch.from_numpy(diff4d_nn)  # Keep (N, C, H, W) format
```

**Impact:** Must transpose when comparing TensorFlow vs PyTorch tensors in parity tests.

### 5.3 Complex Number Handling

**TensorFlow:** Native `tf.complex64` support

**PyTorch:** Native `torch.complex64` support (since PyTorch 1.6)

**Compatibility:** ✅ Both support complex tensors

**Validation Required:**
- Verify `Y` patches are `torch.complex64` (not `torch.float64`)
- Test complex multiplication: `probe * illuminated_object`
- Ensure phase extraction: `torch.angle(Y)` matches `tf.math.angle(Y)`

---

## 6. Performance & Memory Considerations

| Aspect | TensorFlow | PyTorch | Notes |
|--------|------------|---------|-------|
| **Memory Mapping** | Not used | `tensordict.MemoryMappedTensor` | PyTorch advantage for large datasets |
| **Grouping Algorithm** | O(nsamples·K) sample-then-group | O(n_points·K) group-all | TensorFlow faster for small nsamples |
| **Cache Strategy** | None (cache-free design) | Persistent memory maps | Different trade-offs |
| **Multi-Process** | TensorFlow data pipelines | DataLoader with `num_workers` | Framework-specific |

**Recommendation:** Delegate grouping to TensorFlow (faster algorithm), then convert to PyTorch tensors.

---

## 7. Summary: Critical Blockers for Phase C

| # | Gap | Severity | Phase |
|---|-----|----------|-------|
| 1 | No `RawDataTorch` wrapper delegating to `ptycho/raw_data.py` | **CRITICAL** | C.C1 |
| 2 | No `PtychoDataContainerTorch` class with TF-equivalent attributes | **CRITICAL** | C.C2 |
| 3 | Ground truth (`Y`) patches not loaded from NPZ | **CRITICAL** | C.C3 |
| 4 | `gridsize` (int) vs `grid_size` (tuple) type mismatch | **CRITICAL** | B.B3 (config bridge) |
| 5 | `n_groups` / `n_subsample` semantic collision | **HIGH** | C.C4 |
| 6 | No single-NPZ-file loading (only directories) | **HIGH** | C.C1 |
| 7 | Non-deterministic sampling (no seed control) | **MEDIUM** | C.C1 |
| 8 | `sequential_sampling` flag not implemented | **MEDIUM** | C.C1 |
| 9 | Dimension order (NCHW vs NHWC) requires transpose | **LOW** | C.C2 |

**Dependency Chain:**
1. Config bridge (Phase B) must complete first → provides `gridsize` translation
2. `RawDataTorch` (C.C1) → depends on config bridge
3. `PtychoDataContainerTorch` (C.C2) → depends on `RawDataTorch`
4. Memory-mapped integration (C.C3) → depends on both above

---

## 8. Recommended Implementation Strategy (Phase C.C)

### Option A: Full Delegation (Recommended)

**Architecture:**
```
PyTorch Entry Point
    ↓
RawDataTorch (thin wrapper)
    ↓
ptycho.raw_data.RawData (reuse TensorFlow)
    ↓
NumPy dict (grouped data)
    ↓
PtychoDataContainerTorch (tensor conversion)
    ↓
torch.Tensor outputs
```

**Advantages:**
- ✅ Code reuse (single grouping implementation)
- ✅ Identical `nn_indices` across backends (perfect parity)
- ✅ Inherits TensorFlow bug fixes automatically

### Option B: Native PyTorch Rewrite

**Architecture:**
```
PyTorch Entry Point
    ↓
PtychoDatasetTorch (native implementation)
    ↓
patch_generator.group_coords() (PyTorch-only)
    ↓
TensorDict (memory-mapped)
    ↓
torch.Tensor outputs
```

**Advantages:**
- ✅ No TensorFlow dependency
- ✅ Memory mapping optimization

**Disadvantages:**
- ❌ Must re-implement sample-then-group algorithm
- ❌ Risk of behavioral divergence
- ❌ Duplicate maintenance burden

**Recommendation:** **Option A** for Phase C to minimize risk; defer Option B optimization to future phase.

---

## File References

**PyTorch Implementation:**
- `/ptycho_torch/dset_loader_pt_mmap.py` — Main dataset class
- `/ptycho_torch/patch_generator.py` — Coordinate grouping logic
- `/ptycho_torch/config_params.py` — Singleton configuration
- `/ptycho_torch/config_bridge.py` — Config translation (from Phase B)

**TensorFlow Reference:**
- `/ptycho/raw_data.py` — Authoritative grouping implementation
- `/ptycho/loader.py` — PtychoDataContainer definition
- `/ptycho/config/config.py` — Dataclass bridge

**Specifications:**
- `/specs/data_contracts.md` — NPZ schema requirements
- `/specs/ptychodus_api_spec.md` — §4.3 data ingestion contract

**Gap Analysis:**
- `/plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T020000Z/parity_map.md` — Gap #2 summary

---

**Next Steps (Phase C.C):** Implement `RawDataTorch` wrapper (C.C1) and `PtychoDataContainerTorch` (C.C2) per Option A strategy.
