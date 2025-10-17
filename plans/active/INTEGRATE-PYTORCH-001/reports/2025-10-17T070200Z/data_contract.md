# TensorFlow Data Pipeline Contract (Phase C.A1)

**Initiative:** INTEGRATE-PYTORCH-001
**Phase:** C.A — Canonical Data Contract Alignment
**Date:** 2025-10-17
**Purpose:** Document TensorFlow data pipeline requirements as normative baseline for PyTorch parity.

---

## 1. NPZ Input Schema

**Authoritative Source:** `specs/data_contracts.md:7-70`

### Required Keys

| Key Name | Shape | Data Type | Description | Status |
|----------|-------|-----------|-------------|--------|
| `diffraction` | `(n_images, H, W)` | `float32` | Measured diffraction patterns (**amplitude**, not intensity) | **Required** |
| `Y` | `(n_images, H, W)` | `complex64` | Ground truth real-space object patches (3D, squeeze channel) | **Required for supervised** |
| `objectGuess` | `(M, M)` | `complex64` | Full un-patched ground truth object (M >> N) | **Required** |
| `probeGuess` | `(H, W)` | `complex64` | Ground truth probe function | **Required** |
| `xcoords` | `(n_images,)` | `float64` | X-coordinates of scan positions | **Required** |
| `ycoords` | `(n_images,)` | `float64` | Y-coordinates of scan positions | **Required** |
| `scan_index` | `(n_images,)` | `int` | Index of scan point for each pattern | **Optional** |

### Critical Normalization Requirements

**Source:** `specs/data_contracts.md:23-70`

1. **Amplitude vs Intensity:**
   - `diffraction` MUST contain **amplitude** (sqrt of intensity), not raw intensity
   - Conversion: `diffraction = np.sqrt(intensity)`

2. **Data Normalization:**
   - Diffraction patterns MUST be normalized with max values typically < 1.0
   - nphotons parameter controls physics scaling during training, NOT data values

3. **NO Pre-scaling:**
   - Do NOT pre-apply photon scaling to data
   - Example: Even for nphotons=1e6, diffraction data remains normalized

4. **Data Type Strictness:**
   - `Y` MUST be `complex64` (not `float64`) — historical silent conversion bug (FINDING-DATA-001)
   - `diffraction` MUST be `float32` (not `uint16` or `float64`)

**Validation Commands:**
```python
import numpy as np
data = np.load('dataset.npz')
assert np.max(data['diffraction']) < 10.0, "Data appears unnormalized"
assert np.min(data['diffraction']) >= 0.0, "Amplitude should be non-negative"
assert data['diffraction'].dtype == np.float32, "Should be float32"
assert data['Y'].dtype == np.complex64, "Y must be complex64"
```

---

## 2. RawData.generate_grouped_data() Contract

**Source:** `ptycho/raw_data.py:365-486`

### Function Signature

```python
def generate_grouped_data(
    self,
    N: int,                              # Size of solution region
    K: int = 4,                          # Number of nearest neighbors
    nsamples: int = 1,                   # Number of samples/groups
    dataset_path: Optional[str] = None,  # (Deprecated, kept for compatibility)
    seed: Optional[int] = None,          # Random seed for reproducibility
    sequential_sampling: bool = False,   # Force sequential vs random sampling
    gridsize: Optional[int] = None       # Explicit gridsize (prefers param)
) -> dict
```

### Configuration Dependencies (CRITICAL)

**Mandatory Initialization Sequence:**

```python
# 1. Create modern dataclass configuration
from ptycho.config.config import TrainingConfig, ModelConfig, update_legacy_dict
from ptycho import params as p

config = TrainingConfig(
    model=ModelConfig(N=64, gridsize=2),
    n_groups=512,
    neighbor_count=4
)

# 2. MANDATORY: Bridge to legacy system BEFORE any data loading
update_legacy_dict(p.cfg, config)

# 3. NOW it is safe to call data functions
from ptycho import raw_data
raw_data_obj = raw_data.RawData.from_file('data.npz')
grouped_data = raw_data_obj.generate_grouped_data(N=64, K=4, nsamples=512)
```

**Critical Bug Signature (CONFIG-001):**
```
Generated shape: (1000, 64, 64, 1) instead of (1000, 64, 64, 4) with gridsize=2
Root cause: update_legacy_dict() not called before generate_grouped_data()
→ params.cfg['gridsize'] defaults to 1 instead of 2
→ C = gridsize² = 1 instead of 4
Fix: Call update_legacy_dict(params.cfg, config) BEFORE any data loading
```

**Documented Locations:**
- `ptycho/raw_data.py:412-425` (dependency warning)
- `ptycho/config/config.py:270-275` (bridge docstring)
- `docs/debugging/QUICK_REFERENCE_PARAMS.md` (troubleshooting)

### Returned Dictionary Structure

**Required Keys:**

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `diffraction` | `(nsamples, N, N, gridsize²)` | `float32` | 4D grouped diffraction (channel format) |
| `X_full` | `(nsamples, N, N, gridsize²)` | `float32` | Normalized diffraction (model-ready) |
| `coords_offsets` | `(nsamples, 1, 2, 1)` | `float32` | Mean coordinates for each group |
| `coords_relative` | `(nsamples, 1, 2, gridsize²)` | `float32` | Relative coordinates within group |
| `nn_indices` | `(nsamples, gridsize²)` | `int32` | Selected coordinate indices |

**Optional Keys:**

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `Y` | `(nsamples, N, N, gridsize²)` | `complex64` | Ground truth object patches (if objectGuess provided) |
| `coords_start_offsets` | `(nsamples, 1, 2, 1)` | `float32` | Start coordinate offsets (if available) |
| `coords_start_relative` | `(nsamples, 1, 2, gridsize²)` | `float32` | Start coordinate relatives |
| `objectGuess` | `(M, M)` | `complex64` | Original full object (reference) |

### Data Transformation Pipeline

**Source:** `ptycho/raw_data.py:506-553`

```python
# 1. Index selection via K-nearest neighbors
nn_indices = selected_groups  # shape: (nsamples, gridsize²)

# 2. 3D → 4D transformation (channel layout)
diff4d_nn = np.transpose(self.diff3d[nn_indices], [0, 2, 3, 1])
# Input:  (n_points, N, N)
# Output: (nsamples, N, N, gridsize²)

# 3. Coordinate reshaping
coords_nn = np.transpose(
    np.array([self.xcoords[nn_indices], self.ycoords[nn_indices]]),
    [1, 0, 2]
)[:, None, :, :]
# Result: (nsamples, 1, 2, gridsize²)

# 4. Normalization
X_full_norm = np.sqrt(((N / 2)**2) / np.mean(X_full**2))
X_full = X_full_norm * diffraction
```

### Sampling Strategy: Sample-Then-Group

**Algorithm:** `ptycho/raw_data.py:436-486`

1. **Set gridsize:** Explicit parameter or `params.cfg['gridsize']` (defaults to 1)
2. **Calculate C:** `C = gridsize²` (patch cardinality)
3. **Determine oversampling:** `needs_oversampling = (nsamples > n_points) and (C > 1)`
4. **Select strategy:**
   - `sequential_sampling=True` → First N points
   - `sequential_sampling=False` → Random sampling with seed
5. **Route to implementation:**
   - Standard: `_generate_groups_efficiently()`
   - Oversampling: `_generate_groups_with_oversampling()` (K choose C combinations)

**Performance:** 10-100x faster than legacy cache-based approaches (no cache files generated)

---

## 3. PtychoDataContainer Contract

**Source:** `ptycho/loader.py:93-138`

### Class Attributes

| Attribute | Type | Shape | Dtype | Description |
|-----------|------|-------|-------|-------------|
| **X** | TensorFlow | `(n_images, N, N, gridsize²)` | `tf.float32` | Diffraction patterns (normalized amplitude) |
| **Y** | TensorFlow | `(n_images, N, N, gridsize²)` | `tf.complex64` | Combined complex ground truth |
| **Y_I** | TensorFlow | `(n_images, N, N, gridsize²)` | `tf.float32` | Ground truth amplitude patches |
| **Y_phi** | TensorFlow | `(n_images, N, N, gridsize²)` | `tf.float32` | Ground truth phase patches |
| **coords_nominal** | TensorFlow | `(n_images, 2)` | `tf.float32` | Scan coordinates (nominal/relative) |
| **coords_true** | TensorFlow | `(n_images, 2)` | `tf.float32` | True scan coordinates |
| **coords** | TensorFlow | `(n_images, 2)` | `tf.float32` | Alias for coords_nominal |
| **probe** | TensorFlow | `(N, N)` | `tf.complex64` | Probe function |
| **norm_Y_I** | NumPy | Scalar/array | `float32` | Normalization factors (from scale_nphotons) |
| **nn_indices** | NumPy | `(n_images, gridsize²)` | `int32` | Nearest neighbor indices |
| **global_offsets** | NumPy | `(n_images, 1, 2, 1)` | `float32` | Global coordinate offsets |
| **local_offsets** | NumPy | `(n_images, 1, 2, gridsize²)` | `float32` | Local coordinate offsets |
| **YY_full** | NumPy | Variable | `complex64` | Full object reconstruction (or None) |

### Multi-Channel Tensor Layout

For `gridsize > 1`, tensors preserve last dimension as "channel" dimension:
- **4D Tensors (X, Y_I, Y_phi, Y):** Last dimension = `gridsize²` (e.g., 4 for gridsize=2, 9 for gridsize=3)
- **Coordinate Reshaping:** `(n_groups, 1, 2, gridsize²)` → `(n_groups, 2)` via flattening
- **Channel Validation:** `loader.load()` verifies `X.shape[-1] == Y.shape[-1]` (line 325-326)

### Initialization Factory Method

**Source:** `ptycho/loader.py:157-185`

```python
@staticmethod
def from_raw_data_without_pc(
    xcoords, ycoords, diff3d, probeGuess, scan_index,
    objectGuess=None, N=None, K=7, nsamples=1
) -> PtychoDataContainer:
    """Create container from NPZ-like arrays."""
    # 1. Create RawData
    # 2. Generate grouped data
    # 3. Call load() for tensor conversion
    # 4. Return fully initialized container
```

---

## 4. Loader Transformation Pipeline

**Source:** `ptycho/loader.py:250-341` (`load()` function)

### Function Signature

```python
def load(
    cb: Callable,              # Callback returning grouped data dict
    probeGuess: tf.Tensor,     # Initial probe (complex64, N×N)
    which: str,                # 'train' or 'test'
    create_split: bool         # Enable train/test splitting
) -> PtychoDataContainer
```

### Data Flow Steps

**Step 1: Invoke Callback** (lines 285-288)
```python
if create_split:
    dset, train_frac = cb()  # Returns (dict, float)
else:
    dset = cb()              # Returns dict
```

**Step 2: Extract Core Arrays** (lines 290-295)
```python
X_full = dset['X_full']                      # Shape: (nsamples, N, N, gridsize²)
global_offsets = dset['coords_offsets']      # Shape: (nsamples, 1, 2, 1)
coords_nominal = dset['coords_relative']     # Shape: (nsamples, 1, 2, gridsize²)
```

**Step 3: Train/Test Splitting** (lines 298-302)
- Uses `split_tensor()` for global_offsets
- Uses `split_data()` for X_full, coords_nominal, coords_true
- Applies consistent indices across arrays

**Step 4: TensorFlow Conversion** (lines 305-307)
```python
X = tf.convert_to_tensor(X_full_split, dtype=tf.float32)
coords_nominal = tf.convert_to_tensor(coords_nominal, dtype=tf.float32)
```

**Step 5: Ground Truth Handling** (lines 310-322)
```python
if dset['Y'] is None:
    Y = tf.ones_like(X, dtype=tf.complex64)  # Dummy placeholder
else:
    Y_split = split_data(Y_full, ...)
    Y = tf.convert_to_tensor(Y_split, dtype=tf.complex64)
```

**Step 6: Extract Amplitude & Phase** (lines 329-330)
```python
Y_I = tf.math.abs(Y)
Y_phi = tf.math.angle(Y)
```

**Step 7: Calculate Intensity Scaling** (line 332)
```python
norm_Y_I = datasets.scale_nphotons(X)  # Calls ptycho/diffsim.py
```

**Step 8: Construct Container** (lines 337-338)
```python
container = PtychoDataContainer(
    X, Y_I, Y_phi, norm_Y_I, YY_full,
    coords_nominal, coords_true,
    nn_indices, global_offsets, local_offsets,
    probeGuess
)
```

### Channel Validation (CRITICAL)

**Source:** `ptycho/loader.py:325-326`

```python
if X.shape[-1] != Y.shape[-1]:
    raise ValueError(f"Channel mismatch: X={X.shape[-1]}, Y={Y.shape[-1]}")
```

**Failure Mode:** All tensors must have matching last dimension (`gridsize²`)

---

## 5. Cache Behavior

**Source:** `ptycho/raw_data.py:1-100, 386`

**Status:** **ELIMINATED** in current implementation

- **Previous:** `.groups_cache.npz` files generated for expensive group discovery
- **Current:** No cache files generated or used
- **Performance:** "10-100x faster than cache-based approaches; zero cache files"
- **Algorithm:** Sample-Then-Group strategy makes caching unnecessary (O(nsamples·K) vs O(n_points·K))
- **dataset_path Parameter:** Kept for backward compatibility but no longer functional

**For PyTorch Parity:** No cache file compatibility requirements; delegate grouping to TensorFlow RawData.

---

## 6. Configuration Bridge Requirements

**Source:** `ptycho/config/config.py:267-294`, `specs/ptychodus_api_spec.md:§4.3`

### KEY_MAPPINGS Translation

**Source:** `ptycho/config/config.py:231-241`

| Modern Field | Legacy Key | Applied By |
|--------------|------------|------------|
| `object_big` | `object.big` | `dataclass_to_legacy_dict()` |
| `probe_big` | `probe.big` | `dataclass_to_legacy_dict()` |
| `probe_mask` | `probe.mask` | `dataclass_to_legacy_dict()` |
| `probe_trainable` | `probe.trainable` | `dataclass_to_legacy_dict()` |
| `intensity_scale_trainable` | `intensity_scale.trainable` | `dataclass_to_legacy_dict()` |
| `output_dir` | `output_prefix` | `dataclass_to_legacy_dict()` |
| `train_data_file` | `train_data_file_path` | `dataclass_to_legacy_dict()` |
| `test_data_file` | `test_data_file_path` | `dataclass_to_legacy_dict()` |

### Required params.cfg Keys for Data Pipeline

| Legacy Key | Modern Field | Consumer | Default |
|-----------|--------------|----------|---------|
| `gridsize` | `ModelConfig.gridsize` | `raw_data.generate_grouped_data()` | `1` ⚠️ |
| `N` | `ModelConfig.N` | `raw_data.get_image_patches()` | Retrieved from params |
| `neighbor_count` | `TrainingConfig.neighbor_count` | `workflows/components.py` | `4` |
| `n_groups` | `TrainingConfig.n_groups` | `workflows/components.py` | `512` |
| `sequential_sampling` | `TrainingConfig.sequential_sampling` | `raw_data.generate_grouped_data()` | `False` |

---

## 7. Minimal Fixture & ROI for Tests

**Recommendation:** Use synthetic data for deterministic unit tests

### Synthetic Fixture Pattern

**Source:** `tests/test_coordinate_grouping.py` pattern

```python
import numpy as np
from ptycho.raw_data import RawData

def create_minimal_raw_data(n_points=100, N=64, gridsize=2):
    """Create synthetic RawData for testing."""
    # Deterministic coordinate grid
    x = np.linspace(0, 10, int(np.sqrt(n_points)))
    y = np.linspace(0, 10, int(np.sqrt(n_points)))
    xv, yv = np.meshgrid(x, y)
    xcoords = xv.flatten()[:n_points]
    ycoords = yv.flatten()[:n_points]

    # Random diffraction patterns
    np.random.seed(42)
    diff3d = np.random.rand(n_points, N, N).astype(np.float32)

    # Simple probe and object
    probe = np.ones((N, N), dtype=np.complex64)
    obj = np.ones((N*2, N*2), dtype=np.complex64)

    return RawData(xcoords, ycoords, diff3d, probe, obj, scan_index=None)
```

### ROI Parameters

| Aspect | Unit Test | Integration | Rationale |
|--------|-----------|-------------|-----------|
| N (grid size) | 32-64 | 64 | Larger = slower forward pass |
| gridsize | 1-2 | 2 | Standard is 2x2 patches |
| n_groups | 64-128 | 512 | Fewer = faster loading |
| n_patterns | 10-50 | 200-1000 | Keep unit tests <100ms |

**Real Dataset Status:**
- `datasets/fly/fly001_transposed.npz`: **DOES NOT EXIST** (only PNGs present)
- Alternative: Use synthetic fixtures for unit tests; reserve real data for integration

---

## 8. Critical Gotchas Checklist

**Before calling ANY data loading functions:**

- [ ] Call `update_legacy_dict(params.cfg, config)` with initialized `TrainingConfig`/`InferenceConfig`
- [ ] Verify `params.cfg['gridsize']` matches expected value (not defaulting to 1)
- [ ] Verify `params.cfg['N']` matches diffraction pattern size
- [ ] Verify NPZ keys match `specs/data_contracts.md` schema
- [ ] Verify `diffraction` dtype is `float32` (not `uint16` or `float64`)
- [ ] Verify `Y` dtype is `complex64` (not `float64`)
- [ ] Verify `diffraction` contains amplitude, not intensity
- [ ] Verify max(diffraction) < 10.0 (normalized)

---

## File References

**Normative Contracts:**
- `/specs/data_contracts.md` — NPZ schema, normalization requirements
- `/specs/ptychodus_api_spec.md` — §4 data ingestion, §5.2 training config

**TensorFlow Implementation:**
- `/ptycho/raw_data.py` — RawData class, generate_grouped_data(), sample-then-group algorithm
- `/ptycho/loader.py` — PtychoDataContainer, load(), tensor conversion pipeline
- `/ptycho/config/config.py` — update_legacy_dict(), KEY_MAPPINGS, dataclasses

**Architecture & Debugging:**
- `/docs/architecture.md` — §3 data pipeline flow
- `/docs/DEVELOPER_GUIDE.md` — §3 data pipeline, §10 params lifecycle
- `/docs/debugging/QUICK_REFERENCE_PARAMS.md` — CONFIG-001 gotcha
- `/docs/findings.md` — DATA-001, CONFIG-001, NORMALIZATION-001 findings

**Testing Patterns:**
- `/tests/test_coordinate_grouping.py` — Synthetic fixture pattern
- `/tests/conftest.py` — Global pytest fixtures
- `/docs/TESTING_GUIDE.md` — TDD methodology, fixture guidance
