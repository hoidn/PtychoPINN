# Inference Pipeline Architecture

**Version:** 1.0.0
**Last Updated:** 2025-01-09

This document describes the architecture, data flow, and design rationale for the PtychoPINN inference pipeline. For normative contracts and function signatures, see `specs/spec-inference-pipeline.md`.

---

## 1. Component Architecture

### 1.1. High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    scripts/inference/inference.py                           │
│                         (Entry Point)                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  CLI Arguments → InferenceConfig → Backend Dispatch → Inference → Output    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│    Backend    │          │   Workflow    │          │ Data Pipeline │
│   Selector    │          │  Components   │          │               │
├───────────────┤          ├───────────────┤          ├───────────────┤
│load_inference_│          │load_data()    │          │RawData        │
│bundle_with_   │          │DiffractionTo- │          │.from_file()   │
│backend()      │          │ObjectAdapter  │          │.generate_     │
└───────────────┘          └───────────────┘          │grouped_data() │
        │                          │                  └───────────────┘
        ▼                          ▼                          │
┌───────────────┐          ┌───────────────┐                  ▼
│ ModelManager  │          │  loader.py    │          ┌───────────────┐
│ .load_        │          │  .load()      │          │PtychoData-    │
│ multiple_     │          │  PtychoData-  │          │Container      │
│ models()      │          │  Container    │          │ .X, .Y, ...   │
└───────────────┘          └───────────────┘          └───────────────┘
        │                          │                          │
        └──────────────────────────┼──────────────────────────┘
                                   ▼
                          ┌───────────────┐
                          │  nbutils.py   │
                          │reconstruct_   │
                          │image()        │
                          └───────────────┘
                                   │
                                   ▼
                          ┌───────────────┐
                          │ tf_helper.py  │
                          │reassemble_    │
                          │position()     │
                          └───────────────┘
```

### 1.2. Module Responsibilities

| Module | Responsibility | Key Functions |
|--------|---------------|---------------|
| `backend_selector.py` | Backend dispatch (TF/PyTorch) | `load_inference_bundle_with_backend()` |
| `components.py` | Workflow orchestration | `load_data()`, `load_inference_bundle()` |
| `raw_data.py` | Data ingestion, coordinate grouping | `RawData`, `generate_grouped_data()` |
| `loader.py` | NumPy→TensorFlow conversion | `load()`, `PtychoDataContainer` |
| `nbutils.py` | Inference execution | `reconstruct_image()` |
| `tf_helper.py` | Tensor ops, reassembly | `reassemble_position()`, `shift_and_sum()` |
| `model_manager.py` | Model serialization | `load_multiple_models()` |

---

## 2. Data Flow: Load → Inference → Stitch

### 2.1. Phase 1: Model Loading

```
model_dir/wts.h5.zip
    │
    ▼ ModelManager.load_multiple_models()
    │   ├── Extracts models from zip archive
    │   ├── Restores params.cfg from saved state ← CONFIG-001 compliance
    │   └── Returns dict: {'diffraction_to_obj': tf.keras.Model, ...}
    │
    ▼ DiffractionToObjectAdapter (wrapper)
        └── Syncs params.cfg['gridsize'] from input tensor channels
            at predict() time ← Prevents MODULE-SINGLETON-001
```

**Design Rationale:** The `DiffractionToObjectAdapter` wrapper exists because legacy model code captures `gridsize` at import time. The adapter inspects input tensor shapes and updates `params.cfg` before delegation, ensuring the model's internal layers see consistent configuration.

### 2.2. Phase 2: Data Loading

```
test_data.npz
    │
    ▼ load_data(file_path, n_images, n_subsample, ...)
    │   ├── Loads: xcoords, ycoords, diff3d, probeGuess, objectGuess
    │   ├── Applies coordinate transforms (flip_x, flip_y, swap_xy, coord_scale)
    │   ├── Subsamples if n_subsample < dataset_size
    │   └── Returns: RawData instance
    │
    ▼ RawData.generate_grouped_data(N, K, nsamples, gridsize)
    │   ├── Builds KDTree from (xcoords, ycoords)
    │   ├── Samples seed points (random or sequential)
    │   ├── Finds K nearest neighbors per seed
    │   ├── Forms groups of size C = gridsize²
    │   └── Returns: GroupedDataDict
    │
    ▼ loader.load(cb, probeGuess, ...)
        ├── Converts NumPy → TensorFlow tensors (lazy)
        └── Returns: PtychoDataContainer
```

**Design Rationale:** The "sample-then-group" strategy in `generate_grouped_data()` provides O(nsamples × K × log M) complexity vs O(M × K) for the old cache-based approach. The lazy tensor conversion in `PtychoDataContainer` prevents GPU OOM for large datasets.

### 2.3. Phase 3: Inference

```
PtychoDataContainer + Model
    │
    ▼ reconstruct_image(test_data, diffraction_to_obj)
    │   │
    │   ├── Extracts: global_offsets, local_offsets from container
    │   │
    │   ├── Scales input: test_data.X * params['intensity_scale']
    │   │
    │   └── model.predict([scaled_X, local_offsets])
    │       │
    │       └── Returns: obj_tensor_full
    │           Shape: (B, N, N, C) complex64
    │
    ▼ Returns: (obj_tensor_full, global_offsets)
```

**Design Rationale:** The model receives both diffraction patterns AND local offsets as inputs. The offsets enable position-aware reconstruction where the network learns to account for sub-pixel positioning.

### 2.4. Phase 4: Stitching/Reassembly

```
obj_tensor_full + global_offsets
    │
    ▼ reassemble_position(obj_tensor_full, global_offsets, M=20)
    │   │
    │   ├── Crops central M×M region from each N×N patch
    │   │   └── Reduces boundary artifacts from patch edges
    │   │
    │   ├── Computes center of mass of offsets
    │   │   └── Centers reconstruction on canvas
    │   │
    │   ├── Adjusts offsets: adjusted = offsets - center_of_mass
    │   │
    │   ├── Determines canvas size from max offset
    │   │   └── dynamic_pad = ceil(max(|adjusted_offsets|))
    │   │
    │   └── shift_and_sum(obj_tensor, global_offsets, M)
    │       │
    │       └── Batched translation + accumulation
    │           ├── Chunks of 1024 patches for memory efficiency
    │           ├── Bilinear interpolation for sub-pixel shifts
    │           └── Overlap normalization (divide by count)
    │
    ▼ Returns: assembled_image (H, W, 1) complex64
```

**Design Rationale:** The M parameter controls the "effective patch size" used for stitching. Using M < N discards edge regions where neural network predictions are less reliable. The shift_and_sum implementation uses chunking to avoid GPU OOM while maintaining 20-44x speedup over iterative approaches.

---

## 3. Tensor Format System

### 3.1. Three Canonical Representations

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TENSOR FORMAT SYSTEM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  GRID FORMAT: (B, gridsize, gridsize, N, N, 1)                             │
│  ════════════                                                               │
│  Physical 2D arrangement of patches. Used for visualization                 │
│  and understanding spatial relationships.                                   │
│                                                                             │
│      ┌───┬───┐                                                              │
│      │0,0│0,1│  Each cell is an (N, N, 1) patch                            │
│      ├───┼───┤                                                              │
│      │1,0│1,1│  gridsize=2 example                                         │
│      └───┴───┘                                                              │
│                                                                             │
│  CHANNEL FORMAT: (B, N, N, C) where C = gridsize²                          │
│  ══════════════                                                             │
│  Neural network compatible (HWC layout). Primary format for                │
│  model input/output and training.                                          │
│                                                                             │
│      [patch_0, patch_1, patch_2, patch_3] as channels                      │
│      Channel index c → grid position: row=c//gridsize, col=c%gridsize      │
│                                                                             │
│  FLAT FORMAT: (B*C, N, N, 1)                                               │
│  ═══════════                                                                │
│  Individual patches as batch elements. Required for physics                │
│  simulation (illuminate_and_diffract) which operates per-patch.            │
│                                                                             │
│      B groups × C patches/group = B*C individual patches                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2. Conversion Functions

```
_togrid()           : Flat → Grid
_fromgrid()         : Grid → Flat
_grid_to_channel()  : Grid → Channel
_channel_to_flat()  : Channel → Flat
_flat_to_channel()  : Flat → Channel
```

**When to Use Each Format:**

| Format | Use Case |
|--------|----------|
| Channel | Model input/output, training, inference |
| Flat | Physics simulation (`illuminate_and_diffract`) |
| Grid | Visualization, debugging, understanding spatial layout |

---

## 4. Backend Dispatch Pattern

### 4.1. Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Backend Dispatcher Pattern                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. Client calls run_cdi_example_with_backend(train_data, test_data, config)│
│                              │                                              │
│  2. Dispatcher calls ────────┼──► update_legacy_dict(params.cfg, config)   │
│     CONFIG-001 gate          │                                              │
│                              │                                              │
│  3. Dispatcher inspects ─────┼──► config.backend field                     │
│                              │                                              │
│  4. Routes to backend: ──────┴──────────────────────┐                      │
│                                                      │                      │
│     ┌────────────────────────┬───────────────────────┤                      │
│     │                        │                       │                      │
│     ▼                        ▼                       │                      │
│  ┌──────────────┐    ┌──────────────┐               │                      │
│  │ TensorFlow   │    │  PyTorch     │               │                      │
│  │ components   │    │  components  │               │                      │
│  └──────────────┘    └──────────────┘               │                      │
│                                                      │                      │
│  5. Results returned with 'backend' field injected ◄┘                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2. Design Rationale

- **CONFIG-001 compliance:** The dispatcher ALWAYS calls `update_legacy_dict()` before backend dispatch, ensuring consistent state regardless of which backend is used.
- **Torch-optional design:** TensorFlow path has zero PyTorch dependencies. PyTorch uses guarded imports with actionable error messages.
- **API parity:** Both backends implement identical function signatures for interchangeability.

---

## 5. Coordinate System Conventions

### 5.1. Coordinate Axis Order

```
coords_nn shape: (B, 1, 2, C)
                      │
                      └── Axis 2 contains [x, y] (NOT [row, col])
```

**Critical:** The coordinate axis order is `[x, y]`, where:
- x = horizontal position (column in image terms)
- y = vertical position (row in image terms)

### 5.2. Offset Sign Convention

```python
local_offset_sign = -1  # Module constant in raw_data.py

# Relative coords calculated as:
coords_relative = -1 * (coords_nn - coords_offsets)

# Translation uses negated offset:
translated = translate(patch, -offset)
```

**Why the Sign Inversion?** The convention arose from the physical interpretation: offsets represent "how far to shift the image to center the patch," which is the negative of "where the patch is located."

### 5.3. Channel-to-Grid Mapping

```
For gridsize=2 (C=4 channels):

Channel 0 → Grid position (0, 0) → row=0//2=0, col=0%2=0
Channel 1 → Grid position (0, 1) → row=1//2=0, col=1%2=1
Channel 2 → Grid position (1, 0) → row=2//2=1, col=2%2=0
Channel 3 → Grid position (1, 1) → row=3//2=1, col=3%2=1

Formula: row = c // gridsize, col = c % gridsize
```

---

## 6. Critical Design Patterns

### 6.1. Lazy Tensor Conversion (PINN-CHUNKED-001)

```python
class PtychoDataContainer:
    def __init__(self, X, ...):
        # Store as NumPy internally
        self._X_np = X.numpy() if tf.is_tensor(X) else X
        self._tensor_cache = {}

    @property
    def X(self):
        # Convert to TensorFlow only on first access
        if 'X' not in self._tensor_cache:
            self._tensor_cache['X'] = tf.convert_to_tensor(self._X_np)
        return self._tensor_cache['X']
```

**Why:** Prevents GPU OOM for large datasets (20k+ images). Tensors are only created when accessed.

### 6.2. Gridsize Synchronization (MODULE-SINGLETON-001)

```python
class DiffractionToObjectAdapter(tf.keras.Model):
    def predict(self, *args, **kwargs):
        # Sync gridsize from input shape before prediction
        channels = inputs[0].shape[-1]
        gridsize = int(round(math.sqrt(channels)))
        params.cfg['gridsize'] = gridsize  # Critical!
        return self._model.predict(*args, **kwargs)
```

**Why:** Legacy model code captures `gridsize` at import time via Lambda layers. The adapter ensures runtime synchronization.

### 6.3. Chunked Stitching for Memory Efficiency

```python
@tf.function
def shift_and_sum(obj_tensor, global_offsets, M):
    chunk_size = 1024  # Process in chunks to avoid OOM
    for start in range(0, B * C, chunk_size):
        chunk = obj_tensor_flat[start:start+chunk_size]
        # ... translate and accumulate
```

**Why:** Full batched translation would require O(B × C × H × W) memory. Chunking reduces peak memory while maintaining vectorization benefits.

---

## 7. Performance Characteristics

### 7.1. Complexity Analysis

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| `generate_grouped_data` | O(nsamples × K × log M) | O(nsamples × C) |
| `shift_and_sum` | O(B × M²) | O(padded_size²) |
| `loader.load` | O(B × N² × C) | O(B × N² × C) |

### 7.2. Measured Performance

| Operation | Speedup vs Iterative | Notes |
|-----------|---------------------|-------|
| `shift_and_sum` | 20-44x | Batched with chunking |
| `generate_grouped_data` | 10-100x | vs cache-based approach |
| Lazy tensor loading | N/A | Prevents OOM for B > 10k |

---

## 8. Error Handling Strategy

### 8.1. Fail-Fast with Context

```python
# Good: Include configuration state in error
raise ValueError(
    f"Expected shape (*,*,*,4) for gridsize=2, got {shape}. "
    f"Check params.cfg['gridsize']={params.cfg.get('gridsize')}"
)

# Bad: Generic error
raise ValueError("Shape mismatch")
```

### 8.2. Precondition Validation

Critical preconditions are validated at function entry:

```python
def generate_grouped_data(self, N, K, nsamples, gridsize=None):
    if gridsize is None:
        gridsize = params.get('gridsize', 1)

    if K < gridsize ** 2:
        raise ValueError(
            f"K={K} must be >= gridsize²={gridsize**2}"
        )
```

---

## 9. References

- `specs/spec-inference-pipeline.md` — Normative contracts
- `docs/DEVELOPER_GUIDE.md` §12 — Practical code examples
- `docs/DATA_GENERATION_GUIDE.md` §4 — Alternative data flows
- `docs/findings.md` — CONFIG-001, MODULE-SINGLETON-001 policies
- `ptycho/tf_helper.py` lines 1-116 — Tensor format documentation
