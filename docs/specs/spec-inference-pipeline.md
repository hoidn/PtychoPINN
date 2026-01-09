# Inference Pipeline Specification

**Status:** Normative
**Version:** 1.0.0
**Last Updated:** 2025-01-09

This specification defines the contracts, invariants, and error conditions for the PtychoPINN inference pipeline. All implementations MUST conform to these contracts.

---

## 1. Global State Contracts

### 1.1. Configuration Singleton

```
singleton params.cfg : Dict[str, Any]
```

The global configuration dictionary accessed by all pipeline components.

**Required Keys for Inference:**

| Key | Type | Description | Invariant |
|-----|------|-------------|-----------|
| `N` | int32 | Patch size in pixels | `N > 0 and N % 2 == 0` |
| `gridsize` | int32 | sqrt(channels per group) | `gridsize >= 1` |
| `intensity_scale` | float32 | Diffraction scaling factor | `intensity_scale > 0` |
| `offset` | int32 | Patch stride | `0 < offset <= N` |

**Optional Keys:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `nphotons` | float64 | None | Photon count for physics simulation |
| `probe_mask` | Tensor | None | Probe support mask |
| `default_probe_scale` | float32 | 4.0 | Probe lowpass filter scale |

### 1.2. CONFIG-001: Legacy Dictionary Synchronization

```
procedure update_legacy_dict(
    target : inout Dict[str, Any],
    config : TrainingConfig | InferenceConfig
) -> void
```

**Preconditions:**
- `config` is a valid dataclass instance

**Postconditions:**
- `target['N'] == config.model.N`
- `target['gridsize'] == config.model.gridsize`
- All ModelConfig fields are mapped to target

**Side Effects:**
- Mutates global `params.cfg`

**CRITICAL:** This procedure MUST be called before any operation that:
- Calls `RawData.generate_grouped_data()`
- Loads a model via `ModelManager`
- Performs inference with `model.predict()`

---

## 2. Configuration Dataclasses

### 2.1. ModelConfig

```
dataclass ModelConfig {
    N           : int32 = 64
    gridsize    : int32 = 1
    model_type  : Literal['pinn', 'supervised'] = 'pinn'
}
```

**Invariants:**
- `N > 0 and N % 2 == 0`
- `gridsize >= 1`

**Derived Properties:**
- `channels = gridsize * gridsize`

### 2.2. InferenceConfig

```
dataclass InferenceConfig {
    model           : ModelConfig
    model_path      : Path
    test_data_file  : Path
    n_images        : Optional[int32] = None
    n_subsample     : Optional[int32] = None
    subsample_seed  : Optional[int32] = None
    backend         : Literal['tensorflow', 'pytorch'] = 'tensorflow'
    output_dir      : Path = Path('inference_outputs')
}
```

**Preconditions:**
- `model_path.exists() and model_path.is_dir()`
- `test_data_file.exists()`

**Invariants:**
- `backend in {'tensorflow', 'pytorch'}`

---

## 3. Data Container Contracts

### 3.1. RawData

Core container for raw experimental/simulated data.

```
class RawData {
    xcoords       : ndarray[float64, (M,)]
    ycoords       : ndarray[float64, (M,)]
    xcoords_start : ndarray[float64, (M,)]
    ycoords_start : ndarray[float64, (M,)]
    diff3d        : ndarray[float32, (M, N, N)]
    probeGuess    : ndarray[complex64, (N, N)]
    scan_index    : ndarray[int64, (M,)]
    objectGuess   : Optional[ndarray[complex64, (H, W)]]
    Y             : Optional[ndarray[complex64, (M, N, N, 1)]]
}
```

**Invariants:**
- `len(xcoords) == len(ycoords) == len(diff3d) == M`
- `diff3d.shape[1] == diff3d.shape[2] == N`
- `probeGuess.shape == (N, N)`

#### 3.1.1. RawData.from_file()

```
static function from_file(
    path            : str | Path,
    validate_config : bool = False
) -> RawData
```

**Preconditions:**
- `Path(path).exists()`
- `Path(path).suffix == '.npz'`

**Postconditions:**
- `result.diff3d is not None`
- `result.probeGuess is not None`

#### 3.1.2. RawData.generate_grouped_data()

```
function generate_grouped_data(
    N                   : int32,
    K                   : int32 = 4,
    nsamples            : int32 = 1,
    dataset_path        : Optional[str] = None,
    seed                : Optional[int32] = None,
    sequential_sampling : bool = False,
    gridsize            : Optional[int32] = None,
    enable_oversampling : bool = False,
    neighbor_pool_size  : Optional[int32] = None
) -> GroupedDataDict
```

**Preconditions:**
- `N > 0`
- `K >= gridsize²` (when gridsize provided)
- `len(self.xcoords) >= gridsize²`
- `enable_oversampling implies neighbor_pool_size >= gridsize²`

**Postconditions:**
- `result['diffraction'].shape == (nsamples, N, N, C)` where `C = gridsize²`
- `result['coords_offsets'].shape == (nsamples, 1, 2, 1)`
- `result['coords_relative'].shape == (nsamples, 1, 2, C)`
- `result['nn_indices'].shape == (nsamples, C)`
- `result['X_full'].shape == result['diffraction'].shape`

**Side Effects:**
- Reads `params.cfg['gridsize']` if gridsize argument is None

### 3.2. GroupedDataDict

Return type of `generate_grouped_data()`.

```
typedef GroupedDataDict = Dict {
    // Required keys
    'diffraction'          : ndarray[float32, (B, N, N, C)]
    'X_full'               : ndarray[float32, (B, N, N, C)]
    'coords_offsets'       : ndarray[float64, (B, 1, 2, 1)]
    'coords_relative'      : ndarray[float64, (B, 1, 2, C)]
    'coords_start_offsets' : ndarray[float64, (B, 1, 2, 1)]
    'coords_start_relative': ndarray[float64, (B, 1, 2, C)]
    'coords_nn'            : ndarray[float64, (B, 1, 2, C)]
    'coords_start_nn'      : ndarray[float64, (B, 1, 2, C)]
    'nn_indices'           : ndarray[int32, (B, C)]
    'objectGuess'          : Optional[ndarray[complex64, (H, W)]]

    // Conditional keys
    'Y'                    : Optional[ndarray[complex64, (B, N, N, C)]]
    'sample_indices'       : Optional[ndarray[int32, (B,)]]
}
```

**Invariants:**
- `C == gridsize²`
- Coordinate axis order is `[x, y]` (NOT `[row, col]`)

### 3.3. PtychoDataContainer

TensorFlow-ready container with lazy tensor conversion.

```
class PtychoDataContainer {
    // Lazy-loaded TensorFlow tensors
    property X              : Tensor[float32, (B, N, N, C)]
    property Y_I            : Tensor[float32, (B, N, N, C)]
    property Y_phi          : Tensor[float32, (B, N, N, C)]
    property Y              : Tensor[complex64, (B, N, N, C)]
    property coords_nominal : Tensor[float32, (B, 1, 2, C)]
    property coords_true    : Tensor[float32, (B, 1, 2, C)]
    property probe          : Tensor[complex64, (N, N)]

    // NumPy attributes
    global_offsets : ndarray[float64, (B, 1, 2, C)]
    local_offsets  : ndarray[float64, (B, 1, 2, C)]
    nn_indices     : ndarray[int32, (B, C)]
    norm_Y_I       : float32
}
```

**Invariants:**
- `X.shape[-1] == Y_I.shape[-1] == C`
- `coords_nominal.shape == (B, 1, 2, C)`
- `global_offsets.shape == local_offsets.shape`

---

## 4. Model Loading Contracts

### 4.1. load_inference_bundle_with_backend()

```
function load_inference_bundle_with_backend(
    bundle_dir : str | Path,
    config     : InferenceConfig,
    model_name : str = 'diffraction_to_obj'
) -> Tuple[Model, Dict[str, Any]]
```

**Preconditions:**
- `Path(bundle_dir).exists()`
- `(Path(bundle_dir) / 'wts.h5.zip').exists()`
- `config.backend in {'tensorflow', 'pytorch'}`

**Postconditions:**
- `result[0]` is callable model with `.predict()` method
- `result[1]` contains restored `params.cfg` state
- `params.cfg` is updated from saved state (CONFIG-001 satisfied)

**Raises:**
- `FileNotFoundError` if `wts.h5.zip` missing
- `KeyError` if `model_name` not in archive
- `RuntimeError` if `backend='pytorch'` and torch unavailable

### 4.2. DiffractionToObjectAdapter

Wrapper that synchronizes `params.cfg['gridsize']` with input tensor channels.

```
class DiffractionToObjectAdapter extends tf.keras.Model {
    _model : tf.keras.Model

    function predict(
        inputs : Tensor | List[Tensor],
        **kwargs
    ) -> Tensor[complex64, (B, N, N, C)]
}
```

**Preconditions:**
- `inputs[0].shape[-1] == C` (channels)

**Side Effects:**
- Sets `params.cfg['gridsize'] = sqrt(C)`

**Postconditions:**
- `params.cfg['gridsize']² == inputs[0].shape[-1]`

---

## 5. Data Loading Contracts

### 5.1. load_data()

```
function load_data(
    file_path      : str,
    n_images       : Optional[int32] = None,
    n_subsample    : Optional[int32] = None,
    flip_x         : bool = False,
    flip_y         : bool = False,
    swap_xy        : bool = False,
    n_samples      : int32 = 1,
    coord_scale    : float64 = 1.0,
    subsample_seed : Optional[int32] = None
) -> RawData
```

**Preconditions:**
- `Path(file_path).exists()`
- `n_subsample is None or n_subsample > 0`
- `n_images is None or n_images > 0`

**Postconditions:**
- `len(result.xcoords) <= original_dataset_size`
- If `n_subsample` provided: `len(result.xcoords) == min(n_subsample, original)`
- If `flip_x`: `result.xcoords == -original.xcoords`
- If `swap_xy`: x and y coordinates are swapped

**Side Effects:**
- Persists indices to `tmp/subsample_seed{N}_indices.txt` if seed provided

### 5.2. loader.load()

```
function load(
    cb           : Callable[[], GroupedDataDict],
    probeGuess   : ndarray[complex64, (N, N)],
    which        : Optional[Literal['train', 'test']] = None,
    create_split : bool = False
) -> PtychoDataContainer
```

**Preconditions:**
- `cb()` returns valid `GroupedDataDict`
- `probeGuess.shape == (N, N)`
- `create_split implies which in {'train', 'test'}`

**Postconditions:**
- `result.X.shape[-1] == result.Y_I.shape[-1]` (channel match)
- `result.probe.shape == probeGuess.shape`

**Raises:**
- `ValueError` if X and Y channel counts mismatch

---

## 6. Inference Contracts

### 6.1. reconstruct_image()

```
function reconstruct_image(
    test_data          : PtychoDataContainer,
    diffraction_to_obj : Optional[Model] = None
) -> Tuple[
    ndarray[complex64, (B, N, N, C)],
    ndarray[float64, (B, 1, 2, C)]
]
```

**Preconditions:**
- `test_data.X is not None`
- `test_data.local_offsets is not None`
- `params.cfg['intensity_scale']` is set

**Postconditions:**
- `result[0].shape == test_data.X.shape` (but complex64)
- `result[1].shape == test_data.global_offsets.shape`

**Side Effects:**
- Reads `params.cfg['intensity_scale']`

**Implementation:**
```python
scaled_X = test_data.X * params.cfg['intensity_scale']
obj_tensor_full = model.predict([scaled_X, test_data.local_offsets])
return (obj_tensor_full, test_data.global_offsets)
```

---

## 7. Stitching/Reassembly Contracts

### 7.1. reassemble_position()

Primary reassembly function with overlap normalization.

```
function reassemble_position(
    obj_tensor     : ndarray[complex64, (B, N, N, C)],
    global_offsets : ndarray[float64, (B, 1, 2, C)],
    M              : int32 = 10
) -> Tensor[complex64, (H, W, 1)]
```

**Preconditions:**
- `obj_tensor.dtype == complex64`
- `global_offsets.dtype == float64`
- `obj_tensor.ndim == 4`
- `global_offsets.ndim == 4`
- `M <= N and M > 0`

**Postconditions:**
- Result is normalized (divided by overlap count)
- `H == W == M + 2 * dynamic_pad` where `dynamic_pad = ceil(max(|adjusted_offsets|))`

**Implementation:**
```python
ones = tf.ones_like(obj_tensor)
numerator = shift_and_sum(obj_tensor, global_offsets, M)
denominator = shift_and_sum(ones, global_offsets, M) + 1e-9
return numerator / denominator
```

### 7.2. shift_and_sum()

Core stitching primitive with batched translation.

```
@tf.function(reduce_retracing=True)
function shift_and_sum(
    obj_tensor     : ndarray[complex64, (B, N, N, C)],
    global_offsets : ndarray[float64, (B, 1, 2, C)],
    M              : int32 = 10
) -> Tensor[complex64, (H, W, 1)]
```

**Preconditions:**
- `obj_tensor.dtype == complex64`
- `global_offsets.dtype == float64`
- `M > 0 and M <= N`

**Postconditions:**
- `result.shape == (padded_size, padded_size, 1)`

**Implementation Steps:**
1. Crop central M×M: `obj[:, N//2-M//2 : N//2+M//2, ...]`
2. Center offsets: `adjusted = offsets - mean(offsets)`
3. Compute canvas: `padded_size = M + 2*ceil(max(|adjusted|))`
4. Stream chunks of 1024 patches with translation and accumulation
5. Return accumulated canvas

### 7.3. translate()

Image translation with bilinear interpolation.

```
function translate(
    images       : Tensor[T, (B, H, W, C)],
    translations : Tensor[float32, (B, 2)]
) -> Tensor[T, (B, H, W, C)]
```

**Preconditions:**
- `translations.shape == (B, 2)`
- `images.shape[0] == translations.shape[0]` or broadcast

**Postconditions:**
- `result.shape == images.shape`
- Out-of-bounds pixels are zero

**Supported Types:**
- `T in {float32, complex64}`
- For complex64: applies to real and imag separately via `@complexify_function`

---

## 8. Tensor Format Conversion Contracts

### 8.1. Format Definitions

```
GRID FORMAT:    (B, gridsize, gridsize, N, N, 1)
                Physical 2D arrangement of patches

CHANNEL FORMAT: (B, N, N, C) where C = gridsize²
                Neural network compatible (HWC layout)

FLAT FORMAT:    (B*C, N, N, 1)
                Individual patches as batch elements
```

### 8.2. Conversion Functions

```
function _channel_to_flat(
    img : Tensor[T, (B, N, N, C)]
) -> Tensor[T, (B*C, N, N, 1)]

function _flat_to_channel(
    img      : Tensor[T, (B*C, N, N, 1)],
    N        : Optional[int32] = None,
    gridsize : Optional[int32] = None
) -> Tensor[T, (B, N, N, C)]
```

**Postconditions:**
- `_channel_to_flat(img).shape[0] == img.shape[0] * img.shape[-1]`

**Side Effects:**
- `_flat_to_channel` reads `params.cfg` if parameters not provided

---

## 9. Coordinate System Contracts

### 9.1. get_relative_coords()

```
function get_relative_coords(
    coords_nn : ndarray[float64, (B, 1, 2, C)]
) -> Tuple[
    ndarray[float64, (B, 1, 2, 1)],
    ndarray[float64, (B, 1, 2, C)]
]
```

**Preconditions:**
- `coords_nn.ndim == 4`
- `coords_nn.shape[1:3] == (1, 2)`

**Postconditions:**
- `result[0] == mean(coords_nn, axis=3)[..., None]`
- `result[1] == -1 * (coords_nn - result[0])`

**Module Constant:**
- `local_offset_sign = -1`

---

## 10. Error Taxonomy

```
enum InferenceError {
    CONFIG_001_VIOLATION      // params.cfg not initialized before use
    MODULE_SINGLETON_001      // gridsize mismatch in model singleton
    SHAPE_MISMATCH           // Tensor dimension inconsistency
    CHANNEL_MISMATCH         // X and Y channel counts differ
    BACKEND_UNAVAILABLE      // PyTorch not installed
    MODEL_NOT_FOUND          // wts.h5.zip or model key missing
    DATA_FILE_NOT_FOUND      // NPZ file doesn't exist
    INSUFFICIENT_DATA        // nsamples > available points without oversampling
    INVALID_GRIDSIZE         // gridsize < 1 or non-integer sqrt(C)
}
```

**Error Conditions:**

| Error | Trigger Condition |
|-------|-------------------|
| `CONFIG_001_VIOLATION` | `generate_grouped_data()` called before `update_legacy_dict()` |
| `MODULE_SINGLETON_001` | `model.autoencoder` accessed after changing gridsize |
| `SHAPE_MISMATCH` | `X.shape[-1] != Y.shape[-1]` in `loader.load()` |
| `CHANNEL_MISMATCH` | `coords.shape[-1] != X.shape[-1]` |

---

## 11. Performance Contracts

### 11.1. generate_grouped_data

```
performance_contract generate_grouped_data {
    time_complexity  : O(nsamples * K * log(M))
    space_complexity : O(nsamples * C)
    note            : "10-100x faster than cache-based approaches"
}
```

### 11.2. shift_and_sum

```
performance_contract shift_and_sum {
    time_complexity  : O(B * M²)
    space_complexity : O(padded_size²)
    speedup         : "20-44x over iterative implementation"
    accuracy        : "0.00e+00 numerical error vs reference"
}
```

### 11.3. PtychoDataContainer

```
performance_contract PtychoDataContainer {
    lazy_loading   : true
    gpu_memory     : "Only allocated on property access"
    recommendation : "Use as_tf_dataset(batch_size) for B > 10000"
}
```

---

## 12. References

- `docs/architecture_inference.md` — Data flow diagrams and design rationale
- `docs/DEVELOPER_GUIDE.md` §12 — Practical usage patterns
- `docs/DATA_GENERATION_GUIDE.md` §4 — Alternative data creation flows
- `docs/specs/data_contracts.md` — NPZ format specification
- `docs/findings.md` — Known issues and policies (CONFIG-001, MODULE-SINGLETON-001)
