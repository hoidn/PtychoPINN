## Ptychodus Data Contracts

### 1. Scope

This document defines the normative on‑disk data format for Ptychodus product files written and read by `H5ProductFileIO` (file filter: "Ptychodus Product Files (*.h5 *.hdf5)"). The format is HDF5 with a fixed set of root attributes and datasets capturing metadata, scan geometry, probe(s), object, and training loss history.

The specification is derived from and authoritative over the reference I/O implementation (`H5ProductFileIO`). Readers and writers must follow the rules below for field names, units, shapes, and types. Implementations must ignore unknown attributes/datasets and must not rely on unspecified fields.

### 2. File Identification

- Container: HDF5 (`.h5` or `.hdf5`).
- Root object stores global metadata as attributes; data arrays are root datasets named below.
- Optional but recommended root attribute: `format_id = 'ptychodus.product.hdf5'` and `format_version = '1.0'`. Readers must not require these for backward compatibility.

### 3. Global Metadata (root attributes)

Required attributes
- `name` (string): Human‑readable product name.
- `comments` (string): Free‑form notes.
- `detector_object_distance_m` (float64): Distance from detector to object in meters.
- `probe_energy_eV` (float64): Probe photon energy in electron volts.
- `exposure_time_s` (float64): Exposure time in seconds.

Optional attributes
- `probe_photon_count` (float64): Expected photon count per exposure.
- `mass_attenuation_m2_kg` (float64): Mass attenuation coefficient in m²/kg.
- `tomography_angle_deg` (float64): Rotation angle in degrees for tomographic acquisition (default 0 if absent).

Encoding
- Strings are UTF‑8 variable‑length. Numerics are IEEE‑754; writers should use `float64` for real values.

### 4. Scan Geometry (root datasets)

All arrays below have length `N_scan` and must be the same length.

- `probe_position_indexes` (int32 or int64) [N_scan]
  - Zero‑based index selecting a probe entry (see §5.1) for each scan point.
  - Values must be in `[0, K-1]`, where `K = len(probes)`.

- `probe_position_x_m` (float64) [N_scan]
  - X positions in meters (object plane/world frame).

- `probe_position_y_m` (float64) [N_scan]
  - Y positions in meters (object plane/world frame).

Coordinate conventions
- Positions are in the same world coordinate system as the object center (§6). The mapping from world to object pixel indices follows `ObjectGeometry.map_scan_point_to_object_point()`.

### 5. Probe

Dataset
- `probe` (complex64 or complex128)
  - Shape options:
    - 2D: `[H, W]` (single coherent, single incoherent mode)
    - 3D: `[I, H, W]` (single coherent, `I` incoherent modes)
    - 4D: `[C, I, H, W]` (`C` coherent modes, `I` incoherent modes) — canonical form used internally by `ProbeSequence`.
  - Writers should prefer complex64 to reduce size; readers must accept complex64 or complex128.

Attributes (required)
- `pixel_width_m` (float64) > 0: Physical width of a pixel.
- `pixel_height_m` (float64) > 0: Physical height of a pixel.

Attributes (optional)
- `opr_weights` (float32/float64) [K, C]
  - Orthonormal probe reconstruction (OPR) weights; each row is a length‑`C` weight vector to combine `C` coherent modes into a single effective mode for that probe entry.
  - `K` defines the number of probe entries addressable by `probe_position_indexes` (see §4). If absent, `len(probes) == 1` and no per‑scan OPR combining is applied.
  - Each row should be non‑negative and L1‑normalized (recommended, not enforced).

Note: The `opr_weights` location is normative as a probe dataset attribute. Implementations must not store it as a separate root dataset.

### 6. Object

Dataset
- `object` (complex64 or complex128)
  - Shape options:
    - 2D: `[H, W]` for a single layer (reader promotes to `[1, H, W]`).
    - 3D: `[L, H, W]` for `L` layers (canonical form used internally by `Object`).
  - Writers should prefer complex64; readers must accept complex64 or complex128.

Attributes (required)
- `center_x_m` (float64): World X coordinate of the object center.
- `center_y_m` (float64): World Y coordinate of the object center.
- `pixel_width_m` (float64) > 0: Object pixel width in meters.
- `pixel_height_m` (float64) > 0: Object pixel height in meters.

Auxiliary dataset (required)
- `object_layer_spacing_m` (float64)
  - Shape: `[L-1]` for `L` object layers; empty (`[0]`) if `L == 1`.
  - Each value is the axial spacing (meters) between adjacent layers along the beam direction.

Coordinate conventions
- Array indexing is row‑major: axis order `[layer, y, x]` where `y` is rows (height), `x` is columns (width).
- The world position of pixel `(y, x)` is computed via `ObjectGeometry` using the object center and pixel sizes; array origin is not the world origin.

### 7. Loss History

Datasets
- `loss_values` (float64) [E] (required)
  - Scalar loss per epoch; monotonic decreasing is not required.

- `loss_epochs` (int32 or int64) [E] (optional)
  - Epoch indices corresponding to `loss_values`. If absent, assume `0..E-1`.

Backward compatibility
- Readers must accept `costs` (float array) as an alias for `loss_values` when `loss_values` is absent. Writers must emit `loss_values` and are encouraged to include `loss_epochs`.

### 8. Types, Units, and Validation Rules

- Complex arrays: complex64 preferred; complex128 allowed.
- Real arrays/attributes: float64 preferred; float32 allowed for attributes if necessary (readers must upcast as needed).
- Index arrays: int32 or int64. Use zero‑based indexing.
- Units: meters (`*_m`), seconds (`*_s`), electron volts (`*_eV`), degrees (`*_deg`), square meters per kilogram (`*_m2_kg`).
- Consistency checks (recommended):
  - `len(probe_position_indexes) == len(probe_position_x_m) == len(probe_position_y_m)`.
  - If `opr_weights` present with shape `[K, C]`, then `0 ≤ probe_position_indexes[i] < K` for all `i` and the probe array has `C` coherent modes (i.e., `probe.shape[0] == C` in the 4D case, or implied `C == 1` for 2D/3D arrays).
  - `object_layer_spacing_m.shape[0] == (L - 1)` where `L` is the number of object layers after promoting 2D to 3D.

### 9. Example Layout

```
/
  @ name: "My Reconstruction"                  (string)
  @ comments: "Phase retrieval run ..."         (string)
  @ detector_object_distance_m: 0.750           (float64)
  @ probe_energy_eV: 8000.0                     (float64)
  @ probe_photon_count: 1.0e6                   (float64, optional)
  @ exposure_time_s: 0.1                        (float64)
  @ mass_attenuation_m2_kg: 0.0                 (float64, optional)
  @ tomography_angle_deg: 0.0                   (float64, optional)

  probe: complex64 [C, I, H, W]
    @ pixel_width_m: 1.25e-7                    (float64)
    @ pixel_height_m: 1.25e-7                   (float64)
    @ opr_weights: float64 [K, C]               (optional)

  object: complex64 [L, H, W]
    @ center_x_m: 0.0                           (float64)
    @ center_y_m: 0.0                           (float64)
    @ pixel_width_m: 5.0e-8                     (float64)
    @ pixel_height_m: 5.0e-8                    (float64)

  object_layer_spacing_m: float64 [L-1]

  probe_position_indexes: int32 [N_scan]
  probe_position_x_m: float64 [N_scan]
  probe_position_y_m: float64 [N_scan]

  loss_epochs: int32 [E]                        (optional)
  loss_values: float64 [E]
```

### 10. Compliance Notes for Implementers

- Writers must attach `opr_weights` as an attribute of the `probe` dataset (not as a root dataset) to match reader expectations.
- Writers should emit `loss_values`/`loss_epochs`; readers must accept `costs` in place of `loss_values` for backward compatibility.
- Writers should prefer complex64 for arrays and float64 for real values to balance size and precision.
- Readers should upcast real/complex types as needed for internal processing; do not assume exact storage dtypes.

### 11. Non‑Normative Python Snippets

Creating the minimal structure with `h5py`:

```python
import h5py, numpy as np

with h5py.File('product.h5', 'w') as f:
    # Root attributes
    f.attrs['name'] = 'Example'
    f.attrs['comments'] = 'Demo reconstruction'
    f.attrs['detector_object_distance_m'] = 0.75
    f.attrs['probe_energy_eV'] = 8000.0
    f.attrs['exposure_time_s'] = 0.1

    # Scan
    N = 10
    f.create_dataset('probe_position_indexes', data=np.zeros(N, dtype=np.int32))
    f.create_dataset('probe_position_x_m', data=np.linspace(-1e-6, 1e-6, N))
    f.create_dataset('probe_position_y_m', data=np.linspace(-1e-6, 1e-6, N))

    # Probe (single coherent & incoherent mode as 2D)
    H, W = 64, 64
    p = (np.random.randn(H, W) + 1j*np.random.randn(H, W)).astype(np.complex64)
    dset_probe = f.create_dataset('probe', data=p)
    dset_probe.attrs['pixel_width_m'] = 1.25e-7
    dset_probe.attrs['pixel_height_m'] = 1.25e-7

    # Object (single layer)
    obj = (np.random.randn(H, W) + 1j*np.random.randn(H, W)).astype(np.complex64)
    dset_obj = f.create_dataset('object', data=obj)
    dset_obj.attrs['center_x_m'] = 0.0
    dset_obj.attrs['center_y_m'] = 0.0
    dset_obj.attrs['pixel_width_m'] = 5.0e-8
    dset_obj.attrs['pixel_height_m'] = 5.0e-8
    f.create_dataset('object_layer_spacing_m', data=np.array([], dtype=np.float64))

    # Losses
    f.create_dataset('loss_values', data=np.array([1.0, 0.9, 0.85], dtype=np.float64))
    f.create_dataset('loss_epochs', data=np.array([0, 1, 2], dtype=np.int32))
```

### 12. Optional raw_data Bundle (Extension)

Some workflows may embed the source dataset alongside the product in a single file. This extension preserves full compatibility with the product contract by writing raw data under a namespaced group.

- **Group:** `/raw_data` (extension; optional)
- **Required (if present):**
  - `diffraction`: canonical NHW array, shape `[N, H, W]`, dtype float32/float64
    - Attributes:
      - `axis_canonical = 'NHW'`
      - `original_axis_order` (optional): original axis tag (`'NHW'|'HNW'|'HWN'`)
  - `xcoords`, `ycoords`: pixel coordinates (float), shape `[N]`
  - `scan_index`: int32/int64, shape `[N]`
- **Optional:**
  - `probeGuess`, `objectGuess`: may be HDF5 hard links to root datasets `/probe` and `/object`
  - `_metadata`: JSON string with source metadata (e.g., nphotons)

Writers should canonicalize `diffraction` to NHW (first axis length equals `len(xcoords)`); readers must not assume presence of `/raw_data`.
