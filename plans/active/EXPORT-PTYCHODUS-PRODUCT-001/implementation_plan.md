## EXPORT-PTYCHODUS-PRODUCT-001 — Implementation Plan

### 1. Overview

Add TensorFlow-side save/load for the Ptychodus product format and convert the sample Run1084 NPZ dataset into an HDF5 product. This work does not modify core physics or touch fly64 datasets.

Key constraints (per request):
- Format: HDF5 only (no NPZ product writer initially)
- Coordinates: NPZ coords are relative to object pixels; convert to meters using object pixel size
- Pixel sizes and some physics fields may be unavailable; use dummy defaults when absent
- Derive object center from metadata if available; otherwise default to (0.0, 0.0) meters
 - Loss history: emit empty loss datasets per spec (write empty `loss_values` and optionally empty `loss_epochs` when no history is available)
- Convert Run1084 only; do not convert fly64 datasets

### 2. Authoritative References

- Product data contract: `specs/data_contracts.md`
- Ptychodus HDF5 I/O reference: `ptychodus/src/ptychodus/plugins/h5_product_file.py`
- TF NPZ contracts and containers: `ptycho/raw_data.py`, `ptycho/loader.py`
- NPZ metadata manager: `ptycho/metadata.py`

### 3. Deliverables

- Code
  - `ptycho/io/ptychodus_product_io.py`
    - `export_product_from_rawdata(raw: RawData, out_path: Path, meta: ExportMeta | None) -> None`
    - `import_product_to_rawdata(in_path: Path) -> RawData`
    - `ExportMeta` dataclass: HDF5 root attributes and pixel geometry
  - `scripts/tools/convert_to_ptychodus_product.py`
    - CLI to convert an input NPZ (Run1084) to HDF5 product; no losses

- Tests
  - `tests/io/test_ptychodus_product_io.py` (unit + smoke)
  - Evidence captured under `plans/active/EXPORT-PTYCHODUS-PRODUCT-001/reports/<timestamp>/`

- Docs
  - Short snippet in `docs/DATA_MANAGEMENT_GUIDE.md` (export usage example)

### 4. Data Mapping (RawData → Product)

- Positions (length = number of points)
  - `probe_position_indexes`: `RawData.scan_index` if present; otherwise zeros
  - `probe_position_x_m`, `probe_position_y_m`: pixel → meter using object pixel size from meta
    - `x_m = xcoords_px * object_pixel_width_m`, `y_m = ycoords_px * object_pixel_height_m`

- Probe
  - `probe` dataset: `RawData.probeGuess` (complex64 preferred; downcast complex128)
  - Attributes: `pixel_width_m`, `pixel_height_m` from `ExportMeta`
  - `opr_weights`: not emitted (not available from RawData)

- Object
  - `object` dataset: `RawData.objectGuess` if present (complex64)
  - Attributes: `center_x_m`, `center_y_m`: from metadata if present, else `0.0`
  - Attributes: `pixel_width_m`, `pixel_height_m` from `ExportMeta`
  - `object_layer_spacing_m`: empty (single layer)

- Metadata (root attributes)
  - `name`, `comments`, `detector_object_distance_m`, `probe_energy_eV`, `exposure_time_s`, `probe_photon_count`, `mass_attenuation_m2_kg`, `tomography_angle_deg`
  - Use `ExportMeta` and fallback dummy values when not provided

 - Losses
  - Emit empty datasets when losses are unavailable
    - `loss_values`: write an empty float64 array (required by spec)
    - `loss_epochs`: write an empty int array (optional; recommended for compatibility)

### 5. Data Mapping (Product → RawData)

- Read HDF5 using `h5py` in accordance with `specs/data_contracts.md`.
- Produce `RawData` with:
  - `xcoords`, `ycoords` (pixels) = meters / object pixel size from object attributes
  - `probeGuess` from `probe`
  - `objectGuess` from `object` if present
  - `scan_index` from `probe_position_indexes`
  - `diff3d=None`, `Y=None`, `norm_Y_I=None` (product files do not contain diffraction)

### 6. Defaults & Heuristics

- Pixel sizes (used when metadata absent):
  - `object_pixel_width_m = object_pixel_height_m = 5.0e-8` (50 nm)
  - `probe_pixel_width_m = probe_pixel_height_m = 1.25e-7` (125 nm)
  - Annotate defaults in root `comments` (e.g., "estimated pixel sizes")

- Coordinates: treat NPZ coords as pixels relative to object pixels (per request)

- Object center: derive from NPZ metadata if available via `MetadataManager`; else `(0.0, 0.0)`

- Physics fields: when not available, use zeros (e.g., `detector_object_distance_m = 0.0`, `probe_photon_count = 0.0`, `mass_attenuation_m2_kg = 0.0`, `tomography_angle_deg = 0.0`)

### 7. Work Breakdown

- Phase A — Design & Setup
  - Finalize `ExportMeta` schema and default policy
  - Create initiative scaffold and test strategy (see `test_strategy.md`)

- Phase B — Exporter (RawData → HDF5)
  - Implement `export_product_from_rawdata()`
  - Conversions: pixels→meters; dtype downcast; attach attributes
  - Unit tests: synthetic RawData → HDF5 → validate via ptychodus reader

- Phase C — Importer (HDF5 → RawData)
  - Implement `import_product_to_rawdata()`
  - Conversions: meters→pixels using object pixel sizes
  - Unit tests: export→import round-trip sanity

- Phase D — Conversion CLI (Run1084 only)
  - Implement `scripts/tools/convert_to_ptychodus_product.py`
  - Input: `datasets/Run1084_recon3_postPC_shrunk_3.npz`
  - Output: `outputs/ptychodus_products/run1084_product.h5`
  - Metadata: attempt to derive from NPZ metadata, else apply defaults
  - Exclude fly64 datasets explicitly

- Phase E — Evidence & Docs
  - pytest logs → `plans/active/EXPORT-PTYCHODUS-PRODUCT-001/reports/<timestamp>/pytest.log`
  - `summary.md` with pass/skip counts and artifact paths
  - Add a brief how-to to `docs/DATA_MANAGEMENT_GUIDE.md`

### 8. Acceptance Criteria

- HDF5 product conforms to `specs/data_contracts.md`
- File loads with `H5ProductFileIO.read()` without errors; fields populated as mapped
- Importer yields a `RawData` with expected coords and guesses; diffraction is None
- Run1084 conversion succeeds and passes a read smoke test
- Tests PASS; evidence stored under the initiative reports directory

### 9. Risks & Mitigations

- Unknown pixel sizes → Use documented defaults; annotate in comments; allow CLI overrides later
- Coord unit mismatch → Fixed to pixels→meters (per request); document in CLI help
- Missing objectGuess → Current scope assumes present for Run1084; add allow-missing flag in a follow-up if needed
- HDF5 OPR weights placement → Exporter does not emit root `opr_weights`; aligns with spec

### 10. Out of Scope

- Writing NPZ product format variant
- Embedding diffraction or loss history in product files
- Updating upstream ptychodus writers/readers

### 11. Open Question Resolutions (from user)

- Pixel sizes not available → Use dummy defaults (section 6)
- Coord units → Relative to object pixels (convert using object pixel size)
- Dataset scope → Convert Run1084 only; not fly64
- Object center → Derive from metadata if possible; else (0.0, 0.0)
- Losses → No (skip)
- Default format → HDF5

### 12. Next Actions

- Author `test_strategy.md` for this initiative
- Implement exporter/importer skeletons with unit tests (RED)
- Prepare CLI scaffold for Run1084 conversion
