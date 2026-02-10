# PtychoViT Interop Contract Specification

## 1. Scope

This specification defines the normative contract for PtychoPINN <-> PtychoViT interop in the
`pinn_ptychovit` model arm. It covers:

- NPZ -> paired HDF5 conversion semantics
- Runtime normalization configuration requirements
- Reconstruction assembly behavior required for parity with upstream PtychoViT

Unless explicitly overridden by an approved plan, implementations must follow this document.

## 2. Source Pin

Reference upstream implementation:

```yaml
source_repo: /home/ollie/Documents/ptycho-vit
source_commit: 2316b378006ef330e18af343d10dc8a7b821b0a8
source_paths:
  - data.py
  - training.py
  - utils/ptychi_utils.py
  - scripts/make_normalization_dict.py
validated_on: 2026-02-10
```

## 3. Paired HDF5 Contract

### 3.1 Required files

For each object name `<name>`:

- `<name>_dp.hdf5` containing dataset `dp`
- `<name>_para.hdf5` containing datasets `object`, `probe`, `probe_position_x_m`, `probe_position_y_m`

### 3.2 `dp` dataset

- Rank: 3
- Shape: `[N_scan, H, W]`
- Semantics: intensity (not amplitude)
- Mapping rule: `dp = diffraction_amplitude**2`

### 3.3 `object` dataset

- Rank: 3
- Shape: `[1, H_obj, W_obj]`
- Dtype: complex64
- Required attrs: `pixel_height_m`, `pixel_width_m`

NPZ source selection (normative priority):

1. `YY_full` (preferred; scan-consistent object geometry)
2. `YY_ground_truth` (fallback only when `YY_full` absent)

If selected object contains extra leading sample dimension, use the first sample deterministically.

### 3.4 `probe` dataset

- Rank: 4
- Shape: `(1, M, H, W)`
- Dtype: complex64
- Required attrs: `pixel_height_m`, `pixel_width_m`

Probe conversion must ensure loader compatibility with upstream padding logic.

### 3.5 Probe positions

- `probe_position_x_m`: shape `[N_scan]`, float
- `probe_position_y_m`: shape `[N_scan]`, float
- Length must equal `dp.shape[0]`

Position source (required):

- `coords_offsets` OR `coords_start_offsets`
- Local-only `coords_nominal` / `coords_true` are not valid as the primary source.

Frame semantics (required):

- Stored HDF5 positions must be in the centered frame expected by upstream `PtychographyDataset`.
- Upstream loader converts meters -> pixels, then adds object origin (`round(H/2)+0.5`, `round(W/2)+0.5`).
- If input offsets are absolute top-left-origin pixels, convert to centered frame before write.

## 4. Runtime Normalization Contract

Bridge runtime config must set both:

- `data.normalization_dict_path`
- `data.test_normalization`

to the same generated pickle path.

Normalization dict format:

- Python pickle containing `dict[str, float]`
- Keys: object names
- Values: `max(dp)` for each paired dataset

A run is invalid if stdout contains `Normalization file not found`.

## 5. Reconstruction Assembly Contract

### 5.1 Required behavior

Bridge inference must reconstruct object-space output using scan-position-aware stitching, equivalent to upstream logic:

- place predicted patches by probe position
- accumulate occupancy buffer
- divide accumulated object by occupancy (clipped minimum 1)

### 5.2 Non-compliant behavior

The following is not contract-compliant for PtychoViT object reconstruction:

- scan-wise mean aggregation (simple `mean` across scan predictions) without positional placement

This can produce flat/low-information reconstructions even when data contracts are otherwise valid.

## 6. Validation Checklist

An interop output is compliant only if all checks pass:

1. Paired file/key presence and required attrs
2. `dp` rank-3 intensity tensor
3. Probe rank-4 complex tensor
4. Non-degenerate, finite probe position vectors with scan-count parity
5. Position frame consistent with centered-coordinate expectation
6. Runtime normalization dict exists and is referenced by both config keys
7. No normalization fallback warning in runtime logs
8. Reconstruction path uses position-aware stitching (not scan-wise mean)

## 7. Non-Goals

- Defining upstream PtychoViT training policy
- Physical-unit harmonization beyond the existing pixel-space comparison workflow
- Supporting arbitrary non-256 diffraction patch sizes for `pinn_ptychovit` in this contract version
