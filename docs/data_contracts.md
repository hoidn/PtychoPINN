# Data Contracts for the PtychoPINN Pipeline

This document defines the official format for key data artifacts used in this project. All tools that generate or consume these files MUST adhere to these contracts.

---

## 1. Canonical Ptychography Dataset (`.npz` format)

This contract applies to any dataset that is considered "ready for training":
the files read by `ptycho.raw_data.RawData.from_file` (all training and
inference entry points) and written by the synthetic pipeline
(`datasets/train.npz` / `test.npz` under a run's output root).

**File Naming Convention:** `*_train.npz`, `*_test.npz`, `*_prepared.npz`

| Key Name        | Shape              | Data Type   | Description                                                          | Notes |
| :-------------- | :----------------- | :---------- | :------------------------------------------------------------------- | :---- |
| `diff3d`        | `(n_images, H, W)` | `float32`   | The stack of measured diffraction patterns. Measurement domain (amplitude vs. detector counts) is governed by the dataset's scale contract — see below. | Canonical key. `diffraction` is accepted as a compatibility alias. |
| `xcoords`       | `(n_images,)`      | `float64`   | The x-coordinates of each scan position.                             | **Required.** |
| `ycoords`       | `(n_images,)`      | `float64`   | The y-coordinates of each scan position.                             | **Required.** |
| `probeGuess`    | `(H, W)`           | `complex64` | The probe. For legacy normalized-amplitude data, the transformed simulation probe; for count-intensity data, the CI-scaled physical forward probe in the same count convention as `diff3d`. | **Required.** |
| `objectGuess`   | `(M, M)`           | `complex64` | The full, un-patched ground truth object.                            | Optional at load; required for truth-based evaluation. |
| `scan_index`    | `(n_images,)`      | `int`       | The scan-point index for each diffraction pattern. Values may repeat. | Optional (defaults to zeros). The synthetic writer always emits it. |
| `object_index`  | `(n_images,)`      | `int`       | Independent-object bank membership for each pattern. | Optional (defaults to zeros). Grouping never crosses an object bank. |
| `xcoords_start`, `ycoords_start` | `(n_images,)` | `float64` | Pre-offset scan coordinates.                            | Optional (default to `xcoords`/`ycoords`). |
| `Y`             | `(n_images, H, W)` | `complex64` | Ground truth real-space object patches.                              | Supervised training only; not part of flat synthetic outputs. Generate with `scripts/tools/generate_patches_tool.py`. **MUST be 3D** (squeeze any channel dimension). |
| `_metadata`     | scalar (JSON string) | `str`     | Provenance metadata (e.g. `nphotons`), managed by `ptycho.metadata.MetadataManager`. | Optional, recommended. |

### Key-naming note: `diff3d` vs `diffraction`

`diff3d` is the canonical standalone-NPZ key and the synthetic writer emits
it. The shared acquisition decoder also accepts `diffraction` for compatibility.
If both keys are present they must describe the same canonical stack or loading
fails. Canonical `(n_images, H, W)`, legacy `(H, W, n_images)`, and either with
a trailing singleton channel are accepted when coordinates disambiguate the
layout; downstream code receives canonical `(n_images, H, W)` data.

### Measurement Domain and Normalization

**⚠️ CRITICAL:** what `diff3d` contains depends on the dataset's scale
contract. Two conventions are supported; mixing them up is a common source
of errors. Synthetic outputs record the governing contract in the run's
`datasets/manifest.json` (`scale_contract_version`, `measurement_domain`).

#### Legacy normalized amplitude (`legacy_v1`)

The default convention (e.g. the `synthetic-lines` profile).

1. **Diffraction patterns MUST be normalized**
   - Data should be in a normalized range (typically with max values < 1.0)
   - The `nphotons` parameter controls physics scaling during training, NOT data values
   - Example: Even for nphotons=1e6, diffraction data remains normalized

2. **Intensity vs Amplitude**
   - `diff3d` MUST contain amplitude (square root of intensity)
   - If you have intensity data: `diff3d = np.sqrt(intensity)`
   - The model applies intensity scaling internally for physics calculations

3. **DO NOT pre-apply photon scaling**
   ```python
   # WRONG - Don't scale by photon count in the data
   diff3d = np.sqrt(intensity) * photon_scale

   # CORRECT - Keep data normalized
   diff3d = np.sqrt(intensity)
   # Set nphotons in config for physics modeling
   ```

#### Count intensity (`ci_intensity_v2`)

The count-intensity contract (e.g. the `cnn-lines-ci` profile).

- `diff3d` contains Poisson-realized detector **counts** (intensity, not
  square-root intensity) and is **not** normalized to a unit range.
- `probeGuess` is the CI-scaled physical forward probe in the same count
  convention.
- The legacy validation below does not apply; count datasets are validated
  against their recorded manifest digests and contract version instead.

#### Validation (legacy amplitude datasets only)

```python
import numpy as np

# Load your dataset
data = np.load('your_dataset.npz')

# Check normalization
assert np.max(data['diff3d']) < 10.0, "Data appears unnormalized"
assert np.min(data['diff3d']) >= 0.0, "Amplitude should be non-negative"

# Check data type
assert data['diff3d'].dtype == np.float32, "Should be float32"

# Check for amplitude (not intensity)
# Amplitude data typically has smaller dynamic range than intensity
ratio = np.max(data['diff3d']) / np.mean(data['diff3d'])
assert ratio < 100, "May be intensity instead of amplitude"
```

**For detailed normalization information:** See <doc-ref type="guide">docs/DATA_NORMALIZATION_GUIDE.md</doc-ref>

---

## 2. Experimental and Raw Dataset Formats

Some datasets may not initially conform to the canonical format above and require preprocessing before use with PtychoPINN. These are typically raw experimental datasets or legacy formats.

### Raw Dataset Format (requires preprocessing)

Raw experimental datasets often store diffraction as unconverted intensity:

| Key Name      | Shape                 | Data Type      | Description                                                              | Action Required                                                    |
| :------------ | :-------------------- | :------------- | :----------------------------------------------------------------------- | :----------------------------------------------------------------- |
| `diff3d`      | `(n_images, H, W)`    | `uint16`       | Raw diffraction patterns as intensity data.                              | **Convert to `float32` amplitude (`np.sqrt(intensity)`), keeping the `diff3d` key, before training use.** |
| Missing `Y`   | N/A                   | N/A            | Ground truth patches not pre-computed.                                  | **For supervised training only, generate using <code-ref type="tool">scripts/tools/generate_patches_tool.py</code-ref>** |

### Preprocessing Requirements

1. **Data Type Conversion:** `uint16` intensity → `float32` amplitude (legacy contract)
2. **Key Naming:** producers write `diff3d`; ingestion also accepts the
   `diffraction` compatibility alias (see the key-naming note in §1)
3. **Array Reshaping:** Ensure Y arrays are 3D (squeeze any singleton dimensions)

For the `diffraction`-keyed peripheral consumers (tike/pty-chi
reconstruction scripts, PtychoViT interop), convert with:

```bash
python scripts/tools/transpose_rename_convert_tool.py raw_dataset.npz converted_dataset.npz
```

The tool's output renames `diff3d → diffraction`; the shared acquisition
decoder accepts that alias.

### Experimental Dataset Documentation

For detailed preprocessing workflows for specific experimental datasets, see:
- <doc-ref type="guide">docs/FLY64_DATASET_GUIDE.md</doc-ref> - FLY64 experimental dataset guide
