# Data Contracts for the PtychoPINN Pipeline

This document defines the official format for key data artifacts used in this project. All tools that generate or consume these files MUST adhere to these contracts.

---

## 1. Canonical Ptychography Dataset (`.npz` format)

This contract applies to any dataset that is considered "ready for training" or is the final output of a preparation pipeline (e.g., from `prepare.sh`).

**File Naming Convention:** `*_train.npz`, `*_test.npz`, `*_prepared.npz`

| Key Name      | Shape                 | Data Type      | Description                                                              | Notes                                                              |
| :------------ | :-------------------- | :------------- | :----------------------------------------------------------------------- | :----------------------------------------------------------------- |
| `diffraction` | `(n_images, H, W)`    | `float32`      | The stack of measured diffraction patterns (amplitude, not intensity). **MUST be normalized** - see normalization requirements below.   | **Required.** Formerly `diff3d`. Must be 3D.                       |
| `Y`           | `(n_images, H, W)`    | `complex64`    | The stack of ground truth real-space object patches.                     | **Required for supervised training.** **MUST be 3D.** Squeeze any channel dimension. |
| `objectGuess` | `(M, M)`              | `complex64`    | The full, un-patched ground truth object.                                | **Required.**                                                      |
| `probeGuess`  | `(H, W)`              | `complex64`    | The ground truth probe.                                                  | **Required.**                                                      |
| `xcoords`     | `(n_images,)`         | `float64`      | The x-coordinates of each scan position.                                 | **Required.**                                                      |
| `ycoords`     | `(n_images,)`         | `float64`      | The y-coordinates of each scan position.                                 | **Required.**                                                      |
| `scan_index`  | `(n_images,)`         | `int`          | The index of the scan point for each diffraction pattern.                | Optional, but recommended.                                         |

### Normalization Requirements

**⚠️ CRITICAL:** PtychoPINN expects data in a specific normalization state. Incorrect normalization is a common source of errors.

#### Required Data State

1. **Diffraction patterns MUST be normalized**
   - Data should be in a normalized range (typically with max values < 1.0)
   - The `nphotons` parameter controls physics scaling during training, NOT data values
   - Example: Even for nphotons=1e6, diffraction data remains normalized

2. **Intensity vs Amplitude**
   - `diffraction` array MUST contain amplitude (square root of intensity)
   - If you have intensity data: `diffraction = np.sqrt(intensity)`
   - The model applies intensity scaling internally for physics calculations

3. **DO NOT pre-apply photon scaling**
   ```python
   # WRONG - Don't scale by photon count in the data
   diffraction = np.sqrt(intensity) * photon_scale
   
   # CORRECT - Keep data normalized
   diffraction = np.sqrt(intensity)
   # Set nphotons in config for physics modeling
   ```

#### Validation

To verify your data meets normalization requirements:

```python
import numpy as np

# Load your dataset
data = np.load('your_dataset.npz')

# Check normalization
assert np.max(data['diffraction']) < 10.0, "Data appears unnormalized"
assert np.min(data['diffraction']) >= 0.0, "Amplitude should be non-negative"

# Check data type
assert data['diffraction'].dtype == np.float32, "Should be float32"

# Check for amplitude (not intensity)
# Amplitude data typically has smaller dynamic range than intensity
ratio = np.max(data['diffraction']) / np.mean(data['diffraction'])
assert ratio < 100, "May be intensity instead of amplitude"
```

**For detailed normalization information:** See <doc-ref type="guide">docs/DATA_NORMALIZATION_GUIDE.md</doc-ref>

---

## 2. Experimental and Raw Dataset Formats

Some datasets may not initially conform to the canonical format above and require preprocessing before use with PtychoPINN. These are typically raw experimental datasets or legacy formats.

### Raw Dataset Format (requires preprocessing)

Raw experimental datasets often use legacy naming conventions and data types that require conversion:

| Key Name      | Shape                 | Data Type      | Description                                                              | Action Required                                                    |
| :------------ | :-------------------- | :------------- | :----------------------------------------------------------------------- | :----------------------------------------------------------------- |
| `diff3d`      | `(n_images, H, W)`    | `uint16`       | Legacy diffraction patterns as intensity data.                          | **Convert to `diffraction` with float32 amplitude using <code-ref type="tool">scripts/tools/transpose_rename_convert_tool.py</code-ref>** |
| Missing `Y`   | N/A                   | N/A            | Ground truth patches not pre-computed.                                  | **Generate using <code-ref type="tool">scripts/tools/generate_patches_tool.py</code-ref>** |

### Preprocessing Requirements

Raw datasets must undergo format conversion to ensure PtychoPINN compatibility:

1. **Data Type Conversion:** `uint16` intensity → `float32` amplitude
2. **Key Renaming:** `diff3d` → `diffraction`
3. **Array Reshaping:** Ensure Y arrays are 3D (squeeze any singleton dimensions)

**Essential preprocessing command:**
```bash
python scripts/tools/transpose_rename_convert_tool.py raw_dataset.npz converted_dataset.npz
```

### Experimental Dataset Documentation

For detailed preprocessing workflows for specific experimental datasets, see:
- <doc-ref type="guide">docs/FLY64_DATASET_GUIDE.md</doc-ref> - FLY64 experimental dataset guide
