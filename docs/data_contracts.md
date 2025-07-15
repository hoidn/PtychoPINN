# Data Contracts for the PtychoPINN Pipeline

This document defines the official format for key data artifacts used in this project. All tools that generate or consume these files MUST adhere to these contracts.

---

## 1. Canonical Ptychography Dataset (`.npz` format)

This contract applies to any dataset that is considered "ready for training" or is the final output of a preparation pipeline (e.g., from `prepare.sh`).

**File Naming Convention:** `*_train.npz`, `*_test.npz`, `*_prepared.npz`

| Key Name      | Shape                 | Data Type      | Description                                                              | Notes                                                              |
| :------------ | :-------------------- | :------------- | :----------------------------------------------------------------------- | :----------------------------------------------------------------- |
| `diffraction` | `(n_images, H, W)`    | `float32`      | The stack of measured diffraction patterns (amplitude, not intensity).   | **Required.** Formerly `diff3d`. Must be 3D.                       |
| `Y`           | `(n_images, H, W)`    | `complex64`    | The stack of ground truth real-space object patches.                     | **Required for supervised training.** **MUST be 3D.** Squeeze any channel dimension. |
| `objectGuess` | `(M, M)`              | `complex64`    | The full, un-patched ground truth object.                                | **Required.**                                                      |
| `probeGuess`  | `(H, W)`              | `complex64`    | The ground truth probe.                                                  | **Required.**                                                      |
| `xcoords`     | `(n_images,)`         | `float64`      | The x-coordinates of each scan position.                                 | **Required.**                                                      |
| `ycoords`     | `(n_images,)`         | `float64`      | The y-coordinates of each scan position.                                 | **Required.**                                                      |
| `scan_index`  | `(n_images,)`         | `int`          | The index of the scan point for each diffraction pattern.                | Optional, but recommended.                                         |

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

---