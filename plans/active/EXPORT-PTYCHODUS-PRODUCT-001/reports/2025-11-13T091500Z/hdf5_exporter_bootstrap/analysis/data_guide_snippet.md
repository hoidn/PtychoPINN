# Ptychodus Product Export

## Overview

PtychoPINN can export reconstruction results to the Ptychodus HDF5 product format for interoperability with Ptychodus visualization and analysis tools. The product format is defined in `specs/data_contracts.md`.

## Converting NPZ to Ptychodus Product

Use the `convert_to_ptychodus_product.py` script to convert a reconstruction NPZ file to HDF5 product format:

```bash
python scripts/tools/convert_to_ptychodus_product.py \
  --input-npz datasets/Run1084_recon3_postPC_shrunk_3.npz \
  --output-product outputs/ptychodus_products/run1084_product.h5 \
  --name "Run1084" \
  --comments "Run1084 reconstruction export" \
  --detector-distance-m 0.0 \
  --probe-energy-eV 8000.0 \
  --exposure-time-s 0.1 \
  --object-pixel-size-m 5e-8 \
  --probe-pixel-size-m 1.25e-7 \
  --object-center-x-m 0.0 \
  --object-center-y-m 0.0 \
  --include-diffraction
```

### Required Parameters

- `--input-npz`: Path to source NPZ file containing reconstruction data
- `--output-product`: Path for output HDF5 file (*.h5 or *.hdf5)

### Metadata Parameters

The following parameters provide metadata for the HDF5 product. Use values appropriate to your experimental setup, or use defaults (zeros) when actual values are unknown:

- `--name`: Product name (default: "")
- `--comments`: Product comments/description (default: "")
- `--detector-distance-m`: Detector-to-object distance in meters (default: 0.0)
- `--probe-energy-eV`: Probe photon energy in eV (default: 0.0)
- `--exposure-time-s`: Exposure time per position in seconds (default: 0.0)
- `--object-pixel-size-m`: Object pixel size in meters (default: 5e-8, i.e., 50 nm)
- `--probe-pixel-size-m`: Probe pixel size in meters (default: 1.25e-7, i.e., 125 nm)
- `--object-center-x-m`, `--object-center-y-m`: Object center coordinates in meters (default: 0.0)

### Raw Data Inclusion

By default, the converter includes raw diffraction patterns and coordinates in a `/raw_data` group within the HDF5 file. To exclude raw data and create a smaller file with only the reconstruction results:

```bash
python scripts/tools/convert_to_ptychodus_product.py \
  --input-npz datasets/Run1084_recon3_postPC_shrunk_3.npz \
  --output-product outputs/ptychodus_products/run1084_product.h5 \
  --no-include-diffraction
```

## Product File Structure

The generated HDF5 file conforms to `specs/data_contracts.md` and contains:

### Root Attributes
- `name`, `comments`: User-provided metadata
- Physics parameters: `detector_object_distance_m`, `probe_energy_eV`, `exposure_time_s`, etc.

### Required Datasets
- `probe`: Complex-valued probe reconstruction [H, W] with pixel geometry attributes
- `object`: Complex-valued object reconstruction [H, W] with center and pixel geometry attributes
- `probe_position_x_m`, `probe_position_y_m`: Scan positions in meters
- `probe_position_indexes`: Scan position indices
- `loss_values`, `loss_epochs`: Empty arrays (loss history not tracked)
- `object_layer_spacing_m`: Empty array (single layer)

### Optional Raw Data Group (`/raw_data`)
When `--include-diffraction` is enabled:
- `diffraction`: Raw diffraction patterns [N, H, W] with canonical axis order
- `xcoords`, `ycoords`: Scan positions in pixels
- `scan_index`: Scan indices
- `probeGuess`, `objectGuess`: Links to root datasets

## Storage Policy

**Important**: Product files (*.h5) should be stored under `outputs/ptychodus_products/` which is git-ignored. Do not commit large HDF5 files to the repository.

## Programmatic Access

For programmatic export/import within Python:

```python
from pathlib import Path
from ptycho.io.ptychodus_product_io import (
    ExportMeta,
    export_product_from_rawdata,
    import_product_to_rawdata,
)
from ptycho.raw_data import RawData

# Export RawData to HDF5 product
meta = ExportMeta(
    name="My Reconstruction",
    object_pixel_width_m=5e-8,
    object_pixel_height_m=5e-8,
    probe_pixel_width_m=1.25e-7,
    probe_pixel_height_m=1.25e-7,
)
export_product_from_rawdata(
    raw=my_rawdata,
    out_path=Path("outputs/my_product.h5"),
    meta=meta,
    include_raw=True,
)

# Import HDF5 product back to RawData
raw = import_product_to_rawdata(Path("outputs/my_product.h5"))
```

## Verification

To verify a generated product file:

```python
import h5py
import json

with h5py.File("outputs/ptychodus_products/run1084_product.h5", "r") as f:
    print(f"Name: {f.attrs['name']}")
    print(f"Scan positions: {len(f['probe_position_indexes'])}")
    print(f"Probe shape: {f['probe'].shape}")
    print(f"Object shape: {f['object'].shape}")
    if 'raw_data' in f:
        print(f"Diffraction shape: {f['raw_data/diffraction'].shape}")
```

## References

- Product format specification: `specs/data_contracts.md`
- Exporter/importer implementation: `ptycho/io/ptychodus_product_io.py`
- Test suite: `tests/io/test_ptychodus_product_io.py`
