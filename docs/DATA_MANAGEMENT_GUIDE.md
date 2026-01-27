# Data Management Guide

## 🚨 CRITICAL: Never Commit Data Files to Git

**This is the #1 rule for PtychoPINN development:** Data files (NPZ, HDF5, etc.) must NEVER be committed to the Git repository. These files are often hundreds of megabytes and will bloat the repository permanently.

## Quick Reference

### Files That Should NEVER Be Committed
- `*.npz` - NumPy data archives
- `*.h5` - HDF5 model files
- `*.hdf5` - Alternative HDF5 format
- `*.groups_cache.npz` - Coordinate grouping caches
- `*.dill` - Serialized Python objects
- Large image files (>1MB)
- Any files in `datasets/` (except documentation)
- Model outputs (`wts.h5.zip`, `baseline_model.h5`)

### Before Every Commit
```bash
# Check for accidentally staged data files
git status | grep -E "\.(npz|h5|hdf5)$"

# If found, unstage them
git reset HEAD <file>

# Add to .gitignore if needed
echo "path/to/file.npz" >> .gitignore
```

## Directory Structure

### Standard Data Locations

```
PtychoPINN/
├── datasets/                # Primary dataset storage (git-ignored)
│   ├── fly/                # Production datasets
│   ├── fly64/              # Experimental datasets
│   └── probes/             # Probe data
├── outputs/                 # Training/inference outputs (git-ignored)
├── simulation_outputs/      # Simulated data (git-ignored)
└── memoized_data/          # Cache directory (auto-generated)
```

### Output Directory Convention

Each workflow creates its own output directory:
```
<output_dir>/
├── wts.h5.zip              # Trained model archive
├── history.dill            # Training history
├── params.dill             # Configuration snapshot
├── metrics.csv             # Evaluation metrics
├── logs/                   # Log files
│   └── debug.log          # Complete debug log
└── reconstructed_*.png    # Visualization outputs
```

## Data File Types

### NPZ Files (NumPy Archives)

**Format:** Compressed archives containing multiple NumPy arrays

**Standard Keys** (see [data_contracts.md](data_contracts.md)):
- `diffraction`: `(n_images, H, W)` - Amplitude data
- `objectGuess`: `(M, M)` - Complex object
- `probeGuess`: `(H, W)` - Complex probe
- `xcoords`, `ycoords`: `(n_images,)` - Scan positions

**Size:** Typically 50-500 MB per file

### HDF5 Files (Model Storage)

**Types:**
- `*.h5` - TensorFlow/Keras models
- `wts.h5.zip` - Compressed model archives

**Size:** 10-100 MB depending on model architecture

### Cache Files

**Pattern:** `<hash>.groups_cache.npz`

**Purpose:** Speed up coordinate grouping calculations

**Management:** Safe to delete; regenerated automatically

### Memoization Controls

PtychoPINN also uses disk memoization for expensive grid simulations (e.g., `diffsim.mk_simdata`).
The cache is safe to delete and will be regenerated as needed.

**Environment variables:**
- `PTYCHO_DISABLE_MEMOIZE=1` disables memoization entirely (recommended for sweeps).
- `PTYCHO_MEMOIZE_KEY_MODE=dataset` uses dataset-defining inputs only to compute cache keys.

## Best Practices

### 1. Use .gitignore Properly

Add these patterns to `.gitignore`:
```gitignore
# Data files
*.npz
*.h5
*.hdf5
*.groups_cache.npz
*.dill

# Data directories
datasets/
outputs/
*_outputs/
*_output/
memoized_data/
memoized_simulated_data/

# Temporary files
tmp/
*.tmp
```

### 2. Check File Sizes Before Committing

```bash
# Find large files
find . -type f -size +1M -not -path "./.git/*"

# Check specific directory
du -h datasets/ | sort -h
```

### 3. Data Sharing Strategies

**For Small Test Data (<10MB):**
- OK to commit if essential for tests
- Place in `tests/data/` directory
- Document purpose clearly

**For Large Datasets:**
- Use external storage (cloud, shared drives)
- Document download instructions in README
- Provide scripts for data retrieval

**Example Download Script:**
```python
# scripts/download_datasets.py
import urllib.request
import os

DATASETS = {
    "fly001": "https://example.com/fly001.npz",
    "fly64": "https://example.com/fly64.npz"
}

def download_datasets():
    os.makedirs("datasets", exist_ok=True)
    for name, url in DATASETS.items():
        target = f"datasets/{name}.npz"
        if not os.path.exists(target):
            print(f"Downloading {name}...")
            urllib.request.urlretrieve(url, target)
```

### 4. Clean Up Temporary Files

```bash
# Remove all cache files
find . -name "*.groups_cache.npz" -delete

# Clean simulation outputs
rm -rf simulation_outputs/

# Remove temporary experiment directories
rm -rf test_*/
```

### 5. Handle Accidental Commits

If you accidentally commit a large file:

```bash
# Remove from current commit (if not pushed)
git reset HEAD^ --soft
git reset HEAD <large_file>
git commit -c ORIG_HEAD

# If already pushed (requires force push - coordinate with team!)
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch path/to/large_file.npz" \
  --prune-empty --tag-name-filter cat -- --all
```

## Model Artifact Management

### Saving Models

Models are automatically saved as `wts.h5.zip` in the output directory:
```python
# Models are saved with metadata
model_manager.save_models(output_dir, {
    'main_model': model,
    'inference_model': inference_model
})
```

### Sharing Models

1. **Don't commit models to Git**
2. **Use releases** for stable model versions
3. **Document model location** in experiment logs
4. **Include configuration** with saved models

## Data Validation

### Before Using Data Files

Always validate data format:
```python
import numpy as np

def validate_dataset(filepath):
    """Validate NPZ file format."""
    data = np.load(filepath)
    
    # Check required keys
    required = ['diffraction', 'objectGuess', 'probeGuess', 
                'xcoords', 'ycoords']
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Missing keys: {missing}")
    
    # Validate shapes
    n_images = data['diffraction'].shape[0]
    assert data['xcoords'].shape == (n_images,)
    assert data['ycoords'].shape == (n_images,)
    
    print(f"Dataset valid: {n_images} images")
```

### Data Contract Compliance

See [data_contracts.md](data_contracts.md) for:
- Required array shapes and dtypes
- Normalization requirements
- Coordinate system conventions
- Metadata standards

## Ptychodus Product Export

PtychoPINN can export reconstruction results to the Ptychodus HDF5 product format for interoperability with Ptychodus visualization and analysis tools. The product format is defined in `specs/data_contracts.md`.

### Converting NPZ to Ptychodus Product

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

#### Required Parameters

- `--input-npz`: Path to source NPZ file containing reconstruction data
- `--output-product`: Path for output HDF5 file (*.h5 or *.hdf5)

#### Metadata Parameters

The following parameters provide metadata for the HDF5 product. Use values appropriate to your experimental setup, or use defaults (zeros) when actual values are unknown:

- `--name`: Product name (default: "")
- `--comments`: Product comments/description (default: "")
- `--detector-distance-m`: Detector-to-object distance in meters (default: 0.0)
- `--probe-energy-eV`: Probe photon energy in eV (default: 0.0)
- `--exposure-time-s`: Exposure time per position in seconds (default: 0.0)
- `--object-pixel-size-m`: Object pixel size in meters (default: 5e-8, i.e., 50 nm)
- `--probe-pixel-size-m`: Probe pixel size in meters (default: 1.25e-7, i.e., 125 nm)
- `--object-center-x-m`, `--object-center-y-m`: Object center coordinates in meters (default: 0.0)

#### Raw Data Inclusion

By default, the converter includes raw diffraction patterns and coordinates in a `/raw_data` group within the HDF5 file. To exclude raw data and create a smaller file with only the reconstruction results:

```bash
python scripts/tools/convert_to_ptychodus_product.py \
  --input-npz datasets/Run1084_recon3_postPC_shrunk_3.npz \
  --output-product outputs/ptychodus_products/run1084_product.h5 \
  --no-include-diffraction
```

### Product File Structure

The generated HDF5 file conforms to `specs/data_contracts.md` and contains:

#### Root Attributes
- `name`, `comments`: User-provided metadata
- Physics parameters: `detector_object_distance_m`, `probe_energy_eV`, `exposure_time_s`, etc.

#### Required Datasets
- `probe`: Complex-valued probe reconstruction [H, W] with pixel geometry attributes
- `object`: Complex-valued object reconstruction [H, W] with center and pixel geometry attributes
- `probe_position_x_m`, `probe_position_y_m`: Scan positions in meters
- `probe_position_indexes`: Scan position indices
- `loss_values`, `loss_epochs`: Empty arrays (loss history not tracked)
- `object_layer_spacing_m`: Empty array (single layer)

#### Optional Raw Data Group (`/raw_data`)
When `--include-diffraction` is enabled:
- `diffraction`: Raw diffraction patterns [N, H, W] with canonical axis order
- `xcoords`, `ycoords`: Scan positions in pixels
- `scan_index`: Scan indices
- `probeGuess`, `objectGuess`: Links to root datasets

### Storage Policy

**Important**: Product files (*.h5) should be stored under `outputs/ptychodus_products/` which is git-ignored. Do not commit large HDF5 files to the repository.

### Programmatic Access

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

### Verification

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

### References

- Product format specification: `specs/data_contracts.md`
- Exporter/importer implementation: `ptycho/io/ptychodus_product_io.py`
- Test suite: `tests/io/test_ptychodus_product_io.py`

## Troubleshooting

### "File too large" Git Error
```bash
# Find the large file
git status
# Remove from staging
git reset HEAD <large_file>
# Add to .gitignore
echo "<large_file>" >> .gitignore
```

### Out of Memory During Data Loading
- Use the sampling parameters to load subset
- Implement data generators for large datasets
- See [memory.md](memory.md) for optimization strategies

### Cache Files Taking Too Much Space
```bash
# Find all cache files
find . -name "*.groups_cache.npz" -exec du -h {} \;
# Remove all cache files (safe - will regenerate)
find . -name "*.groups_cache.npz" -delete
```

## Summary Checklist

Before committing:
- [ ] No `*.npz` files in `git status`
- [ ] No `*.h5` or `*.hdf5` files staged
- [ ] Large directories excluded in `.gitignore`
- [ ] Cache files not included
- [ ] File sizes checked (<1MB for non-docs)
- [ ] Data download instructions documented
- [ ] Model artifacts saved locally, not committed

## Related Documentation

- [Data Contracts](data_contracts.md) - Data format specifications
- [Memory Guide](memory.md) - Memory optimization strategies
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues and solutions
