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