# Ptychography Data Preprocessing Tools

This directory contains essential tools for preparing ptychography datasets for use with PtychoPINN. These tools handle format conversion, data validation, and dataset preparation workflows.

## Tool Overview

### Core Preprocessing Tools

| Tool | Purpose | When to Use |
|------|---------|-------------|
| `transpose_rename_convert_tool.py` | Format standardization and data type conversion | **Always required** for raw datasets with uint16 diffraction data |
| `generate_patches_tool.py` | Ground truth patch generation for supervised learning | When Y patches are missing from prepared datasets |
| `prepare_data_tool.py` | Advanced data preparation (smoothing, interpolation, apodization) | For probe/object enhancement before simulation |
| `downsample_data_tool.py` | K-space cropping and real-space binning | To reduce dataset size while maintaining physical consistency |
| `split_dataset_tool.py` | Train/test dataset splitting | To create training and testing splits from a single dataset |
| `pad_to_even_tool.py` | Dimension padding to even numbers | When arrays have odd dimensions that cause processing issues |
| `visualize_dataset.py` | Dataset inspection and visualization | For quality assurance and debugging dataset issues |
| `update_tool.py` | NPZ file updating with new reconstructions | To add reconstruction results to existing datasets |
| `generate_test_index.py` | Generates the Markdown test suite index | Run after adding or renaming tests to refresh documentation |

## Essential Preprocessing Pipeline

### For Raw Experimental Datasets

Most raw experimental datasets require format standardization before use with PtychoPINN:

```bash
# Step 1: Format conversion (REQUIRED for raw data)
python scripts/tools/transpose_rename_convert_tool.py \
    raw_dataset.npz \
    converted_dataset.npz

# Step 2: Generate ground truth patches (if needed for supervised training)
python scripts/tools/generate_patches_tool.py \
    converted_dataset.npz \
    final_dataset.npz
```

### For Prepared Datasets

If working with datasets that already follow the data contracts:

```bash
# Optional: Split into train/test
python scripts/tools/split_dataset_tool.py \
    prepared_dataset.npz \
    output_directory \
    --split-fraction 0.5
```

## Tool Details

### `transpose_rename_convert_tool.py`

**Purpose:** Converts raw datasets to comply with PtychoPINN data contracts.

**Key Operations:**
- Converts `uint16` diffraction data to `float32` for numerical stability
- Renames `diff3d` key to `diffraction` for contract compliance
- Squeezes 4D Y arrays to 3D by removing singleton dimensions
- Optional coordinate flipping for coordinate system alignment

**Usage:**
```bash
python scripts/tools/transpose_rename_convert_tool.py input.npz output.npz [--flipx] [--flipy]
```

**When to Use:**
- **Always** for raw experimental datasets with uint16 diffraction data
- When datasets use legacy `diff3d` key naming
- When coordinate systems need alignment

### `generate_patches_tool.py`

**Purpose:** Generates ground truth object patches (Y array) from the full object for supervised learning.

**Key Operations:**
- Extracts patches from `objectGuess` at scan positions
- Creates properly shaped Y array for supervised training
- Adds coordinate offset metadata for advanced workflows

**Usage:**
```bash
python scripts/tools/generate_patches_tool.py input.npz output.npz [--patch-size 64] [--k-neighbors 7]
```

**When to Use:**
- When Y patches are missing from a dataset
- For supervised learning workflows requiring ground truth patches
- When switching from unsupervised to supervised training

### `prepare_data_tool.py`

**Purpose:** Advanced data preparation including smoothing, interpolation, and apodization.

**Operations:**
- `--apodize`: Smoothly tapers probe edges to zero
- `--smooth`: Applies Gaussian filtering to probe or object
- `--interpolate`: Upsamples data via spline interpolation

**Usage:**
```bash
# Interpolate then smooth (chainable operations)
python scripts/tools/prepare_data_tool.py input.npz temp.npz --interpolate --zoom-factor 2.0
python scripts/tools/prepare_data_tool.py temp.npz output.npz --smooth --target probe --sigma 1.5
```

**When to Use:**
- Before simulation to enhance probe/object quality
- To reduce artifacts in probe functions
- For upsampling low-resolution data

### `downsample_data_tool.py`

**Purpose:** Physically consistent dataset size reduction.

**Operations:**
- K-space cropping with proper frequency domain handling
- Real-space binning maintaining physical relationships
- Coordinate system updates for reduced dimensions

**Usage:**
```bash
python scripts/tools/downsample_data_tool.py input.npz output.npz --crop-factor 2 --bin-factor 2
```

**When to Use:**
- To reduce computational requirements
- For creating smaller test datasets
- When memory constraints require size reduction

## Common Workflows

### Raw Dataset → Ready for Training

```bash
# Complete preprocessing pipeline
python scripts/tools/transpose_rename_convert_tool.py raw.npz converted.npz
python scripts/tools/generate_patches_tool.py converted.npz ready.npz
python scripts/tools/split_dataset_tool.py ready.npz datasets/ --split-fraction 0.8
```

### Dataset Quality Assurance

```bash
# Inspect dataset before training
python scripts/tools/visualize_dataset.py dataset.npz
```

### Creating Simulation-Ready Data

```bash
# Enhance data before simulation
python scripts/tools/prepare_data_tool.py raw.npz enhanced.npz --smooth --target probe --sigma 1.0
python scripts/tools/transpose_rename_convert_tool.py enhanced.npz simulation_ready.npz
```

## Documentation Utilities

Keep the developer-facing test documentation aligned with the repository by regenerating the automated index whenever tests are added, renamed, or removed:

```bash
python scripts/tools/generate_test_index.py > docs/development/TEST_SUITE_INDEX.md
```

Commit the regenerated Markdown alongside any test suite changes so humans and agents can discover the latest validation coverage.

## Troubleshooting

### Common Issues

**"KeyError: diffraction"**
- Solution: Use `transpose_rename_convert_tool.py` to rename `diff3d` → `diffraction`

**"TypeError: uint16 not supported"**
- Solution: Use `transpose_rename_convert_tool.py` to convert `uint16` → `float32`

**"Y array has wrong shape"**
- Solution: Use `generate_patches_tool.py` to create proper ground truth patches

**"Odd dimension errors"**
- Solution: Use `pad_to_even_tool.py` to ensure even dimensions

### Data Format Validation

Always verify your dataset follows the data contracts before training:

```python
import numpy as np
data = np.load('your_dataset.npz')

# Check required keys
required_keys = ['diffraction', 'objectGuess', 'probeGuess', 'xcoords', 'ycoords']
missing = [key for key in required_keys if key not in data.keys()]
if missing:
    print(f"Missing keys: {missing}")

# Check data types
print(f"diffraction dtype: {data['diffraction'].dtype}")  # Should be float32
print(f"Y dtype: {data['Y'].dtype if 'Y' in data else 'Missing'}")  # Should be complex64
```

## Workflow Recovery Patterns

When preprocessing workflows are interrupted or fail, use these recovery strategies:

### Partial Pipeline Recovery

**Step-by-step verification:**
```bash
# Check which steps have completed
ls -la *.npz  # See which output files exist
python scripts/tools/visualize_dataset.py intermediate_file.npz  # Verify quality

# Resume from last successful step
python scripts/tools/generate_patches_tool.py intermediate.npz final.npz
```

**File corruption or bad outputs:**
```bash
# Start over from last known good file
cp backup_file.npz current.npz
# Re-run failed step with different parameters
python scripts/tools/prepare_data_tool.py current.npz output.npz --smooth --sigma 0.5
```

### Dataset Preparation Recovery

**When dataset generation fails:**
```bash
# Use existing validated dataset instead of regenerating
cp datasets/fly/fly001_transposed.npz working_dataset.npz

# Or recover partial simulation
python scripts/tools/split_dataset_tool.py partial_sim.npz recovered/ --split-fraction 0.8
```

**Memory issues during processing:**
```bash
# Use downsampling to reduce memory requirements
python scripts/tools/downsample_data_tool.py large_dataset.npz smaller.npz --crop-factor 2
# Then proceed with normal workflow
python scripts/tools/transpose_rename_convert_tool.py smaller.npz ready.npz
```

### Validation and Debugging

**Dataset validation after recovery:**
```bash
# Always verify data integrity after recovery
python scripts/tools/visualize_dataset.py recovered_dataset.npz

# Check data contracts compliance
python -c "
import numpy as np
data = np.load('recovered_dataset.npz')
print('Keys:', list(data.keys()))
print('Diffraction shape:', data['diffraction'].shape)
print('Diffraction dtype:', data['diffraction'].dtype)
"
```

**Tool selection for recovery:**
- **Use existing datasets**: When generation fails, prefer `datasets/fly/fly001_transposed.npz` (~10k images)
- **Memory constraints**: Use `downsample_data_tool.py` before other processing steps
- **Tool hierarchy**: `split_dataset_tool.py` is documented and reliable for creating train/test splits

## See Also

- <doc-ref type="contract">specs/data_contracts.md</doc-ref> - Official data format specifications
- <doc-ref type="guide">docs/FLY64_DATASET_GUIDE.md</doc-ref> - Experimental dataset preprocessing guide
- <doc-ref type="workflow-guide">scripts/studies/GENERALIZATION_STUDY_GUIDE.md</doc-ref> - Model evaluation workflows
