# Simulation Workflow Agent Guide

## Quick Context
- **Two-stage architecture**: Input generation → Diffraction simulation
- **Core tool**: `simulate_and_save.py` (refactored with modular workflow)
- **Quick start**: `run_with_synthetic_lines.py` (for 'lines' objects)
- **Purpose**: Generate synthetic training/test datasets
- **Key improvement**: Now supports gridsize > 1 correctly

## Essential Commands

### Synthetic Lines (Quickest Start)
```bash
# Generate synthetic "lines" object with diffraction patterns
python scripts/simulation/run_with_synthetic_lines.py --output-dir sim_outputs --n-images 2000

# With visualization
python scripts/simulation/run_with_synthetic_lines.py --output-dir sim_outputs --n-images 1000 --visualize
```

### General Simulation (Two-Stage)
```bash
# Stage 1: Create input .npz with object + probe
# (Use existing .npz or create programmatically)

# Stage 2: Generate diffraction patterns
python scripts/simulation/simulate_and_save.py \
    --input-file input_with_obj_probe.npz \
    --output-file sim_data.npz \
    --n-images 2000 \
    --gridsize 1

# With visualization
python scripts/simulation/simulate_and_save.py \
    --input-file input.npz \
    --output-file output.npz \
    --n-images 1000 \
    --visualize

# With external probe (new feature)
python scripts/simulation/simulate_and_save.py \
    --input-file input.npz \
    --output-file output.npz \
    --probe-file external_probe.npy \
    --n-images 1000
```

## Two-Stage Architecture

### Stage 1: Input Generation
**Purpose**: Create `.npz` file with `objectGuess` and `probeGuess`

**Methods**:
- Use existing reconstruction as input
- Generate synthetic objects programmatically
- Use `run_with_synthetic_lines.py` (handles both stages)

### Stage 2: Diffraction Simulation
**Purpose**: Generate `diffraction`, `xcoords`, `ycoords` from object/probe

**Tool**: `simulate_and_save.py`
- Takes object + probe → produces full training dataset
- Follows data contracts automatically

## Common Patterns

### Quick Synthetic Dataset
```bash
# One command for complete synthetic dataset
python scripts/simulation/run_with_synthetic_lines.py \
    --output-dir synthetic_lines_data \
    --n-images 5000 \
    --visualize
```

### From Existing Reconstruction
```bash
# Use trained model output as simulation input
python scripts/simulation/simulate_and_save.py \
    --input-file training_outputs/my_model/reconstructed_object.npz \
    --output-file sim_from_reconstruction.npz \
    --n-images 3000
```

### Custom Grid Sampling
```bash
# Dense grid sampling (gridsize > 1) - NOW WORKING!
python scripts/simulation/simulate_and_save.py \
    --input-file input.npz \
    --output-file dense_grid.npz \
    --n-images 2000 \
    --gridsize 2  # 2x2 grid sampling (fixed!)

# Gridsize 3 also supported
python scripts/simulation/simulate_and_save.py \
    --input-file input.npz \
    --output-file dense_grid_3x3.npz \
    --n-images 1000 \
    --gridsize 3  # 3x3 grid sampling
```

## Object Types

### Synthetic Lines
- **Command**: `run_with_synthetic_lines.py`
- **Use case**: Quick testing, baseline experiments
- **Properties**: Simple geometric patterns, known ground truth

### GRF (Gaussian Random Field)
```python
# Programmatic generation
from ptycho.diffsim import sim_object_image
obj = sim_object_image(data_source='grf', N_obj=192, ...)
```

### Custom Objects
```python
# Create custom input .npz
import numpy as np
np.savez('custom_input.npz', 
         objectGuess=my_object,  # Complex (M, M) array
         probeGuess=my_probe)    # Complex (N, N) array
```

## Advanced Workflows

### Decoupled Probe Simulation
```bash
# Use external probe for controlled studies
python scripts/simulation/simulate_and_save.py \
    --input-file object_data.npz \
    --output-file sim_with_custom_probe.npz \
    --probe-file experimental_probe.npz \
    --n-images 2000

# Combine with hybrid probe tool
python scripts/tools/create_hybrid_probe.py \
    amplitude_source.npy phase_source.npy \
    --output hybrid.npy

python scripts/simulation/simulate_and_save.py \
    --input-file object_data.npz \
    --output-file sim_with_hybrid.npz \
    --probe-file hybrid.npy \
    --n-images 2000
```

### Noise-Free Simulation
```bash
# High photon count for minimal noise
python scripts/simulation/simulate_and_save.py \
    --input-file input.npz \
    --output-file clean_data.npz \
    --n-images 1000 \
    --nphotons 10000  # High photon count
```

### Multiple Datasets
```bash
# Generate training and test sets
python scripts/simulation/simulate_and_save.py \
    --input-file input.npz \
    --output-file train_data.npz \
    --n-images 8000

python scripts/simulation/simulate_and_save.py \
    --input-file input.npz \
    --output-file test_data.npz \
    --n-images 2000
```

## Parameter Control

### Key Parameters
| Parameter | Effect | Typical Values |
|-----------|--------|----------------|
| `n-images` | Dataset size | 1000-10000 |
| `gridsize` | Sampling density | 1 (sparse), 2+ (dense) |
| `nphotons` | Noise level | 1000 (noisy), 10000 (clean) |

### Data Format Output
- Automatically follows data contracts
- Includes: `diffraction`, `xcoords`, `ycoords`, `objectGuess`, `probeGuess`
- Ready for training without preprocessing

## Troubleshooting

### Input File Errors
**Problem**: Cannot load input .npz file  
**Solutions**:
- Verify file contains `objectGuess` and `probeGuess`
- Check arrays are complex-valued
- Ensure `objectGuess` is larger than `probeGuess`

### Memory Issues
**Problem**: Out of memory during simulation  
**Solutions**:
- Reduce `n-images`
- Use smaller object/probe sizes
- Process in batches

### Poor Quality Simulation
**Problem**: Unrealistic diffraction patterns  
**Solutions**:
- Check object contrast and features
- Verify probe is reasonable (focused, finite support)
- Adjust `nphotons` for desired noise level

## Integration with Training

### Simulation → Training Pipeline
```bash
# 1. Generate synthetic dataset
python scripts/simulation/run_with_synthetic_lines.py --output-dir sim_data --n-images 5000

# 2. Train model on synthetic data
ptycho_train --train_data_file sim_data/synthetic_lines_data.npz --output_dir trained_on_synthetic

# 3. Test on real data
ptycho_inference --model_path trained_on_synthetic --test_data real_test_data.npz --output_dir synthetic_to_real_test
```

## Architecture Notes (Updated 2025-08-02)

### Modular Workflow
The `simulate_and_save.py` script now uses a modular, explicit workflow instead of the monolithic `RawData.from_simulation` method:

1. **Coordinate Generation**: Uses `ptycho.raw_data.group_coords()` for spatial grouping
2. **Patch Extraction**: Uses `ptycho.raw_data.get_image_patches()` in Channel Format
3. **Format Conversion**: Explicit Channel ↔ Flat format conversion via `tf_helper`
4. **Physics Simulation**: Direct use of `ptycho.diffsim.illuminate_and_diffract()`
5. **Assembly**: Proper handling of coordinate expansion for gridsize > 1

### Migration from Legacy Code
If you have code using the deprecated `RawData.from_simulation`:
- Replace with direct calls to `simulate_and_save.py`
- The new approach is more transparent and debuggable
- Gridsize > 1 now works correctly

### Debug Mode
```bash
# Enable debug logging to trace tensor shapes
python scripts/simulation/simulate_and_save.py \
    --input-file input.npz \
    --output-file output.npz \
    --gridsize 2 \
    --debug
```

## Cross-References

- **Data format specs**: <doc-ref type="contract">docs/data_contracts.md</doc-ref>
- **Training workflow**: <doc-ref type="workflow-guide">scripts/training/CLAUDE.md</doc-ref>
- **Detailed simulation docs**: <doc-ref type="workflow-guide">scripts/simulation/README.md</doc-ref>
- **Tool selection**: <doc-ref type="guide">docs/TOOL_SELECTION_GUIDE.md</doc-ref>