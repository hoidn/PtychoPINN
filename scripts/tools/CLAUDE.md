# Data Tools Agent Guide

## Quick Context
- **Essential pipeline**: transpose_rename_convert_tool.py → generate_patches_tool.py
- **Purpose**: Raw data → training-ready format
- **Critical rule**: Always use transpose_rename_convert_tool.py for raw uint16 data
- **Philosophy**: Modular tools for flexible data processing

## Essential Pipeline

### Core Workflow
```bash
# 1. Convert raw data format
python scripts/tools/transpose_rename_convert_tool.py raw_data.npz converted.npz

# 2. Generate training patches
python scripts/tools/generate_patches_tool.py converted.npz training_ready.npz

# 3. Split into train/test sets
python scripts/tools/split_dataset_tool.py training_ready.npz output_dir/ --split-fraction 0.8
```

### Quality Check Pipeline
```bash
# Visualize at each step
python scripts/tools/visualize_dataset.py raw_data.npz
python scripts/tools/visualize_dataset.py converted.npz
python scripts/tools/visualize_dataset.py training_ready.npz
```

## Critical Tools

### transpose_rename_convert_tool.py
**Purpose**: Convert raw experimental data to training format  
**When**: Always first step for raw uint16 data  
**Effect**: Handles dtype conversion, key renaming, transposition

```bash
# Essential for raw data
python scripts/tools/transpose_rename_convert_tool.py raw.npz converted.npz

# Check the conversion
python scripts/tools/visualize_dataset.py converted.npz
```

### generate_patches_tool.py  
**Purpose**: Extract object patches (Y) from full reconstructions  
**When**: After transpose_rename_convert, before training  
**Effect**: Creates supervised learning targets

```bash
# Generate training patches
python scripts/tools/generate_patches_tool.py input.npz output.npz

# Verify patch extraction
python scripts/tools/visualize_dataset.py output.npz
```

### split_dataset_tool.py
**Purpose**: Create train/test splits  
**When**: Final step before training  
**Effect**: Maintains data integrity during splitting

```bash
# 80/20 train/test split
python scripts/tools/split_dataset_tool.py dataset.npz output_dir/ --split-fraction 0.8

# Custom split sizes
python scripts/tools/split_dataset_tool.py dataset.npz output_dir/ --split-fraction 0.9
```

## Data Format Troubleshooting

### KeyError: 'diffraction'
**Problem**: Missing expected keys in .npz file  
**Solution**: Use `transpose_rename_convert_tool.py`
```bash
python scripts/tools/transpose_rename_convert_tool.py problematic.npz fixed.npz
```

### "uint16 not supported"
**Problem**: Raw data has unsupported data type  
**Solution**: Use `transpose_rename_convert_tool.py` 
```bash
python scripts/tools/transpose_rename_convert_tool.py uint16_data.npz converted.npz
```

### "Y array wrong shape" 
**Problem**: Missing object patches for supervised training  
**Solution**: Use `generate_patches_tool.py`
```bash
python scripts/tools/generate_patches_tool.py no_patches.npz with_patches.npz
```

### Memory errors
**Problem**: Dataset too large for processing  
**Solution**: Use `downsample_data_tool.py` first
```bash
python scripts/tools/downsample_data_tool.py large.npz smaller.npz --factor 2
```

## Advanced Tools

### downsample_data_tool.py
**Purpose**: Reduce dataset size while maintaining physical consistency  
**Use case**: Memory constraints, quick testing

```bash
# Downsample by factor of 2
python scripts/tools/downsample_data_tool.py input.npz output.npz --factor 2

# Crop k-space and bin real-space
python scripts/tools/downsample_data_tool.py input.npz output.npz --factor 4
```

### prepare_data_tool.py  
**Purpose**: Object/probe conditioning (smoothing, apodization)  
**Use case**: Simulation preparation, noise reduction

```bash
# Smooth probe and apodize object
python scripts/tools/prepare_data_tool.py input.npz prepared.npz --smooth-probe --apodize-object
```

### update_tool.py
**Purpose**: Update .npz files with new reconstruction results  
**Use case**: Adding model outputs to existing datasets

```bash
# Add reconstruction to dataset
python scripts/tools/update_tool.py dataset.npz updated.npz --reconstruction model_output.npz
```

### shuffle_dataset_tool.py
**Purpose**: Randomize dataset order for sampling studies  
**Use case**: Spatial bias analysis, generalization studies

```bash
# Shuffle scan order
python scripts/tools/shuffle_dataset_tool.py ordered.npz shuffled.npz --seed 42
```

### create_hybrid_probe.py
**Purpose**: Create synthetic probes by mixing amplitude and phase from different sources  
**Use case**: Controlled experiments on probe sensitivity, aberration studies

```bash
# Basic usage - combine amplitude from one probe with phase from another
python scripts/tools/create_hybrid_probe.py amplitude_source.npz phase_source.npz

# With visualization and custom output
python scripts/tools/create_hybrid_probe.py \
    datasets/default_probe.npz \
    datasets/fly64/fly001_64_train_converted.npz \
    --output hybrid_probe.npy \
    --visualize

# With power normalization (preserve total intensity)
python scripts/tools/create_hybrid_probe.py \
    ideal_probe.npy \
    aberrated_probe.npy \
    --output normalized_hybrid.npy \
    --normalize \
    --visualize
```

**Key features**:
- Handles probes of different sizes (automatic resizing)
- Supports .npy and .npz file formats
- Optional power normalization
- Visualization of source and hybrid probes
- Validates output for physical plausibility

## Common Workflows

### Experimental Data → Training
```bash
# Complete pipeline for experimental data
python scripts/tools/transpose_rename_convert_tool.py experimental_raw.npz step1.npz
python scripts/tools/generate_patches_tool.py step1.npz step2.npz  
python scripts/tools/split_dataset_tool.py step2.npz training_data/ --split-fraction 0.8

# Quality check
python scripts/tools/visualize_dataset.py training_data/train.npz
python scripts/tools/visualize_dataset.py training_data/test.npz
```

### Reconstruction → Simulation Input
```bash
# Prepare reconstruction for simulation
python scripts/tools/prepare_data_tool.py reconstruction.npz sim_input.npz --smooth-probe

# Generate simulation dataset  
python scripts/simulation/simulate_and_save.py --input-file sim_input.npz --output-file sim_data.npz --n-images 5000
```

### Dataset Size Management
```bash
# Large dataset → manageable size
python scripts/tools/downsample_data_tool.py huge_dataset.npz manageable.npz --factor 2
python scripts/tools/split_dataset_tool.py manageable.npz working_data/ --split-fraction 0.8
```

## Recovery Patterns

### Failed Training Due to Data Format
```bash
# 1. Diagnose the issue
python scripts/tools/visualize_dataset.py problematic_dataset.npz

# 2. Apply appropriate fix
python scripts/tools/transpose_rename_convert_tool.py problematic_dataset.npz fixed_step1.npz
python scripts/tools/generate_patches_tool.py fixed_step1.npz fixed_step2.npz

# 3. Verify fix
python scripts/tools/visualize_dataset.py fixed_step2.npz

# 4. Retry training
ptycho_train --train_data_file fixed_step2.npz --output_dir retry_training
```

### Dataset Corruption Recovery
```bash
# Regenerate from earlier stage if possible
python scripts/tools/generate_patches_tool.py last_good_stage.npz recovered.npz
python scripts/tools/split_dataset_tool.py recovered.npz recovered_split/ --split-fraction 0.8
```

## Data Validation

### Quick Validation
```bash
# Check data contract compliance
python scripts/tools/visualize_dataset.py dataset.npz
# Look for: probeGuess, objectGuess, diffraction, xcoords, ycoords
# Verify: Complex dtypes, correct shapes, reasonable value ranges
```

### Deep Validation
```bash
# Check statistical properties
python -c "
import numpy as np
data = np.load('dataset.npz')
print('Keys:', list(data.keys()))
print('Shapes:', {k: v.shape for k, v in data.items()})
print('Dtypes:', {k: v.dtype for k, v in data.items()})
print('Diffraction stats:', np.mean(data['diffraction']), np.std(data['diffraction']))
"
```

## Cross-References

- **Data format specs**: <doc-ref type="contract">docs/data_contracts.md</doc-ref>
- **Simulation workflow**: <doc-ref type="workflow-guide">scripts/simulation/CLAUDE.md</doc-ref>
- **Training workflow**: <doc-ref type="workflow-guide">scripts/training/CLAUDE.md</doc-ref>
- **Detailed tool docs**: <doc-ref type="workflow-guide">scripts/tools/README.md</doc-ref>
- **Tool selection guide**: <doc-ref type="guide">docs/TOOL_SELECTION_GUIDE.md</doc-ref>