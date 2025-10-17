# Synthetic Low-Photon Dataset Generation Plan

## Overview

This plan outlines the approach for generating a synthetic ptychography dataset with 1e4 average photons per diffraction pattern, following PtychoPINN's best practices and data contracts.

## Motivation

Current datasets use 1e9 photons (prepare.sh default), which represents high-flux conditions. A 1e4 photon dataset would:
- Test model performance under severe photon-limited conditions
- Simulate realistic experimental scenarios with limited beam time
- Evaluate noise robustness of PINN vs baseline approaches

## Implementation Plan

### Stage 1: Create Low-Photon Simulation Script
**Goal**: Develop `scripts/simulation/generate_low_photon_synthetic.sh`
**Success Criteria**: 
- Script generates complete dataset with proper NPZ format
- Maintains 1e4 average photons per pattern
- Follows project directory organization
**Status**: Not Started

### Stage 2: Implement Supporting Python Tool
**Goal**: Create `scripts/simulation/run_low_photon_synthetic.py`
**Success Criteria**:
- Generates synthetic object and probe programmatically
- Configurable object types (lines, siemens star, random phases)
- Proper integration with existing simulation pipeline
**Status**: Not Started

### Stage 3: Dataset Validation and Documentation
**Goal**: Validate dataset quality and create usage documentation
**Success Criteria**:
- Dataset passes data contract validation
- Metrics confirm 1e4 average photon count
- Documentation added to relevant guides
**Status**: Not Started

## Technical Approach

### Directory Structure
```
synthetic_low_photon_10k/
├── source/
│   ├── synthetic_object_probe.npz    # Original synthetic object/probe
│   └── generation_params.yaml        # Configuration used
├── simulated/
│   ├── raw_simulated.npz             # Raw simulation output
│   └── simulation_visualization.png  # Quality check plots
├── prepared/
│   ├── dataset_train.npz            # Training split (80%)
│   ├── dataset_test.npz             # Test split (20%)
│   └── dataset_stats.json           # Photon statistics
└── README.md                         # Dataset documentation
```

### Workflow Pipeline

1. **Synthetic Object/Probe Generation**
   - Use existing `sim_object_image()` for object creation
   - Use `get_default_probe()` for probe generation
   - Object size: 232×232 (standard for 64×64 probe)
   - Complex-valued with realistic phase structure

2. **Low-Photon Simulation**
   - Call `generate_simulated_data()` with `nphotons=1e4`
   - Generate 10,000 diffraction patterns for statistical validity
   - Apply Poisson noise modeling inherent in pipeline
   - Save with proper amplitude format (not intensity)

3. **Data Preparation**
   - Format conversion using `transpose_rename_convert_tool.py`
   - Generate Y patches for supervised learning
   - Split into train/test sets (80/20 or 50/50 based on use case)

4. **Quality Validation**
   - Verify average photon count: ~1e4 per pattern
   - Check SNR and dynamic range
   - Validate against data contracts
   - Generate visualization plots

### Script Template: `generate_low_photon_synthetic.sh`

```bash
#!/bin/bash
set -e

# Configuration
OUTPUT_ROOT="synthetic_low_photon_10k"
N_IMAGES=10000
N_PHOTONS=1e4
PROBE_SIZE=64
OBJECT_TYPE="lines"  # or "siemens", "random"
SEED=42

# Stage 1: Generate synthetic object/probe
python scripts/simulation/run_low_photon_synthetic.py \
    --output-dir "$OUTPUT_ROOT/source" \
    --probe-size $PROBE_SIZE \
    --object-type $OBJECT_TYPE \
    --save-only

# Stage 2: Run simulation with low photon count
python scripts/simulation/simulate_and_save.py \
    --input-file "$OUTPUT_ROOT/source/synthetic_object_probe.npz" \
    --output-file "$OUTPUT_ROOT/simulated/raw_simulated.npz" \
    --n-images $N_IMAGES \
    --n-photons $N_PHOTONS \
    --seed $SEED \
    --visualize

# Stage 3: Format conversion
python scripts/tools/transpose_rename_convert_tool.py \
    "$OUTPUT_ROOT/simulated/raw_simulated.npz" \
    "$OUTPUT_ROOT/simulated/formatted.npz"

# Stage 4: Generate Y patches
python scripts/tools/generate_patches_tool.py \
    "$OUTPUT_ROOT/simulated/formatted.npz" \
    "$OUTPUT_ROOT/simulated/with_patches.npz"

# Stage 5: Create train/test split
python scripts/tools/split_dataset_tool.py \
    "$OUTPUT_ROOT/simulated/with_patches.npz" \
    "$OUTPUT_ROOT/prepared" \
    --split-fraction 0.8 \
    --split-axis random

# Stage 6: Compute and save statistics
python scripts/tools/compute_dataset_stats.py \
    "$OUTPUT_ROOT/prepared/dataset_train.npz" \
    "$OUTPUT_ROOT/prepared/dataset_test.npz" \
    --output "$OUTPUT_ROOT/prepared/dataset_stats.json"

echo "Low-photon synthetic dataset generated successfully!"
echo "Location: $OUTPUT_ROOT/prepared/"
```

## Integration Points

### With Existing Tools
- Leverages `simulate_and_save.py` for core simulation
- Uses standard data preparation tools from `scripts/tools/`
- Follows established NPZ data contracts
- Compatible with training/inference pipelines

### With Studies Framework
- Can be integrated into `run_complete_generalization_study.sh`
- Suitable for comparison studies (PINN vs baseline at low photon counts)
- Enables photon-scaling studies (1e4, 1e5, 1e6, etc.)

## Best Practices Adherence

### Data Management
- ✅ No data files committed to Git
- ✅ Clear directory structure with stages
- ✅ Metadata preserved (generation parameters, statistics)

### Code Organization
- ✅ Reuses existing simulation infrastructure
- ✅ Follows script naming conventions
- ✅ Proper error handling with `set -e`

### Documentation
- ✅ Inline comments in scripts
- ✅ README in dataset directory
- ✅ Integration with existing guides

### Testing Strategy
1. **Unit validation**: Each stage produces verifiable output
2. **Integration test**: Full pipeline execution
3. **Quality metrics**: Photon statistics, SNR analysis
4. **Training validation**: Confirm models can train on dataset

## Advantages of This Approach

1. **Modularity**: Each stage can be modified independently
2. **Reproducibility**: Fixed seeds and saved configurations
3. **Flexibility**: Easy to vary photon counts or object types
4. **Compatibility**: Works with all existing workflows
5. **Traceability**: Complete record from source to final dataset

## Next Steps

1. Implement `run_low_photon_synthetic.py` with configurable object generation
2. Create `compute_dataset_stats.py` tool for photon statistics
3. Test pipeline with small dataset (100 images)
4. Scale to full 10,000 image dataset
5. Run comparison study between PINN and baseline models
6. Document findings and update relevant guides

## Expected Outcomes

- Dataset with realistic low-photon noise characteristics
- Baseline for photon-limited ptychography studies
- Insights into model robustness under severe noise
- Potential improvements to training strategies for noisy data

## Risk Mitigation

- **Risk**: Models may fail to converge with extreme noise
  - **Mitigation**: Test with intermediate photon counts (1e5, 1e6)
  
- **Risk**: Simulation may be computationally expensive for 10k images
  - **Mitigation**: Use batched processing, consider cloud compute
  
- **Risk**: Dataset may not represent realistic experimental conditions
  - **Mitigation**: Validate against known low-photon experimental data