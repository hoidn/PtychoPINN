# Prepare.sh Path Parameterization Plan

## Current State Analysis

### Fixed Elements
- **Input**: `tike_outputs/fly001/fly001_reconstructed.npz` (hardcoded)
- **Output Root**: Split between `tike_outputs/` and `datasets/`
- **Naming**: All directories named as `${BASE_NAME}_${STAGE}`
- **BASE_NAME**: Extracted from input filename

### Path Dependencies
```
Input: tike_outputs/fly001/fly001_reconstructed.npz
  ↓
BASE_NAME="fly001_reconstructed"
  ↓
Stages:
  1. tike_outputs/${BASE_NAME}_padded/
  2. tike_outputs/${BASE_NAME}_transposed/
  3. tike_outputs/${BASE_NAME}_interpolated/
  4. tike_outputs/${BASE_NAME}_interp_smooth_probe/
  5. tike_outputs/${BASE_NAME}_final_prepared/
  6. tike_outputs/${BASE_NAME}_final_simulated/
  7. tike_outputs/${BASE_NAME}_final_downsampled/
  8. datasets/${BASE_NAME}_prepared/  [FINAL OUTPUT]
```

## Proposed Parameterization

### New CLI Parameters

```bash
# Required (or with smart default)
--input-file PATH         # Input NPZ file (default: tike_outputs/fly001/fly001_reconstructed.npz)

# Optional  
--output-root DIR         # Root directory for all outputs (default: project root)
--experiment-name NAME    # Custom experiment identifier (default: derived from input)
--stages-dir DIR          # Intermediate stages directory (default: ${output_root}/pipeline_stages)
--final-dir DIR           # Final dataset directory (default: ${output_root}/datasets)

# Existing (already added)
--sim-images N            # Number of images to simulate
--sim-photons P           # Photons per image
```

### Directory Structure Options

#### Option 1: Experiment-Centric (Recommended)
```
${OUTPUT_ROOT}/${EXPERIMENT_NAME}/
├── pipeline_stages/          # All intermediate outputs
│   ├── 01_padded/
│   ├── 02_transposed/
│   ├── 03_interpolated/
│   ├── 04_smooth_probe/
│   ├── 05_smooth_object/
│   ├── 06_simulated/
│   └── 07_downsampled/
├── dataset/                  # Final train/test splits
│   ├── train.npz
│   └── test.npz
├── config.json               # Parameters used
└── pipeline.log              # Execution log
```

#### Option 2: Stage-Centric (Current Style)
```
${OUTPUT_ROOT}/
├── ${EXPERIMENT_NAME}_padded/
├── ${EXPERIMENT_NAME}_transposed/
├── ${EXPERIMENT_NAME}_interpolated/
├── ${EXPERIMENT_NAME}_interp_smooth_probe/
├── ${EXPERIMENT_NAME}_final_prepared/
├── ${EXPERIMENT_NAME}_final_simulated/
├── ${EXPERIMENT_NAME}_final_downsampled/
└── datasets/${EXPERIMENT_NAME}_prepared/
```

#### Option 3: Hybrid (Best of Both)
```
${OUTPUT_ROOT}/
├── experiments/
│   └── ${EXPERIMENT_NAME}/
│       ├── stages/           # Numbered for clarity
│       │   ├── 01_padded/
│       │   ├── 02_transposed/
│       │   └── ...
│       └── dataset/          # Final output
│           ├── train.npz
│           └── test.npz
└── datasets/                 # Symlink for backward compatibility
    └── ${EXPERIMENT_NAME} -> ../experiments/${EXPERIMENT_NAME}/dataset/
```

## Implementation Strategy

### Phase 1: Minimal Changes (Backward Compatible)
```bash
# Add basic parameters
--input-file PATH     # Defaults to current hardcoded path
--output-prefix PATH  # Optional prefix for all output paths

# Usage examples:
./prepare.sh  # Current behavior unchanged

./prepare.sh --input-file my_data.npz  
# Creates: tike_outputs/my_data_padded/, etc.

./prepare.sh --input-file my_data.npz --output-prefix experiments/low_photon/
# Creates: experiments/low_photon/my_data_padded/, etc.
```

### Phase 2: Full Parameterization
```bash
# Complete control over paths
--input-file PATH
--experiment-name NAME
--output-root DIR
--keep-intermediates  # Flag to preserve intermediate files

# Example: Low-photon experiment
./prepare.sh \
  --input-file data/synthetic_object.npz \
  --experiment-name low_photon_1e4 \
  --output-root experiments \
  --sim-photons 1e4 \
  --sim-images 10000

# Creates:
# experiments/low_photon_1e4/
#   ├── stages/[01-07]_*/
#   ├── dataset/{train,test}.npz
#   └── config.json
```

### Phase 3: Configuration File Support
```yaml
# prepare_config.yaml
input_file: data/synthetic_object.npz
output_root: experiments
experiment_name: low_photon_study_1e4
simulation:
  n_images: 10000
  n_photons: 1e4
  seed: 42
processing:
  upsample_factor: 2.0
  object_blur_sigma: 0.5
  probe_blur_sigma: 0.0
  downsample_crop: 2
  downsample_bin: 2
split:
  fraction: 0.5
  axis: y
```

```bash
# Run with config file
./prepare.sh --config prepare_config.yaml

# Override config values via CLI
./prepare.sh --config prepare_config.yaml --sim-photons 1e5
```

## Benefits of Parameterization

### For Low-Photon Studies
```bash
# Generate multiple photon conditions systematically
for photons in 1e4 1e5 1e6 1e7; do
  ./prepare.sh \
    --input-file synthetic_base.npz \
    --experiment-name photon_study_${photons} \
    --output-root studies/photon_scaling \
    --sim-photons $photons \
    --sim-images 10000
done
```

### For Different Objects
```bash
# Process multiple reconstructions
for obj in fly001 fly002 siemens_star; do
  ./prepare.sh \
    --input-file reconstructions/${obj}.npz \
    --experiment-name ${obj}_processed \
    --output-root datasets/prepared
done
```

### For Parameter Sweeps
```bash
# Test different blur settings
for sigma in 0.0 0.5 1.0 1.5; do
  ./prepare.sh \
    --input-file base_object.npz \
    --experiment-name blur_sigma_${sigma} \
    --output-root studies/blur_effects \
    --object-blur-sigma $sigma  # Would need to add this param
done
```

## Migration Path

### Step 1: Add Input File Parameter (Minimal Risk)
- Add `--input-file` with default to current hardcoded path
- Everything else remains the same
- Allows immediate use for different inputs

### Step 2: Add Output Organization (Low Risk)
- Add `--output-dir` (single parameter, not split)
- Default behavior unchanged if not specified
- Improves organization for studies

### Step 2.5: Update Documentation (Critical)
Based on comprehensive codebase search, the following files need updates:

#### **Immediate Updates Required:**
1. **`docs/COMMANDS_REFERENCE.md`**
   - Update Golden Path 2 examples with new parameter syntax
   - Add examples showing `--input-file` and `--output-dir` usage
   - Document backward compatibility

2. **`scripts/studies/run_complete_generalization_study.sh`** (Line 482)
   - Update the `prep_cmd` to pass through new parameters:
   ```bash
   # OLD: local prep_cmd="bash scripts/prepare.sh"
   # NEW: local prep_cmd="bash scripts/prepare.sh --input-file $input --output-dir $prep_dir"
   ```

3. **`docs/studies/GENERALIZATION_STUDY_GUIDE.md`**
   - Update workflow examples to show parameterized usage
   - Add section on using prepare.sh for custom datasets
   
4. **Create `scripts/prepare_examples.md`**
   - Document common use cases with new parameters
   - Show photon study workflows
   - Include backward compatibility examples

#### **Secondary Updates:**
5. **`SYNTHETIC_LOW_PHOTON_DATASET_PLAN.md`**
   - Update to use new parameterized prepare.sh
   - Simplify workflow with proper output organization

6. **`scripts/simulation/README.md`**
   - Add note about prepare.sh parameterization
   - Link to new examples document

#### **Documentation Template for Updates:**
```markdown
## Dataset Preparation

### Basic Usage (Backward Compatible)
```bash
# Original behavior preserved
./scripts/prepare.sh
```

### Custom Input/Output
```bash
# Specify custom input and organized output
./scripts/prepare.sh \
  --input-file path/to/reconstruction.npz \
  --output-dir experiments/my_study \
  --sim-images 10000 \
  --sim-photons 1e4
```

### Photon Studies
```bash
# Generate datasets with different photon counts
for photons in 1e4 1e5 1e6; do
  ./scripts/prepare.sh \
    --input-file synthetic.npz \
    --output-dir studies/photons_${photons} \
    --sim-photons $photons
done
```
```

### Step 3: Add Processing Parameters (Medium Risk)
- Make blur sigma, crop/bin factors configurable
- Add parameter validation
- Document parameter ranges

### Step 4: Config File Support (Enhancement)
- YAML/JSON config file option
- Parameter precedence: CLI > config > defaults
- Export config from successful runs

## Backward Compatibility

Ensure the script works without any parameters:
```bash
./prepare.sh  # Must work exactly as before
```

Default values:
```bash
INPUT_FILE="${INPUT_FILE:-tike_outputs/fly001/fly001_reconstructed.npz}"
OUTPUT_ROOT="${OUTPUT_ROOT:-.}"  # Current directory
EXPERIMENT_NAME="${EXPERIMENT_NAME:-$(basename ${INPUT_FILE%.npz})}"
```

## Testing Strategy

1. **No parameters**: Verify identical to current behavior
2. **Only photon/image params**: Test already-added features
3. **Custom input**: Test with different NPZ files
4. **Custom outputs**: Verify directory creation
5. **Full parameters**: Complete custom configuration
6. **Edge cases**: Missing input, write permissions, etc.

## Recommended Implementation Order

1. **Immediate**: Add `--input-file` parameter
   - Enables using different inputs
   - Zero risk to existing workflows
   
2. **Next Sprint**: Add `--output-root` and `--experiment-name`
   - Improves organization
   - Critical for multiple experiments
   
3. **Future**: Processing parameters and config files
   - Based on actual study needs
   - After validating core parameterization

## Example: Low-Photon Dataset Generation

With full parameterization:
```bash
#!/bin/bash
# generate_photon_study.sh

PHOTON_COUNTS="1e4 1e5 1e6 1e7 1e8 1e9"
BASE_INPUT="synthetic_object.npz"
STUDY_ROOT="studies/photon_scaling_$(date +%Y%m%d)"

for photons in $PHOTON_COUNTS; do
  echo "Generating dataset with $photons photons..."
  
  ./prepare.sh \
    --input-file "$BASE_INPUT" \
    --experiment-name "photons_${photons}" \
    --output-root "$STUDY_ROOT" \
    --sim-photons "$photons" \
    --sim-images 10000
    
  # Link to standard location for training scripts
  ln -s "$STUDY_ROOT/photons_${photons}/dataset" \
        "datasets/photon_study_${photons}"
done

echo "Photon scaling study complete!"
echo "Datasets available in: $STUDY_ROOT"
```

This approach maintains the power of the existing pipeline while enabling systematic studies and better experiment organization.