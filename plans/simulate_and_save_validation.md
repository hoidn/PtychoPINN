# Validation Plan: simulate_and_save.py for Grid Study

## Objective

Before running the full photon grid study (1e3 to 1e9), validate that `simulate_and_save.py` produces correct, trainable synthetic datasets with controllable photon counts.

## Validation Stages

### Stage 1: Basic Functionality Test
**Goal**: Verify simulate_and_save.py runs without errors  
**Test Conditions**: 
- Input: `datasets/fly/fly001.npz` (known-good reference)
- Photons: 1e6 (moderate noise level)
- Images: 1000 (small for quick test)

```bash
# Test 1.1: Basic execution
python scripts/simulation/simulate_and_save.py \
    --input-file datasets/fly/fly001.npz \
    --output-file validation/test_1e6_basic.npz \
    --n-images 1000 \
    --n-photons 1e6 \
    --seed 42 \
    --visualize

# Expected outputs:
# - validation/test_1e6_basic.npz created
# - visualization/test_1e6_basic_visualization.png created
# - No errors in console
```

### Stage 2: Data Contract Validation
**Goal**: Verify output NPZ matches expected format from `specs/data_contracts.md`

```python
# Test 2.1: Validate NPZ structure
import numpy as np

data = np.load('validation/test_1e6_basic.npz')

# Required keys per data contract
required_keys = ['diff3d', 'objectGuess', 'probeGuess', 'xcoords', 'ycoords']
assert all(k in data for k in required_keys), f"Missing keys: {set(required_keys) - set(data.keys())}"

# Shape validation
assert data['diff3d'].shape == (1000, 64, 64), f"Wrong diffraction shape: {data['diff3d'].shape}"
assert data['objectGuess'].shape == (232, 232), f"Wrong object shape: {data['objectGuess'].shape}"
assert data['probeGuess'].shape == (64, 64), f"Wrong probe shape: {data['probeGuess'].shape}"
assert len(data['xcoords']) == 1000, f"Wrong number of x coordinates: {len(data['xcoords'])}"

# Data type validation
assert data['diff3d'].dtype == np.float32, f"Wrong diffraction dtype: {data['diff3d'].dtype}"
assert np.iscomplexobj(data['objectGuess']), "Object should be complex"
assert np.iscomplexobj(data['probeGuess']), "Probe should be complex"

print("✓ Data contract validation passed")
```

### Stage 3: Photon Statistics Validation
**Goal**: Verify photon count control works correctly

```python
# Test 3.1: Check photon statistics
import numpy as np

data = np.load('validation/test_1e6_basic.npz')
diffraction = data['diff3d']

# Convert amplitude to intensity (photon counts)
intensity = diffraction ** 2

# Calculate statistics
mean_photons = np.mean(intensity.sum(axis=(1,2)))
std_photons = np.std(intensity.sum(axis=(1,2)))
min_photons = np.min(intensity.sum(axis=(1,2)))
max_photons = np.max(intensity.sum(axis=(1,2)))

print(f"Photon statistics for 1e6 target:")
print(f"  Mean: {mean_photons:.2e} (expected ~1e6)")
print(f"  Std:  {std_photons:.2e}")
print(f"  Min:  {min_photons:.2e}")
print(f"  Max:  {max_photons:.2e}")

# Validation criteria
assert 0.5e6 < mean_photons < 2e6, f"Mean photons {mean_photons:.2e} outside expected range"
print("✓ Photon statistics validation passed")
```

### Stage 4: Low Photon Regime Test
**Goal**: Verify extreme low photon case (1e3) produces valid data

```bash
# Test 4.1: Generate low photon dataset
python scripts/simulation/simulate_and_save.py \
    --input-file datasets/fly/fly001.npz \
    --output-file validation/test_1e3_extreme.npz \
    --n-images 1000 \
    --n-photons 1e3 \
    --seed 42

# Test 4.2: Validate low photon data
python -c "
import numpy as np
data = np.load('validation/test_1e3_extreme.npz')
intensity = data['diff3d'] ** 2
mean_photons = np.mean(intensity.sum(axis=(1,2)))
print(f'Low photon mean: {mean_photons:.2e}')
assert 500 < mean_photons < 2000, f'Low photon mean {mean_photons:.2e} out of range'
print('✓ Low photon validation passed')
"
```

### Stage 5: Reproducibility Test
**Goal**: Verify seed parameter ensures reproducible results

```bash
# Test 5.1: Generate two datasets with same seed
python scripts/simulation/simulate_and_save.py \
    --input-file datasets/fly/fly001.npz \
    --output-file validation/test_seed_1.npz \
    --n-images 100 \
    --n-photons 1e6 \
    --seed 123

python scripts/simulation/simulate_and_save.py \
    --input-file datasets/fly/fly001.npz \
    --output-file validation/test_seed_2.npz \
    --n-images 100 \
    --n-photons 1e6 \
    --seed 123

# Test 5.2: Verify datasets are identical
python -c "
import numpy as np
d1 = np.load('validation/test_seed_1.npz')
d2 = np.load('validation/test_seed_2.npz')
assert np.allclose(d1['diff3d'], d2['diff3d']), 'Diffraction patterns not identical'
assert np.allclose(d1['xcoords'], d2['xcoords']), 'Coordinates not identical'
print('✓ Reproducibility validation passed')
"
```

### Stage 6: Model Training Validation
**Goal**: Verify models can train on generated data

```bash
# Test 6.1: Train PINN on synthetic data
ptycho_train \
    --train_data validation/test_1e6_basic.npz \
    --test_data validation/test_1e6_basic.npz \
    --n_images 500 \
    --nepochs 5 \
    --model_type pinn \
    --output_dir validation/pinn_test \
    --quiet

# Test 6.2: Train Baseline on synthetic data  
ptycho_train \
    --train_data validation/test_1e6_basic.npz \
    --test_data validation/test_1e6_basic.npz \
    --n_images 500 \
    --nepochs 5 \
    --model_type supervised \
    --output_dir validation/baseline_test \
    --quiet

# Test 6.3: Verify training completed
python -c "
from pathlib import Path
assert (Path('validation/pinn_test/wts.h5.zip').exists()), 'PINN weights not saved'
assert (Path('validation/baseline_test/wts.h5.zip').exists()), 'Baseline weights not saved'
print('✓ Model training validation passed')
"
```

### Stage 7: Comparison with Known Baseline
**Goal**: Compare synthetic data quality against existing datasets

```bash
# Test 7.1: Generate high-quality reference (1e9 photons)
python scripts/simulation/simulate_and_save.py \
    --input-file datasets/fly/fly001.npz \
    --output-file validation/test_1e9_reference.npz \
    --n-images 1000 \
    --n-photons 1e9 \
    --seed 42

# Test 7.2: Visual comparison
python scripts/tools/visualize_dataset.py \
    --input-file validation/test_1e3_extreme.npz \
    --output-file validation/viz_1e3.png \
    --title "1e3 photons (extreme noise)"

python scripts/tools/visualize_dataset.py \
    --input-file validation/test_1e6_basic.npz \
    --output-file validation/viz_1e6.png \
    --title "1e6 photons (moderate noise)"

python scripts/tools/visualize_dataset.py \
    --input-file validation/test_1e9_reference.npz \
    --output-file validation/viz_1e9.png \
    --title "1e9 photons (low noise)"

echo "Check visualizations in validation/ directory for quality assessment"
```

## Validation Script

Create `scripts/validation/validate_simulate_and_save.sh`:

```bash
#!/bin/bash
set -e

VALIDATION_DIR="validation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$VALIDATION_DIR"
cd "$VALIDATION_DIR"

echo "=== Stage 1: Basic Functionality ==="
python ../scripts/simulation/simulate_and_save.py \
    --input-file ../datasets/fly/fly001.npz \
    --output-file test_1e6_basic.npz \
    --n-images 1000 \
    --n-photons 1e6 \
    --seed 42

echo "=== Stage 2: Data Contract ==="
python -c "
import numpy as np
import sys
data = np.load('test_1e6_basic.npz')
# ... [validation code from Stage 2]
"

echo "=== Stage 3: Photon Statistics ==="
# ... [continue with all stages]

echo "=== All Validations Passed ==="
echo "Results saved in $VALIDATION_DIR"
```

## Success Criteria

All stages must pass before proceeding with full grid study:

- [ ] Stage 1: Script runs without errors
- [ ] Stage 2: Output matches data contract
- [ ] Stage 3: Photon statistics correct (±50% of target)
- [ ] Stage 4: Low photon regime produces valid data
- [ ] Stage 5: Seed ensures reproducibility
- [ ] Stage 6: Both models train successfully
- [ ] Stage 7: Visual quality matches expectations

## Contingency Plans

### If Stage 3 Fails (Wrong Photon Count)
- Check if photon parameter is properly passed to diffsim layer
- Verify Poisson noise application in illuminate_and_diffract
- May need to adjust expectations based on object/probe overlap

### If Stage 4 Fails (Low Photon Invalid)
- Check for numerical issues at extreme low counts
- May need minimum photon threshold (e.g., 1e2)
- Consider log-scaling issues in visualization

### If Stage 6 Fails (Training Issues)
- Check if Y patches are generated correctly
- Verify data normalization in loader.py
- May need to add explicit patch generation step

## Next Steps

1. Run validation script: `./scripts/validation/validate_simulate_and_save.sh`
2. Review all outputs in validation directory
3. If all passes, proceed with full grid study
4. If failures, debug specific stage and document fixes
5. Update this plan with findings

## Expected Timeline

- Validation execution: 30-60 minutes
- Review and debugging: 1-2 hours if issues found
- Documentation: 30 minutes

Total: 2-3 hours before grid study can proceed with confidence