# Plan: Fix Probe Parameterization Study Workflow

**Objective:** Correct the probe parameterization study to properly test how different probe characteristics in training data affect model performance.

## Current Issues

1. **Probe Generation Order**: The script extracts probes from the dataset instead of using pre-created probes as simulation inputs
2. **Simulation Logic**: The script uses the same base dataset for all experiments instead of creating separate simulations
3. **Probe Confusion**: The "default" and "hybrid" probes are being generated after the fact, not used as inputs

## Proposed Solution

### Phase 1: Create Probe Generation Script
**Goal:** Create a standalone script to generate the probe pair before any simulation

**Tasks:**
1. Create `scripts/studies/prepare_probe_study.py`:
   - Generate default probe (experimental amplitude + flat phase)
   - Generate hybrid probe (experimental amplitude + aberrated phase)
   - Save both probes to study directory
   - Visualize probes for verification

### Phase 2: Modify Study Orchestration Script
**Goal:** Fix `run_2x2_probe_study.sh` to use pre-created probes correctly

**Changes Required:**
1. Remove probe extraction logic from dataset
2. Remove probe generation using `create_hybrid_probe.py`
3. Add probe preparation step that runs the new script
4. Modify simulation commands to use correct probe for each experiment:
   - gs1_default: Use default_probe.npy
   - gs1_hybrid: Use hybrid_probe.npy
   - gs2_default: Use default_probe.npy
   - gs2_hybrid: Use hybrid_probe.npy

### Phase 3: Create Simplified Study Runner
**Goal:** Create a cleaner version that demonstrates the correct workflow

**New Script:** `scripts/studies/run_probe_study_corrected.sh`
- Clear separation of phases
- Explicit probe usage
- Better documentation

## Implementation Details

### 1. Probe Preparation Script Structure
```python
# prepare_probe_study.py
def create_probe_pair(amplitude_source, phase_source, output_dir):
    """
    Create default and hybrid probes for study.
    
    Default: amplitude_source amplitude + flat phase
    Hybrid: amplitude_source amplitude + phase_source phase
    """
    # Load sources
    # Create default probe (flat phase)
    # Create hybrid probe (aberrated phase)
    # Save both probes
    # Generate visualization
```

### 2. Modified Simulation Flow
```bash
# For each experiment (gs1_default, gs1_hybrid, etc.)
PROBE_FILE="${OUTPUT_DIR}/${PROBE_TYPE}_probe.npy"
OBJECT_SOURCE="${DATASET}"  # Use object from base dataset

# Run simulation with specific probe
python simulate_and_save.py \
    --input-file "${OBJECT_SOURCE}" \
    --probe-file "${PROBE_FILE}" \
    --output-file "${EXP_DIR}/simulated_data.npz" \
    --n-images "${N_TRAIN}" \
    --gridsize "${GRIDSIZE}"
```

### 3. Validation Steps
- Verify each simulated dataset contains the correct probe
- Check that diffraction patterns differ between experiments
- Ensure probe characteristics are preserved through simulation

## Success Criteria

1. **Correct Probe Usage**: Each experiment uses its designated probe for simulation
2. **Physical Consistency**: Diffraction patterns match the probe used
3. **Clear Workflow**: Probe creation → Simulation → Training → Evaluation
4. **Reproducibility**: Study can be rerun with different probe sources

## Risk Mitigation

1. **Backward Compatibility**: Keep original script, create new corrected version
2. **Validation**: Add checks to verify probe usage at each step
3. **Documentation**: Clear inline comments explaining the workflow

## Timeline

- Phase 1: 30 minutes (create probe preparation script)
- Phase 2: 45 minutes (modify orchestration script)
- Phase 3: 30 minutes (create simplified runner)
- Testing: 30 minutes (verify with quick test)

Total: ~2.5 hours

## Next Steps

1. Implement Phase 1 - Create probe preparation script
2. Test probe generation with different sources
3. Proceed with Phase 2 modifications
4. Run quick test to validate workflow
5. Run full study with corrected workflow