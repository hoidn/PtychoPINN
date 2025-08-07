# 2x2 Probe Parameterization Study Report - Corrected Workflow

**Date:** 2025-08-01  
**Initiative:** Probe Parameterization Study  
**Status:** Workflow Clarification

## Corrected Workflow Understanding

### The Correct Order of Operations

1. **Probe Creation (Pre-simulation)**
   - **Default Probe**: Idealized probe with disk-like amplitude and flat (zero) phase
   - **Hybrid Probe**: Same disk-like amplitude but with experimental phase aberrations

2. **Simulation Phase**
   - Each probe is used to simulate its own dataset:
     - `simulate_and_save.py --probe-file default_probe.npy` → creates dataset A
     - `simulate_and_save.py --probe-file hybrid_probe.npy` → creates dataset B
   - The probe used in simulation becomes part of the output NPZ file

3. **Training Phase**
   - Models are trained on each dataset separately
   - Each model learns to reconstruct based on the probe characteristics in its training data

4. **Evaluation**
   - Compare reconstruction quality between models trained on different probe conditions

### Key Insight

The probe parameterization study tests how different probe characteristics (specifically phase aberrations) in the training data affect the model's ability to learn accurate reconstructions. It's NOT about using different probes at inference time, but about training on data created with different probes.

### Implications

- Each experimental condition (e.g., gs1_default, gs1_hybrid) has its own simulated dataset
- The diffraction patterns in each dataset are physically consistent with the probe used to create them
- The study measures how probe aberrations in the training data impact learning and reconstruction quality

## Next Steps

1. Modify the `run_2x2_probe_study.sh` script to:
   - Skip probe extraction from dataset
   - Use pre-created probes as simulation inputs
   - Ensure each experiment uses its designated probe for simulation

2. Run the corrected workflow with proper probe usage

3. Analyze results to understand impact of training data probe characteristics on model performance