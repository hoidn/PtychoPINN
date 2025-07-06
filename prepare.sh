#!/bin/bash
#
# This script orchestrates a multi-step ptychography data preparation, simulation,
# and downsampling pipeline.
#
#   Step 0: Pads object/probe to have even dimensions if they are odd.
#   Step 1: Upsamples the object and probe.
#   Step 2: A no-op placeholder for smoothing the probe.
#   Step 3: Applies a light Gaussian blur to the upsampled object.
#   Step 4: Runs a NEW simulation using the prepared object/probe.
#   Step 5: Downsamples the simulated data back to the original dimensions.
#
set -e # Exit immediately if any command fails

# --- FILE PATHS ---

# The starting point: can now have odd or even dimensions.
ORIGINAL_NPZ="tike_outputs/fly001/fly001_reconstructed.npz"

# NEW: Intermediate file after padding to even dimensions.
PADDED_DIR="tike_outputs/fly001_padded"
PADDED_NPZ="$PADDED_DIR/fly001_padded.npz"

# Intermediate file after upsampling (interpolation).
INTERP_DIR="tike_outputs/fly001_interpolated"
INTERP_NPZ="$INTERP_DIR/fly001_interpolated_2x.npz"

# Intermediate file after the (no-op) probe smoothing step.
SMOOTH_PROBE_DIR="tike_outputs/fly001_interp_smooth_probe"
SMOOTH_PROBE_NPZ="$SMOOTH_PROBE_DIR/fly001_interp_smooth_probe.npz"

# The result of the preparation steps. This file is the input for the simulation.
PREPARED_INPUT_DIR="tike_outputs/fly001_final_prepared"
PREPARED_INPUT_NPZ="$PREPARED_INPUT_DIR/fly001_interp_smooth_both.npz"

# The high-resolution simulated data from Step 4.
SIMULATED_DIR="tike_outputs/fly001_final_simulated"
SIMULATED_NPZ="$SIMULATED_DIR/fly001_final_simulated_data.npz"

# The final output of the entire pipeline, downsampled to original dimensions.
DOWNSAMPLED_DIR="tike_outputs/fly001_final_downsampled"
DOWNSAMPLED_NPZ="$DOWNSAMPLED_DIR/fly001_final_downsampled_data.npz"


# --- SIMULATION PARAMETERS (for Step 4) ---
SIM_IMAGES=2000
SIM_PHOTONS=1e9
SIM_SEED=42


# ==============================================================================
# STEP 0: Pad Object and Probe to Even Dimensions
# Ensures all subsequent steps work with even-sized arrays.
# ==============================================================================
echo "--- Step 0: Padding to Even Dimensions ---"
mkdir -p "$PADDED_DIR"
python scripts/tools/pad_to_even_tool.py \
    "$ORIGINAL_NPZ" \
    "$PADDED_NPZ"


# ==============================================================================
# STEP 1: Interpolate the Padded Data (upsample object and probe)
# ==============================================================================
echo -e "\n--- Step 1: Interpolating Data (2x Zoom) ---"
mkdir -p "$INTERP_DIR"
python scripts/tools/prepare_data_tool.py \
    "$PADDED_NPZ" \
    "$INTERP_NPZ" \
    --interpolate --zoom-factor 2.0


# ==============================================================================
# STEP 2: Smooth the probe in the upsampled file (No-Op)
# ==============================================================================
echo -e "\n--- Step 2: Smoothing Probe (No-Op, sigma=0) ---"
mkdir -p "$SMOOTH_PROBE_DIR"
python scripts/tools/prepare_data_tool.py \
    "$INTERP_NPZ" \
    "$SMOOTH_PROBE_NPZ" \
    --smooth --target probe --sigma 0.


# ==============================================================================
# STEP 3: Smooth the object in the probe-smoothed file
# ==============================================================================
echo -e "\n--- Step 3: Smoothing Object (sigma=0.5) ---"
mkdir -p "$PREPARED_INPUT_DIR"
python scripts/tools/prepare_data_tool.py \
    "$SMOOTH_PROBE_NPZ" \
    "$PREPARED_INPUT_NPZ" \
    --smooth --target object --sigma 0.5


# ==============================================================================
# STEP 4: Simulate New Diffraction Data from the Prepared Object/Probe
# ==============================================================================
echo -e "\n--- Step 4: Simulating New Diffraction Data ---"
mkdir -p "$SIMULATED_DIR"
python scripts/simulation/simulate_and_save.py \
    --input-file "$PREPARED_INPUT_NPZ" \
    --output-file "$SIMULATED_NPZ" \
    --n-images "$SIM_IMAGES" \
    --n-photons "$SIM_PHOTONS" \
    --seed "$SIM_SEED" \
    --visualize


# ==============================================================================
# STEP 5: Downsample the Simulated Data
# ==============================================================================
echo -e "\n--- Step 5: Downsampling Simulated Data ---"
mkdir -p "$DOWNSAMPLED_DIR"
python scripts/tools/downsample_data_tool.py \
    "$SIMULATED_NPZ" \
    "$DOWNSAMPLED_NPZ" \
    --crop-factor 2 \
    --bin-factor 2


# --- Final Confirmation ---
echo -e "\n--- Workflow Complete ---"
echo "The final, downsampled, and self-consistent dataset is available at:"
echo "$DOWNSAMPLED_NPZ"
