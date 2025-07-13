#!/bin/bash
#
# This script orchestrates a multi-step ptychography data preparation, simulation,
# and downsampling pipeline.
#
#   Step 0: Pads object/probe to have even dimensions if they are odd.
#   Step 1 (NEW): Canonicalizes the NPZ format (renames, converts dtypes).
#   Step 2: Upsamples the object and probe.
#   Step 3: A no-op placeholder for smoothing the probe.
#   Step 4: Applies a light Gaussian blur to the upsampled object.
#   Step 5: Runs a NEW simulation using the prepared object/probe.
#   Step 6: Downsamples the simulated data back to the original dimensions.
#   Step 7 (NEW): Splits the final dataset into training and testing sets.
#
set -e # Exit immediately if any command fails

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root directory
cd "$PROJECT_ROOT"

# --- CONFIGURABLE PARAMETERS ---
# The starting point: can now have odd or even dimensions.
ORIGINAL_NPZ="tike_outputs/fly001/fly001_reconstructed.npz"

# --- NEW: Train/Test Split Configuration ---
SPLIT_FRACTION="0.5" 
SPLIT_AXIS="y"       # 'y' for a top/bottom split, 'x' for left/right

# --- FILE PATHS (AUTOMATICALLY DERIVED) ---
BASE_NAME=$(basename "${ORIGINAL_NPZ%.npz}")

PADDED_DIR="tike_outputs/${BASE_NAME}_padded"
PADDED_NPZ="$PADDED_DIR/${BASE_NAME}_padded.npz"

TRANSPOSED_DIR="tike_outputs/${BASE_NAME}_transposed"
TRANSPOSED_NPZ="$TRANSPOSED_DIR/${BASE_NAME}_transposed.npz"

INTERP_DIR="tike_outputs/${BASE_NAME}_interpolated"
INTERP_NPZ="$INTERP_DIR/${BASE_NAME}_interpolated_2x.npz"

SMOOTH_PROBE_DIR="tike_outputs/${BASE_NAME}_interp_smooth_probe"
SMOOTH_PROBE_NPZ="$SMOOTH_PROBE_DIR/${BASE_NAME}_interp_smooth_probe.npz"

PREPARED_INPUT_DIR="tike_outputs/${BASE_NAME}_final_prepared"
PREPARED_INPUT_NPZ="$PREPARED_INPUT_DIR/${BASE_NAME}_interp_smooth_both.npz"

SIMULATED_DIR="tike_outputs/${BASE_NAME}_final_simulated"
SIMULATED_NPZ="$SIMULATED_DIR/${BASE_NAME}_final_simulated_data.npz"

DOWNSAMPLED_DIR="tike_outputs/${BASE_NAME}_final_downsampled"
DOWNSAMPLED_NPZ="$DOWNSAMPLED_DIR/${BASE_NAME}_final_downsampled_data.npz"

# NEW: Final output directory for the split files
FINAL_DATA_DIR="datasets/${BASE_NAME}_prepared"


# --- SIMULATION PARAMETERS (for Step 5) ---
# Generate 20,000 images total (will be split into 10,000 train + 10,000 test)
SIM_IMAGES=20000
SIM_PHOTONS=1e9
SIM_SEED=42


# ==============================================================================
# STEP 0: Pad Object and Probe to Even Dimensions
# ==============================================================================
echo "--- Step 0: Padding to Even Dimensions ---"
mkdir -p "$PADDED_DIR"
python scripts/tools/pad_to_even_tool.py \
    "$ORIGINAL_NPZ" \
    "$PADDED_NPZ"

# ==============================================================================
# STEP 1 (NEW): Canonicalize NPZ Format
# ==============================================================================
echo -e "\n--- Step 1: Canonicalizing NPZ Format ---"
mkdir -p "$TRANSPOSED_DIR"
python scripts/tools/transpose_rename_convert_tool.py \
    "$PADDED_NPZ" \
    "$TRANSPOSED_NPZ"

# ==============================================================================
# STEP 2: Interpolate the Canonical Data (upsample object and probe)
# ==============================================================================
echo -e "\n--- Step 2: Interpolating Data (2x Zoom) ---"
mkdir -p "$INTERP_DIR"
python scripts/tools/prepare_data_tool.py \
    "$TRANSPOSED_NPZ" \
    "$INTERP_NPZ" \
    --interpolate --zoom-factor 2.0

# ==============================================================================
# STEP 3: Smooth the probe in the upsampled file (No-Op)
# ==============================================================================
echo -e "\n--- Step 3: Smoothing Probe (No-Op, sigma=0) ---"
mkdir -p "$SMOOTH_PROBE_DIR"
python scripts/tools/prepare_data_tool.py \
    "$INTERP_NPZ" \
    "$SMOOTH_PROBE_NPZ" \
    --smooth --target probe --sigma 0.

# ==============================================================================
# STEP 4: Smooth the object in the probe-smoothed file
# ==============================================================================
echo -e "\n--- Step 4: Smoothing Object (sigma=0.5) ---"
mkdir -p "$PREPARED_INPUT_DIR"
python scripts/tools/prepare_data_tool.py \
    "$SMOOTH_PROBE_NPZ" \
    "$PREPARED_INPUT_NPZ" \
    --smooth --target object --sigma 0.5

# ==============================================================================
# STEP 5: Simulate New Diffraction Data from the Prepared Object/Probe
# ==============================================================================
echo -e "\n--- Step 5: Simulating New Diffraction Data ---"
mkdir -p "$SIMULATED_DIR"
python scripts/simulation/simulate_and_save.py \
    --input-file "$PREPARED_INPUT_NPZ" \
    --output-file "$SIMULATED_NPZ" \
    --n-images "$SIM_IMAGES" \
    --n-photons "$SIM_PHOTONS" \
    --seed "$SIM_SEED" \
    --visualize

# ==============================================================================
# STEP 6: Downsample the Simulated Data
# ==============================================================================
echo -e "\n--- Step 6: Downsampling Simulated Data ---"
mkdir -p "$DOWNSAMPLED_DIR"
python scripts/tools/downsample_data_tool.py \
    "$SIMULATED_NPZ" \
    "$DOWNSAMPLED_NPZ" \
    --crop-factor 2 \
    --bin-factor 2

# ==============================================================================
# STEP 7 (NEW): Split the Final Dataset into Train and Test Sets
# ==============================================================================
echo -e "\n--- Step 7: Splitting Final Dataset into Train/Test Sets ---"
mkdir -p "$FINAL_DATA_DIR"
python scripts/tools/split_dataset_tool.py \
    "$DOWNSAMPLED_NPZ" \
    "$FINAL_DATA_DIR" \
    --split-fraction "$SPLIT_FRACTION" \
    --split-axis "$SPLIT_AXIS"


# --- Final Confirmation ---
echo -e "\n--- Workflow Complete ---"
echo "The final, prepared, and split datasets are available in:"
echo "$FINAL_DATA_DIR/"
echo "  - ${BASE_NAME}_final_downsampled_data_train.npz"
echo "  - ${BASE_NAME}_final_downsampled_data_test.npz"
