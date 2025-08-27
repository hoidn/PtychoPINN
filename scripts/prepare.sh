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
# Usage:
#   ./prepare.sh [--sim-images N] [--sim-photons P]
#
# Options:
#   --sim-images N    Number of images to simulate (default: 35000)
#   --sim-photons P   Number of photons per image (default: 1e9)
#
# Examples:
#   ./prepare.sh                           # Use defaults (35000 images, 1e9 photons)
#   ./prepare.sh --sim-images 10000        # 10k images with default photons
#   ./prepare.sh --sim-photons 1e4         # Low-photon dataset
#   ./prepare.sh --sim-images 10000 --sim-photons 1e4  # Both custom
#
set -e # Exit immediately if any command fails

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root directory
cd "$PROJECT_ROOT"

# --- CONFIGURABLE PARAMETERS ---
# Default input file (can be overridden via --input-file)
DEFAULT_INPUT="tike_outputs/fly001/fly001_reconstructed.npz"
ORIGINAL_NPZ="${DEFAULT_INPUT}"

# Default output directory (can be overridden via --output-dir)
OUTPUT_DIR=""  # Empty means use default behavior

# --- NEW: Train/Test Split Configuration ---
SPLIT_FRACTION="0.5" 
SPLIT_AXIS="y"       # 'y' for a top/bottom split, 'x' for left/right

# --- SIMULATION PARAMETERS (for Step 5) ---
# Default values (can be overridden via command-line arguments)
SIM_IMAGES=35000    # Default: 35,000 images total (will be split into train/test)
SIM_PHOTONS=1e9     # Default: 1e9 photons (high flux)
SIM_SEED=42

# --- PARSE COMMAND-LINE ARGUMENTS ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --input-file)
            ORIGINAL_NPZ="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --sim-images)
            SIM_IMAGES="$2"
            shift 2
            ;;
        --sim-photons)
            SIM_PHOTONS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --input-file PATH   Input NPZ file (default: $DEFAULT_INPUT)"
            echo "  --output-dir DIR    Output directory for all results (default: automatic)"
            echo "  --sim-images N      Number of images to simulate (default: 35000)"
            echo "  --sim-photons P     Number of photons per image (default: 1e9)"
            echo ""
            echo "Examples:"
            echo "  $0                           # Use all defaults"
            echo "  $0 --input-file my_data.npz # Custom input"
            echo "  $0 --output-dir experiments/my_study --sim-photons 1e4"
            echo "  $0 --input-file synthetic.npz --output-dir studies/low_photon --sim-photons 1e4"
            echo ""
            echo "Output Structure:"
            echo "  When --output-dir is specified:"
            echo "    DIR/stages/         # Intermediate processing stages"
            echo "    DIR/dataset/        # Final train/test splits"
            echo "  Default (no --output-dir):"
            echo "    tike_outputs/*/     # Intermediate stages"
            echo "    datasets/*/         # Final train/test splits"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# --- FILE PATHS (AUTOMATICALLY DERIVED) ---
# This MUST come after argument parsing so OUTPUT_DIR is set correctly
BASE_NAME=$(basename "${ORIGINAL_NPZ%.npz}")

# Determine path structure based on whether output-dir was specified
if [[ -n "$OUTPUT_DIR" ]]; then
    # Custom output directory specified - use organized structure
    STAGES_DIR="${OUTPUT_DIR}/stages"
    FINAL_DATA_DIR="${OUTPUT_DIR}/dataset"
    
    PADDED_DIR="${STAGES_DIR}/01_padded"
    PADDED_NPZ="$PADDED_DIR/${BASE_NAME}_padded.npz"
    
    TRANSPOSED_DIR="${STAGES_DIR}/02_transposed"
    TRANSPOSED_NPZ="$TRANSPOSED_DIR/${BASE_NAME}_transposed.npz"
    
    INTERP_DIR="${STAGES_DIR}/03_interpolated"
    INTERP_NPZ="$INTERP_DIR/${BASE_NAME}_interpolated_2x.npz"
    
    SMOOTH_PROBE_DIR="${STAGES_DIR}/04_smooth_probe"
    SMOOTH_PROBE_NPZ="$SMOOTH_PROBE_DIR/${BASE_NAME}_interp_smooth_probe.npz"
    
    PREPARED_INPUT_DIR="${STAGES_DIR}/05_smooth_object"
    PREPARED_INPUT_NPZ="$PREPARED_INPUT_DIR/${BASE_NAME}_interp_smooth_both.npz"
    
    SIMULATED_DIR="${STAGES_DIR}/06_simulated"
    SIMULATED_NPZ="$SIMULATED_DIR/${BASE_NAME}_final_simulated_data.npz"
    
    DOWNSAMPLED_DIR="${STAGES_DIR}/07_downsampled"
    DOWNSAMPLED_NPZ="$DOWNSAMPLED_DIR/${BASE_NAME}_final_downsampled_data.npz"
else
    # Default behavior - maintain backward compatibility
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
    
    # Final output directory for the split files
    FINAL_DATA_DIR="datasets/${BASE_NAME}_prepared"
fi

# Display configuration
echo "=== Configuration ==="
echo "Input file: $ORIGINAL_NPZ"
if [[ -n "$OUTPUT_DIR" ]]; then
    echo "Output directory: $OUTPUT_DIR"
else
    echo "Output directory: (default structure)"
fi
echo "Images to simulate: $SIM_IMAGES"
echo "Photons per image: $SIM_PHOTONS"
echo "Random seed: $SIM_SEED"
echo "====================="
echo ""


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

# If using custom output directory, rename to simple names
if [[ -n "$OUTPUT_DIR" ]]; then
    # Move the generated files to simpler names
    mv "$FINAL_DATA_DIR/${BASE_NAME}_final_downsampled_data_train.npz" "$FINAL_DATA_DIR/train.npz" 2>/dev/null || true
    mv "$FINAL_DATA_DIR/${BASE_NAME}_final_downsampled_data_test.npz" "$FINAL_DATA_DIR/test.npz" 2>/dev/null || true
fi


# --- Final Confirmation ---
echo -e "\n--- Workflow Complete ---"
echo "The final, prepared, and split datasets are available in:"
echo "$FINAL_DATA_DIR/"
if [[ -n "$OUTPUT_DIR" ]]; then
    echo "  - train.npz"
    echo "  - test.npz"
else
    echo "  - ${BASE_NAME}_final_downsampled_data_train.npz"
    echo "  - ${BASE_NAME}_final_downsampled_data_test.npz"
fi
