# Input file
ORIGINAL_NPZ="tike_outputs/fly001/fly001_reconstructed.npz"

# Intermediate file after interpolation
INTERP_DIR="tike_outputs/fly001_interpolated"
INTERP_NPZ="$INTERP_DIR/fly001_interpolated_2x.npz"

# Intermediate file after smoothing the probe
SMOOTH_PROBE_DIR="tike_outputs/fly001_interp_smooth_probe"
SMOOTH_PROBE_NPZ="$SMOOTH_PROBE_DIR/fly001_interp_smooth_probe.npz"

# Final output file after smoothing the object
FINAL_DIR="tike_outputs/fly001_final_prepared"
FINAL_NPZ="$FINAL_DIR/fly001_interp_smooth_both.npz"


# --- Step 1: Interpolate the original data ---
echo "--- Step 1: Interpolating Data ---"
mkdir -p "$INTERP_DIR"
python scripts/tools/prepare_data_tool.py \
    "$ORIGINAL_NPZ" \
    "$INTERP_NPZ" \
    --interpolate --zoom-factor 2.0


# --- Step 2: Smooth the probe in the upsampled file ---
echo -e "\n--- Step 2: Smoothing Probe ---"
mkdir -p "$SMOOTH_PROBE_DIR"
python scripts/tools/prepare_data_tool.py \
    "$INTERP_NPZ" \
    "$SMOOTH_PROBE_NPZ" \
    --smooth --target probe --sigma 0.


# --- Step 3: Smooth the object in the probe-smoothed file ---
echo -e "\n--- Step 3: Smoothing Object ---"
mkdir -p "$FINAL_DIR"
python scripts/tools/prepare_data_tool.py \
    "$SMOOTH_PROBE_NPZ" \
    "$FINAL_NPZ" \
    --smooth --target object --sigma 0.5


# --- Final Confirmation ---
echo -e "\n--- Workflow Complete ---"
echo "Final prepared data is available at: $FINAL_NPZ"
