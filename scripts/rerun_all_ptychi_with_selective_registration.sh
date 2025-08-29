#!/bin/bash
# Rerun all existing pty-chi comparisons with selective registration
# This applies registration only to pty-chi while keeping PINN/Baseline unregistered

echo "=========================================="
echo "Rerunning Pty-chi Comparisons with Selective Registration"
echo "=========================================="
echo ""
echo "Strategy: "
echo "  - Pty-chi: WITH registration (improves SSIM from -0.079 to 0.334)"
echo "  - PtychoPINN: WITHOUT registration (better performance: 0.328 vs 0.604)"
echo "  - Baseline: WITHOUT registration"
echo ""

# Find all directories with pty-chi reconstructions
PTYCHI_DIRS=$(find . -type f -name "ptychi_reconstruction.npz" -path "*/ptychi_run/*" 2>/dev/null | sed 's|/ptychi_run/ptychi_reconstruction.npz||' | sort -u)

if [ -z "$PTYCHI_DIRS" ]; then
    echo "No pty-chi reconstructions found."
    exit 1
fi

echo "Found pty-chi reconstructions in:"
echo "$PTYCHI_DIRS" | sed 's/^/  - /'
echo ""

for trial_dir in $PTYCHI_DIRS; do
    echo "Processing: $trial_dir"
    
    # Extract paths
    PINN_DIR="$trial_dir/pinn_run"
    BASELINE_DIR="$trial_dir/baseline_run"
    PTYCHI_RECON="$trial_dir/ptychi_run/ptychi_reconstruction.npz"
    
    # Find the actual baseline model directory (may have timestamp)
    if [ ! -d "$BASELINE_DIR" ]; then
        BASELINE_DIR=$(find "$trial_dir" -type d -name "*baseline*" | head -1)
    fi
    
    # Check if all required files exist
    if [ ! -d "$PINN_DIR" ]; then
        echo "  ⚠️  Skipping - PtychoPINN model not found: $PINN_DIR"
        continue
    fi
    
    if [ ! -d "$BASELINE_DIR" ]; then
        echo "  ⚠️  Skipping - Baseline model not found"
        continue
    fi
    
    if [ ! -f "$PTYCHI_RECON" ]; then
        echo "  ⚠️  Skipping - Pty-chi reconstruction not found"
        continue
    fi
    
    # Determine test data location (look for common patterns)
    TEST_DATA=""
    if [ -f "prepare_1e4_photons_5k/dataset/test.npz" ]; then
        TEST_DATA="prepare_1e4_photons_5k/dataset/test.npz"
    elif [ -f "prepare_1e4_photons/dataset/test.npz" ]; then
        TEST_DATA="prepare_1e4_photons/dataset/test.npz"
    elif [ -f "datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_test.npz" ]; then
        TEST_DATA="datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_test.npz"
    else
        # Try to find it from the study config or log
        STUDY_DIR=$(dirname "$trial_dir")
        if [ -f "$STUDY_DIR/study_config.txt" ]; then
            TEST_DATA=$(grep "TEST_DATA=" "$STUDY_DIR/study_config.txt" | cut -d'=' -f2 | tr -d '"')
        fi
    fi
    
    if [ -z "$TEST_DATA" ] || [ ! -f "$TEST_DATA" ]; then
        echo "  ⚠️  Skipping - Could not find test data file"
        continue
    fi
    
    # Determine n_test_images from directory name or config
    N_TEST_IMAGES=""
    if [[ "$trial_dir" =~ train_([0-9]+) ]]; then
        N_TEST_IMAGES="${BASH_REMATCH[1]}"
    fi
    
    # Create output directory with _selective suffix
    OUTPUT_DIR="${trial_dir}_selective"
    
    echo "  ✓ Found all required files"
    echo "    - PtychoPINN: $PINN_DIR"
    echo "    - Baseline: $BASELINE_DIR"
    echo "    - Pty-chi: $PTYCHI_RECON"
    echo "    - Test data: $TEST_DATA"
    echo "    - N test images: ${N_TEST_IMAGES:-default}"
    echo "    - Output: $OUTPUT_DIR"
    
    # Run selective registration comparison
    CMD="python scripts/run_selective_registration_comparison.py"
    CMD="$CMD --pinn_dir '$PINN_DIR'"
    CMD="$CMD --baseline_dir '$BASELINE_DIR'"
    CMD="$CMD --test_data '$TEST_DATA'"
    CMD="$CMD --tike_recon_path '$PTYCHI_RECON'"
    CMD="$CMD --output_dir '$OUTPUT_DIR'"
    
    if [ -n "$N_TEST_IMAGES" ]; then
        CMD="$CMD --n_test_images $N_TEST_IMAGES"
    fi
    
    echo "  Running selective registration comparison..."
    eval $CMD
    
    if [ $? -eq 0 ]; then
        echo "  ✅ Success! Results saved to $OUTPUT_DIR"
        
        # Show improvement summary
        if [ -f "$OUTPUT_DIR/comparison_metrics.csv" ]; then
            echo "  Summary of results:"
            python3 -c "
import pandas as pd
df = pd.read_csv('$OUTPUT_DIR/comparison_metrics.csv')
for model in ['PtychoPINN', 'Baseline', 'Pty-chi (ePIE)']:
    ssim = df[(df['model'] == model) & (df['metric'] == 'ssim')]['amplitude'].values
    if len(ssim) > 0:
        print(f'    {model:15s}: SSIM = {ssim[0]:.3f}')
"
        fi
    else
        echo "  ❌ Failed to process $trial_dir"
    fi
    
    echo ""
done

echo "=========================================="
echo "Selective Registration Complete"
echo "=========================================="
echo ""
echo "Look for directories with '_selective' suffix for updated results."
echo "These results show each model with its optimal registration strategy:"
echo "  - Pty-chi WITH registration"
echo "  - PtychoPINN and Baseline WITHOUT registration"