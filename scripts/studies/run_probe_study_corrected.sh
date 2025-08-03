#!/bin/bash
#
# run_probe_study_corrected.sh - Corrected 2x2 probe parameterization study
#
# This script demonstrates the correct workflow for probe parameterization studies:
# 1. Create probe pair (default and hybrid) FIRST
# 2. Run separate simulations with each probe
# 3. Train models on each simulated dataset
# 4. Compare results
#
# The key insight: each probe creates its own dataset with physically consistent
# diffraction patterns. We're testing how probe characteristics in the training
# data affect model performance.
#
# Usage:
#   ./scripts/studies/run_probe_study_corrected.sh --output-dir DIRECTORY [OPTIONS]
#
# Required Options:
#   --output-dir DIRECTORY     Output directory for study results
#
# Optional Options:
#   --amplitude-source PATH    Source for probe amplitude (default: synthetic lines)
#   --phase-source PATH        Source for aberrated phase (default: datasets/fly/fly001_transposed.npz)
#   --object-source PATH       Source for object/coordinates (default: synthetic lines)
#   --quick-test              Run in quick test mode (fewer images, epochs)
#   --gridsize N              Gridsize to test (default: 1)
#   --skip-completed          Skip already completed steps
#   --help                    Show this help message
#

set -e  # Exit on error

# Default values
OUTPUT_DIR=""
AMPLITUDE_SOURCE=""
PHASE_SOURCE="datasets/fly/fly001_transposed.npz"
OBJECT_SOURCE=""
QUICK_TEST=false
GRIDSIZE=1
SKIP_COMPLETED=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --amplitude-source)
            AMPLITUDE_SOURCE="$2"
            shift 2
            ;;
        --phase-source)
            PHASE_SOURCE="$2"
            shift 2
            ;;
        --object-source)
            OBJECT_SOURCE="$2"
            shift 2
            ;;
        --quick-test)
            QUICK_TEST=true
            shift
            ;;
        --gridsize)
            GRIDSIZE="$2"
            shift 2
            ;;
        --skip-completed)
            SKIP_COMPLETED=true
            shift
            ;;
        --help)
            grep "^#" "$0" | grep -v "^#!/bin/bash" | sed 's/^# *//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: --output-dir is required"
    exit 1
fi

# Set parameters based on mode
if [ "$QUICK_TEST" = true ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_QUICK_TEST"
    N_TRAIN=512
    N_TEST=128
    EPOCHS=5
    echo "Running in quick test mode"
else
    N_TRAIN=5000
    N_TEST=1000
    EPOCHS=50
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Corrected Probe Parameterization Study"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Phase source: $PHASE_SOURCE"
echo "Gridsize: $GRIDSIZE"
echo "Quick test mode: $QUICK_TEST"
echo "Parameters: N_TRAIN=$N_TRAIN, N_TEST=$N_TEST, EPOCHS=$EPOCHS"
echo "=========================================="

# Function to check if step is completed
check_completed() {
    local marker_file="$1"
    if [ "$SKIP_COMPLETED" = true ] && [ -f "$marker_file" ]; then
        echo "  [SKIP] Step already completed (found $marker_file)"
        return 0
    else
        return 1
    fi
}

# Function to run command with logging
run_command() {
    local description="$1"
    local command="$2"
    local log_file="$3"
    
    echo "  $description"
    echo "  Command: $command"
    
    if [ -n "$log_file" ]; then
        echo "  Output will be saved to: $log_file"
        if ! eval "$command" > "$log_file" 2>&1; then
            echo "  ERROR: Command failed. Check $log_file for details"
            tail -20 "$log_file"
            exit 1
        fi
    else
        if ! eval "$command"; then
            echo "  ERROR: Command failed"
            exit 1
        fi
    fi
    
    echo "  Completed successfully"
}

# Step 0: Generate synthetic object if needed
if [ -z "$AMPLITUDE_SOURCE" ] || [ -z "$OBJECT_SOURCE" ]; then
    echo ""
    echo "[Step 0] Generating synthetic lines dataset..."
    
    SYNTHETIC_DIR="${OUTPUT_DIR}/synthetic_input"
    
    if check_completed "${SYNTHETIC_DIR}/simulated_data.npz"; then
        AMPLITUDE_SOURCE="${SYNTHETIC_DIR}/simulated_data.npz"
        OBJECT_SOURCE="${SYNTHETIC_DIR}/simulated_data.npz"
    else
        mkdir -p "${SYNTHETIC_DIR}"
        run_command \
            "Creating synthetic lines object and probe" \
            "python scripts/simulation/run_with_synthetic_lines.py --output-dir ${SYNTHETIC_DIR} --n-images 100" \
            "${SYNTHETIC_DIR}/generation.log"
        
        AMPLITUDE_SOURCE="${SYNTHETIC_DIR}/simulated_data.npz"
        OBJECT_SOURCE="${SYNTHETIC_DIR}/simulated_data.npz"
        
        echo "  Using synthetic data from: $SYNTHETIC_DIR"
    fi
fi

# Step 1: Prepare probe pair
echo ""
echo "[Step 1] Preparing probe pair..."

if check_completed "${OUTPUT_DIR}/default_probe.npy" && check_completed "${OUTPUT_DIR}/hybrid_probe.npy"; then
    echo "  Probes already prepared"
else
    run_command \
        "Creating default (flat phase) and hybrid (aberrated phase) probes" \
        "python scripts/studies/prepare_probe_study.py \
            --amplitude-source '${AMPLITUDE_SOURCE}' \
            --phase-source '${PHASE_SOURCE}' \
            --output-dir '${OUTPUT_DIR}' \
            --visualize" \
        "${OUTPUT_DIR}/probe_preparation.log"
fi

# Load and display probe statistics
echo "  Probe statistics:"
python -c "
import numpy as np
default = np.load('${OUTPUT_DIR}/default_probe.npy')
hybrid = np.load('${OUTPUT_DIR}/hybrid_probe.npy')
print(f'    Default: amp mean={np.abs(default).mean():.4f}, phase std={np.angle(default).std():.4f}')
print(f'    Hybrid:  amp mean={np.abs(hybrid).mean():.4f}, phase std={np.angle(hybrid).std():.4f}')
"

# Step 2: Run experiments
echo ""
echo "[Step 2] Running experiments..."

# Define experiments
declare -a EXPERIMENTS=("default" "hybrid")

for PROBE_TYPE in "${EXPERIMENTS[@]}"; do
    EXP_NAME="gs${GRIDSIZE}_${PROBE_TYPE}"
    EXP_DIR="${OUTPUT_DIR}/${EXP_NAME}"
    
    echo ""
    echo "  Experiment: $EXP_NAME"
    echo "  ========================"
    
    mkdir -p "$EXP_DIR"
    
    # Step 2a: Simulate with specific probe
    echo "  [2a] Running simulation..."
    if check_completed "${EXP_DIR}/simulated_data.npz"; then
        echo "    Simulation already completed"
    else
        PROBE_FILE="${OUTPUT_DIR}/${PROBE_TYPE}_probe.npy"
        
        run_command \
            "Simulating with ${PROBE_TYPE} probe" \
            "python scripts/simulation/simulate_and_save.py \
                --input-file '${OBJECT_SOURCE}' \
                --probe-file '${PROBE_FILE}' \
                --output-file '${EXP_DIR}/simulated_data.npz' \
                --n-images $N_TRAIN \
                --gridsize $GRIDSIZE" \
            "${EXP_DIR}/simulation.log"
        
        # Verify the probe in the dataset
        echo "    Verifying probe in simulated dataset:"
        python -c "
import numpy as np
data = np.load('${EXP_DIR}/simulated_data.npz')
probe = data['probeGuess']
print(f'      Shape: {probe.shape}, Phase std: {np.angle(probe).std():.4f}')
"
    fi
    
    # Step 2b: Train model
    echo "  [2b] Training model..."
    if check_completed "${EXP_DIR}/model/wts.h5.zip"; then
        echo "    Model already trained"
    else
        run_command \
            "Training PtychoPINN model" \
            "ptycho_train \
                --train_data_file '${EXP_DIR}/simulated_data.npz' \
                --output_dir '${EXP_DIR}/model' \
                --nepochs $EPOCHS \
                --batch_size 32 \
                --model_type pinn" \
            "${EXP_DIR}/training.log"
    fi
    
    # Step 2c: Create test subset and evaluate
    echo "  [2c] Evaluating model..."
    if check_completed "${EXP_DIR}/evaluation/comparison_metrics.csv"; then
        echo "    Evaluation already completed"
    else
        # Create test subset
        TEST_FILE="${EXP_DIR}/test_data.npz"
        if [ ! -f "$TEST_FILE" ]; then
            echo "    Creating test subset with $N_TEST images..."
            # Use Python to create a subset
            python -c "
import numpy as np
data = np.load('${EXP_DIR}/simulated_data.npz')
n_test = min($N_TEST, len(data['xcoords']))
test_data = {}
for key in data.keys():
    if hasattr(data[key], 'shape') and len(data[key].shape) > 0 and data[key].shape[0] > n_test:
        test_data[key] = data[key][-n_test:]  # Take last n_test samples
    else:
        test_data[key] = data[key]
np.savez('$TEST_FILE', **test_data)
print(f'Created test subset with {n_test} images')
"
        fi
        
        run_command \
            "Running model evaluation" \
            "python scripts/compare_models.py \
                --pinn_dir '${EXP_DIR}/model' \
                --test_data '$TEST_FILE' \
                --output_dir '${EXP_DIR}/evaluation' \
                --n-test-images $N_TEST" \
            "${EXP_DIR}/evaluation.log"
    fi
    
    echo "  Experiment $EXP_NAME completed"
done

# Step 3: Generate results summary
echo ""
echo "[Step 3] Generating results summary..."

# Aggregate results
python scripts/studies/aggregate_2x2_results.py "$OUTPUT_DIR"

# Generate visualizations
python scripts/studies/generate_2x2_visualization.py "$OUTPUT_DIR"

# Create final report
cat > "${OUTPUT_DIR}/study_report.md" << EOF
# Probe Parameterization Study Results

**Date:** $(date)
**Study Type:** Gridsize ${GRIDSIZE} comparison
**Training images:** $N_TRAIN
**Test images:** $N_TEST
**Epochs:** $EPOCHS

## Summary

This study compared model performance when trained on data simulated with:
- **Default probe**: Idealized probe with flat phase
- **Hybrid probe**: Same amplitude but with experimental phase aberrations

## Results

$(cat "${OUTPUT_DIR}/summary_table.txt")

## Visualizations

- Probe comparison: ![Probes](probe_pair_visualization.png)
- Reconstruction comparison: ![Reconstructions](2x2_reconstruction_comparison.png)

## Conclusion

The results show how phase aberrations in the training data affect the model's ability to learn accurate reconstructions.
EOF

echo ""
echo "=========================================="
echo "Study Complete!"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR"
echo "Summary report: ${OUTPUT_DIR}/study_report.md"
echo ""