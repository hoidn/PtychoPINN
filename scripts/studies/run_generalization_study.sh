#!/bin/bash
# run_generalization_study.sh - Orchestrate multiple training runs with varying dataset sizes
# to study model generalization and data efficiency
#
# Usage: ./scripts/studies/run_generalization_study.sh <train_data.npz> <test_data.npz> <output_dir> [options]
#
# Options:
#   --train-sizes "SIZE1 SIZE2 ..."   List of training set sizes (default: "512 1024 2048 4096")
#   --test-size NUM                   Fixed test set size (default: 1000)
#   --skip-if-exists                  Skip runs if output directory already exists

set -e  # Exit on error

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Change to project root directory
cd "$PROJECT_ROOT"

# Function to display usage
usage() {
    echo "Usage: $0 <train_data.npz> <test_data.npz> <output_dir> [options]"
    echo ""
    echo "Required arguments:"
    echo "  train_data.npz    Path to training dataset (should be large)"
    echo "  test_data.npz     Path to test dataset (fixed size)"
    echo "  output_dir        Base directory for all output files"
    echo ""
    echo "Options:"
    echo "  --train-sizes \"SIZE1 SIZE2 ...\"   List of training set sizes (default: \"512 1024 2048 4096\")"
    echo "  --test-size NUM                   Fixed test set size (default: 1000)"
    echo "  --skip-if-exists                  Skip runs if output directory already exists"
    echo ""
    echo "Example:"
    echo "  $0 datasets/fly001_prepared/fly001_train.npz datasets/fly001_prepared/fly001_test.npz generalization_study"
    echo "  $0 data/train.npz data/test.npz study_output --train-sizes \"256 512 1024 2048 4096 8192\""
    exit 1
}

# Check minimum arguments
if [ "$#" -lt 3 ]; then
    usage
fi

# Parse positional arguments
TRAIN_DATA="$1"
TEST_DATA="$2"
BASE_OUTPUT_DIR="$3"
shift 3

# Default values
TRAIN_SIZES="512 1024 2048 4096"
TEST_SIZE="1000"
SKIP_IF_EXISTS=false

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --train-sizes)
            TRAIN_SIZES="$2"
            shift 2
            ;;
        --test-size)
            TEST_SIZE="$2"
            shift 2
            ;;
        --skip-if-exists)
            SKIP_IF_EXISTS=true
            shift
            ;;
        *)
            echo "Error: Unknown option $1"
            usage
            ;;
    esac
done

# Verify input files exist
if [ ! -f "$TRAIN_DATA" ]; then
    echo "Error: Training data file not found: $TRAIN_DATA"
    exit 1
fi

if [ ! -f "$TEST_DATA" ]; then
    echo "Error: Test data file not found: $TEST_DATA"
    exit 1
fi

# Create base output directory
mkdir -p "$BASE_OUTPUT_DIR"

# Log file for the study
LOG_FILE="$BASE_OUTPUT_DIR/study_log.txt"

echo "============================================" | tee "$LOG_FILE"
echo "Model Generalization Study" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
echo "Train data: $TRAIN_DATA" | tee -a "$LOG_FILE"
echo "Test data: $TEST_DATA" | tee -a "$LOG_FILE"
echo "Base output directory: $BASE_OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Training sizes: $TRAIN_SIZES" | tee -a "$LOG_FILE"
echo "Test size: $TEST_SIZE" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Convert train sizes string to array
TRAIN_SIZES_ARRAY=($TRAIN_SIZES)

# Count total runs
TOTAL_RUNS=${#TRAIN_SIZES_ARRAY[@]}
CURRENT_RUN=0

# Run comparison for each training size
for TRAIN_SIZE in $TRAIN_SIZES; do
    CURRENT_RUN=$((CURRENT_RUN + 1))
    
    # Create output directory for this run
    RUN_OUTPUT_DIR="$BASE_OUTPUT_DIR/train_${TRAIN_SIZE}"
    
    echo "============================================" | tee -a "$LOG_FILE"
    echo "Run $CURRENT_RUN/$TOTAL_RUNS: Training with $TRAIN_SIZE images" | tee -a "$LOG_FILE"
    echo "============================================" | tee -a "$LOG_FILE"
    
    # Check if we should skip this run
    if [ "$SKIP_IF_EXISTS" = true ] && [ -d "$RUN_OUTPUT_DIR" ] && [ -f "$RUN_OUTPUT_DIR/comparison_metrics.csv" ]; then
        echo "Skipping: Output directory already exists with results" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
        continue
    fi
    
    # Run the comparison workflow
    echo "Running comparison with $TRAIN_SIZE training images..." | tee -a "$LOG_FILE"
    echo "Output directory: $RUN_OUTPUT_DIR" | tee -a "$LOG_FILE"
    
    # Execute run_comparison.sh with specified training and test sizes
    if ./scripts/run_comparison.sh \
        "$TRAIN_DATA" \
        "$TEST_DATA" \
        "$RUN_OUTPUT_DIR" \
        --n-train-images "$TRAIN_SIZE" \
        --n-test-images "$TEST_SIZE" 2>&1 | tee -a "$LOG_FILE"; then
        
        echo "Run completed successfully!" | tee -a "$LOG_FILE"
    else
        echo "ERROR: Run failed for training size $TRAIN_SIZE" | tee -a "$LOG_FILE"
        echo "Continuing with next size..." | tee -a "$LOG_FILE"
    fi
    
    echo "" | tee -a "$LOG_FILE"
done

echo "============================================" | tee -a "$LOG_FILE"
echo "All runs completed!" | tee -a "$LOG_FILE"
echo "Completed at: $(date)" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"

# Run aggregation and plotting
echo "" | tee -a "$LOG_FILE"
echo "Aggregating results and creating plots..." | tee -a "$LOG_FILE"

if python scripts/studies/aggregate_and_plot_results.py \
    "$BASE_OUTPUT_DIR" \
    --metric psnr \
    --part phase 2>&1 | tee -a "$LOG_FILE"; then
    
    echo "Results aggregation completed!" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "Final outputs:" | tee -a "$LOG_FILE"
    echo "  - Study log: $LOG_FILE" | tee -a "$LOG_FILE"
    echo "  - Aggregated results: $BASE_OUTPUT_DIR/results.csv" | tee -a "$LOG_FILE"
    echo "  - Comparison plot: $BASE_OUTPUT_DIR/generalization_plot.png" | tee -a "$LOG_FILE"
else
    echo "ERROR: Results aggregation failed" | tee -a "$LOG_FILE"
    exit 1
fi

echo "" | tee -a "$LOG_FILE"
echo "Study complete!" | tee -a "$LOG_FILE"