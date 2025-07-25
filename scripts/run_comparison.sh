#!/bin/bash
# run_comparison.sh - Master script to train and compare PtychoPINN vs Baseline models
#
# This script orchestrates a complete comparison between PtychoPINN and baseline models:
# 1. Trains both models with identical hyperparameters
# 2. Runs comparison analysis with side-by-side visualizations
# 3. Generates quantitative metrics (MAE, MSE, PSNR, FRC)
#
# Usage: ./scripts/run_comparison.sh <train_data.npz> <test_data.npz> <output_dir> [options]
#
# New feature: Supports --n-train-images and --n-test-images to control dataset sizes
# for generalization studies and quick testing.
#
# Enhanced in Phase 1 of Model Generalization Study implementation.

set -e  # Exit on error

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root directory
cd "$PROJECT_ROOT"

# Parse command line arguments
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <train_data.npz> <test_data.npz> <output_dir> [pinn_phase_vmin] [pinn_phase_vmax] [baseline_phase_vmin] [baseline_phase_vmax] [--n-train-images N] [--n-test-images N]"
    echo ""
    echo "Required arguments:"
    echo "  train_data.npz    Path to training dataset"
    echo "  test_data.npz     Path to test dataset"
    echo "  output_dir        Directory for outputs"
    echo ""
    echo "Optional arguments:"
    echo "  pinn_phase_vmin   Minimum phase value for PtychoPINN visualization"
    echo "  pinn_phase_vmax   Maximum phase value for PtychoPINN visualization"
    echo "  baseline_phase_vmin  Minimum phase value for baseline visualization"
    echo "  baseline_phase_vmax  Maximum phase value for baseline visualization"
    echo "  --n-train-images N   Number of training images to use (overrides config)"
    echo "  --n-test-images N    Number of test images to use (overrides config)"
    echo ""
    echo "Examples:"
    echo "  $0 datasets/fly/fly001_transposed.npz datasets/fly/fly001_transposed.npz comparison_results"
    echo "  $0 datasets/fly/fly001_transposed.npz datasets/fly/fly001_transposed.npz comparison_results --n-train-images 512"
    echo "  $0 datasets/fly/fly001_transposed.npz datasets/fly/fly001_transposed.npz comparison_results -3.14159 3.14159 -3.14159 3.14159 --n-train-images 1024 --n-test-images 500"
    exit 1
fi

TRAIN_DATA="$1"
TEST_DATA="$2"
OUTPUT_DIR="$3"

# Initialize variables for optional parameters
PINN_PHASE_VMIN=""
PINN_PHASE_VMAX=""
BASELINE_PHASE_VMIN=""
BASELINE_PHASE_VMAX=""
N_TRAIN_IMAGES=""
N_TEST_IMAGES=""

# Parse remaining arguments (mix of positional and named)
shift 3  # Remove the first 3 arguments we already processed

while [[ $# -gt 0 ]]; do
    case $1 in
        --n-train-images)
            if [[ -z "$2" ]] || [[ "$2" =~ ^-- ]]; then
                echo "Error: --n-train-images requires a numeric argument"
                exit 1
            fi
            if ! [[ "$2" =~ ^[0-9]+$ ]] || [[ "$2" -le 0 ]]; then
                echo "Error: --n-train-images must be a positive integer, got: $2"
                exit 1
            fi
            N_TRAIN_IMAGES="$2"
            shift 2
            ;;
        --n-test-images)
            if [[ -z "$2" ]] || [[ "$2" =~ ^-- ]]; then
                echo "Error: --n-test-images requires a numeric argument"
                exit 1
            fi
            if ! [[ "$2" =~ ^[0-9]+$ ]] || [[ "$2" -le 0 ]]; then
                echo "Error: --n-test-images must be a positive integer, got: $2"
                exit 1
            fi
            N_TEST_IMAGES="$2"
            shift 2
            ;;
        *)
            # Handle positional phase control parameters
            if [[ -z "$PINN_PHASE_VMIN" ]]; then
                PINN_PHASE_VMIN="$1"
            elif [[ -z "$PINN_PHASE_VMAX" ]]; then
                PINN_PHASE_VMAX="$1"
            elif [[ -z "$BASELINE_PHASE_VMIN" ]]; then
                BASELINE_PHASE_VMIN="$1"
            elif [[ -z "$BASELINE_PHASE_VMAX" ]]; then
                BASELINE_PHASE_VMAX="$1"
            else
                echo "Error: Unknown argument: $1"
                exit 1
            fi
            shift
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

# Validate n_images against dataset size if specified
if [[ -n "$N_TRAIN_IMAGES" ]]; then
    # Get the number of images in the training dataset
    TRAIN_DATASET_SIZE=$(python -c "
import numpy as np
data = np.load('$TRAIN_DATA')
diff_key = 'diff3d' if 'diff3d' in data else 'diffraction'
print(data[diff_key].shape[0])
" 2>/dev/null)
    
    if [[ -z "$TRAIN_DATASET_SIZE" ]] || ! [[ "$TRAIN_DATASET_SIZE" =~ ^[0-9]+$ ]]; then
        echo "Warning: Could not determine training dataset size. Proceeding with specified n_train_images=$N_TRAIN_IMAGES"
    elif [[ "$N_TRAIN_IMAGES" -gt "$TRAIN_DATASET_SIZE" ]]; then
        echo "Error: Requested training images ($N_TRAIN_IMAGES) exceeds dataset size ($TRAIN_DATASET_SIZE)"
        exit 1
    fi
fi

if [[ -n "$N_TEST_IMAGES" ]]; then
    # Get the number of images in the test dataset
    TEST_DATASET_SIZE=$(python -c "
import numpy as np
data = np.load('$TEST_DATA')
diff_key = 'diff3d' if 'diff3d' in data else 'diffraction'
print(data[diff_key].shape[0])
" 2>/dev/null)
    
    if [[ -z "$TEST_DATASET_SIZE" ]] || ! [[ "$TEST_DATASET_SIZE" =~ ^[0-9]+$ ]]; then
        echo "Warning: Could not determine test dataset size. Proceeding with specified n_test_images=$N_TEST_IMAGES"
    elif [[ "$N_TEST_IMAGES" -gt "$TEST_DATASET_SIZE" ]]; then
        echo "Error: Requested test images ($N_TEST_IMAGES) exceeds dataset size ($TEST_DATASET_SIZE)"
        exit 1
    fi
fi

# Create output directories
echo "Creating output directories..."
mkdir -p "$OUTPUT_DIR"
PINN_DIR="$OUTPUT_DIR/pinn_run"
BASELINE_DIR="$OUTPUT_DIR/baseline_run"
mkdir -p "$PINN_DIR"
mkdir -p "$BASELINE_DIR"

# Path to shared config
CONFIG_FILE="configs/comparison_config.yaml"

# Extract gridsize from config file for baseline training
GRIDSIZE_OVERRIDE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['gridsize'])")

echo "=========================================="
echo "Starting PtychoPINN vs Baseline Comparison"
echo "=========================================="
echo "Train data: $TRAIN_DATA"
echo "Test data: $TEST_DATA"
echo "Output directory: $OUTPUT_DIR"
echo "Config file: $CONFIG_FILE"

# Display image count information
if [[ -n "$N_TRAIN_IMAGES" ]]; then
    echo "Training images: $N_TRAIN_IMAGES (override)"
else
    # Extract default from config file
    DEFAULT_N_IMAGES=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['n_images'])")
    echo "Training images: $DEFAULT_N_IMAGES (from config)"
fi

if [[ -n "$N_TEST_IMAGES" ]]; then
    echo "Test images: $N_TEST_IMAGES (override)"
else
    echo "Test images: using full test dataset"
fi

if [ -n "$PINN_PHASE_VMIN" ] || [ -n "$PINN_PHASE_VMAX" ] || [ -n "$BASELINE_PHASE_VMIN" ] || [ -n "$BASELINE_PHASE_VMAX" ]; then
    echo "Phase control: PtychoPINN [$PINN_PHASE_VMIN, $PINN_PHASE_VMAX], Baseline [$BASELINE_PHASE_VMIN, $BASELINE_PHASE_VMAX]"
fi
echo ""

# Step 1: Train PtychoPINN model
echo "Step 1/3: Training PtychoPINN model..."
echo "----------------------------------------"

# Build PtychoPINN training command
PINN_CMD="python scripts/training/train.py \
    --config \"$CONFIG_FILE\" \
    --train_data_file \"$TRAIN_DATA\" \
    --test_data_file \"$TEST_DATA\" \
    --output_dir \"$PINN_DIR\" \
    --model_type pinn"

# Add n_images parameter if specified
if [[ -n "$N_TRAIN_IMAGES" ]]; then
    PINN_CMD="$PINN_CMD --n_images $N_TRAIN_IMAGES"
fi

# Execute PtychoPINN training
eval $PINN_CMD

echo ""
echo "PtychoPINN training complete!"
echo ""

# Step 2: Train Baseline model
echo "Step 2/3: Training Baseline model..."
echo "------------------------------------"

# Build baseline training command
BASELINE_CMD="python scripts/run_baseline.py \
    --config \"$CONFIG_FILE\" \
    --train_data_file \"$TRAIN_DATA\" \
    --test_data_file \"$TEST_DATA\" \
    --output_dir \"$BASELINE_DIR\" \
    --gridsize \"$GRIDSIZE_OVERRIDE\""

# Add n_images parameter if specified
if [[ -n "$N_TRAIN_IMAGES" ]]; then
    BASELINE_CMD="$BASELINE_CMD --n_images $N_TRAIN_IMAGES"
fi

# Execute baseline training
eval $BASELINE_CMD

echo ""
echo "Baseline training complete!"
echo ""

# Step 3: Run comparison analysis
echo "Step 3/3: Running comparison analysis..."
echo "---------------------------------------"

# Build the command with optional phase parameters
COMPARE_CMD="python scripts/compare_models.py \
    --pinn_dir \"$PINN_DIR\" \
    --baseline_dir \"$BASELINE_DIR\" \
    --test_data \"$TEST_DATA\" \
    --output_dir \"$OUTPUT_DIR\""

# Add n-test-images parameter if specified
if [[ -n "$N_TEST_IMAGES" ]]; then
    COMPARE_CMD="$COMPARE_CMD --n-test-images $N_TEST_IMAGES"
fi

# Add phase control parameters if provided
if [ -n "$PINN_PHASE_VMIN" ]; then
    COMPARE_CMD="$COMPARE_CMD --pinn_phase_vmin $PINN_PHASE_VMIN"
fi
if [ -n "$PINN_PHASE_VMAX" ]; then
    COMPARE_CMD="$COMPARE_CMD --pinn_phase_vmax $PINN_PHASE_VMAX"
fi
if [ -n "$BASELINE_PHASE_VMIN" ]; then
    COMPARE_CMD="$COMPARE_CMD --baseline_phase_vmin $BASELINE_PHASE_VMIN"
fi
if [ -n "$BASELINE_PHASE_VMAX" ]; then
    COMPARE_CMD="$COMPARE_CMD --baseline_phase_vmax $BASELINE_PHASE_VMAX"
fi

# Execute the comparison command
eval $COMPARE_CMD

echo ""
echo "=========================================="
echo "Comparison workflow complete!"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR"
echo "  - comparison_metrics.csv: Performance metrics for both models"
echo "  - comparison_plot.png: Visual comparison of reconstructions"
echo ""