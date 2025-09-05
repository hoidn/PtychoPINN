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
    echo "Usage: $0 <train_data.npz> <test_data.npz> <output_dir> [options]"
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
    echo "  --n-train-groups N   Number of training groups to generate"
    echo "  --n-train-subsample N Number of images to subsample for training"
    echo "  --n-test-groups N    Number of test groups to generate for evaluation"
    echo "  --n-test-subsample N  Number of images to subsample for testing"
    echo "  --neighbor-count K   Number of nearest neighbors for K choose C oversampling"
    echo "  --skip-training      Skip training and use existing models"
    echo "  --pinn-model PATH    Path to existing PtychoPINN model (with --skip-training)"
    echo "  --baseline-model PATH Path to existing baseline model (with --skip-training)"
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
N_TRAIN_GROUPS=""
N_TRAIN_SUBSAMPLE=""
N_TEST_GROUPS=""
N_TEST_SUBSAMPLE=""
NEIGHBOR_COUNT=""
SKIP_TRAINING=false
PINN_MODEL=""
BASELINE_MODEL=""
SKIP_REGISTRATION=""
REGISTER_PTYCHI_ONLY=""
TIKE_RECON_PATH=""
STITCH_CROP_SIZE=""

# Parse remaining arguments (mix of positional and named)
shift 3  # Remove the first 3 arguments we already processed

while [[ $# -gt 0 ]]; do
    case $1 in
        --n-train-images)
            echo "Warning: --n-train-images is deprecated. Use --n-train-groups instead."
            N_TRAIN_GROUPS="$2"
            shift 2
            ;;
        --n-train-groups)
            N_TRAIN_GROUPS="$2"
            shift 2
            ;;
        --n-train-subsample)
            N_TRAIN_SUBSAMPLE="$2"
            shift 2
            ;;
        --n-test-groups)
            N_TEST_GROUPS="$2"
            shift 2
            ;;
        --n-test-subsample)
            N_TEST_SUBSAMPLE="$2"
            shift 2
            ;;
        --neighbor-count)
            NEIGHBOR_COUNT="$2"
            shift 2
            ;;
        --n-test-images)
            echo "Warning: --n-test-images is deprecated. Use --n-test-groups instead."
            N_TEST_GROUPS="$2"
            shift 2
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --pinn-model)
            PINN_MODEL="$2"
            shift 2
            ;;
        --baseline-model)
            BASELINE_MODEL="$2"
            shift 2
            ;;
        --skip-registration)
            SKIP_REGISTRATION="--skip-registration"
            shift
            ;;
        --register-ptychi-only)
            REGISTER_PTYCHI_ONLY="--register-ptychi-only"
            shift
            ;;
        --tike_recon_path|--tike-recon-path)
            TIKE_RECON_PATH="$2"
            shift 2
            ;;
        --stitch-crop-size)
            STITCH_CROP_SIZE="$2"
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

# Validate n_groups/n_subsample against dataset size if specified
if [[ -n "$N_TRAIN_SUBSAMPLE" ]]; then
    # Get the number of images in the training dataset
    TRAIN_DATASET_SIZE=$(python -c "
import numpy as np
data = np.load('$TRAIN_DATA')
diff_key = 'diff3d' if 'diff3d' in data else 'diffraction'
print(data[diff_key].shape[0])
" 2>/dev/null)
    
    if [[ -z "$TRAIN_DATASET_SIZE" ]] || ! [[ "$TRAIN_DATASET_SIZE" =~ ^[0-9]+$ ]]; then
        echo "Warning: Could not determine training dataset size. Proceeding with specified n_train_subsample=$N_TRAIN_SUBSAMPLE"
    elif [[ "$N_TRAIN_SUBSAMPLE" -gt "$TRAIN_DATASET_SIZE" ]]; then
        echo "Error: Requested training subsample ($N_TRAIN_SUBSAMPLE) exceeds dataset size ($TRAIN_DATASET_SIZE)"
        exit 1
    fi
fi

if [[ -n "$N_TEST_SUBSAMPLE" ]]; then
    # Get the number of images in the test dataset
    TEST_DATASET_SIZE=$(python -c "
import numpy as np
data = np.load('$TEST_DATA')
diff_key = 'diff3d' if 'diff3d' in data else 'diffraction'
print(data[diff_key].shape[0])
" 2>/dev/null)
    
    if [[ -z "$TEST_DATASET_SIZE" ]] || ! [[ "$TEST_DATASET_SIZE" =~ ^[0-9]+$ ]]; then
        echo "Warning: Could not determine test dataset size. Proceeding with specified n_test_subsample=$N_TEST_SUBSAMPLE"
    elif [[ "$N_TEST_SUBSAMPLE" -gt "$TEST_DATASET_SIZE" ]]; then
        echo "Error: Requested test subsample ($N_TEST_SUBSAMPLE) exceeds dataset size ($TEST_DATASET_SIZE)"
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

# Extract stitch_crop_size from config file for comparison
M_STITCH_SIZE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE')).get('stitch_crop_size', 20))")

echo "=========================================="
echo "Starting PtychoPINN vs Baseline Comparison"
echo "=========================================="
echo "Train data: $TRAIN_DATA"
echo "Test data: $TEST_DATA"
echo "Output directory: $OUTPUT_DIR"
echo "Config file: $CONFIG_FILE"

# Display sampling information
if [[ -n "$N_TRAIN_SUBSAMPLE" ]]; then
    echo "Training subsample: $N_TRAIN_SUBSAMPLE images"
fi
if [[ -n "$N_TRAIN_GROUPS" ]]; then
    echo "Training groups: $N_TRAIN_GROUPS"
else
    # Extract default from config file
    DEFAULT_N_GROUPS=$(python -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config.get('n_groups', config.get('n_images', 512)))" 2>/dev/null || echo "512")
    echo "Training groups: $DEFAULT_N_GROUPS (from config)"
fi

if [[ -n "$N_TEST_SUBSAMPLE" ]]; then
    echo "Test subsample: $N_TEST_SUBSAMPLE images"
fi
if [[ -n "$N_TEST_GROUPS" ]]; then
    echo "Test groups: $N_TEST_GROUPS"
else
    echo "Test groups: using full test dataset"
fi

if [ -n "$PINN_PHASE_VMIN" ] || [ -n "$PINN_PHASE_VMAX" ] || [ -n "$BASELINE_PHASE_VMIN" ] || [ -n "$BASELINE_PHASE_VMAX" ]; then
    echo "Phase control: PtychoPINN [$PINN_PHASE_VMIN, $PINN_PHASE_VMAX], Baseline [$BASELINE_PHASE_VMIN, $BASELINE_PHASE_VMAX]"
fi
echo ""

# Handle skip-training mode
if [ "$SKIP_TRAINING" = true ]; then
    if [ -n "$PINN_MODEL" ]; then
        PINN_DIR="$PINN_MODEL"
    fi
    if [ -n "$BASELINE_MODEL" ]; then
        BASELINE_DIR="$BASELINE_MODEL"
    fi
    echo "Skipping training, using existing models:"
    echo "  PtychoPINN: $PINN_DIR"
    echo "  Baseline: $BASELINE_DIR"
    echo ""
else
    # Step 1: Train PtychoPINN model
    echo "Step 1/3: Training PtychoPINN model (subsample=$N_TRAIN_SUBSAMPLE, groups=$N_TRAIN_GROUPS)..."
    echo "----------------------------------------"

    # Build PtychoPINN training command
    PINN_CMD="python scripts/training/train.py \\
        --config \"$CONFIG_FILE\" \\
        --train_data_file \"$TRAIN_DATA\" \\
        --test_data_file \"$TEST_DATA\" \\
        --output_dir \"$PINN_DIR\" \\
        --model_type pinn"

    # Add training sampling parameters
    if [[ -n "$N_TRAIN_SUBSAMPLE" ]]; then
        PINN_CMD="$PINN_CMD --n_subsample $N_TRAIN_SUBSAMPLE"
    fi
    if [[ -n "$N_TRAIN_GROUPS" ]]; then
        PINN_CMD="$PINN_CMD --n_groups $N_TRAIN_GROUPS"
    fi
    if [[ -n "$NEIGHBOR_COUNT" ]]; then
        PINN_CMD="$PINN_CMD --neighbor_count $NEIGHBOR_COUNT"
    fi

    # Execute PtychoPINN training
    eval $PINN_CMD

    echo ""
    echo "PtychoPINN training complete!"
    echo ""

    # Step 2: Train Baseline model
    echo "Step 2/3: Training Baseline model (subsample=$N_TRAIN_SUBSAMPLE, groups=$N_TRAIN_GROUPS)..."
    echo "------------------------------------"

    # Build baseline training command
    BASELINE_CMD="python scripts/run_baseline.py \\
        --config \"$CONFIG_FILE\" \\
        --train_data_file \"$TRAIN_DATA\" \\
        --test_data_file \"$TEST_DATA\" \\
        --output_dir \"$BASELINE_DIR\" \\
        --gridsize \"$GRIDSIZE_OVERRIDE\""

    # Add training sampling parameters for baseline
    if [[ -n "$N_TRAIN_SUBSAMPLE" ]]; then
        BASELINE_CMD="$BASELINE_CMD --n_subsample $N_TRAIN_SUBSAMPLE"
    fi
    if [[ -n "$N_TRAIN_GROUPS" ]]; then
        BASELINE_CMD="$BASELINE_CMD --n_groups $N_TRAIN_GROUPS"
    fi
    if [[ -n "$NEIGHBOR_COUNT" ]]; then
        BASELINE_CMD="$BASELINE_CMD --neighbor_count $NEIGHBOR_COUNT"
    fi

    # Execute baseline training
    eval $BASELINE_CMD

    echo ""
    echo "Baseline training complete!"
    echo ""
fi  # End of training block

# Step 3: Run comparison analysis
echo "Step 3/3: Running comparison analysis (subsample=$N_TEST_SUBSAMPLE, groups=$N_TEST_GROUPS)..."
echo "---------------------------------------"

# Build the command with optional phase parameters
COMPARE_CMD="python scripts/compare_models.py \
    --pinn_dir \"$PINN_DIR\" \
    --baseline_dir \"$BASELINE_DIR\" \
    --test_data \"$TEST_DATA\" \
    --output_dir \"$OUTPUT_DIR\" \
    --stitch-crop-size \"$M_STITCH_SIZE\""

# Add test sampling parameters
if [[ -n "$N_TEST_SUBSAMPLE" ]]; then
    COMPARE_CMD="$COMPARE_CMD --n-test-subsample $N_TEST_SUBSAMPLE"
fi
if [[ -n "$N_TEST_GROUPS" ]]; then
    COMPARE_CMD="$COMPARE_CMD --n-test-groups $N_TEST_GROUPS"
fi

# Add phase control parameters if provided
if [ ! -z "$PINN_PHASE_VMIN" ]; then
    COMPARE_CMD="$COMPARE_CMD --pinn_phase_vmin $PINN_PHASE_VMIN"
fi
if [ ! -z "$PINN_PHASE_VMAX" ]; then
    COMPARE_CMD="$COMPARE_CMD --pinn_phase_vmax $PINN_PHASE_VMAX"
fi
if [ ! -z "$BASELINE_PHASE_VMIN" ]; then
    COMPARE_CMD="$COMPARE_CMD --baseline_phase_vmin $BASELINE_PHASE_VMIN"
fi
if [ ! -z "$BASELINE_PHASE_VMAX" ]; then
    COMPARE_CMD="$COMPARE_CMD --baseline_phase_vmax $BASELINE_PHASE_VMAX"
fi

# Add additional parameters if provided
if [ ! -z "$SKIP_REGISTRATION" ]; then
    COMPARE_CMD="$COMPARE_CMD $SKIP_REGISTRATION"
fi
if [ ! -z "$REGISTER_PTYCHI_ONLY" ]; then
    COMPARE_CMD="$COMPARE_CMD $REGISTER_PTYCHI_ONLY"
fi
if [ ! -z "$TIKE_RECON_PATH" ]; then
    COMPARE_CMD="$COMPARE_CMD --tike_recon_path $TIKE_RECON_PATH"
fi
if [ ! -z "$STITCH_CROP_SIZE" ]; then
    COMPARE_CMD="$COMPARE_CMD --stitch-crop-size $STITCH_CROP_SIZE"
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