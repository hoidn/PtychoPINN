#!/bin/bash
# 3-Way Comparison Recipe: PtychoPINN vs Baseline (supervised) vs Pty-Chi
#
# This script:
#   1. Trains PtychoPINN (self-supervised, physics-informed)
#   2. Trains Baseline (supervised)
#   3. Runs Pty-Chi reconstruction
#   4. Runs the 3-way comparison
#
# Usage:
#   ./run_3way_comparison_recipe.sh <train_data> <test_data> <output_dir> [options]
#
# Example:
#   ./run_3way_comparison_recipe.sh \
#       datasets/fly64/fly64_top_half_shuffled.npz \
#       datasets/fly64/fly64_shuffled.npz \
#       results/3way_fly64_512

set -e  # Exit on error

# Default parameters
N_IMAGES=512
N_GROUPS=512
NEIGHBOR_COUNT=7
NEPOCHS=50
PTYCHI_ALGORITHM="LSQML"
PTYCHI_EPOCHS=200
SKIP_REGISTRATION=false
SAMPLING_SEED=42

# Parse arguments
print_usage() {
    echo "Usage: $0 <train_data> <test_data> <output_dir> [options]"
    echo ""
    echo "Required arguments:"
    echo "  train_data    Path to training NPZ file"
    echo "  test_data     Path to test NPZ file (for comparison)"
    echo "  output_dir    Output directory for all results"
    echo ""
    echo "Options:"
    echo "  --n-images N          Number of images for training/comparison (default: 512)"
    echo "  --n-groups N          Number of groups for training (default: 512)"
    echo "  --neighbor-count N    Neighbor count for training (default: 7)"
    echo "  --nepochs N           Training epochs (default: 50)"
    echo "  --ptychi-algorithm A  Pty-Chi algorithm: DM, LSQML, PIE (default: LSQML)"
    echo "  --ptychi-epochs N     Pty-Chi reconstruction epochs (default: 200)"
    echo "  --skip-registration   Skip registration in comparison"
    echo "  --seed N              Random seed for sampling (default: 42)"
    echo "  --skip-pinn           Skip PtychoPINN training (use existing)"
    echo "  --skip-baseline       Skip Baseline training (use existing)"
    echo "  --skip-ptychi         Skip pty-chi reconstruction (use existing)"
    echo ""
    echo "Example:"
    echo "  $0 datasets/fly64/fly64_top_half_shuffled.npz \\"
    echo "     datasets/fly64/fly64_shuffled.npz \\"
    echo "     results/3way_fly64_512 \\"
    echo "     --n-images 512 --nepochs 50"
}

if [[ $# -lt 3 ]]; then
    print_usage
    exit 1
fi

TRAIN_DATA="$1"
TEST_DATA="$2"
OUTPUT_DIR="$3"
shift 3

SKIP_PINN=false
SKIP_BASELINE=false
SKIP_PTYCHI=false

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --n-images)
            N_IMAGES="$2"
            shift 2
            ;;
        --n-groups)
            N_GROUPS="$2"
            shift 2
            ;;
        --neighbor-count)
            NEIGHBOR_COUNT="$2"
            shift 2
            ;;
        --nepochs)
            NEPOCHS="$2"
            shift 2
            ;;
        --ptychi-algorithm)
            PTYCHI_ALGORITHM="$2"
            shift 2
            ;;
        --ptychi-epochs)
            PTYCHI_EPOCHS="$2"
            shift 2
            ;;
        --skip-registration)
            SKIP_REGISTRATION=true
            shift
            ;;
        --seed)
            SAMPLING_SEED="$2"
            shift 2
            ;;
        --skip-pinn)
            SKIP_PINN=true
            shift
            ;;
        --skip-baseline)
            SKIP_BASELINE=true
            shift
            ;;
        --skip-ptychi)
            SKIP_PTYCHI=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Derived paths
PINN_DIR="${OUTPUT_DIR}/pinn_run"
BASELINE_DIR="${OUTPUT_DIR}/baseline_run"
PTYCHI_DIR="${OUTPUT_DIR}/ptychi_recon"
COMPARISON_DIR="${OUTPUT_DIR}/comparison"

echo "=============================================="
echo "3-Way Comparison Recipe"
echo "=============================================="
echo "Train data:       $TRAIN_DATA"
echo "Test data:        $TEST_DATA"
echo "Output dir:       $OUTPUT_DIR"
echo "N images:         $N_IMAGES"
echo "N groups:         $N_GROUPS"
echo "Training epochs:  $NEPOCHS"
echo "Pty-Chi algo:     $PTYCHI_ALGORITHM"
echo "Pty-Chi epochs:   $PTYCHI_EPOCHS"
echo "Skip registration: $SKIP_REGISTRATION"
echo "=============================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Save configuration
cat > "${OUTPUT_DIR}/recipe_config.json" << EOF
{
    "train_data": "$TRAIN_DATA",
    "test_data": "$TEST_DATA",
    "n_images": $N_IMAGES,
    "n_groups": $N_GROUPS,
    "neighbor_count": $NEIGHBOR_COUNT,
    "nepochs": $NEPOCHS,
    "ptychi_algorithm": "$PTYCHI_ALGORITHM",
    "ptychi_epochs": $PTYCHI_EPOCHS,
    "skip_registration": $SKIP_REGISTRATION,
    "sampling_seed": $SAMPLING_SEED,
    "timestamp": "$(date -Iseconds)"
}
EOF

# ============================================
# Step 1: Train PtychoPINN (self-supervised)
# ============================================
if [[ "$SKIP_PINN" == "false" ]]; then
    echo ""
    echo "=============================================="
    echo "Step 1: Training PtychoPINN (self-supervised)"
    echo "=============================================="
    echo ""

    python scripts/training/train.py \
        --train_data_file "$TRAIN_DATA" \
        --test_data_file "$TRAIN_DATA" \
        --n_groups "$N_GROUPS" \
        --n_subsample "$N_IMAGES" \
        --neighbor_count "$NEIGHBOR_COUNT" \
        --nepochs "$NEPOCHS" \
        --model_type pinn \
        --output_dir "$PINN_DIR"

    echo ""
    echo "PtychoPINN training complete. Model saved to: $PINN_DIR"
else
    echo ""
    echo "Skipping PtychoPINN training (--skip-pinn specified)"
    echo "Using existing model at: $PINN_DIR"
fi

# ============================================
# Step 2: Train Baseline (supervised)
# ============================================
if [[ "$SKIP_BASELINE" == "false" ]]; then
    echo ""
    echo "=============================================="
    echo "Step 2: Training Baseline (supervised)"
    echo "=============================================="
    echo ""

    python scripts/training/train.py \
        --train_data_file "$TRAIN_DATA" \
        --test_data_file "$TRAIN_DATA" \
        --n_groups "$N_GROUPS" \
        --n_subsample "$N_IMAGES" \
        --neighbor_count "$NEIGHBOR_COUNT" \
        --nepochs "$NEPOCHS" \
        --model_type supervised \
        --output_dir "$BASELINE_DIR"

    echo ""
    echo "Baseline training complete. Model saved to: $BASELINE_DIR"
else
    echo ""
    echo "Skipping Baseline training (--skip-baseline specified)"
    echo "Using existing model at: $BASELINE_DIR"
fi

# ============================================
# Step 3: Run Pty-Chi reconstruction
# ============================================
if [[ "$SKIP_PTYCHI" == "false" ]]; then
    echo ""
    echo "=============================================="
    echo "Step 3: Running Pty-Chi reconstruction"
    echo "=============================================="
    echo ""

    python scripts/reconstruction/ptychi_reconstruct_tike.py \
        --input-npz "$TEST_DATA" \
        --output-dir "$PTYCHI_DIR" \
        --algorithm "$PTYCHI_ALGORITHM" \
        --num-epochs "$PTYCHI_EPOCHS" \
        --n-images "$N_IMAGES"

    echo ""
    echo "Pty-Chi reconstruction complete. Saved to: $PTYCHI_DIR"
else
    echo ""
    echo "Skipping pty-chi reconstruction (--skip-ptychi specified)"
    echo "Using existing reconstruction at: $PTYCHI_DIR"
fi

# ============================================
# Step 4: Run 3-way comparison
# ============================================
echo ""
echo "=============================================="
echo "Step 4: Running 3-way comparison"
echo "=============================================="
echo ""

COMPARISON_ARGS=(
    --pinn_dir "$PINN_DIR"
    --baseline_dir "$BASELINE_DIR"
    --test_data "$TEST_DATA"
    --tike_recon_path "${PTYCHI_DIR}/ptychi_reconstruction.npz"
    --output_dir "$COMPARISON_DIR"
    --n-test-subsample "$N_IMAGES"
    --n-test-groups "$N_IMAGES"
)

if [[ "$SKIP_REGISTRATION" == "true" ]]; then
    COMPARISON_ARGS+=(--skip-registration)
fi

python scripts/compare_models.py "${COMPARISON_ARGS[@]}"

echo ""
echo "=============================================="
echo "3-Way Comparison Complete!"
echo "=============================================="
echo ""
echo "Results saved to:"
echo "  - PtychoPINN model (self-supervised): $PINN_DIR"
echo "  - Baseline model (supervised):        $BASELINE_DIR"
echo "  - Pty-Chi reconstruction:             $PTYCHI_DIR"
echo "  - Comparison results:                 $COMPARISON_DIR"
echo ""
echo "Key output files:"
echo "  - ${COMPARISON_DIR}/comparison_plot.png"
echo "  - ${COMPARISON_DIR}/comparison_metrics.csv"
echo ""
