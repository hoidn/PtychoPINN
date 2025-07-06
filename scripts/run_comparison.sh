#!/bin/bash
# run_comparison.sh - Master script to train and compare PtychoPINN vs Baseline models
# Usage: ./scripts/run_comparison.sh <train_data.npz> <test_data.npz> <output_dir>

set -e  # Exit on error

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root directory
cd "$PROJECT_ROOT"

# Parse command line arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <train_data.npz> <test_data.npz> <output_dir>"
    echo "Example: $0 datasets/fly/fly001_transposed.npz datasets/fly/fly001_transposed.npz comparison_results"
    exit 1
fi

TRAIN_DATA="$1"
TEST_DATA="$2"
OUTPUT_DIR="$3"

# Verify input files exist
if [ ! -f "$TRAIN_DATA" ]; then
    echo "Error: Training data file not found: $TRAIN_DATA"
    exit 1
fi

if [ ! -f "$TEST_DATA" ]; then
    echo "Error: Test data file not found: $TEST_DATA"
    exit 1
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

echo "=========================================="
echo "Starting PtychoPINN vs Baseline Comparison"
echo "=========================================="
echo "Train data: $TRAIN_DATA"
echo "Test data: $TEST_DATA"
echo "Output directory: $OUTPUT_DIR"
echo "Config file: $CONFIG_FILE"
echo ""

# Step 1: Train PtychoPINN model
echo "Step 1/3: Training PtychoPINN model..."
echo "----------------------------------------"
python scripts/training/train.py \
    --config "$CONFIG_FILE" \
    --train_data_file "$TRAIN_DATA" \
    --test_data_file "$TEST_DATA" \
    --output_dir "$PINN_DIR" \
    --model_type pinn

echo ""
echo "PtychoPINN training complete!"
echo ""

# Step 2: Train Baseline model
echo "Step 2/3: Training Baseline model..."
echo "------------------------------------"
python scripts/run_baseline.py \
    --config "$CONFIG_FILE" \
    --train_data_file "$TRAIN_DATA" \
    --test_data_file "$TEST_DATA" \
    --output_dir "$BASELINE_DIR" \
    --gridsize 1

echo ""
echo "Baseline training complete!"
echo ""

# Step 3: Run comparison analysis
echo "Step 3/3: Running comparison analysis..."
echo "---------------------------------------"
python scripts/compare_models.py \
    --pinn_dir "$PINN_DIR" \
    --baseline_dir "$BASELINE_DIR" \
    --test_data "$TEST_DATA" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "Comparison workflow complete!"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR"
echo "  - comparison_metrics.csv: Performance metrics for both models"
echo "  - comparison_plot.png: Visual comparison of reconstructions"
echo ""