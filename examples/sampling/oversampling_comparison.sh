#!/bin/bash
set -euo pipefail

# Compare standard grouping with explicit K-choose-C oversampling.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

echo "=================================================="
echo "K Choose C Oversampling Comparison"
echo "Oversampling begins only when groups exceed selected raw rows"
echo "=================================================="

# Check if datasets exist
TRAIN_DATA="prepare_1e4_photons_5k/dataset/train.npz"
TEST_DATA="prepare_1e4_photons_5k/dataset/test.npz"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "Error: Training dataset $TRAIN_DATA not found"
    echo "Please ensure you have the required dataset"
    exit 1
fi

if [ ! -f "$TEST_DATA" ]; then
    echo "Error: Test dataset $TEST_DATA not found"
    echo "Please ensure you have the required dataset"
    exit 1
fi

echo ""
echo "Configuration:"
echo "- Training Dataset: $TRAIN_DATA"
echo "- Test Dataset: $TEST_DATA"
echo "- Selected raw rows: 512"
echo "- Gridsize: 2 (4 images per group)"
echo "- Epochs: 50"
echo ""

# Standard grouping: at most one group per selected seed point.
echo "=================================================="
echo "Example 1: Standard grouping"
echo "Creating 512 groups from 512 selected rows"
echo "=================================================="

ptycho_train \
    --config "$SCRIPT_DIR/oversampling_standard.yaml" \
    --data.train_data_file "$TRAIN_DATA" \
    --data.test_data_file "$TEST_DATA" \
    --output_dir traditional_512groups \
    --do_stitching

echo ""
echo "Standard grouping complete. Check logs for:"
echo "- 'Using efficient random sample-then-group strategy'"
echo "- 512 groups created from 512 selected rows"
echo ""

# K choose C oversampling with the same selected pool.
echo "=================================================="
echo "Example 2: K Choose C Oversampling (2x groups)"
echo "Creating 1024 groups from the same 512 rows using K=7"
echo "=================================================="

ptycho_train \
    --config "$SCRIPT_DIR/oversampling_2x.yaml" \
    --data.train_data_file "$TRAIN_DATA" \
    --data.test_data_file "$TEST_DATA" \
    --output_dir oversampled_1024groups \
    --do_stitching

echo ""
echo "2x oversampling complete. Check logs for:"
echo "- 'Using K choose C oversampling strategy'"
echo "- 1024 groups created from 512 selected rows"
echo ""

# Example 3: Extreme oversampling
echo "=================================================="
echo "Example 3: Extreme K Choose C Oversampling (4x groups)"
echo "Creating 2048 groups from the same 512 rows using K=7"
echo "=================================================="

ptycho_train \
    --config "$SCRIPT_DIR/oversampling_4x.yaml" \
    --data.train_data_file "$TRAIN_DATA" \
    --data.test_data_file "$TEST_DATA" \
    --output_dir extreme_oversampled_2048groups \
    --do_stitching

echo ""
echo "4x oversampling complete. Check logs for:"
echo "- 'Using K choose C oversampling strategy'"
echo "- 2048 groups created from 512 selected rows"
echo ""

# Summary
echo "=================================================="
echo "Comparison Summary"
echo "=================================================="
echo ""
echo "All three runs used the same 512 selected rows, but:"
echo ""
echo "1. Standard (K=4): 512 groups"
echo "   - One group per selected anchor"
echo ""
echo "2. Oversampled 2x (K=7): 1024 groups"
echo "   - Explicit oversampling triggered"
echo "   - Uses K choose C combinations"
echo "   - 2x more training samples from same data"
echo ""
echo "3. Oversampled 4x (K=7): 2048 groups"
echo "   - Each seed can generate C(7,4)=35 combinations"
echo "   - 4x more training samples from same data"
echo ""
echo "Key Insight: Higher K values (e.g., 7) enable more combinations"
echo "from the same data, effectively augmenting your training set"
echo "without needing more raw data."
echo ""
echo "Compare the training curves and final quality in:"
echo "- traditional_512groups/"
echo "- oversampled_1024groups/"
echo "- extreme_oversampled_2048groups/"
echo "=================================================="
