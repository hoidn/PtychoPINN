#!/bin/bash
# K Choose C Oversampling Comparison Example
# This script demonstrates the difference between traditional 1:1 mapping and K choose C oversampling

echo "=================================================="
echo "K Choose C Oversampling Comparison"
echo "Demonstrating automatic oversampling when requesting more groups than available points"
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

# Configuration
CONFIG="configs/gridsize2_minimal.yaml"
SUBSAMPLE=512  # Number of images to subsample from dataset
EPOCHS=50

echo ""
echo "Configuration:"
echo "- Training Dataset: $TRAIN_DATA"
echo "- Test Dataset: $TEST_DATA"
echo "- Subsampling: $SUBSAMPLE images from dataset"
echo "- Gridsize: 2 (4 images per group)"
echo "- Epochs: $EPOCHS"
echo ""

# Example 1: Traditional approach - one group per seed point
echo "=================================================="
echo "Example 1: Traditional 1:1 Mapping"
echo "Creating 128 groups from 512 images (1 group per 4 images)"
echo "=================================================="

ptycho_train \
    --train_data_file "$TRAIN_DATA" \
    --test_data_file "$TEST_DATA" \
    --n_subsample $SUBSAMPLE \
    --n_groups 128 \
    --neighbor_count 4 \
    --gridsize 2 \
    --config "$CONFIG" \
    --output_dir traditional_128groups \
    --nepochs $EPOCHS \
    --do_stitching

echo ""
echo "Traditional approach complete. Check logs for:"
echo "- 'Using efficient random sample-then-group strategy'"
echo "- 128 groups created from 512 subsampled images"
echo ""

# Example 2: K choose C oversampling with same subsample
echo "=================================================="
echo "Example 2: K Choose C Oversampling (2x groups)"
echo "Creating 256 groups from same 512 images using K=7"
echo "=================================================="

ptycho_train \
    --train_data_file "$TRAIN_DATA" \
    --test_data_file "$TEST_DATA" \
    --n_subsample $SUBSAMPLE \
    --n_groups 256 \
    --neighbor_count 7 \
    --gridsize 2 \
    --config "$CONFIG" \
    --output_dir oversampled_256groups \
    --nepochs $EPOCHS \
    --do_stitching

echo ""
echo "2x oversampling complete. Check logs for:"
echo "- 'Using K choose C oversampling strategy'"
echo "- 256 groups created from 512 subsampled images"
echo ""

# Example 3: Extreme oversampling
echo "=================================================="
echo "Example 3: Extreme K Choose C Oversampling (4x groups)"
echo "Creating 512 groups from same 512 images using K=7"
echo "=================================================="

ptycho_train \
    --train_data_file "$TRAIN_DATA" \
    --test_data_file "$TEST_DATA" \
    --n_subsample $SUBSAMPLE \
    --n_groups 512 \
    --neighbor_count 7 \
    --gridsize 2 \
    --config "$CONFIG" \
    --output_dir extreme_oversampled_512groups \
    --nepochs $EPOCHS \
    --do_stitching

echo ""
echo "4x oversampling complete. Check logs for:"
echo "- 'Automatically using K choose C oversampling'"
echo "- 512 groups created from 512 subsampled images"
echo ""

# Summary
echo "=================================================="
echo "Comparison Summary"
echo "=================================================="
echo ""
echo "All three runs used the SAME 512 subsampled images, but:"
echo ""
echo "1. Traditional (K=4): 128 groups"
echo "   - Each seed point generates 1 group"
echo "   - Limited augmentation"
echo ""
echo "2. Oversampled 2x (K=7): 256 groups"
echo "   - Automatic oversampling triggered"
echo "   - Uses K choose C combinations"
echo "   - 2x more training samples from same data"
echo ""
echo "3. Oversampled 4x (K=7): 512 groups"
echo "   - Maximum oversampling"
echo "   - Each seed can generate C(7,4)=35 combinations"
echo "   - 4x more training samples from same data"
echo ""
echo "Key Insight: Higher K values (e.g., 7) enable more combinations"
echo "from the same data, effectively augmenting your training set"
echo "without needing more raw data."
echo ""
echo "Compare the training curves and final quality in:"
echo "- traditional_128groups/"
echo "- oversampled_256groups/"
echo "- extreme_oversampled_512groups/"
echo "=================================================="