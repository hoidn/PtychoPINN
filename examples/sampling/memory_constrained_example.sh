#!/bin/bash
# Memory-Constrained Example
# Use case: Working with limited GPU/CPU memory
# This example shows strategies for memory-limited environments

echo "=================================================="
echo "Memory-Constrained Example"
echo "Goal: Train effectively with limited memory"
echo "=================================================="

# Scenario: Large dataset but limited memory
# Strategy: Load minimal data, use all of it efficiently

# Example 1: Minimal memory footprint
echo ""
echo "Example 1: Minimal memory usage (gridsize=1)"
echo "Loading only 256 images for very limited memory"
ptycho_train \
    --train_data_file datasets/fly/fly001_transposed.npz \
    --n_subsample 256 \
    --n_images 256 \
    --gridsize 1 \
    --batch_size 8 \
    --subsample_seed 42 \
    --output_dir memory_minimal_example \
    --nepochs 10

# Example 2: Balanced memory usage with grouping
echo ""
echo "Example 2: Efficient grouping for moderate memory"
echo "Loading 512 images, creating 128 groups with gridsize=2"
ptycho_train \
    --train_data_file datasets/fly/fly001_transposed.npz \
    --n_subsample 512 \
    --n_images 128 \
    --gridsize 2 \
    --batch_size 16 \
    --subsample_seed 42 \
    --output_dir memory_balanced_example \
    --nepochs 10

# Example 3: Progressive loading strategy
echo ""
echo "Example 3: Progressive training with different seeds"
echo "Train multiple times with different subsamples"

for seed in 42 123 456; do
    echo "Training with seed $seed..."
    ptycho_train \
        --train_data_file datasets/fly/fly001_transposed.npz \
        --n_subsample 1000 \
        --n_images 250 \
        --gridsize 2 \
        --batch_size 8 \
        --subsample_seed $seed \
        --output_dir memory_progressive_seed${seed} \
        --nepochs 5
done

echo ""
echo "=================================================="
echo "Memory-Saving Strategies:"
echo "1. Use smaller n_subsample to load less data"
echo "2. Reduce batch_size for training"
echo "3. Train multiple times with different seeds for diversity"
echo "4. Consider gridsize=1 for maximum memory efficiency"
echo "=================================================="