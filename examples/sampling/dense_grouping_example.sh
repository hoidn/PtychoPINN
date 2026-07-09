#!/bin/bash
# Dense Grouping Example
# Use case: Maximum data utilization with neighbor grouping
# This example shows how to use most of your subsampled data for training

echo "=================================================="
echo "Dense Grouping Example"
echo "Goal: Use as much of the subsampled data as possible for training"
echo "=================================================="

# Scenario: You have a dataset with 10,000 images, but can only load 2,000 into memory
# You want to create as many training groups as possible from these 2,000 images

# For gridsize=2 (4 images per group):
echo ""
echo "Example 1: Dense grouping with gridsize=2"
echo "Loading 2000 images, creating 500 groups (using all 2000 images)"
ptycho_train \
    --train_data_file datasets/fly/fly001_transposed.npz \
    --n_subsample 2000 \
    --n_images 500 \
    --gridsize 2 \
    --subsample_seed 42 \
    --output_dir dense_gs2_example \
    --nepochs 2

# For gridsize=4 (16 images per group):
echo ""
echo "Example 2: Dense grouping with gridsize=4"
echo "Loading 2048 images, creating 128 groups (using all 2048 images)"
ptycho_train \
    --train_data_file datasets/fly/fly001_transposed.npz \
    --n_subsample 2048 \
    --n_images 128 \
    --gridsize 4 \
    --subsample_seed 42 \
    --output_dir dense_gs4_example \
    --nepochs 2

echo ""
echo "=================================================="
echo "Key Points:"
echo "1. n_subsample controls how many images are loaded from disk"
echo "2. n_images controls how many groups are created"
echo "3. For dense grouping: n_images = n_subsample / (gridsize²)"
echo "4. This maximizes data utilization within memory constraints"
echo "=================================================="