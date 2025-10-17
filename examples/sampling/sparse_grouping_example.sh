#!/bin/bash
# Sparse Grouping Example
# Use case: Sample diverse data, use fewer groups for faster training
# This example shows how to load more data for diversity but train on fewer groups

echo "=================================================="
echo "Sparse Grouping Example"
echo "Goal: Load diverse data but train on fewer groups for speed"
echo "=================================================="

# Scenario: You want diverse sampling from 5,000 images
# But only want to train on 100 groups for faster iteration

# Example 1: Load 5000 images, but only create 100 groups
echo ""
echo "Example 1: Sparse grouping with gridsize=2"
echo "Loading 5000 images, but only creating 100 groups (using 400 images)"
ptycho_train \
    --train_data_file datasets/fly/fly001_transposed.npz \
    --n_subsample 5000 \
    --n_images 100 \
    --gridsize 2 \
    --subsample_seed 42 \
    --output_dir sparse_gs2_example \
    --nepochs 5

# Example 2: Even sparser - load many, use few
echo ""
echo "Example 2: Very sparse grouping with gridsize=4"
echo "Loading 8000 images, but only creating 50 groups (using 800 images)"
ptycho_train \
    --train_data_file datasets/fly/fly001_transposed.npz \
    --n_subsample 8000 \
    --n_images 50 \
    --gridsize 4 \
    --subsample_seed 42 \
    --output_dir sparse_gs4_example \
    --nepochs 5

echo ""
echo "=================================================="
echo "Key Points:"
echo "1. n_subsample >> n_images × gridsize² for sparse grouping"
echo "2. Groups are selected from the subsampled pool"
echo "3. Useful for: Quick experiments, hyperparameter tuning"
echo "4. Trade-off: Faster training but uses less of your data"
echo "=================================================="