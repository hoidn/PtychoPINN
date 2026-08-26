#!/bin/bash
# Sparse Grouping Example
# Use case: Sample a diverse candidate pool, use fewer exact groups for faster training
# This example shows how to load more data for diversity but train on fewer groups

echo "=================================================="
echo "Sparse Grouping Example"
echo "Goal: Load a diverse candidate pool but train on fewer groups for speed"
echo "=================================================="

# Scenario: You want diverse candidates from 5,000 rows
# But only want to train on 100 groups (100 unique centers) for faster iteration

# Example 1: Load 5000 candidate rows, but only create 100 groups
echo ""
echo "Example 1: Sparse grouping with gridsize=2"
echo "Loading 5000 candidate rows, creating 100 groups (100 unique centers)"
ptycho_train \
    --train_data_file datasets/fly/fly001_transposed.npz \
    --train_raw_selection 5000 \
    --training_groups 100 \
    --gridsize 2 \
    --neighbor_count 7 \
    --subsample_seed 42 \
    --output_dir sparse_gs2_example \
    --nepochs 5

# Example 2: Even sparser - load many, use few
echo ""
echo "Example 2: Very sparse grouping with gridsize=4"
echo "Loading 8000 candidate rows, creating 50 groups (50 unique centers)"
ptycho_train \
    --train_data_file datasets/fly/fly001_transposed.npz \
    --train_raw_selection 8000 \
    --training_groups 50 \
    --gridsize 4 \
    --neighbor_count 20 \
    --subsample_seed 42 \
    --output_dir sparse_gs4_example \
    --nepochs 5

echo ""
echo "=================================================="
echo "Key Points:"
echo "1. train_raw_selection >> training_groups for sparse grouping"
echo "2. Groups draw their designated centers from the candidate pool"
echo "3. Useful for: Quick experiments, hyperparameter tuning"
echo "4. Trade-off: Faster training but uses fewer centers (neighbors are still shared across groups)"
echo "5. training_groups can never exceed the candidate pool size"
echo "=================================================="