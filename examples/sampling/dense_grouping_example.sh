#!/bin/bash
# Dense Grouping Example
# Use case: Maximum data utilization with centered-nearest grouping
# This example shows how to use every candidate row as a group center

echo "=================================================="
echo "Dense Grouping Example"
echo "Goal: Use as much of the loaded candidate pool as possible for training"
echo "=================================================="

# Scenario: You have a dataset with 10,000 scan rows, but can only load 2,000 into memory
# You want to create as many training groups as possible from these 2,000 candidate rows

# For gridsize=2 (4 rows per group, 1 center + 3 nearest non-center candidates):
echo ""
echo "Example 1: Dense grouping with gridsize=2"
echo "Loading 2000 candidate rows, creating 2000 groups (every row is a center)"
ptycho_train \
    --train_data_file datasets/fly/fly001_transposed.npz \
    --train_raw_selection 2000 \
    --training_groups 2000 \
    --gridsize 2 \
    --neighbor_count 7 \
    --subsample_seed 42 \
    --output_dir dense_gs2_example \
    --nepochs 2

# For gridsize=4 (16 rows per group, 1 center + 15 nearest non-center candidates):
echo ""
echo "Example 2: Dense grouping with gridsize=4"
echo "Loading 2048 candidate rows, creating 2048 groups (every row is a center)"
ptycho_train \
    --train_data_file datasets/fly/fly001_transposed.npz \
    --train_raw_selection 2048 \
    --training_groups 2048 \
    --gridsize 4 \
    --neighbor_count 20 \
    --subsample_seed 42 \
    --output_dir dense_gs4_example \
    --nepochs 2

echo ""
echo "=================================================="
echo "Key Points:"
echo "1. train_raw_selection controls how many rows are loaded as the candidate pool"
echo "2. training_groups is the exact group count (= number of unique centers)"
echo "3. Dense grouping: training_groups = train_raw_selection (every candidate is a center)"
echo "4. Each group is its center plus gridsize^2 - 1 rows chosen from its K nearest non-center candidates"
echo "5. Groups overlap; distinct rows used is at most the candidate pool size"
echo "=================================================="