#!/bin/bash
set -euo pipefail

# Select a broad raw pool but materialize groups for relatively few anchors.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

echo "Sparse grouping: 5000 selected rows -> 100 groups (gridsize=2)"
ptycho_train \
    --config "$SCRIPT_DIR/sparse_grouping.yaml" \
    --data.train_data_file datasets/fly/fly001_transposed.npz \
    --output_dir sparse_gs2_example

echo "The 100 anchors are sampled from the 5000-row pool; neighbor rows may overlap."
