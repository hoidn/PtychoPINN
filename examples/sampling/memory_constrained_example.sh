#!/bin/bash
set -euo pipefail

# Bound raw selection, grouped-array size, and optimizer batch size separately.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

echo "Memory-bounded example: 512 selected rows -> 128 groups, batch size 16"
ptycho_train \
    --config "$SCRIPT_DIR/memory_constrained.yaml" \
    --data.train_data_file datasets/fly/fly001_transposed.npz \
    --output_dir memory_balanced_example

echo "Reduce train_raw_selection, training_groups, or batch_size if the corresponding stage is too large."
