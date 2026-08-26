#!/bin/bash
set -euo pipefail

# Use every selected row once as a grouping anchor. Neighbor rows may overlap.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

echo "Dense anchor coverage: 2000 selected rows -> 2000 groups (gridsize=2)"
ptycho_train \
    --config "$SCRIPT_DIR/dense_grouping.yaml" \
    --data.train_data_file datasets/fly/fly001_transposed.npz \
    --output_dir dense_gs2_example

echo "Each selected row is used once as an anchor; groups are not disjoint."
