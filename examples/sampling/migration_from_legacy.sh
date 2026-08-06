#!/bin/bash
set -euo pipefail

# Print the public flat-to-nested migration without invoking training.

echo "Retired flat form:"
echo "  ptycho_train --train_data_file data.npz --n_images 1000"
echo
echo "Write numeric, Boolean, model, and sampling values in training.yaml:"
echo "  model:"
echo "    gridsize: 2"
echo "  sampling:"
echo "    n_subsample: 2000"
echo "    n_groups: 1000"
echo "    subsample_seed: 42"
echo
echo "Then use path/literal CLI overrides:"
echo "  ptycho_train --config training.yaml --data.train_data_file data.npz --output_dir run"
echo
echo "Deprecated YAML alias: sampling.n_images"
echo "Canonical YAML field: sampling.n_groups"
echo "Supplying unequal alias and canonical values fails validation."
echo
echo "Inference remains a separate flat interface:"
echo "  ptycho_inference --model_path model/ --test_data data.npz --n_subsample 5000 --n_groups 1000"
