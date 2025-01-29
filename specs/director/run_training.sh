#!/bin/bash

# Navigate to the ptycho directory
cd /home/ollie/Documents/scratch/ptycho/ || exit 1

# Remove the build directory if it exists
if [ -d build/ ]; then
    rm -r build/
fi

# Install the package
python -m pip install .

# Return to the previous directory
cd - || exit 1

# Run the training script with the specified data file
python ../../scripts/training/train.py --train_data_file ../../scripts/Run1084_recon3_postPC_shrunk_3.npz
