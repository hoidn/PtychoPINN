#!/bin/bash

# Navigate to the ptycho directory
cd /home/ollie/Documents/PtychoPINN/ || exit 1

# Remove the build directory if it exists
if [ -d build/ ]; then
    rm -r build/
fi

# Install the package
python -m pip install .

# Return to the previous directory
cd - || exit 1

# Run the training script with the specified data file
pytest tests/test_loader.py

