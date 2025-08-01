# PtychoPINN Simulation Script

This script simulates ptychography data, generates visualizations, and performs a comparison between PtychoPINN and baseline reconstructions.

## Prerequisites

- PtychoPINN package

## Usage

python simulation.py <input_file> <output_dir> [options]

## Arguments

- `input_file`: (Required) Path to the input .npz file containing probe and object guesses.
- `output_dir`: (Required) Directory to save output visualizations.

## Options

- `--nimages`: Number of images to simulate. Default is 2000.
- `--seed`: Random seed for reproducibility.
- `--nepochs`: Number of epochs for training. Default is 50.
- `--output_prefix`: Prefix for output files. Default is "tmp".
- `--intensity_scale_trainable`: Make intensity scale trainable. Default is False.
- `--positions_provided`: Positions are provided. Default is True.
- `--probe_big`: Use big probe. Default is True.
- `--probe_mask`: Use probe mask. Default is False.
- `--data_source`: Data source type. Default is "generic".
- `--gridsize`: Grid size. Default is 1.
- `--train_data_file_path`: Path to train data file.
- `--test_data_file_path`: Path to test data file.
- `--N`: Size of the simulation grid. Default is 128.
- `--probe_scale`: Probe scale factor. Default is 4.
- `--nphotons`: Number of photons. Default is 1e9.
- `--mae_weight`: Weight for MAE loss. Default is 1.
- `--nll_weight`: Weight for NLL loss. Default is 0.
- `--config`: Path to YAML configuration file.

## Example

python simulation.py input_data.npz output_results/ --nimages 1000 --seed 42 --nepochs 100 --N 256 

## Input Data Format

The script expects the input data to be in .npz format with the following arrays:

- `probeGuess`: Initial guess of the probe function
- `objectGuess`: Initial guess of the object

## Output

The script generates the following outputs in the specified output directory:

1. Simulated data visualization
2. Random groups visualizations (3 sets)
3. Reconstruction comparison image
4. HTML report (`report.html`) containing:
   - Embedded visualizations
   - Launch command
   - Model parameters

## Process

1. The script simulates ptychography data based on the input file and specified parameters.
2. It generates visualizations of the simulated data and random groups.
3. The script then runs a CDI example using PtychoPINN and compares it with a baseline reconstruction.
4. Finally, it generates an HTML report with all visualizations and parameters.

## Notes

- The script uses logging to provide information about the process. Check the console output for details.
- The HTML report provides a comprehensive overview of the simulation results and parameters used.
- Adjust the simulation parameters to experiment with different scenarios and data characteristics.
