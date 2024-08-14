# PtychoPINN Inference Script

This script performs inference using a trained instance of PtychoPINN on test data and generates reconstruction results.

## Prerequisites

- PtychoPINN package

## Usage

```bash
python inference_script.py --model_prefix <model_prefix> --test_data <test_data_file> [--output_path <output_path>] [--visualize_probe] [--K <K>] [--nsamples <nsamples>]
```

## Arguments

- `--model_prefix`: (Required) Path prefix for the saved model and its configuration.
- `--test_data`: (Required) Path to the .npz file containing test data.
- `--output_path`: (Optional) Path prefix for saving output files and images. Default is './'.
- `--visualize_probe`: (Optional) Flag to generate and save probe visualization.
- `--K`: (Optional) Number of nearest neighbors for grouped data generation. Default is 7.
- `--nsamples`: (Optional) Number of samples for grouped data generation. Default is 1.

## Example

```bash
python inference_script.py --model_prefix models/my_model --test_data data/test_data.npz --output_path results/ --visualize_probe
```

## Input Data Format

The script expects the test data to be in .npz format with the following arrays:

- `xcoords`: x coordinates of the scan points
- `ycoords`: y coordinates of the scan points
- `xcoords_start`: starting x coordinates for the scan
- `ycoords_start`: starting y coordinates for the scan
- `diffraction`: diffraction patterns
- `probeGuess`: initial guess of the probe function
- `objectGuess`: initial guess of the object

## Output

The script generates the following outputs:

1. A comparison image (`reconstruction_comparison.png`) showing:
   - Reconstructed amplitude
   - Reconstructed phase
   - ePIE amplitude
   - ePIE phase
2. If `--visualize_probe` is used, a probe visualization image (`probe_visualization.png`)
3. Log file (`inference.log`) with detailed information about the inference process

All outputs are saved in the specified output directory.

## Process

1. The script first loads the trained model and its configuration.
2. It then loads the test data from the provided .npz file.
3. Inference is performed using the loaded model and test data.
4. The results are processed to generate the comparison image.
5. If requested, a probe visualization is generated.

## Notes

- The script uses logging to provide information about the process. Check the console output and log file for details.
- Ensure that the model files (saved using the training script) are located at the path specified by `model_prefix`.
- The `K` and `nsamples` parameters can be adjusted to experiment with different data grouping strategies.

