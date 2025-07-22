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

The script generates the following outputs in the specified `--output_dir`:

1. **`logs/`**: Directory containing all log files
   - **`debug.log`**: Complete log history (DEBUG level and above) for troubleshooting
2. **`reconstruction_comparison.png`**: Comparison image showing:
   - Reconstructed amplitude and phase
   - ePIE amplitude and phase (if ground truth available)
3. **`reconstructed_amplitude.png`**: Individual amplitude reconstruction
4. **`reconstructed_phase.png`**: Individual phase reconstruction  
5. **`probe_visualization.png`**: (Optional) Generated when `--visualize_probe` flag is used

All outputs are organized within the specified output directory.

## Process

1. The script first loads the trained model and its configuration.
2. It then loads the test data from the provided .npz file.
3. Inference is performed using the loaded model and test data.
4. The results are processed to generate the comparison image.
5. If requested, a probe visualization is generated.

## Enhanced Logging

The script uses an advanced centralized logging system with comprehensive output capture:

**Key Features:**
- **Complete Output Capture**: ALL stdout (including print statements from any module) is captured to log files
- **Tee-style Logging**: Simultaneous console and file output with flexible control
- **Command-line Options**: Control console verbosity without affecting file logging

**File Logging**: The `<output_dir>/logs/debug.log` file contains:
- All logging messages (DEBUG level and above)  
- All print() statements from any imported module
- Model loading information and inference progress
- Complete execution record

**Console Control Options**:
```bash
# Default: INFO level console output + complete file logging
ptycho_inference --model_path model_dir --test_data test.npz --output_dir results

# Quiet mode: suppress console logging (automation-friendly)
ptycho_inference --model_path model_dir --test_data test.npz --output_dir results --quiet

# Verbose mode: DEBUG level console output + complete file logging
ptycho_inference --model_path model_dir --test_data test.npz --output_dir results --verbose
```

**Important**: These flags only affect console output. All messages are ALWAYS captured in the log file.

This centralized approach ensures logs are organized within each inference run's output directory.

## Notes

- Ensure that the model files (saved using the training script) are located at the path specified by `model_prefix`.
- The `K` and `nsamples` parameters can be adjusted to experiment with different data grouping strategies.
- For troubleshooting, check both console output and the debug log file in the output directory's logs subdirectory.

