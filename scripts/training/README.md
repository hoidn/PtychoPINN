# PtychoPINN Training Script

This script trains PtychoPINN from a .npz input and writes the resulting model artifacts to disk.

## Prerequisites

- PtychoPINN installation

## Input Data Format

The training script expects the input data to be in the following format:

- Coordinates (x, y) of the scan points
- Diffraction patterns
- Ground truth of the probe function
- Scan indices for each diffraction pattern
- (Optional) Ground truth of the object

The data should be provided as a NumPy `.npz` file with the following keys:
- `xcoords`: x coordinates of the scan points
- `ycoords`: y coordinates of the scan points
- `xcoords_start`: starting x coordinates for the scan (deprecated, same as `xcoords`)
- `ycoords_start`: starting y coordinates for the scan (deprecated, same as `ycoords`)
- `diff3d`: diffraction patterns with shape `(num_diffraction_patterns, N, N)`, where `N` is the model parameter (typically 64 or 128)
- `probeGuess`: complex-valued probe ground truth
- `scan_index`: array indicating the scan index for each diffraction pattern

Note: The distinction between `xcoords`/`ycoords` and `xcoords_start`/`ycoords_start` is only relevant if the iterative solver used to generate the probe ground truth used position correction. This distinction is deprecated, so `xcoords` and `xcoords_start` (and `ycoords` and `ycoords_start`) can be assumed to be the same.

The height and width of the diffraction patterns are equal and determined by the model parameter `N`, which is typically set to 64 or 128. The value of `N` should be consistent with the model configuration.

## Data Loading

By default, the training script loads up to 512 images from the input data file. This limit is hardcoded but can be modified in the source code if needed.

## Configuration

The training script uses a configuration file (`config.yaml`) to set various parameters. The configuration system supports both new-style configuration and legacy parameters. Key parameters include:

- Number of epochs (`nepochs`)
- Batch size (`batch_size`)
- Output directory (`output_dir`)
- Train data file path (`train_data_file`)
- Test data file path (`test_data_file`, optional)
- Model parameters:
  - N: Size of diffraction patterns (64, 128, or 256)
  - gridsize: Grid size for model - controls number of images processed per solution region (e.g., gridsize=2 means 2²=4 images at a time)
  - n_filters_scale: Scale factor for number of filters
  - model_type: 'pinn' or 'supervised'
  - amp_activation: Activation function ('sigmoid', 'swish', 'softplus', 'relu')

### Important Considerations

- **Subsampling with `gridsize > 1`:** Be aware that using the `--n_images` flag selects the *first N* images from the dataset. If `gridsize` is greater than 1, this will result in training on a spatially biased, non-representative subset of your object. For robust training in this mode, it is better to prepare a smaller, complete dataset rather than subsampling a large one.
  - Various boolean flags for model configuration

You can provide a custom configuration file using the `--config` command-line argument.

## Usage

1. Prepare your ptychographic imaging dataset in the required format.

2. (Optional) Create a configuration file with the desired training parameters.

3. Run the training script:
   ```
   python train.py --train_data_file /path/to/your/train_data.npz [--config /path/to/config.yaml]
   ```
   Note: The script supports both `--train_data_file` and the legacy `--train_data_file_path` arguments.

4. The script will:
   - Load and validate the configuration
   - Load the training data (and test data if specified)
   - Run the CDI example
   - Save the model and outputs
   - Display progress information during training

## Error Handling

The script includes comprehensive error handling:
- All exceptions during execution are caught and logged
- Detailed error messages are written to both the debug log and console
- The script will exit with an error status if any critical errors occur

## Output Structure

The training script generates the following outputs in the specified `--output_dir`:

- **`logs/`**: Directory containing all log files
  - **`debug.log`**: Complete log history (DEBUG level and above) 
- **Model artifacts**: Saved model files and weights
- **Training results**: Including reconstructed amplitude, phase, and metrics
- **Console output**: Real-time training progress (INFO level)

## Logging

The script uses the centralized logging system:
- **File logging**: All messages (DEBUG level and above) are written to `<output_dir>/logs/debug.log`
- **Console logging**: INFO level messages are displayed in real-time for monitoring training progress

This centralized approach ensures logs are organized within each training run's output directory, making it easier to debug specific runs and keep the project root clean.

