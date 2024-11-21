# PtychoPINN Training Script

This script trains PtychoPINN from a .npz input and writes the resulting model artifacts to disk.

## Prerequisites

- PtychoPINN installation

## Input Data Format

The training script expects the input data to be in the following format:

- Coordinates (x, y) of the scan points
- Diffraction patterns
- Initial guess of the probe function
- Scan indices for each diffraction pattern
- (Optional) Initial guess of the object

The data should be provided as a NumPy `.npz` file with the following keys:
- `xcoords`: x coordinates of the scan points
- `ycoords`: y coordinates of the scan points
- `xcoords_start`: starting x coordinates for the scan (deprecated, same as `xcoords`)
- `ycoords_start`: starting y coordinates for the scan (deprecated, same as `ycoords`)
- `diff3d`: diffraction patterns with shape `(num_diffraction_patterns, N, N)`, where `N` is the model parameter (typically 64 or 128)
- `probeGuess`: initial guess of the probe function
- `scan_index`: array indicating the scan index for each diffraction pattern

Note: The distinction between `xcoords`/`ycoords` and `xcoords_start`/`ycoords_start` is only relevant if the iterative solver used to generate the probe ground truth used position correction. This distinction is deprecated, so `xcoords` and `xcoords_start` (and `ycoords` and `ycoords_start`) can be assumed to be the same.

The height and width of the diffraction patterns are equal and determined by the model parameter `N`, which is typically set to 64 or 128. The value of `N` should be consistent with the model configuration.

## Data Loading

By default, the training script loads up to 512 images from the input data file. This limit can be adjusted through the configuration.

## Configuration

The training script uses a configuration file (`config.yaml`) to set various parameters, such as:
- Number of epochs
- Batch size
- Learning rate
- Output directory
- Test data file path (optional)
- Image transformation options:
  - flip_x: Flip images horizontally
  - flip_y: Flip images vertically
  - transpose: Transpose the images
  - M: Image reassembly parameter (default: 20)

You can provide a custom configuration file using the `--config` command-line argument.

## Usage

1. Prepare your ptychographic imaging dataset in the required format.

2. (Optional) Create a configuration file with the desired training parameters.

3. Run the training script:
   ```
   python train.py --train_data_file /path/to/your/train_data.npz [--config /path/to/config.yaml]
   ```
   Replace `/path/to/your/train_data.npz` with the actual path to your training data file.

4. The script will load the data, preprocess it, and start training the model.

5. During training, the script will display progress information, such as the current epoch, loss values, and metrics.

6. After training, the script will save the trained model along with its associated files in the specified output directory.

## Output Structure

The training script generates the following output files:

- Model artifacts saved to the specified output directory
- Debug logs written to 'train_debug.log'
- Console output showing training progress

The output files are organized in the specified output directory. The exact structure and contents may vary depending on the configuration and training process.

## Logging

The script provides two levels of logging:
- Debug information is written to 'train_debug.log'
- Info level messages are displayed in the console

This helps track both detailed debugging information and high-level progress during training.

