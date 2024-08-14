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

## Configuration

The training script uses a configuration file (`config.yaml`) to set various parameters, such as:
- Number of epochs
- Batch size
- Learning rate
- Output directory
- Data-specific parameters

You can provide a custom configuration file using the `--config` command-line argument.

## Usage

1. Prepare your ptychographic imaging dataset in the required format.

2. (Optional) Create a configuration file with the desired training parameters.

3. Run the training script:
   ```
   python train.py --train_data_file_path /path/to/your/train_data.npz [--config /path/to/config.yaml]
   ```
   Replace `/path/to/your/train_data.npz` with the actual path to your training data file.

4. The script will load the data, preprocess it, and start training the model.

5. During training, the script will display progress information, such as the current epoch, loss values, and metrics.

6. After training, the script will save the trained model along with its associated files in the specified output directory.

## Output Structure

The training script generates the following output files for each trained model:

- `model.h5`: The trained model weights saved in HDF5 format.
- `custom_objects.dill`: A dill file containing the custom objects used in the model.
- `params.dill`: A dill file containing the model parameters and configuration.

The output files are organized in a directory structure as follows:
```
<output_directory>_<model_name>/
    ├── model.h5
    ├── custom_objects.dill
    └── params.dill
```

The `model_name` corresponds to the name of the trained model. The script saves two main models:

1. `autoencoder`: This is the full end-to-end model used for training. It takes the input diffraction patterns and positions, and outputs the reconstructed object, predicted amplitude (scaled), and predicted intensity (sampled from a Poisson distribution). The `autoencoder` model is compiled with multiple loss functions, including real-space loss, mean absolute error (MAE), and negative log-likelihood (NLL) of the Poisson distribution.

2. `diffraction_to_obj`: This is a submodel of the `autoencoder` used for inference or evaluation. It takes the input diffraction patterns and positions, and outputs only the reconstructed object. The `diffraction_to_obj` model is not compiled with any loss functions and is not directly used for training.

To load a trained model, you can use the `ModelManager.load_model(model_path, model_name)` function provided in the `model_manager.py` file. This function will load the model weights, custom objects, and parameters, and return the loaded model instance.

