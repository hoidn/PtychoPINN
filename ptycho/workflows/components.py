import argparse
import yaml
import os
import numpy as np
import tensorflow as tf
from ptycho import params as p
from ptycho import probe
from ptycho.loader import RawData, PtychoDataContainer
import logging
import matplotlib.pyplot as plt
from typing import Union, Optional, Dict, Any, Tuple, Literal
from pathlib import Path
from ptycho.config.config import TrainingConfig, ModelConfig, dataclass_to_legacy_dict
from dataclasses import fields
from ptycho import loader, probe
from typing import Union, Optional, Tuple, Dict, Any
from ptycho.raw_data import RawData
from ptycho.loader import PtychoDataContainer
from ptycho.config.config import TrainingConfig, update_legacy_dict
from ptycho import params
from ptycho.image import reassemble_patches

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from dataclasses import fields
from ptycho.config.config import ModelConfig, TrainingConfig

def update_config_from_dict(config_updates: dict):
    """
    Updates the application's configuration from a dictionary, ideal for notebook workflows.

    Args:
        config_updates (dict): A dictionary of parameters to update.
    """
    # 1. Create a mutable dictionary from the default dataclass values
    model_defaults = {f.name: f.default for f in fields(ModelConfig)}
    training_defaults = {f.name: f.default for f in fields(TrainingConfig) if f.name != 'model'}
    
    # Merge them
    full_config_dict = {**model_defaults, **training_defaults}

    # 2. Update with the user's dictionary
    for key, value in config_updates.items():
        if key in full_config_dict:
            full_config_dict[key] = value
        else:
            # Optionally warn about unused keys
            logger.warning(f"Configuration key '{key}' is not a recognized parameter.")

    # 3. Re-construct the dataclasses
    model_args = {k: v for k, v in full_config_dict.items() if k in model_defaults}
    training_args = {k: v for k, v in full_config_dict.items() if k in training_defaults}

    # Handle required Path objects if they are not set
    if training_args.get('train_data_file') is None:
        # Assign a dummy path or handle as an error if it's essential for all workflows
        training_args['train_data_file'] = Path("dummy_path.npz")

    final_model_config = ModelConfig(**model_args)
    final_training_config = TrainingConfig(model=final_model_config, **training_args)
    
    # 4. Update the legacy global params dictionary
    update_legacy_dict(params.cfg, final_training_config)
    
    logger.info("Configuration updated programmatically for interactive session.")
    params.print_params()

def load_data(file_path, n_images=None, flip_x=False, flip_y=False, swap_xy=False, n_samples=1, coord_scale=1.0):
    """
    Load ptychography data from a file and return RawData objects.

    Args:
        file_path (str, optional): Path to the data file. Defaults to the package resource 'datasets/Run1084_recon3_postPC_shrunk_3.npz'.
        n_images (int, optional): Number of data points to include in the training set. Defaults to 512.
        flip_x (bool, optional): If True, flip the sign of x coordinates. Defaults to False.
        flip_y (bool, optional): If True, flip the sign of y coordinates. Defaults to False.
        swap_xy (bool, optional): If True, swap x and y coordinates. Defaults to False.
        n_samples (int, optional): Number of samples to generate. Defaults to 1.
        coord_scale (float, optional): Scale factor for x and y coordinates. Defaults to 1.0.

    Returns:
        RawData: RawData object containing the dataset.
    """
    logger.info(f"Loading data from {file_path} with n_images={n_images}")
    # Load data from file
    data = np.load(file_path)

    # Extract required arrays from loaded data
    xcoords = data['xcoords']
    ycoords = data['ycoords']
    xcoords_start = data['xcoords_start']
    ycoords_start = data['ycoords_start']
    
    # Handle flexible diffraction key and shape
    diff_key = 'diff3d' if 'diff3d' in data else 'diffraction'
    diff_data = data[diff_key]
    
    # Only transpose if the data is in (H, W, N) format, not if it's already (N, H, W)
    if diff_data.shape[0] < diff_data.shape[2]:
        # Data is in (H, W, N) format, transpose to (N, H, W)
        diff3d = np.transpose(diff_data, [2, 0, 1])
    else:
        # Data is already in (N, H, W) format
        diff3d = diff_data
    
    probeGuess = data['probeGuess']
    objectGuess = data.get('objectGuess', None)
    
    # --- FIX: Load the 'Y' array if it exists ---
    Y_patches = data['Y'] if 'Y' in data else None
    # ---------------------------------------------

    # Apply coordinate transformations
    if flip_x:
        xcoords = -xcoords
        xcoords_start = -xcoords_start
        #probeGuess = probeGuess[::-1, :]
    if flip_y:
        ycoords = -ycoords
        ycoords_start = -ycoords_start
        #probeGuess = probeGuess[:, ::-1]
    if swap_xy:
        xcoords, ycoords = ycoords, xcoords
        xcoords_start, ycoords_start = ycoords_start, xcoords_start
        #probeGuess = np.transpose(probeGuess)

    # Apply coordinate scaling
    xcoords *= coord_scale
    ycoords *= coord_scale
    xcoords_start *= coord_scale
    ycoords_start *= coord_scale

    # Create scan_index array
    scan_index = np.zeros(diff3d.shape[0], dtype=int)

    if n_images is None:
        n_images = xcoords.shape[0]
    
    # For gridsize > 1, pass full dataset to enable grouping-aware subsampling
    # For gridsize = 1, use traditional sequential slicing for backward compatibility
    from ptycho import params
    gridsize = params.cfg.get('gridsize', 1)
    
    if gridsize == 1:
        # Traditional behavior: sequential slicing
        logger.info(f"Using sequential slicing for gridsize=1: selecting first {n_images} images")
        selected_indices = slice(None, n_images)
    else:
        # Grouping-aware subsampling behavior: pass full dataset, let generate_grouped_data handle selection
        logger.info(f"Using grouping-aware subsampling for gridsize={gridsize}: passing full dataset for group-first sampling")
        selected_indices = slice(None)  # Full dataset
    
    # Create RawData object with appropriate data subset
    ptycho_data = RawData(xcoords[selected_indices], ycoords[selected_indices],
                          xcoords_start[selected_indices], ycoords_start[selected_indices],
                          diff3d[selected_indices], probeGuess,
                          scan_index[selected_indices], objectGuess=objectGuess,
                          # --- FIX: Pass the loaded Y array to the constructor ---
                          Y=(Y_patches[selected_indices] if Y_patches is not None else None))

    return ptycho_data

def parse_arguments():
    """Parse command-line arguments based on TrainingConfig fields."""
    from ptycho.cli_args import add_logging_arguments
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description="Non-grid CDI Example Script")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--do_stitching", action='store_true', default=False,
                        help="Perform image stitching after training (default: False)")
    
    # Add logging arguments
    add_logging_arguments(parser)
    
    # Add arguments based on TrainingConfig fields
    for field in fields(TrainingConfig):
        if field.name == 'model':
            # Handle ModelConfig fields
            for model_field in fields(ModelConfig):
                # Special handling for Literal types
                if hasattr(model_field.type, "__origin__") and model_field.type.__origin__ is Literal:
                    choices = list(model_field.type.__args__)
                    parser.add_argument(
                        f"--{model_field.name}",
                        type=str,
                        choices=choices,
                        default=model_field.default,
                        help=f"Model parameter: {model_field.name}, choices: {choices}"
                    )
                else:
                    parser.add_argument(
                        f"--{model_field.name}",
                        type=model_field.type,
                        default=model_field.default,
                        help=f"Model parameter: {model_field.name}"
                    )
        else:
            # Handle path fields specially
            if field.type == Path or str(field.type).startswith("typing.Optional[pathlib.Path"):
                logger.debug(f"Field: {field.name}")
                logger.debug(f"Field type: {field.type}")
                logger.debug(f"Field default: {field.default}")
                parser.add_argument(
                    f"--{field.name}",
                    type=lambda x: (logger.debug(f"Converting path value: {x}"), Path(x) if x is not None else None)[1],
                    default=None if field.default == None else str(field.default),
                    help=f"Path for {field.name}"
                )
            else:
                # Special handling for specific parameters to provide better help text
                if field.name == 'n_images':
                    parser.add_argument(
                        f"--{field.name}",
                        type=field.type,
                        default=field.default,
                        help=f"Number of images to use from the dataset (default: {field.default})"
                    )
                else:
                    parser.add_argument(
                        f"--{field.name}",
                        type=field.type,
                        default=field.default,
                        help=f"Training parameter: {field.name}"
                    )
    
    return parser.parse_args()

def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except (yaml.YAMLError, IOError) as e:
        logger.error(f"Error loading YAML config: {e}")
        raise


#def validate_config(config: Dict[str, Any]) -> None:
#    """Validate the configuration."""
#    if 'train_data_file_path' not in config or config['train_data_file_path'] is None:
#        raise ValueError("train_data_file_path is a required parameter and must be provided")

def setup_configuration(args: argparse.Namespace, yaml_path: Optional[str]) -> TrainingConfig:
    """Set up the configuration by merging defaults, YAML file, and command-line arguments."""
    try:
        yaml_config = load_yaml_config(yaml_path) if yaml_path else None
        args_config = vars(args)
        
        # Convert string paths to Path objects
        for key in ['train_data_file', 'test_data_file', 'output_dir']:
            if key in args_config and args_config[key] is not None:
                args_config[key] = Path(args_config[key])
        
        # Create ModelConfig from args
        model_fields = {f.name for f in fields(ModelConfig)}
        model_args = {k: v for k, v in args_config.items() if k in model_fields}
        model_config = ModelConfig(**model_args)
        
        # Create TrainingConfig
        training_fields = {f.name for f in fields(TrainingConfig)}
        training_args = {k: v for k, v in args_config.items() 
                        if k in training_fields and k != 'model'}
        
        # Log the CLI arguments for debugging
        logger.debug(f"CLI arguments: {args_config}")
        logger.debug(f"Training arguments: {training_args}")
        
        config = TrainingConfig(model=model_config, **training_args)
        
        # Update the global configuration
        update_legacy_dict(params.cfg, config)
        
        logger.info("Configuration setup complete")
        logger.info(f"Final configuration: {config}")
        
        return config
    except (yaml.YAMLError, IOError, ValueError) as e:
        logger.error(f"Error setting up configuration: {e}")
        raise

def load_and_prepare_data(data_file_path: str) -> Tuple[RawData, RawData, Any]:
    """
    Load and prepare the data from a single file path.

    Args:
        data_file_path (str): Path to the data file

    Returns:
        Tuple[RawData, RawData, Any]: A tuple containing the full dataset, training subset, and additional data
    """
    # TODO deprecated
    from ptycho.loader import load_xpp_npz
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"Data file not found: {data_file_path}")

    try:
        return load_xpp_npz(data_file_path)
    except Exception as e:
        logger.error(f"Error loading data from {data_file_path}: {str(e)}")
        raise

from typing import Union
from ptycho.loader import RawData, PtychoDataContainer

def create_ptycho_data_container(data: Union[RawData, PtychoDataContainer], config: TrainingConfig) -> PtychoDataContainer:
    """
    Factory function to create or return a PtychoDataContainer.

    Args:
        data (Union[RawData, PtychoDataContainer]): Input data, either RawData or PtychoDataContainer.
        config (TrainingConfig): Training configuration object.

    Returns:
        PtychoDataContainer: The resulting PtychoDataContainer.

    Raises:
        TypeError: If the input data is neither RawData nor PtychoDataContainer.
    """
    if isinstance(data, PtychoDataContainer):
        return data
    elif isinstance(data, RawData):
        # Use config.n_images for nsamples - this is the interpreted value from the training script
        dataset = data.generate_grouped_data(
            config.model.N, 
            K=4, 
            nsamples=config.n_images,  # Use interpreted n_images
            dataset_path=str(config.train_data_file) if config.train_data_file else None
        )
        return loader.load(lambda: dataset, data.probeGuess, which=None, create_split=False)
    else:
        raise TypeError("data must be either RawData or PtychoDataContainer")

def train_cdi_model(
    train_data: Union[RawData, PtychoDataContainer],
    test_data: Optional[Union[RawData, PtychoDataContainer]],
    config: TrainingConfig
) -> Dict[str, Any]:
    """
    Train the CDI model.

    Args:
        train_data (Union[RawData, PtychoDataContainer]): Training data.
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        Dict[str, Any]: Results dictionary containing training history.
    """
    from ptycho.loader import PtychoDataset
    from ptycho import train_pinn
    # Convert input data to PtychoDataContainer
    train_container = create_ptycho_data_container(train_data, config)
    if test_data is not None:
        test_container = create_ptycho_data_container(test_data, config)
    else:
        test_container = None

    # Initialize probe
    probe.set_probe_guess(None, train_container.probe)

#    # Calculate intensity scale
#    intensity_scale = train_pinn.calculate_intensity_scale(train_container)

    # Train the model
    results = train_pinn.train_eval(PtychoDataset(train_container, test_container))
    results['train_container'] = train_container
    results['test_container'] = test_container
    #history = train_pinn.train(train_container)
    
    return results

def reassemble_cdi_image(
    test_data: Union[RawData, PtychoDataContainer],
    config: TrainingConfig,
    flip_x: bool = False,
    flip_y: bool = False,
    transpose: bool = False,
    M: int = 20,
    coord_scale: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Reassemble the CDI image using the trained model.

    Args:
        test_data (Union[RawData, PtychoDataContainer]): Test data.
        config (Dict[str, Any]): Configuration dictionary.
        flip_x (bool): Whether to flip the x coordinates. Default is False.
        flip_y (bool): Whether to flip the y coordinates. Default is False.
        transpose (bool): Whether to transpose the image by swapping the 1st and 2nd dimensions. Default is False.
        M (int): Parameter for reassemble_position function. Default is 20.
        coord_scale (float): Scale factor for x and y coordinates. Default is 1.0.

    Returns:
        Tuple[np.ndarray, np.ndarray, Dict[str, Any]]: 
        Reconstructed amplitude, reconstructed phase, and results dictionary.
    """
    # TODO use train_pinn.eval to get reconstructed diffraction amplitude
    test_container = create_ptycho_data_container(test_data, config)
    
    from ptycho import nbutils
    obj_tensor_full, global_offsets = nbutils.reconstruct_image(test_container)
    
    # Log the shape of global_offsets
    logger.info(f"Shape of global_offsets: {global_offsets.shape}")

    # Assert that obj_tensor_full is a 4D tensor
    assert obj_tensor_full.ndim == 4, f"Expected obj_tensor_full to be a 4D tensor, but got shape {obj_tensor_full.shape}"

    # Transpose the image if requested
    if transpose:
        obj_tensor_full = np.transpose(obj_tensor_full, (0, 2, 1, 3))

    # Flip coordinates if requested
    if flip_x:
        global_offsets[:, 0, 0, :] = -global_offsets[:, 0, 0, :]
    if flip_y:
        global_offsets[:, 0, 1, :] = -global_offsets[:, 0, 1, :]
    
    # Scale coordinates
    global_offsets *= coord_scale
    
    from ptycho import tf_helper as hh
    obj_image = hh.reassemble_position(obj_tensor_full, global_offsets, M=M)
    
    recon_amp = np.absolute(obj_image)
    recon_phase = np.angle(obj_image)
    
    results = {
        "obj_tensor_full": obj_tensor_full,
        "global_offsets": global_offsets,
        "recon_amp": recon_amp,
        "recon_phase": recon_phase
    }
    
    return recon_amp, recon_phase, results

def run_cdi_example(
    train_data: Union[RawData, PtychoDataContainer],
    test_data: Optional[Union[RawData, PtychoDataContainer]],
    config: TrainingConfig,
    flip_x: bool = False,
    flip_y: bool = False,
    transpose: bool = False,
    M: int = 20,
    do_stitching: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """
    Run the main CDI example execution flow.

    Args:
        train_data: Training data
        test_data: Optional test data
        config: Training configuration parameters
        flip_x: Whether to flip the x coordinates
        flip_y: Whether to flip the y coordinates
        transpose: Whether to transpose the image by swapping dimensions
        M: Parameter for reassemble_position function
        do_stitching: Whether to perform image stitching after training

    Returns:
        Tuple containing:
        - reconstructed amplitude (or None)
        - reconstructed phase (or None)
        - results dictionary
    """
    # Update global params with new-style config at entry point
    update_legacy_dict(params.cfg, config)
    
    # Train the model
    train_results = train_cdi_model(train_data, test_data, config)
    
    recon_amp, recon_phase = None, None
    
    # Reassemble test image if stitching is enabled, test data is provided, and reconstructed_obj is available
    if do_stitching and test_data is not None and 'reconstructed_obj' in train_results:
        logger.info("Performing image stitching...")
        recon_amp, recon_phase, reassemble_results = reassemble_cdi_image(
            test_data, config, flip_x, flip_y, transpose, M=M
        )
        train_results.update(reassemble_results)
    else:
        logger.info("Skipping image stitching (disabled or no test data available)")
    
    return recon_amp, recon_phase, train_results


def save_outputs(amplitude: Optional[np.ndarray], phase: Optional[np.ndarray], results: Dict[str, Any], output_prefix: str) -> None:
    """Save the generated images and results."""
    os.makedirs(output_prefix, exist_ok=True)
    
    # TODO Save training history with tensorboard / mlflow
    
    # Save test results if available
    if amplitude is not None and phase is not None:
        logger.info(f"Amplitude array shape: {amplitude.shape}")
        logger.info(f"Phase array shape: {phase.shape}")
        
        # Squeeze any extra dimensions
        amplitude = np.squeeze(amplitude)
        phase = np.squeeze(phase)
        
        logger.info(f"Squeezed amplitude shape: {amplitude.shape}")
        logger.info(f"Squeezed phase shape: {phase.shape}")
        
        # Save as PNG files using plt.figure() to handle 2D arrays properly
        plt.figure(figsize=(8,8))
        plt.imshow(amplitude, cmap='gray')
        plt.colorbar()
        plt.savefig(os.path.join(output_prefix, "reconstructed_amplitude.png"))
        plt.close()
        
        plt.figure(figsize=(8,8))
        plt.imshow(phase, cmap='viridis')
        plt.colorbar()
        plt.savefig(os.path.join(output_prefix, "reconstructed_phase.png"))
        plt.close()
        
    logger.info(f"Outputs saved to {output_prefix}")
