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
from typing import Union, Optional, Dict, Any, Tuple
from ptycho import loader, probe

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the mapping between command-line argument names and config keys with defaults
ARG_TO_CONFIG_MAP = {
    "nepochs": ("nepochs", 50),
    "output_prefix": ("output_prefix", "tmp"),
    "intensity_scale_trainable": ("intensity_scale.trainable", True),
    "positions_provided": ("positions.provided", True),
    "probe_big": ("probe.big", True),
    "probe_mask": ("probe.mask", False),
    "data_source": ("data_source", "generic"),
    "gridsize": ("gridsize", 1),
    "probe_scale": ("probe_scale", 4),
    "train_data_file_path": ("train_data_file_path", None),
    "test_data_file_path": ("test_data_file_path", None),
    "N": ("N", 64)
}

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
    # Load data from file
    data = np.load(file_path)

    # Extract required arrays from loaded data
    xcoords = data['xcoords']
    ycoords = data['ycoords']
    xcoords_start = data['xcoords_start']
    ycoords_start = data['ycoords_start']
    diff3d = np.transpose(data['diffraction'], [2, 0, 1])
    probeGuess = data['probeGuess']
    objectGuess = data['objectGuess']

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

    # Create RawData object for the training subset
    ptycho_data = RawData(xcoords[:n_images], ycoords[:n_images],
                          xcoords_start[:n_images], ycoords_start[:n_images],
                          diff3d[:n_images], probeGuess,
                          scan_index[:n_images], objectGuess=objectGuess)

    return ptycho_data

def update_params(new_config):
    for k2, new_value in new_config.items():
        for arg, (config_k2, _) in ARG_TO_CONFIG_MAP.items():
            if config_k2 == k2:
                ARG_TO_CONFIG_MAP[arg] = (k2, new_value)
                break
        else:
            p.set(k2, new_value)

def parse_arguments():
    """Parse command-line arguments for the CDI script."""
    parser = argparse.ArgumentParser(description="Non-grid CDI Example Script")
    
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    
    for arg_name, (_, default) in ARG_TO_CONFIG_MAP.items():
        if __name__ == '__main__':
            parser.add_argument(f"--{arg_name}", type=str, required=True, 
                                help="Path to the training data file")
        parser.add_argument(f"--{arg_name}", type=type(default) if default is not None else str, 
                            default=default, help=f"Default: {default}")
    
    return parser.parse_args()

def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except (yaml.YAMLError, IOError) as e:
        logger.error(f"Error loading YAML config: {e}")
        raise

def merge_configs(yaml_config: Optional[Dict[str, Any]], args_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge configurations with explicit precedence: defaults -> YAML -> command-line args."""
    config = p.cfg.copy()  # Start with default configuration
    
    if yaml_config:
        config.update(yaml_config)  # Update with YAML configuration
    
    # Update with command-line args, using ARG_TO_CONFIG_MAP
    for arg_name, (cfg_key, default) in ARG_TO_CONFIG_MAP.items():
        if args_config[arg_name] is not None:
            config[cfg_key] = args_config[arg_name]
        elif cfg_key not in config:
            config[cfg_key] = default

    return config

def validate_config(config: Dict[str, Any]) -> None:
    """Validate the configuration."""
    if 'train_data_file_path' not in config or config['train_data_file_path'] is None:
        raise ValueError("train_data_file_path is a required parameter and must be provided")

def setup_configuration(args: argparse.Namespace, yaml_path: Optional[str]) -> Dict[str, Any]:
    """Set up the configuration by merging defaults, YAML file, and command-line arguments."""
    try:
        yaml_config = load_yaml_config(yaml_path) if yaml_path else None
        args_config = vars(args)
        config = merge_configs(yaml_config, args_config)
        validate_config(config)
        p.cfg.update(config)  # Update the global configuration
        
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

def create_ptycho_data_container(data: Union[RawData, PtychoDataContainer], config: Dict[str, Any]) -> PtychoDataContainer:
    """
    Factory function to create or return a PtychoDataContainer.

    Args:
        data (Union[RawData, PtychoDataContainer]): Input data, either RawData or PtychoDataContainer.
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        PtychoDataContainer: The resulting PtychoDataContainer.

    Raises:
        TypeError: If the input data is neither RawData nor PtychoDataContainer.
    """
    if isinstance(data, PtychoDataContainer):
        return data
    elif isinstance(data, RawData):
        dataset = data.generate_grouped_data(config['N'], K=7, nsamples=config.get('n_samples', 1))
        return loader.load(lambda: dataset, data.probeGuess, which=None, create_split=False)
    else:
        raise TypeError("data must be either RawData or PtychoDataContainer")

def train_cdi_model(
    train_data: Union[RawData, PtychoDataContainer],
    test_data: Optional[Union[RawData, PtychoDataContainer]],
    config: Dict[str, Any]
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
    config: Dict[str, Any],
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
    
    obj_image = loader.reassemble_position(obj_tensor_full, global_offsets, M=M)
    
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
    config: Dict[str, Any],
    flip_x: bool = False,
    flip_y: bool = False,
    transpose: bool = False,
    M: int = 20
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """
    Run the main CDI example execution flow.

    Args:
        train_data (Union[RawData, PtychoDataContainer]): Training data.
        test_data (Optional[Union[RawData, PtychoDataContainer]]): Test data, or None.
        config (Dict[str, Any]): Configuration dictionary.
        flip_x (bool): Whether to flip the x coordinates. Default is False.
        flip_y (bool): Whether to flip the y coordinates. Default is False.
        transpose (bool): Whether to transpose the image by swapping the 1st and 2nd dimensions. Default is False.

    Returns:
        Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]: 
        Reconstructed amplitude, reconstructed phase, and results dictionary.
    """
    # Train the model
    train_results = train_cdi_model(train_data, test_data, config)
    
    recon_amp, recon_phase = None, None
    
    # Reassemble test image if test data is provided and reconstructed_obj is available
    if test_data is not None and 'reconstructed_obj' in train_results:
        recon_amp, recon_phase, reassemble_results = reassemble_cdi_image(test_data, config, flip_x, flip_y, transpose, M=M)
        train_results.update(reassemble_results)
    
    return recon_amp, recon_phase, train_results


def save_outputs(amplitude: Optional[np.ndarray], phase: Optional[np.ndarray], results: Dict[str, Any], output_prefix: str) -> None:
    """Save the generated images and results."""
    os.makedirs(output_prefix, exist_ok=True)
    
    # TODO Save training history with tensorboard / mlflow
    
    # Save test results if available
    if amplitude is not None and phase is not None:
#        # Save as NumPy files
#        np.save(os.path.join(output_prefix, "reconstructed_amplitude.npy"), amplitude)
#        np.save(os.path.join(output_prefix, "reconstructed_phase.npy"), phase)
#        np.save(os.path.join(output_prefix, "obj_tensor_full.npy"), results["obj_tensor_full"])
#        np.save(os.path.join(output_prefix, "global_offsets.npy"), results["global_offsets"])
        
        # Save as PNG files
        plt.imsave(os.path.join(output_prefix, "reconstructed_amplitude.png"), amplitude, cmap='gray')
        plt.imsave(os.path.join(output_prefix, "reconstructed_phase.png"), phase, cmap='viridis')
        
    logger.info(f"Outputs saved to {output_prefix}")
