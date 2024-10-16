#!/usr/bin/env python

import argparse
import yaml
import os
import numpy as np
import tensorflow as tf
from ptycho import params as p
from ptycho import probe
from ptycho import xpp
from ptycho.loader import RawData
import logging
from typing import Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt

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
    "probe_scale": ("probe_scale", 5),
    "train_data_file_path": ("train_data_file_path", None),
    "test_data_file_path": ("test_data_file_path", None),
    "N": ("N", 64)
}

def parse_arguments():
    """Parse command-line arguments for the CDI script."""
    parser = argparse.ArgumentParser(description="Non-grid CDI Example Script")
    
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    
    for arg_name, (_, default) in ARG_TO_CONFIG_MAP.items():
        if arg_name == "train_data_file_path":
            parser.add_argument(f"--{arg_name}", type=str, required=True, 
                                help="Path to the training data file")
        else:
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
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"Data file not found: {data_file_path}")

    try:
        return xpp.load_ptycho_data(data_file_path)
    except Exception as e:
        logger.error(f"Error loading data from {data_file_path}: {str(e)}")
        raise

def run_cdi_example(train_data: RawData, test_data: Optional[RawData], config: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """Run the main CDI example execution flow."""
    # Initialize
    probe.set_probe_guess(None, train_data.probeGuess)
    
    
    # Setup model and training
    from ptycho import loader, train_pinn
    tf.random.set_seed(45)
    np.random.seed(45)

    # Generate grouped data for training
    train_dataset = train_data.generate_grouped_data(config['N'], K=7, nsamples=1)
    train_data_container = loader.load(lambda: train_dataset, train_data.probeGuess,
                                       which=None, create_split=False)

    # Train the model
    intensity_scale = train_pinn.calculate_intensity_scale(train_data_container)
    history = train_pinn.train(train_data_container, intensity_scale)
    
    results = {"history": history}
    
    # Reconstruct test image if test data is provided
    if test_data is not None:
        raise NotImplementedError
#        test_dataset = test_data.generate_grouped_data(config['N'], K=7, nsamples=1)
#        obj_tensor_full, global_offsets = evaluation.reconstruct_image(test_dataset)
#        obj_image = xpp.loader.reassemble_position(obj_tensor_full, global_offsets[:, :, :, :], M=20)
#        
#        recon_amp = np.absolute(obj_image)
#        recon_phase = np.angle(obj_image)
#        
#        results.update({
#            "obj_tensor_full": obj_tensor_full,
#            "global_offsets": global_offsets,
#            "recon_amp": recon_amp,
#            "recon_phase": recon_phase
#        })
    else:
        recon_amp, recon_phase = None, None
    
    return recon_amp, recon_phase, results

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

def main(args) -> None:
    config = setup_configuration(args, args.config)
    
    try:
        ptycho_data, ptycho_data_train, obj = load_and_prepare_data(config['train_data_file_path'])
        
        test_data = None
        if config['test_data_file_path']:
            test_ptycho_data, test_ptycho_data_train, test_obj = load_and_prepare_data(config['test_data_file_path'])
            test_data = test_ptycho_data
        
        recon_amp, recon_phase, results = run_cdi_example(ptycho_data, test_data, config)
        save_outputs(recon_amp, recon_phase, results, config['output_prefix'])
    except Exception as e:
        logger.error(f"An error occurred during execution: {e}")
        raise

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
