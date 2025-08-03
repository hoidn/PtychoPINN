#!/usr/bin/env python
"""
Main training script for PtychoPINN models using modern configuration system.

This is the primary entry point for training both PINN (physics-informed) and supervised
baseline models for ptychographic reconstruction. The script supports flexible configuration
through YAML files or command-line arguments, with automatic parameter validation and
organized output directory structure.

Key Features:
- Modern dataclass-based configuration with YAML support
- Automatic interpretation of --n_images based on gridsize
- Comprehensive logging and error handling
- Optional stitching for full object reconstruction
- Support for both training-only and train+test workflows

Usage:
    ptycho_train [--config CONFIG_FILE] [ARGUMENTS]

Arguments:
    --config: Path to YAML configuration file (recommended approach)
    --train-data-file: Path to training NPZ file (required if no config)
    --test-data-file: Path to test NPZ file (optional)
    --output-dir: Directory for outputs (default: current directory)
    --n-images: Number of images/groups to use (see note below)
    --nepochs: Number of training epochs (default: 50)
    --batch-size: Batch size for training (default: 16)
    --model-type: Model type - 'pinn' or 'supervised' (default: pinn)
    --do-stitching: Enable patch stitching for visualization
    
    Additional arguments control model architecture, physics parameters, and
    training hyperparameters. Use --help for complete list.

Critical Note on --n-images:
    The interpretation of --n-images depends on gridsize:
    - gridsize=1: Refers to individual diffraction patterns
    - gridsize>1: Refers to number of neighbor groups
      (total patterns = n_images * gridsize^2)
    
    Example: --n-images=100 with gridsize=2 results in 400 total patterns

Examples:
    # Example 1: Quick verification test with minimal data
    ptycho_train --train-data-file datasets/fly/fly001_transposed.npz \\
                 --n-images 512 --output-dir verification_run

    # Example 2: Full training using YAML configuration (recommended)
    ptycho_train --config configs/experiment_config.yaml

    # Example 3: Override specific parameters from YAML
    ptycho_train --config configs/base_config.yaml \\
                 --n-images 2000 --nepochs 100 --output-dir custom_run

    # Example 4: Train supervised baseline with test data
    ptycho_train --train-data-file train_data.npz \\
                 --test-data-file test_data.npz \\
                 --model-type supervised \\
                 --n-images 5000 \\
                 --output-dir baseline_model

Input Requirements:
    Training data must be in NPZ format with required keys:
    - 'diffraction': Amplitude data (not intensity), shape (n, N, N)
    - 'objectGuess': Complex object array
    - 'probeGuess': Complex probe array
    - 'xcoords', 'ycoords': Scan position coordinates
    
    For supervised models, additionally requires:
    - 'Y': Pre-extracted object patches
    
    See docs/data_contracts.md for complete specifications.

Output Structure:
    The output directory will contain:
    - output_dir/
        - wts.h5.zip: Trained model weights (main output)
        - history.dill: Training history for plotting
        - params.dill: Configuration snapshot
        - reconstructed_amplitude.png: Final reconstruction visualization
        - reconstructed_phase.png: Phase reconstruction
        - metrics.csv: Evaluation metrics (if test data provided)
        - logs/
            - debug.log: Complete training log with DEBUG level detail
"""

import logging
import sys

# Set up file handler for debug logging
file_handler = logging.FileHandler('train_debug.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Set up console handler for info logging
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Configure root logger
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger().addHandler(file_handler)
logging.getLogger().addHandler(console_handler)

from ptycho.workflows.components import (
    parse_arguments,
    setup_configuration,
    load_data,
    run_cdi_example,
    save_outputs,
    logger
)
from ptycho.config.config import TrainingConfig, update_legacy_dict
from ptycho import model_manager, params

def interpret_n_images_parameter(n_images: int, gridsize: int) -> tuple[int, str]:
    """
    Interpret --n-images parameter based on gridsize.
    
    For gridsize=1: n_images refers to individual images (traditional behavior)
    For gridsize>1: n_images refers to number of neighbor groups
    
    Args:
        n_images: User-specified number from --n-images
        gridsize: Current gridsize setting
        
    Returns:
        tuple: (actual_n_images, interpretation_message)
    """
    if gridsize == 1:
        message = f"Parameter interpretation: --n-images={n_images} refers to individual images (gridsize=1)"
        return n_images, message
    else:
        total_patterns = n_images * gridsize * gridsize
        message = f"Parameter interpretation: --n-images={n_images} refers to neighbor groups (gridsize={gridsize}, total patterns={total_patterns})"
        return n_images, message
def main() -> None:
    """Main function to orchestrate the CDI example script execution."""
    args = parse_arguments()
    
    # Handle legacy argument name
    if hasattr(args, 'train_data_file_path'):
        args.train_data_file = args.train_data_file_path
        delattr(args, 'train_data_file_path')
        
    config = setup_configuration(args, args.config)
    
    # Interpret n_images parameter based on gridsize
    interpreted_n_images, interpretation_message = interpret_n_images_parameter(
        config.n_images, config.model.gridsize
    )
    logger.info(interpretation_message)
    
    # Update config with interpreted value
    config = config.__class__(
        **{**config.__dict__, 'n_images': interpreted_n_images}
    )
    
    # Update global params with new-style config at entry point
    update_legacy_dict(params.cfg, config)
    
    try:
        logger.info(f"Starting training with n_images={config.n_images}, stitching={'enabled' if args.do_stitching else 'disabled'}")

        #ptycho_data, ptycho_data_train, obj = load_and_prepare_data(config['train_data_file_path'])
        ptycho_data = load_data(str(config.train_data_file), n_images=config.n_images)
        
        test_data = None
        if config.test_data_file:
            test_data = load_data(str(config.test_data_file))
            logger.info(f"Loaded test data from {config.test_data_file}")

        recon_amp, recon_phase, results = run_cdi_example(ptycho_data, test_data, config, do_stitching=args.do_stitching)
        model_manager.save(str(config.output_dir))
        save_outputs(recon_amp, recon_phase, results, str(config.output_dir))
    except Exception as e:
        logger.error(f"An error occurred during execution: {e}")
        raise

if __name__ == "__main__":
    main()
