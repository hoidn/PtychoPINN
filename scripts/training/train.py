#!/usr/bin/env python

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

from ptycho.raw_data import RawData
from ptycho.config.config import (
    ModelConfig, 
    TrainingConfig,
    load_yaml_config,
    validate_training_config
)
from ptycho.workflows.components import run_cdi_example, load_and_prepare_data

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args() -> TrainingConfig:
    """Parse command line arguments and YAML config into TrainingConfig."""
    parser = argparse.ArgumentParser(description="CDI Training Script")
    
    # Required args
    parser.add_argument("--train-data-file", type=Path, required=True,
                       help="Path to training data file")
    parser.add_argument("--test-data-file", type=Path, required=False,
                       help="Optional path to test data file")
    
    # Optional config file
    parser.add_argument("--config", type=Path, 
                       help="Path to YAML configuration file")
    
    # Model config options
    parser.add_argument("--N", type=int, choices=[64, 128, 256],
                       help="Grid size for reconstruction")
    parser.add_argument("--gridsize", type=int, help="Solution region grid size")
    parser.add_argument("--model-type", choices=['pinn', 'supervised'],
                       help="Model architecture type")
    parser.add_argument("--amp-activation", 
                       choices=['sigmoid', 'swish', 'softplus', 'relu'],
                       help="Amplitude activation function")
    parser.add_argument("--object-big", type=bool, help="Use big object")
    parser.add_argument("--probe-big", type=bool, help="Use big probe")
    parser.add_argument("--probe-mask", type=bool, help="Use probe mask")
    parser.add_argument("--pad-object", type=bool, help="Pad reconstruction")
    parser.add_argument("--probe-scale", type=float, help="Probe scale factor")

    # Training config options  
    parser.add_argument("--batch-size", type=int, help="Training batch size")
    parser.add_argument("--nepochs", type=int, help="Number of epochs")
    parser.add_argument("--mae-weight", type=float, help="MAE loss weight")
    parser.add_argument("--nll-weight", type=float, help="NLL loss weight")
    parser.add_argument("--realspace-mae-weight", type=float, 
                       help="Realspace MAE loss weight")
    parser.add_argument("--realspace-weight", type=float, 
                       help="Realspace loss weight")
    parser.add_argument("--nphotons", type=float, help="Photon count")
    parser.add_argument("--probe-trainable", type=bool, 
                       help="Make probe trainable")
    parser.add_argument("--intensity-scale-trainable", type=bool,
                       help="Make intensity scale trainable")
    parser.add_argument("--output-dir", type=Path, help="Output directory")

    args = parser.parse_args()

    # Start with empty config dict
    config_dict = {}
    
    # Load YAML config if provided
    if args.config:
        config_dict.update(load_yaml_config(args.config))

    # Update with CLI args where specified (not None)
    for key, value in vars(args).items():
        if value is not None:
            # Convert hyphenated CLI arg names to underscores
            dict_key = key.replace('-', '_')
            config_dict[dict_key] = value

    # Create ModelConfig
    model_fields = {
        k: v for k, v in config_dict.items()
        if k in ModelConfig.__dataclass_fields__
    }
    model_config = ModelConfig(**model_fields)

    # Create TrainingConfig including ModelConfig
    training_fields = {
        k: v for k, v in config_dict.items()
        if k in TrainingConfig.__dataclass_fields__ and k != 'model'
    }
    training_config = TrainingConfig(
        model=model_config,
        **training_fields
    )

    # Validate the complete config
    validate_training_config(training_config)

    return training_config

def load_data(config: TrainingConfig) -> Tuple[RawData, Optional[RawData]]:
    """Load training and optional test data."""
    train_raw_data, _, _ = load_and_prepare_data(config.train_data_file)
    
    test_raw_data = None
    if config.test_data_file:
        test_raw_data, _, _ = load_and_prepare_data(config.test_data_file)
    
    return train_raw_data, test_raw_data

def main() -> None:
    """Main training execution flow."""
    try:
        # Parse args into TrainingConfig
        config = parse_args()
        logger.info(f"Created config: {config}")

        # Load data
        train_data, test_data = load_data(config)
        logger.info("Loaded training data" + 
                   (" and test data" if test_data else ""))

        # Run training
        recon_amp, recon_phase, results = run_cdi_example(
            train_data=train_data,
            test_data=test_data,
            config=config
        )

        logger.info(f"Training completed, results saved to {config.output_dir}")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
