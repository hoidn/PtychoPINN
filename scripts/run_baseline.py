import argparse
import os
import sys

# Add the project root to the Python path to allow for package-style imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
import numpy as np
import tensorflow as tf

from ptycho import params as p
from ptycho import misc
from ptycho import baselines as bl
from ptycho.evaluation import save_metrics
from ptycho.export import save_recons
from ptycho.loader import PtychoDataset, PtychoDataContainer, RawData
from ptycho.image import reassemble_patches
from ptycho import probe as probe_module

# Import the modern configuration and data loading components
from ptycho.workflows.components import (
    parse_arguments,
    setup_configuration,
    load_data,
    create_ptycho_data_container,
    logger
)
from ptycho.config.config import TrainingConfig, update_legacy_dict

#def main():
#    """
#    Main function to train and evaluate the baseline model.
#    This script uses the modern configuration and data loading pipeline.
#    """
# 1. --- Configuration and CLI Setup ---
# Use the centralized argument parser which is aware of all TrainingConfig parameters
args = parse_arguments()
config = setup_configuration(args, args.config)

# For compatibility with legacy modules, update the global params dictionary.
update_legacy_dict(p.cfg, config)
p.cfg['label'] = f"baseline_gs{config.model.gridsize}" # Set a specific label for this run
p.cfg['output_prefix'] = misc.get_path_prefix()
out_prefix = p.get('output_prefix')
os.makedirs(out_prefix, exist_ok=True)

logger.info("--- Starting Supervised Baseline Run ---")
logger.info(f"Results will be saved to: {out_prefix}")
p.print_params()

# 2. --- Probe and Data Initialization ---
logger.info("\n[1/6] Initializing probe...")
probe_module.set_default_probe()

logger.info(f"\n[2/6] Loading data...")
# MODIFIED: Logic now depends on whether data file paths are provided, not on 'data_source'.
if config.train_data_file and config.test_data_file:
    logger.info(f"Loading from .npz files: {config.train_data_file}")
    # Load from .npz files using the project's established "generic" loader
    train_data_raw = load_data(str(config.train_data_file), n_images=config.n_images)
    test_data_raw = load_data(str(config.test_data_file))

    train_container = create_ptycho_data_container(train_data_raw, config)
    test_container = create_ptycho_data_container(test_data_raw, config)
    
    ptycho_dataset = PtychoDataset(train_container, test_container)
    YY_ground_truth = test_data_raw.objectGuess[None, ..., None] if test_data_raw.objectGuess is not None else None
else:
    logger.info(f"Generating simulated data...")
    # Fallback to generating data on the fly using the simulation pipeline
    from ptycho import generate_data
    ptycho_dataset = generate_data.ptycho_dataset
    YY_ground_truth = generate_data.YY_ground_truth

# 3. --- Shape Data for Baseline Model ---
logger.info("\n[3/6] Shaping data for the baseline model...")
if config.model.gridsize == 1:
    n_channels = 1
elif config.model.gridsize == 2:
    n_channels = 4
else:
    raise ValueError(f"This baseline script only supports gridsize 1 or 2, but got {config.model.gridsize}.")

# The baseline model expects specific channel depths
X_train_in = ptycho_dataset.train_data.X[..., :n_channels]
Y_I_train_in = ptycho_dataset.train_data.Y_I[..., :n_channels]
Y_phi_train_in = ptycho_dataset.train_data.Y_phi[..., :n_channels]

X_test_in = ptycho_dataset.test_data.X[..., :n_channels]
logger.info(f"Final training input shape: {X_train_in.shape}")

# 4. --- Model Training ---
logger.info(f"\n[4/6] Training the baseline model for {config.nepochs} epochs with batch size {config.batch_size}...")
logger.info(f"Training with {X_train_in.shape[0]} images")
model, history = bl.train(X_train_in, Y_I_train_in, Y_phi_train_in)

model_path = os.path.join(out_prefix, 'baseline_model.h5')
model.save(model_path)
logger.info(f"Trained model saved to {model_path}")

# 5. --- Inference and Evaluation ---
logger.info("\n[5/6] Performing inference and stitching...")
pred_I_patches, pred_phi_patches = model.predict(X_test_in)
reconstructed_patches_complex = tf.cast(pred_I_patches, tf.complex64) * tf.exp(1j * tf.cast(pred_phi_patches, tf.complex64))

try:
    # Use the new stitching module with the configuration dictionary
    stitched_obj = reassemble_patches(reconstructed_patches_complex, p.cfg, part='complex')
    logger.info(f"Stitched object shape: {stitched_obj.shape}")
except (ValueError, TypeError) as e:
    stitched_obj = None
    logger.error(f"Object stitching failed: {e}")

# 6. --- Save Results and Metrics ---
logger.info("\n[6/6] Evaluating reconstruction and saving results...")
if stitched_obj is not None and YY_ground_truth is not None:
    from ptycho.evaluation import eval_reconstruction
    metrics = eval_reconstruction(stitched_obj, YY_ground_truth)
    logger.info("Evaluation Metrics (Amplitude, Phase):")
    logger.info(f"  MAE:  {metrics['mae']}")
    logger.info(f"  PSNR: {metrics['psnr']}")
    save_metrics(stitched_obj, YY_ground_truth, label=p.get('label'))
    save_recons(model_type='supervised', stitched_obj=stitched_obj)
    logger.info("Metrics and reconstruction images saved.")
else:
    logger.warning("Skipping evaluation: stitched object or ground truth was not available.")

logger.info("\n--- Baseline script finished successfully. ---")

#if __name__ == '__main__':
#    main()
