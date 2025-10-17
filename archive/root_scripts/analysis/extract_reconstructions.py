#!/usr/bin/env python3
"""
Extract PtychoPINN reconstruction, Baseline reconstruction, and Ground Truth
for external registration testing.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import ptycho components
from ptycho.workflows.components import load_data, create_ptycho_data_container, logger, load_inference_bundle
from ptycho.config.config import TrainingConfig, ModelConfig, update_legacy_dict
from ptycho import params as p
from ptycho.tf_helper import reassemble_position
from ptycho.image.cropping import align_for_evaluation


def main():
    # Fixed paths for the working test case
    pinn_dir = Path("large_generalization_study_results/train_512/pinn_run")
    baseline_dir = Path("large_generalization_study_results/train_512/baseline_run/07-12-2025-22.14.28_baseline_gs1")
    test_data_path = Path("tike_outputs/fly001_final_downsampled/fly001_final_downsampled_data_transposed.npz")
    output_path = Path("registration_test_data.npz")
    
    logger.info(f"Loading test data from {test_data_path}...")
    test_data_raw = load_data(str(test_data_path))
    
    # Create a minimal config for data container creation
    dummy_config = TrainingConfig(
        model=ModelConfig(N=test_data_raw.probeGuess.shape[0]),
        train_data_file=Path("dummy.npz"),
        n_images=test_data_raw.diff3d.shape[0]
    )
    update_legacy_dict(p.cfg, dummy_config)
    
    # Create data container
    test_container = create_ptycho_data_container(test_data_raw, dummy_config)
    
    # Extract ground truth if available
    ground_truth_obj = test_data_raw.objectGuess[None, ..., None] if test_data_raw.objectGuess is not None else None

    # Load PtychoPINN model using centralized function
    logger.info(f"Loading PtychoPINN model from {pinn_dir}...")
    pinn_model, _ = load_inference_bundle(pinn_dir)

    # Load baseline model
    logger.info(f"Loading Baseline model from {baseline_dir}...")
    baseline_model_path = None
    for root, dirs, files in os.walk(baseline_dir):
        for file in files:
            if file == "baseline_model.h5":
                baseline_model_path = Path(root) / file
                break
        if baseline_model_path:
            break
    
    if baseline_model_path is None or not baseline_model_path.exists():
        raise FileNotFoundError(f"Baseline model not found in: {baseline_dir}")
    
    baseline_model = tf.keras.models.load_model(baseline_model_path)

    # Run inference for PtychoPINN
    logger.info("Running inference with PtychoPINN...")
    pinn_patches = pinn_model.predict(
        [test_container.X * p.get('intensity_scale'), test_container.coords_nominal],
        batch_size=32,
        verbose=1
    )
    
    # Reassemble patches
    logger.info("Reassembling PtychoPINN patches...")
    pinn_recon = reassemble_position(pinn_patches, test_container.global_offsets, M=20)

    # Run inference for Baseline
    logger.info("Running inference with Baseline model...")
    baseline_output = baseline_model.predict(test_container.X, batch_size=32, verbose=1)
    
    # Handle different output formats
    if isinstance(baseline_output, list) and len(baseline_output) == 2:
        baseline_patches_I, baseline_patches_phi = baseline_output
    else:
        raise ValueError("Unexpected baseline model output format")
    
    # Convert to complex representation
    baseline_patches_complex = tf.cast(baseline_patches_I, tf.complex64) * \
                               tf.exp(1j * tf.cast(baseline_patches_phi, tf.complex64))
    
    # Reassemble patches
    logger.info("Reassembling baseline patches...")
    baseline_recon = reassemble_position(baseline_patches_complex, test_container.global_offsets, M=20)

    # Coordinate-based alignment (crop to scanned region) - same as compare_models.py
    if ground_truth_obj is not None:
        logger.info("Performing coordinate-based alignment of ground truth...")
        
        # Squeeze ground truth to 2D
        gt_obj_squeezed = ground_truth_obj.squeeze()
        logger.info(f"Ground truth original shape: {gt_obj_squeezed.shape}")
        
        # Define the stitching parameter
        M_STITCH_SIZE = 20

        # Extract scan coordinates in (y, x) format
        global_offsets = test_container.global_offsets
        scan_coords_xy = np.squeeze(global_offsets)
        scan_coords_yx = scan_coords_xy[:, [1, 0]]

        # Coordinate-based alignment (crop to scanned region)
        pinn_recon_cropped, gt_cropped_for_pinn = align_for_evaluation(
            reconstruction_image=pinn_recon,
            ground_truth_image=ground_truth_obj,
            scan_coords_yx=scan_coords_yx,
            stitch_patch_size=M_STITCH_SIZE
        )
        
        baseline_recon_cropped, gt_cropped_for_baseline = align_for_evaluation(
            reconstruction_image=baseline_recon,
            ground_truth_image=ground_truth_obj,
            scan_coords_yx=scan_coords_yx,
            stitch_patch_size=M_STITCH_SIZE
        )
        
        # Use the first cropped GT (should be identical for both)
        cropped_gt = gt_cropped_for_pinn
        
        # Save the cropped data for external registration testing
        logger.info(f"Saving registration test data to {output_path}...")
        np.savez_compressed(
            output_path,
            pinn_reconstruction=np.squeeze(pinn_recon_cropped),
            baseline_reconstruction=np.squeeze(baseline_recon_cropped), 
            ground_truth=np.squeeze(cropped_gt),
            pinn_amplitude=np.abs(np.squeeze(pinn_recon_cropped)),
            pinn_phase=np.angle(np.squeeze(pinn_recon_cropped)),
            baseline_amplitude=np.abs(np.squeeze(baseline_recon_cropped)),
            baseline_phase=np.angle(np.squeeze(baseline_recon_cropped)),
            gt_amplitude=np.abs(np.squeeze(cropped_gt)),
            gt_phase=np.angle(np.squeeze(cropped_gt)),
            # Also save the full reconstructions before cropping for comparison
            pinn_full=np.squeeze(pinn_recon),
            baseline_full=np.squeeze(baseline_recon),
            gt_full=np.squeeze(gt_obj_squeezed)
        )
        
        logger.info(f"Saved data shapes:")
        logger.info(f"  PtychoPINN reconstruction: {pinn_recon_cropped.shape}")
        logger.info(f"  Baseline reconstruction: {baseline_recon_cropped.shape}")
        logger.info(f"  Ground truth: {cropped_gt.shape}")
        
    else:
        logger.error("No ground truth available - cannot create registration test data")


if __name__ == "__main__":
    main()