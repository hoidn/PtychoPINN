#!/usr/bin/env python3
"""
Extract and crop ground truth images to match reconstruction dimensions.

This script loads ground truth from dataset NPZ files and crops them to match
the actual reconstruction sizes produced by the inference process.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ptycho.image.cropping import align_for_evaluation
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_cropped_ground_truth(dataset_npz_path, recon_npz_path, output_dir=None):
    """
    Extract ground truth and crop it to match reconstruction dimensions.
    
    Args:
        dataset_npz_path: Path to dataset NPZ with ground truth
        recon_npz_path: Path to reconstruction NPZ to match dimensions  
        output_dir: Output directory for cropped images
    """
    dataset_npz_path = Path(dataset_npz_path)
    recon_npz_path = Path(recon_npz_path)
    
    # Load dataset with ground truth
    dataset = np.load(dataset_npz_path)
    if 'objectGuess' not in dataset:
        raise ValueError(f"No objectGuess found in {dataset_npz_path}")
    
    ground_truth = dataset['objectGuess']
    xcoords = dataset['xcoords']
    ycoords = dataset['ycoords']
    
    # Create scan coordinates in yx format
    scan_coords_yx = np.column_stack([ycoords, xcoords])
    
    # Load reconstruction to get target dimensions
    recon_data = np.load(recon_npz_path)
    if 'reconstructed_object' in recon_data:
        recon = recon_data['reconstructed_object']
    elif 'reconstructed_amplitude' in recon_data:
        # Reconstruct complex from amplitude and phase
        amp = recon_data['reconstructed_amplitude']
        phase = recon_data['reconstructed_phase']
        recon = amp * np.exp(1j * phase)
    else:
        raise ValueError(f"No reconstruction found in {recon_npz_path}")
    
    logger.info(f"Ground truth shape: {ground_truth.shape}")
    logger.info(f"Reconstruction shape: {recon.shape}")
    logger.info(f"Number of scan positions: {len(xcoords)}")
    
    # Use align_for_evaluation to crop ground truth to match reconstruction
    # Using M=20 as the stitch patch size (from inference.py)
    stitch_patch_size = 20
    aligned_recon, aligned_gt = align_for_evaluation(
        recon, ground_truth, scan_coords_yx, stitch_patch_size
    )
    
    logger.info(f"Aligned shapes - Recon: {aligned_recon.shape}, GT: {aligned_gt.shape}")
    
    # Extract amplitude and phase from aligned ground truth
    gt_amplitude = np.abs(aligned_gt)
    gt_phase = np.angle(aligned_gt)
    
    # Determine output directory
    if output_dir is None:
        output_dir = recon_npz_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine dataset name from path
    dataset_name = 'unknown'
    train_dataset = None
    test_dataset = None
    
    path_str = str(recon_npz_path).lower()
    
    # Check for cross-dataset scenario
    if 'fly64_trained' in path_str and 'run1084' in path_str:
        train_dataset = 'fly64'
        test_dataset = 'run1084'
    elif 'run1084_trained' in path_str and 'fly64' in path_str:
        train_dataset = 'run1084'
        test_dataset = 'fly64'
    elif 'fly64' in path_str:
        train_dataset = test_dataset = 'fly64'
    elif 'run1084' in path_str:
        train_dataset = test_dataset = 'run1084'
    
    # Build filename
    if train_dataset and test_dataset and train_dataset != test_dataset:
        base_name = f"ground_truth_{test_dataset}_for_{train_dataset}trained"
    elif test_dataset:
        base_name = f"ground_truth_{test_dataset}"
    else:
        base_name = "ground_truth_cropped"
    
    # Save amplitude PNG
    amp_path = output_dir / f"{base_name}_amplitude.png"
    fig = plt.figure(figsize=(4, 4), dpi=100)
    ax = plt.gca()
    im = ax.imshow(gt_amplitude, cmap='viridis', interpolation='nearest', origin='lower')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(amp_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    logger.info(f"Saved amplitude to: {amp_path}")
    
    # Save phase PNG
    phase_path = output_dir / f"{base_name}_phase.png"
    fig = plt.figure(figsize=(4, 4), dpi=100)
    ax = plt.gca()
    im = ax.imshow(gt_phase, cmap='gray', interpolation='nearest', origin='lower')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(phase_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    logger.info(f"Saved phase to: {phase_path}")
    
    # Save as NPZ
    npz_out_path = output_dir / f"{base_name}.npz"
    np.savez_compressed(
        npz_out_path,
        ground_truth_complex=aligned_gt,
        ground_truth_amplitude=gt_amplitude,
        ground_truth_phase=gt_phase,
        metadata={
            'source_dataset': str(dataset_npz_path),
            'matched_to_recon': str(recon_npz_path),
            'original_shape': ground_truth.shape,
            'cropped_shape': aligned_gt.shape,
            'stitch_patch_size': stitch_patch_size
        }
    )
    logger.info(f"Saved NPZ to: {npz_out_path}")
    
    return amp_path, phase_path, npz_out_path

def main():
    parser = argparse.ArgumentParser(
        description="Extract and crop ground truth to match reconstruction dimensions"
    )
    parser.add_argument("dataset_npz", help="Path to dataset NPZ with ground truth")
    parser.add_argument("recon_npz", help="Path to reconstruction NPZ to match")
    parser.add_argument("--output-dir", help="Output directory (default: same as recon)")
    
    args = parser.parse_args()
    
    try:
        paths = extract_cropped_ground_truth(
            args.dataset_npz, args.recon_npz, args.output_dir
        )
        print(f"\nSuccessfully created cropped ground truth images:")
        for path in paths:
            print(f"  - {path}")
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())