#!/usr/bin/env python3
"""
Properly evaluate pty-chi reconstruction using PtychoPINN's existing evaluation tools.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from ptycho.image.cropping import align_for_evaluation
from ptycho.image.registration import register_and_align, find_translation_offset, apply_shift_and_crop
from ptycho.evaluation import eval_reconstruction
from ptycho.loader import RawData
import matplotlib.pyplot as plt
import argparse

def load_ptychi_reconstruction(npz_path):
    """Load pty-chi reconstruction."""
    with np.load(npz_path) as data:
        return data['reconstructed_object']

def load_ground_truth_and_coords(npz_path):
    """Load ground truth and scan coordinates from dataset."""
    with np.load(npz_path) as data:
        obj = data['objectGuess']
        xcoords = data['xcoords']
        ycoords = data['ycoords']
        probe = data['probeGuess']
        
    # Stack coordinates in YX format
    scan_coords_yx = np.stack([ycoords, xcoords], axis=1)
    
    # Determine stitch patch size from probe
    stitch_size = probe.shape[0]
    
    return obj, scan_coords_yx, stitch_size

def evaluate_ptychi_reconstruction(recon_path, gt_path, output_dir=None):
    """
    Evaluate pty-chi reconstruction using PtychoPINN's proper evaluation pipeline.
    """
    print("="*60)
    print("EVALUATING PTY-CHI RECONSTRUCTION WITH PROPER ALIGNMENT")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    ptychi_recon = load_ptychi_reconstruction(recon_path)
    ground_truth, scan_coords_yx, stitch_size = load_ground_truth_and_coords(gt_path)
    
    print(f"   Reconstruction shape: {ptychi_recon.shape}")
    print(f"   Ground truth shape: {ground_truth.shape}")
    print(f"   Number of scan positions: {len(scan_coords_yx)}")
    print(f"   Stitch patch size: {stitch_size}")
    
    # Stage 1: Coordinate-based alignment
    print("\n2. Stage 1: Coordinate-based alignment...")
    try:
        ptychi_aligned, gt_cropped = align_for_evaluation(
            reconstruction_image=ptychi_recon,
            ground_truth_image=ground_truth,
            scan_coords_yx=scan_coords_yx,
            stitch_patch_size=stitch_size
        )
        print(f"   Aligned shapes: recon={ptychi_aligned.shape}, gt={gt_cropped.shape}")
    except Exception as e:
        print(f"   Coordinate alignment failed: {e}")
        print("   Falling back to simple cropping...")
        # Simple size matching as fallback
        min_h = min(ptychi_recon.shape[0], ground_truth.shape[0])
        min_w = min(ptychi_recon.shape[1], ground_truth.shape[1])
        ptychi_aligned = ptychi_recon[:min_h, :min_w]
        gt_cropped = ground_truth[:min_h, :min_w]
    
    # Stage 2: Sub-pixel registration
    print("\n3. Stage 2: Sub-pixel registration...")
    try:
        # Find offset
        offset = find_translation_offset(ptychi_aligned, gt_cropped)
        print(f"   Detected offset: {offset}")
        
        # Apply shift and crop borders
        ptychi_final, gt_final = apply_shift_and_crop(
            ptychi_aligned, gt_cropped, offset, border_crop=2
        )
        print(f"   Final shapes after registration: recon={ptychi_final.shape}, gt={gt_final.shape}")
    except Exception as e:
        print(f"   Registration failed: {e}")
        print("   Using aligned images without registration...")
        ptychi_final = ptychi_aligned
        gt_final = gt_cropped
    
    # Stage 3: Comprehensive evaluation
    print("\n4. Computing metrics with proper phase alignment...")
    
    # Prepare for eval_reconstruction (needs batch and channel dims)
    ptychi_batch = ptychi_final[None, ..., None]  # Add batch and channel dims
    gt_batch = gt_final[..., None]  # Add channel dim
    
    # Run evaluation
    metrics = eval_reconstruction(
        ptychi_batch,
        gt_batch,
        label="PtyChi",
        phase_align_method='plane',  # Remove linear phase trends
        frc_sigma=0.5,  # Gaussian smoothing for FRC
        debug_save_images=output_dir is not None
    )
    
    # Print results
    print("\n" + "="*60)
    print("PROPERLY COMPUTED METRICS (with alignment & phase correction)")
    print("="*60)
    print(f"Amplitude metrics:")
    print(f"  MAE:   {metrics['mae'][0]:.4f}")
    print(f"  PSNR:  {metrics['psnr'][0]:.2f} dB")
    print(f"  SSIM:  {metrics['ssim'][0]:.4f}")
    print(f"  FRC50: {metrics['frc50'][0]}")
    print(f"\nPhase metrics:")
    print(f"  MAE:   {metrics['mae'][1]:.4f}")
    print(f"  PSNR:  {metrics['psnr'][1]:.2f} dB")
    print(f"  SSIM:  {metrics['ssim'][1]:.4f}")
    print(f"  FRC50: {metrics['frc50'][1]}")
    
    # Create visualization if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('Properly Aligned PtyChi Reconstruction Evaluation', fontsize=14)
        
        # Original (before alignment)
        axes[0, 0].imshow(np.abs(ptychi_recon), cmap='gray')
        axes[0, 0].set_title('Original Recon Amplitude')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(np.angle(ptychi_recon), cmap='twilight', vmin=-np.pi, vmax=np.pi)
        axes[0, 1].set_title('Original Recon Phase')
        axes[0, 1].axis('off')
        
        # After alignment
        axes[0, 2].imshow(np.abs(ptychi_final), cmap='gray')
        axes[0, 2].set_title('Aligned Recon Amplitude')
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(np.angle(ptychi_final), cmap='twilight', vmin=-np.pi, vmax=np.pi)
        axes[0, 3].set_title('Aligned Recon Phase')
        axes[0, 3].axis('off')
        
        # Ground truth
        axes[1, 0].imshow(np.abs(gt_final), cmap='gray')
        axes[1, 0].set_title('Ground Truth Amplitude')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(np.angle(gt_final), cmap='twilight', vmin=-np.pi, vmax=np.pi)
        axes[1, 1].set_title('Ground Truth Phase')
        axes[1, 1].axis('off')
        
        # Metrics text
        axes[1, 2].axis('off')
        metrics_text = "AMPLITUDE METRICS:\n\n"
        metrics_text += f"MAE:   {metrics['mae'][0]:.4f}\n"
        metrics_text += f"PSNR:  {metrics['psnr'][0]:.2f} dB\n"
        metrics_text += f"SSIM:  {metrics['ssim'][0]:.4f}\n"
        metrics_text += f"FRC50: {metrics['frc50'][0]}\n"
        axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=12, transform=axes[1, 2].transAxes,
                       verticalalignment='center', fontfamily='monospace')
        
        axes[1, 3].axis('off')
        phase_text = "PHASE METRICS:\n\n"
        phase_text += f"MAE:   {metrics['mae'][1]:.4f}\n"
        phase_text += f"PSNR:  {metrics['psnr'][1]:.2f} dB\n"
        phase_text += f"SSIM:  {metrics['ssim'][1]:.4f}\n"
        phase_text += f"FRC50: {metrics['frc50'][1]}\n"
        axes[1, 3].text(0.1, 0.5, phase_text, fontsize=12, transform=axes[1, 3].transAxes,
                       verticalalignment='center', fontfamily='monospace')
        
        plt.tight_layout()
        fig_path = os.path.join(output_dir, 'proper_evaluation.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {fig_path}")
        plt.close()
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate pty-chi reconstruction properly')
    parser.add_argument('reconstruction', help='Path to pty-chi reconstruction NPZ')
    parser.add_argument('ground_truth', help='Path to ground truth dataset NPZ')
    parser.add_argument('--output', '-o', help='Output directory for results')
    
    args = parser.parse_args()
    
    evaluate_ptychi_reconstruction(args.reconstruction, args.ground_truth, args.output)

if __name__ == "__main__":
    main()