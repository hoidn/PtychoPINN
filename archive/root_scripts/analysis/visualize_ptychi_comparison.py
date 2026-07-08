#!/usr/bin/env python3
"""
Visualize and compare pty-chi reconstructions with ground truth.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def load_reconstruction(npz_path):
    """Load reconstruction data from NPZ file."""
    with np.load(npz_path) as data:
        return {
            'object': data['reconstructed_object'],
            'probe': data['reconstructed_probe'],
            'algorithm': str(data.get('algorithm', 'unknown')),
            'time': float(data.get('reconstruction_time', 0))
        }

def load_ground_truth(npz_path):
    """Load ground truth from original dataset."""
    with np.load(npz_path) as data:
        return {
            'object': data['objectGuess'],
            'probe': data['probeGuess']
        }

def normalize_phase(phase):
    """Normalize phase to [-π, π] range."""
    return np.angle(np.exp(1j * phase))

def compute_metrics(recon, ground_truth):
    """Compute quality metrics between reconstruction and ground truth."""
    # Amplitude error
    amp_recon = np.abs(recon)
    amp_gt = np.abs(ground_truth)
    
    # Normalize for comparison
    amp_recon_norm = amp_recon / np.max(amp_recon)
    amp_gt_norm = amp_gt / np.max(amp_gt)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(amp_recon_norm - amp_gt_norm))
    
    # Peak Signal-to-Noise Ratio
    mse = np.mean((amp_recon_norm - amp_gt_norm) ** 2)
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else np.inf
    
    # Structural Similarity (simplified)
    mean_recon = np.mean(amp_recon_norm)
    mean_gt = np.mean(amp_gt_norm)
    std_recon = np.std(amp_recon_norm)
    std_gt = np.std(amp_gt_norm)
    cov = np.mean((amp_recon_norm - mean_recon) * (amp_gt_norm - mean_gt))
    
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    ssim = ((2 * mean_recon * mean_gt + c1) * (2 * cov + c2)) / \
           ((mean_recon ** 2 + mean_gt ** 2 + c1) * (std_recon ** 2 + std_gt ** 2 + c2))
    
    return {
        'MAE': mae,
        'PSNR': psnr,
        'SSIM': ssim
    }

def visualize_comparison(recon_path, gt_path=None, output_path=None):
    """Create comprehensive visualization comparing reconstruction with ground truth."""
    
    # Load reconstruction
    recon = load_reconstruction(recon_path)
    
    # Load ground truth if provided
    if gt_path:
        gt = load_ground_truth(gt_path)
    else:
        gt = None
    
    # Create figure
    if gt:
        fig = plt.figure(figsize=(16, 10))
        n_rows = 3
    else:
        fig = plt.figure(figsize=(12, 8))
        n_rows = 2
    
    # Title
    title = f"{recon['algorithm']} Reconstruction"
    if recon['time'] > 0:
        title += f" (Time: {recon['time']:.1f}s)"
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Plot reconstruction
    ax1 = plt.subplot(n_rows, 4, 1)
    im = ax1.imshow(np.abs(recon['object']), cmap='gray')
    ax1.set_title('Recon Object Amplitude')
    ax1.axis('off')
    plt.colorbar(im, ax=ax1, fraction=0.046)
    
    ax2 = plt.subplot(n_rows, 4, 2)
    im = ax2.imshow(normalize_phase(np.angle(recon['object'])), cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax2.set_title('Recon Object Phase')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046)
    
    ax3 = plt.subplot(n_rows, 4, 3)
    im = ax3.imshow(np.abs(recon['probe']), cmap='gray')
    ax3.set_title('Recon Probe Amplitude')
    ax3.axis('off')
    plt.colorbar(im, ax=ax3, fraction=0.046)
    
    ax4 = plt.subplot(n_rows, 4, 4)
    im = ax4.imshow(normalize_phase(np.angle(recon['probe'])), cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax4.set_title('Recon Probe Phase')
    ax4.axis('off')
    plt.colorbar(im, ax=ax4, fraction=0.046)
    
    # Plot ground truth if available
    if gt:
        ax5 = plt.subplot(n_rows, 4, 5)
        im = ax5.imshow(np.abs(gt['object']), cmap='gray')
        ax5.set_title('GT Object Amplitude')
        ax5.axis('off')
        plt.colorbar(im, ax=ax5, fraction=0.046)
        
        ax6 = plt.subplot(n_rows, 4, 6)
        im = ax6.imshow(normalize_phase(np.angle(gt['object'])), cmap='twilight', vmin=-np.pi, vmax=np.pi)
        ax6.set_title('GT Object Phase')
        ax6.axis('off')
        plt.colorbar(im, ax=ax6, fraction=0.046)
        
        ax7 = plt.subplot(n_rows, 4, 7)
        im = ax7.imshow(np.abs(gt['probe']), cmap='gray')
        ax7.set_title('GT Probe Amplitude')
        ax7.axis('off')
        plt.colorbar(im, ax=ax7, fraction=0.046)
        
        ax8 = plt.subplot(n_rows, 4, 8)
        im = ax8.imshow(normalize_phase(np.angle(gt['probe'])), cmap='twilight', vmin=-np.pi, vmax=np.pi)
        ax8.set_title('GT Probe Phase')
        ax8.axis('off')
        plt.colorbar(im, ax=ax8, fraction=0.046)
        
        # Compute and show difference
        # Need to crop/pad to match sizes
        obj_recon = recon['object']
        obj_gt = gt['object']
        
        min_h = min(obj_recon.shape[0], obj_gt.shape[0])
        min_w = min(obj_recon.shape[1], obj_gt.shape[1])
        
        obj_recon_crop = obj_recon[:min_h, :min_w]
        obj_gt_crop = obj_gt[:min_h, :min_w]
        
        ax9 = plt.subplot(n_rows, 4, 9)
        diff_amp = np.abs(obj_recon_crop) - np.abs(obj_gt_crop)
        im = ax9.imshow(diff_amp, cmap='RdBu_r', vmin=-np.max(np.abs(diff_amp)), vmax=np.max(np.abs(diff_amp)))
        ax9.set_title('Amplitude Difference')
        ax9.axis('off')
        plt.colorbar(im, ax=ax9, fraction=0.046)
        
        ax10 = plt.subplot(n_rows, 4, 10)
        diff_phase = normalize_phase(np.angle(obj_recon_crop) - np.angle(obj_gt_crop))
        im = ax10.imshow(diff_phase, cmap='RdBu_r', vmin=-np.pi, vmax=np.pi)
        ax10.set_title('Phase Difference')
        ax10.axis('off')
        plt.colorbar(im, ax=ax10, fraction=0.046)
        
        # Compute metrics
        metrics = compute_metrics(obj_recon_crop, obj_gt_crop)
        
        # Add text with metrics
        ax11 = plt.subplot(n_rows, 4, 11)
        ax11.axis('off')
        metrics_text = f"Quality Metrics:\n\n"
        metrics_text += f"MAE: {metrics['MAE']:.4f}\n"
        metrics_text += f"PSNR: {metrics['PSNR']:.2f} dB\n"
        metrics_text += f"SSIM: {metrics['SSIM']:.4f}\n"
        ax11.text(0.1, 0.5, metrics_text, fontsize=12, transform=ax11.transAxes,
                 verticalalignment='center', fontfamily='monospace')
        
        # Add statistics
        ax12 = plt.subplot(n_rows, 4, 12)
        ax12.axis('off')
        stats_text = f"Reconstruction Stats:\n\n"
        stats_text += f"Object shape: {obj_recon.shape}\n"
        stats_text += f"Amp range: [{np.abs(obj_recon).min():.3f}, {np.abs(obj_recon).max():.3f}]\n"
        stats_text += f"Time: {recon['time']:.2f}s\n"
        ax12.text(0.1, 0.5, stats_text, fontsize=12, transform=ax12.transAxes,
                 verticalalignment='center', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    
    plt.show()
    
    return metrics if gt else None

def main():
    parser = argparse.ArgumentParser(description='Visualize pty-chi reconstruction results')
    parser.add_argument('reconstruction', help='Path to reconstruction NPZ file')
    parser.add_argument('--ground-truth', '-gt', help='Path to ground truth NPZ file')
    parser.add_argument('--output', '-o', help='Output path for visualization')
    
    args = parser.parse_args()
    
    metrics = visualize_comparison(args.reconstruction, args.ground_truth, args.output)
    
    if metrics:
        print("\nQuality Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

if __name__ == "__main__":
    main()