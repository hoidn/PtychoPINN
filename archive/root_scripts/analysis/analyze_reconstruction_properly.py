#!/usr/bin/env python3
"""
Properly analyze reconstruction quality with alignment and cropping.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.optimize import minimize
from skimage.metrics import structural_similarity as ssim
import argparse

def load_data(npz_path):
    """Load reconstruction or ground truth data."""
    with np.load(npz_path) as data:
        if 'reconstructed_object' in data:
            return data['reconstructed_object']
        elif 'objectGuess' in data:
            return data['objectGuess']
        else:
            raise KeyError("No object data found")

def crop_center(img, crop_fraction=0.8):
    """Crop to center region to avoid border artifacts."""
    h, w = img.shape
    crop_h = int(h * crop_fraction)
    crop_w = int(w * crop_fraction)
    
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    
    return img[start_h:start_h+crop_h, start_w:start_w+crop_w]

def align_images(img1, img2):
    """Align img2 to img1 using cross-correlation."""
    # Use amplitude for alignment
    amp1 = np.abs(img1)
    amp2 = np.abs(img2)
    
    # Normalize
    amp1 = (amp1 - amp1.mean()) / amp1.std()
    amp2 = (amp2 - amp2.mean()) / amp2.std()
    
    # Cross-correlation to find shift
    corr = np.fft.fftshift(np.fft.ifft2(
        np.fft.fft2(amp1) * np.conj(np.fft.fft2(amp2))
    ))
    
    # Find peak
    peak = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)
    shift = [peak[0] - corr.shape[0]//2, peak[1] - corr.shape[1]//2]
    
    # Apply shift
    img2_aligned = ndimage.shift(img2.real, shift) + 1j * ndimage.shift(img2.imag, shift)
    
    return img2_aligned, shift

def resolve_global_phase(img1, img2):
    """Resolve global phase ambiguity between complex images."""
    # Find global phase that minimizes difference
    def phase_error(phi):
        img2_rotated = img2 * np.exp(1j * phi)
        return np.mean(np.abs(img1 - img2_rotated)**2)
    
    result = minimize(phase_error, 0, bounds=[(-np.pi, np.pi)])
    optimal_phase = result.x[0]
    
    return img2 * np.exp(1j * optimal_phase), optimal_phase

def compute_metrics_properly(recon, gt, crop_fraction=0.8):
    """Compute metrics with proper alignment and cropping."""
    
    # Ensure same size
    min_h = min(recon.shape[0], gt.shape[0])
    min_w = min(recon.shape[1], gt.shape[1])
    recon = recon[:min_h, :min_w]
    gt = gt[:min_h, :min_w]
    
    # Align images
    print("Aligning images...")
    recon_aligned, shift = align_images(gt, recon)
    print(f"  Applied shift: {shift}")
    
    # Resolve global phase
    print("Resolving global phase...")
    recon_aligned, phase_offset = resolve_global_phase(gt, recon_aligned)
    print(f"  Phase offset: {phase_offset:.3f} rad")
    
    # Crop to avoid borders
    print(f"Cropping to center {crop_fraction*100:.0f}%...")
    recon_crop = crop_center(recon_aligned, crop_fraction)
    gt_crop = crop_center(gt, crop_fraction)
    
    # Compute metrics on amplitude
    amp_recon = np.abs(recon_crop)
    amp_gt = np.abs(gt_crop)
    
    # Normalize for fair comparison
    amp_recon_norm = amp_recon / np.max(amp_recon)
    amp_gt_norm = amp_gt / np.max(amp_gt)
    
    # Metrics
    mae = np.mean(np.abs(amp_recon_norm - amp_gt_norm))
    
    mse = np.mean((amp_recon_norm - amp_gt_norm)**2)
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else np.inf
    
    ssim_value = ssim(amp_gt_norm, amp_recon_norm, data_range=1.0)
    
    # Phase metrics (on cropped, aligned data)
    phase_recon = np.angle(recon_crop)
    phase_gt = np.angle(gt_crop)
    phase_mae = np.mean(np.abs(np.angle(np.exp(1j*(phase_recon - phase_gt)))))
    
    metrics = {
        'MAE': mae,
        'PSNR': psnr,
        'SSIM': ssim_value,
        'Phase_MAE': phase_mae,
        'Shift': shift,
        'Phase_offset': phase_offset
    }
    
    return metrics, recon_aligned, recon_crop, gt_crop

def visualize_proper_comparison(recon_path, gt_path, output_path=None):
    """Create visualization with proper alignment."""
    
    # Load data
    recon = load_data(recon_path)
    gt = load_data(gt_path)
    
    # Compute metrics with alignment
    metrics, recon_aligned, recon_crop, gt_crop = compute_metrics_properly(recon, gt)
    
    # Create figure
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Properly Aligned Reconstruction Comparison', fontsize=14, fontweight='bold')
    
    # Original (unaligned)
    axes[0, 0].imshow(np.abs(recon), cmap='gray')
    axes[0, 0].set_title('Original Recon Amplitude')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(np.angle(recon), cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[0, 1].set_title('Original Recon Phase')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(np.abs(gt), cmap='gray')
    axes[0, 2].set_title('Ground Truth Amplitude')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(np.angle(gt), cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[0, 3].set_title('Ground Truth Phase')
    axes[0, 3].axis('off')
    
    # Aligned and cropped
    axes[1, 0].imshow(np.abs(recon_crop), cmap='gray')
    axes[1, 0].set_title('Aligned & Cropped Recon Amp')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(np.angle(recon_crop), cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[1, 1].set_title('Aligned & Cropped Recon Phase')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(np.abs(gt_crop), cmap='gray')
    axes[1, 2].set_title('Cropped GT Amplitude')
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(np.angle(gt_crop), cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[1, 3].set_title('Cropped GT Phase')
    axes[1, 3].axis('off')
    
    # Differences (after alignment)
    diff_amp = np.abs(recon_crop) - np.abs(gt_crop)
    axes[2, 0].imshow(diff_amp, cmap='RdBu_r', vmin=-np.max(np.abs(diff_amp)), vmax=np.max(np.abs(diff_amp)))
    axes[2, 0].set_title('Amplitude Difference (Aligned)')
    axes[2, 0].axis('off')
    
    diff_phase = np.angle(recon_crop * np.conj(gt_crop))
    axes[2, 1].imshow(diff_phase, cmap='RdBu_r', vmin=-np.pi, vmax=np.pi)
    axes[2, 1].set_title('Phase Difference (Aligned)')
    axes[2, 1].axis('off')
    
    # Metrics
    axes[2, 2].axis('off')
    metrics_text = "PROPER METRICS:\n\n"
    metrics_text += f"Alignment shift: {metrics['Shift']}\n"
    metrics_text += f"Phase offset: {metrics['Phase_offset']:.3f} rad\n\n"
    metrics_text += f"After alignment & cropping:\n"
    metrics_text += f"MAE: {metrics['MAE']:.4f}\n"
    metrics_text += f"PSNR: {metrics['PSNR']:.2f} dB\n"
    metrics_text += f"SSIM: {metrics['SSIM']:.4f}\n"
    metrics_text += f"Phase MAE: {metrics['Phase_MAE']:.3f} rad"
    axes[2, 2].text(0.1, 0.5, metrics_text, fontsize=11, transform=axes[2, 2].transAxes,
                    verticalalignment='center', fontfamily='monospace')
    
    # Error histogram
    axes[2, 3].hist(diff_amp.flatten(), bins=50, alpha=0.7, color='blue')
    axes[2, 3].set_title('Amplitude Error Distribution')
    axes[2, 3].set_xlabel('Error')
    axes[2, 3].set_ylabel('Count')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
    
    plt.show()
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Properly analyze reconstruction with alignment')
    parser.add_argument('reconstruction', help='Path to reconstruction NPZ')
    parser.add_argument('ground_truth', help='Path to ground truth NPZ')
    parser.add_argument('--output', '-o', help='Output path for visualization')
    parser.add_argument('--crop', type=float, default=0.8, help='Crop fraction (default: 0.8)')
    
    args = parser.parse_args()
    
    metrics = visualize_proper_comparison(args.reconstruction, args.ground_truth, args.output)
    
    print("\n" + "="*50)
    print("PROPERLY COMPUTED METRICS (with alignment & cropping):")
    print("="*50)
    for key, value in metrics.items():
        if isinstance(value, list):
            print(f"{key}: {value}")
        elif isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()