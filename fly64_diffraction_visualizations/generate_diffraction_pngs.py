#!/usr/bin/env python3
"""
Generate PNG visualizations of diffraction images from fly64 dataset
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def generate_diffraction_pngs():
    """Generate PNG visualizations of a sample of diffraction images"""
    
    # Load the fly64 dataset
    dataset_path = "datasets/fly64/fly64_shuffled.npz"
    if not os.path.exists(dataset_path):
        dataset_path = "datasets/fly64/fly001_64_train_converted.npz"
    
    if not os.path.exists(dataset_path):
        print("❌ Could not find fly64 dataset")
        return
    
    print(f"📂 Loading dataset: {dataset_path}")
    data = np.load(dataset_path)
    
    # Get diffraction data
    diffraction = data['diffraction']
    print(f"📊 Diffraction shape: {diffraction.shape}")
    print(f"📊 Data range: {diffraction.min():.3f} - {diffraction.max():.3f}")
    
    # Select a few representative images
    n_images = min(12, diffraction.shape[0])
    indices = np.linspace(0, diffraction.shape[0]-1, n_images, dtype=int)
    
    print(f"🎯 Generating {n_images} diffraction pattern visualizations...")
    
    # Create individual PNG files
    for i, idx in enumerate(indices):
        img = diffraction[idx]
        
        # Create figure
        plt.figure(figsize=(8, 8))
        
        # Display with log scale for better visibility - use jet colormap and better offset
        log_img = np.log10(img + 0.1)  # Small positive offset for better color distribution
        plt.imshow(log_img, cmap='jet', origin='lower')
        plt.colorbar(label='Log10(Intensity + 0.1)')
        plt.title(f'Diffraction Pattern #{idx} (64x64 pixels)', fontsize=14, fontweight='bold')
        plt.xlabel('Pixel X')
        plt.ylabel('Pixel Y')
        
        # Save individual image
        filename = f"diffraction_pattern_{idx:05d}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {filename}")
    
    # Create a summary grid plot
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        img = diffraction[idx]
        ax = axes[i]
        
        # Display with log scale
        log_img = np.log10(img + 0.1)  # Small positive offset for better color distribution
        im = ax.imshow(log_img, cmap='jet', origin='lower')
        ax.set_title(f'Pattern #{idx}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add colorbar to each subplot
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.suptitle('Fly64 Dataset: Sample Diffraction Patterns (Log Scale - Jet Colormap)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('diffraction_patterns_grid.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: diffraction_patterns_grid.png")
    
    # Create a linear and log comparison for one image
    sample_idx = indices[len(indices)//2]  # Middle image
    sample_img = diffraction[sample_idx]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Linear scale
    im1 = ax1.imshow(sample_img, cmap='jet', origin='lower')
    ax1.set_title(f'Linear Scale - Pattern #{sample_idx}', fontsize=12)
    ax1.set_xlabel('Pixel X')
    ax1.set_ylabel('Pixel Y')
    plt.colorbar(im1, ax=ax1, label='Intensity')
    
    # Log scale
    log_sample = np.log10(sample_img + 0.1)  # Small positive offset for better color distribution
    im2 = ax2.imshow(log_sample, cmap='jet', origin='lower')
    ax2.set_title(f'Log Scale - Pattern #{sample_idx}', fontsize=12)
    ax2.set_xlabel('Pixel X')
    ax2.set_ylabel('Pixel Y')
    plt.colorbar(im2, ax=ax2, label='Log10(Intensity + 0.1)')
    
    plt.suptitle('Diffraction Pattern: Linear vs Log Scale Comparison', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('diffraction_scale_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: diffraction_scale_comparison.png")
    
    # Print summary statistics
    print(f"\n📈 Dataset Statistics:")
    print(f"   • Total diffraction patterns: {diffraction.shape[0]:,}")
    print(f"   • Pattern size: {diffraction.shape[1]}×{diffraction.shape[2]} pixels")
    print(f"   • Data type: {diffraction.dtype}")
    print(f"   • Min intensity: {diffraction.min():.6f}")
    print(f"   • Max intensity: {diffraction.max():.6f}")
    print(f"   • Mean intensity: {diffraction.mean():.6f}")
    print(f"   • Generated {n_images} individual PNG files + 2 summary plots")
    
    data.close()

if __name__ == "__main__":
    generate_diffraction_pngs()