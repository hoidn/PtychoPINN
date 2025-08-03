#!/usr/bin/env python
"""Visual validation script for test outputs."""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def visualize_test_output(npz_path, output_path=None):
    """Create visualization of test output."""
    data = np.load(npz_path)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Test Output Visualization: {os.path.basename(npz_path)}")
    
    # Object amplitude
    ax = axes[0, 0]
    im = ax.imshow(np.abs(data['objectGuess']), cmap='gray')
    ax.set_title('Object Amplitude')
    plt.colorbar(im, ax=ax)
    
    # Object phase
    ax = axes[0, 1]
    im = ax.imshow(np.angle(data['objectGuess']), cmap='hsv')
    ax.set_title('Object Phase')
    plt.colorbar(im, ax=ax)
    
    # Probe
    ax = axes[0, 2]
    im = ax.imshow(np.abs(data['probeGuess']), cmap='hot')
    ax.set_title('Probe Amplitude')
    plt.colorbar(im, ax=ax)
    
    # First diffraction pattern
    ax = axes[1, 0]
    im = ax.imshow(np.log1p(data['diffraction'][0]), cmap='viridis')
    ax.set_title('Diffraction Pattern 0 (log scale)')
    plt.colorbar(im, ax=ax)
    
    # Scan positions
    ax = axes[1, 1]
    ax.scatter(data['xcoords'], data['ycoords'], s=1, alpha=0.5)
    ax.set_title(f'Scan Positions (n={len(data["xcoords"])})')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    
    # Statistics
    ax = axes[1, 2]
    ax.text(0.1, 0.9, f"Diffraction shape: {data['diffraction'].shape}", transform=ax.transAxes)
    ax.text(0.1, 0.8, f"Data type: {data['diffraction'].dtype}", transform=ax.transAxes)
    ax.text(0.1, 0.7, f"Value range: [{np.min(data['diffraction']):.2f}, {np.max(data['diffraction']):.2f}]", transform=ax.transAxes)
    ax.text(0.1, 0.6, f"Coordinate range X: [{np.min(data['xcoords']):.1f}, {np.max(data['xcoords']):.1f}]", transform=ax.transAxes)
    ax.text(0.1, 0.5, f"Coordinate range Y: [{np.min(data['ycoords']):.1f}, {np.max(data['ycoords']):.1f}]", transform=ax.transAxes)
    ax.set_title('Data Statistics')
    ax.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_test_outputs.py <npz_file> [output_image]")
        sys.exit(1)
    
    npz_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    visualize_test_output(npz_path, output_path)
