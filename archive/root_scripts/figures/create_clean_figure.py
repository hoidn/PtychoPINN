#!/usr/bin/env python
"""
Create a clean, minimal figure with proper aspect ratio.
No unnecessary text, let the images speak.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Clean style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0
plt.rcParams['xtick.major.size'] = 0
plt.rcParams['ytick.major.size'] = 0

def create_clean_figure():
    """Create a clean, minimal figure."""
    
    # Figure with proper aspect ratio (wide format for journal)
    fig = plt.figure(figsize=(12, 5), facecolor='white')
    
    # Simple grid: probes on left, reconstructions on right
    gs = gridspec.GridSpec(2, 4, figure=fig,
                          height_ratios=[1, 1],
                          width_ratios=[0.8, 0.8, 1, 1],
                          hspace=0.3, wspace=0.2)
    
    # Load probe data
    fly64 = np.load('datasets/fly64/fly64_shuffled.npz')
    run1084 = np.load('datasets/Run1084_recon3_postPC_shrunk_3.npz')
    
    # ========== LEFT: PROBES ==========
    # Training probe (fly64)
    ax_probe1 = fig.add_subplot(gs[0, 0])
    fly64_amp = np.abs(fly64['probeGuess'])
    im1 = ax_probe1.imshow(fly64_amp, cmap='hot')
    ax_probe1.set_title('Training\n(fly64)', fontsize=9)
    ax_probe1.axis('off')
    
    # Test probe (Run1084)  
    ax_probe2 = fig.add_subplot(gs[1, 0])
    run1084_amp = np.abs(run1084['probeGuess'])
    im2 = ax_probe2.imshow(run1084_amp, cmap='hot')
    ax_probe2.set_title('Test\n(Run1084)', fontsize=9)
    ax_probe2.axis('off')
    
    # Ground truth
    ax_truth = fig.add_subplot(gs[:, 1])
    truth = run1084['objectGuess']
    truth_phase = np.angle(truth)
    # Crop to interesting region
    h, w = truth_phase.shape
    crop = 50
    truth_crop = truth_phase[crop:h-crop, crop:w-crop]
    ax_truth.imshow(truth_crop, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax_truth.set_title('Ground Truth', fontsize=9)
    ax_truth.axis('off')
    
    # ========== RIGHT: RECONSTRUCTIONS ==========
    # Load reconstruction images
    paths = {
        'pinn_id': Path('experiment_outputs/run1084_trained_models/recon_on_run1084_pinn/reconstructed_phase.png'),
        'pinn_ood': Path('experiment_outputs/fly64_trained_models/recon_on_run1084_pinn/reconstructed_phase.png'),
        'base_id': Path('experiment_outputs/run1084_trained_models/recon_on_run1084_baseline/baseline_reconstructed_phase.png'),
        'base_ood': Path('experiment_outputs/fly64_trained_models/recon_on_run1084_baseline/baseline_reconstructed_phase.png')
    }
    
    positions = [(0, 2), (0, 3), (1, 2), (1, 3)]
    titles = ['PtychoPINN\nIn-Dist', 'PtychoPINN\nOut-Dist', 
              'Baseline\nIn-Dist', 'Baseline\nOut-Dist']
    keys = ['pinn_id', 'pinn_ood', 'base_id', 'base_ood']
    
    for pos, title, key in zip(positions, titles, keys):
        ax = fig.add_subplot(gs[pos[0], pos[1]])
        
        if paths[key].exists():
            img = plt.imread(str(paths[key]))
            ax.imshow(img, cmap='viridis' if img.ndim == 2 else None)
        else:
            ax.text(0.5, 0.5, 'Not found', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_facecolor('#f0f0f0')
        
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    
    # Minimal labeling
    fig.text(0.12, 0.95, 'Probes', fontsize=11, fontweight='bold')
    fig.text(0.3, 0.95, 'Reference', fontsize=11, fontweight='bold')
    fig.text(0.65, 0.95, 'Reconstructions', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    fig.savefig('experiment_outputs/clean_figure.pdf', dpi=300, bbox_inches='tight')
    fig.savefig('experiment_outputs/clean_figure.png', dpi=300, bbox_inches='tight')
    
    print("Clean figure saved")
    
    return fig

if __name__ == "__main__":
    create_clean_figure()