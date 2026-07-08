#!/usr/bin/env python
"""
Create publication-quality figure for Nature Methods using CORRECT reconstruction files.
Phase-only visualization as requested.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, Rectangle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Nature Methods style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.linewidth'] = 0.8

def create_final_figure():
    """Create the publication-quality figure with correct data paths."""
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    
    # Main grid: 3 rows
    gs = gridspec.GridSpec(3, 1, figure=fig, 
                          height_ratios=[1, 0.15, 1.5],
                          hspace=0.3)
    
    # ========== TOP ROW: Probe Comparison & Ground Truth ==========
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0],
                                              width_ratios=[1, 1, 1],
                                              wspace=0.2)
    
    # Load probe data
    print("Loading probe data...")
    fly64 = np.load('datasets/fly64/fly64_shuffled.npz')
    run1084 = np.load('datasets/Run1084_recon3_postPC_shrunk_3.npz')
    
    # --- Probe 1: Training (fly64) ---
    ax_probe1 = fig.add_subplot(gs_top[0])
    fly64_probe = fly64['probeGuess']
    fly64_amp = np.abs(fly64_probe)
    fly64_phase = np.angle(fly64_probe)
    
    # Show phase with amplitude as alpha
    fly64_amp_norm = fly64_amp / fly64_amp.max()
    im1 = ax_probe1.imshow(fly64_phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax_probe1.imshow(np.ones_like(fly64_phase), alpha=1-fly64_amp_norm, cmap='gray', vmin=0, vmax=1)
    
    ax_probe1.set_title('Training Probe (fly64)\nμ={:.3f}, σ={:.3f}'.format(
        fly64_amp.mean(), fly64_amp.std()), fontsize=10, fontweight='bold')
    ax_probe1.axis('off')
    
    # --- Ground Truth (center) ---
    ax_truth = fig.add_subplot(gs_top[1])
    truth = run1084['objectGuess']
    truth_phase = np.angle(truth)
    truth_amp = np.abs(truth)
    
    # Crop to interesting region
    h, w = truth_phase.shape
    crop = 30
    truth_phase_crop = truth_phase[crop:h-crop, crop:w-crop]
    
    im_truth = ax_truth.imshow(truth_phase_crop, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax_truth.set_title('Ground Truth (Run1084)\nPhase', fontsize=10, fontweight='bold')
    ax_truth.axis('off')
    
    # --- Probe 2: Test (Run1084) ---
    ax_probe2 = fig.add_subplot(gs_top[2])
    run1084_probe = run1084['probeGuess']
    run1084_amp = np.abs(run1084_probe)
    run1084_phase = np.angle(run1084_probe)
    
    # Show phase with amplitude as alpha
    run1084_amp_norm = run1084_amp / run1084_amp.max()
    im2 = ax_probe2.imshow(run1084_phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax_probe2.imshow(np.ones_like(run1084_phase), alpha=1-run1084_amp_norm, cmap='gray', vmin=0, vmax=1)
    
    ax_probe2.set_title('Test Probe (Run1084)\nμ={:.3f}, σ={:.3f}'.format(
        run1084_amp.mean(), run1084_amp.std()), fontsize=10, fontweight='bold')
    ax_probe2.axis('off')
    
    # Add annotation showing the difference
    fig.text(0.5, 0.68, f'Probe Amplitude Change: {run1084_amp.mean()/fly64_amp.mean():.1f}× increase',
             ha='center', fontsize=10, fontweight='bold', color='red')
    
    # ========== MIDDLE: Arrow ==========
    ax_arrow = fig.add_subplot(gs[1])
    ax_arrow.axis('off')
    
    arrow = FancyArrowPatch((0.15, 0.5), (0.85, 0.5),
                           connectionstyle="arc3", 
                           arrowstyle='->,head_width=0.4,head_length=0.2',
                           lw=3, color='#333333', transform=ax_arrow.transAxes)
    ax_arrow.add_patch(arrow)
    ax_arrow.text(0.5, 0.5, 'Out-of-Distribution Generalization Test', 
                 ha='center', va='center', fontsize=12, 
                 fontweight='bold', transform=ax_arrow.transAxes)
    
    # ========== BOTTOM: 2x2 Reconstructions ==========
    gs_recon = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[2],
                                                hspace=0.2, wspace=0.15)
    
    # Load reconstruction images (phase only)
    reconstructions = {
        'pinn_indist': 'experiment_outputs/run1084_trained_models/recon_on_run1084_pinn/reconstructed_phase.png',
        'pinn_outdist': 'experiment_outputs/fly64_trained_models/recon_on_run1084_pinn/reconstructed_phase.png',
        'base_indist': 'experiment_outputs/run1084_trained_models/recon_on_run1084_baseline/baseline_reconstructed_phase.png',
        'base_outdist': 'experiment_outputs/fly64_trained_models/recon_on_run1084_baseline/baseline_reconstructed_phase.png'
    }
    
    titles = [
        ('PtychoPINN\nIn-Distribution\n(trained on Run1084)', 'pinn_indist', 'green'),
        ('PtychoPINN\nOut-of-Distribution\n(trained on fly64)', 'pinn_outdist', 'green'),
        ('Baseline U-Net\nIn-Distribution\n(trained on Run1084)', 'base_indist', 'blue'),
        ('Baseline U-Net\nOut-of-Distribution\n(trained on fly64)', 'base_outdist', 'red')
    ]
    
    for idx, (title, key, color) in enumerate(titles):
        ax = fig.add_subplot(gs_recon[idx // 2, idx % 2])
        
        # Load the phase reconstruction
        img_path = Path(reconstructions[key])
        if img_path.exists():
            img = plt.imread(str(img_path))
            
            # Display the reconstruction
            ax.imshow(img, cmap='twilight' if img.ndim == 2 else None)
            
            # Special annotation for failed baseline OOD
            if key == 'base_outdist':
                # Check if it's the collapsed output
                if img.ndim == 2:
                    unique_vals = len(np.unique(np.round(img, 3)))
                    if unique_vals < 20:  # It's collapsed
                        ax.text(0.5, 0.5, f'DEGRADED\n({unique_vals} unique values)', 
                               ha='center', va='center', fontsize=11,
                               color='white', fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', 
                                       facecolor='red', alpha=0.7),
                               transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'File not found:\n' + str(img_path.name), 
                   ha='center', va='center', fontsize=8,
                   transform=ax.transAxes)
            ax.set_facecolor('#f0f0f0')
        
        # Add title with color coding
        ax.set_title(title, fontsize=9, color=color, fontweight='bold')
        ax.axis('off')
        
        # Add border for emphasis
        rect = Rectangle((0, 0), 1, 1, linewidth=2, 
                        edgecolor=color, facecolor='none',
                        transform=ax.transAxes)
        ax.add_patch(rect)
    
    # Overall title
    fig.suptitle('Physics-Informed Neural Networks Demonstrate Superior Generalization\nto Out-of-Distribution Probe Functions',
                fontsize=13, fontweight='bold', y=0.98)
    
    # Add panel labels
    fig.text(0.02, 0.94, 'a', fontsize=14, fontweight='bold')
    fig.text(0.02, 0.62, 'b', fontsize=14, fontweight='bold')
    fig.text(0.02, 0.42, 'c', fontsize=14, fontweight='bold')
    
    # Add legend/key
    fig.text(0.5, 0.02, 
            'In-Distribution: Model tested on same dataset it was trained on | ' + 
            'Out-of-Distribution: Model tested on different dataset with different probe',
            ha='center', fontsize=8, style='italic')
    
    plt.tight_layout()
    
    # Save
    fig.savefig('experiment_outputs/nature_figure_final.pdf', 
               dpi=300, bbox_inches='tight', pad_inches=0.1)
    fig.savefig('experiment_outputs/nature_figure_final.png', 
               dpi=300, bbox_inches='tight', pad_inches=0.1)
    
    print("\nFigure saved as nature_figure_final.pdf/png")
    
    # Generate caption
    caption = """
    Figure X. Robustness of physics-informed vs supervised models to probe distribution shift.
    (a) Comparison of probe functions between training datasets. The fly64 probe (left) has 
    significantly lower amplitude (μ=0.086) compared to Run1084 probe (right, μ=0.322), 
    representing a 3.7× amplitude increase. Ground truth phase reconstruction shown in center.
    (b) Schematic of out-of-distribution generalization challenge.
    (c) Phase reconstructions from models trained and tested on different datasets. 
    In-distribution performance (diagonal) shows both models perform well when tested on 
    their training distribution. Out-of-distribution performance reveals PtychoPINN maintains 
    reconstruction quality when trained on fly64 and tested on Run1084, while the baseline 
    U-Net shows significant degradation with quantized output values, demonstrating the 
    importance of physics-based inductive bias for generalization.
    """
    
    with open('experiment_outputs/figure_caption_final.txt', 'w') as f:
        f.write(caption)
    
    print("Caption saved to figure_caption_final.txt")
    
    return fig

if __name__ == "__main__":
    create_final_figure()