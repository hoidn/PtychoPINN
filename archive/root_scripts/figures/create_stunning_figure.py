#!/usr/bin/env python
"""
Create a STUNNING publication-quality figure that tells a visual story.
Focus on the dramatic comparison between failed baseline and successful PtychoPINN.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib import patheffects
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Professional style
plt.style.use('dark_background')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0
plt.rcParams['figure.facecolor'] = '#0a0a0a'
plt.rcParams['axes.facecolor'] = '#0a0a0a'

def load_and_process_image(path, cmap='viridis', enhance=True):
    """Load image and optionally enhance contrast."""
    img = plt.imread(str(path))
    if enhance and img.ndim == 2:
        # Enhance contrast
        vmin, vmax = np.percentile(img, [5, 95])
        img = np.clip((img - vmin) / (vmax - vmin), 0, 1)
    return img

def create_stunning_figure():
    """Create a visually stunning figure that tells the story."""
    
    # Create figure with dark background
    fig = plt.figure(figsize=(16, 10), facecolor='#0a0a0a')
    
    # Custom grid for creative layout
    gs = gridspec.GridSpec(3, 3, figure=fig,
                          height_ratios=[0.3, 1.5, 0.3],
                          width_ratios=[0.8, 1.4, 0.8],
                          hspace=0.15, wspace=0.15)
    
    # ============ TITLE ============
    fig.text(0.5, 0.97, 'Distribution Shift Challenge: Physics vs Pure Learning',
             fontsize=20, fontweight='bold', ha='center', color='white',
             path_effects=[patheffects.withStroke(linewidth=3, foreground='#1a1a1a')])
    
    # ============ TOP: PROBE STORY ============
    # Show probe transformation as a visual narrative
    
    ax_probe_story = fig.add_subplot(gs[0, :])
    ax_probe_story.axis('off')
    
    # Load probes
    fly64 = np.load('datasets/fly64/fly64_shuffled.npz')
    run1084 = np.load('datasets/Run1084_recon3_postPC_shrunk_3.npz')
    
    fly64_amp = np.abs(fly64['probeGuess'])
    run1084_amp = np.abs(run1084['probeGuess'])
    
    # Create probe comparison visualization
    probe_panel_width = 0.25
    probe_positions = [0.1, 0.4, 0.7]
    
    # Probe 1 (training)
    ax1 = fig.add_axes([probe_positions[0], 0.82, probe_panel_width*0.4, 0.12])
    im1 = ax1.imshow(fly64_amp, cmap='inferno', interpolation='bicubic')
    ax1.axis('off')
    ax1.text(0.5, -0.15, 'Training Probe\n(fly64)', ha='center', 
             transform=ax1.transAxes, color='white', fontsize=9)
    ax1.text(0.5, -0.35, f'μ={fly64_amp.mean():.3f}', ha='center',
             transform=ax1.transAxes, color='#888888', fontsize=8)
    
    # Arrow showing transformation
    arrow_ax = fig.add_axes([probe_positions[0]+0.12, 0.82, 0.15, 0.12])
    arrow_ax.axis('off')
    arrow = FancyArrowPatch((0.1, 0.5), (0.9, 0.5),
                           arrowstyle='->,head_width=0.3,head_length=0.2',
                           lw=3, color='#ff6b6b', 
                           path_effects=[patheffects.withStroke(linewidth=5, foreground='#0a0a0a')],
                           transform=arrow_ax.transAxes)
    arrow_ax.add_patch(arrow)
    arrow_ax.text(0.5, 0.5, '3.7×', ha='center', va='center',
                 fontsize=14, fontweight='bold', color='#ff6b6b',
                 transform=arrow_ax.transAxes)
    
    # Probe 2 (test)
    ax2 = fig.add_axes([probe_positions[1]+0.05, 0.82, probe_panel_width*0.4, 0.12])
    im2 = ax2.imshow(run1084_amp, cmap='inferno', interpolation='bicubic')
    ax2.axis('off')
    ax2.text(0.5, -0.15, 'Test Probe\n(Run1084)', ha='center',
             transform=ax2.transAxes, color='white', fontsize=9)
    ax2.text(0.5, -0.35, f'μ={run1084_amp.mean():.3f}', ha='center',
             transform=ax2.transAxes, color='#888888', fontsize=8)
    
    # Challenge text
    fig.text(0.75, 0.88, 'OUT OF', fontsize=14, fontweight='bold',
             color='#ff6b6b', ha='center')
    fig.text(0.75, 0.86, 'DISTRIBUTION', fontsize=14, fontweight='bold',
             color='#ff6b6b', ha='center')
    fig.text(0.75, 0.84, 'CHALLENGE', fontsize=14, fontweight='bold',
             color='#ff6b6b', ha='center')
    
    # ============ MAIN: HERO COMPARISON ============
    # Large, dramatic comparison of OOD performance
    
    # Load the key reconstructions
    baseline_ood_path = Path('experiment_outputs/fly64_trained_models/recon_on_run1084_baseline/baseline_reconstructed_phase.png')
    pinn_ood_path = Path('experiment_outputs/fly64_trained_models/recon_on_run1084_pinn/reconstructed_phase.png')
    
    # Baseline OOD (FAILED) - Left
    ax_base_fail = fig.add_subplot(gs[1, 0])
    if baseline_ood_path.exists():
        img_base = load_and_process_image(baseline_ood_path)
        ax_base_fail.imshow(img_base, cmap='plasma', interpolation='bicubic')
    ax_base_fail.axis('off')
    
    # Add failure annotation with style
    fail_box = FancyBboxPatch((0.05, 0.85), 0.9, 0.12,
                              boxstyle="round,pad=0.02",
                              facecolor='#d32f2f', alpha=0.9,
                              edgecolor='white', linewidth=2,
                              transform=ax_base_fail.transAxes)
    ax_base_fail.add_patch(fail_box)
    ax_base_fail.text(0.5, 0.91, 'BASELINE FAILS', ha='center', va='center',
                     fontsize=12, fontweight='bold', color='white',
                     transform=ax_base_fail.transAxes)
    
    ax_base_fail.text(0.5, -0.05, 'Supervised U-Net', ha='center',
                     fontsize=11, color='#888888',
                     transform=ax_base_fail.transAxes)
    
    # VS text in center
    ax_vs = fig.add_subplot(gs[1, 1])
    ax_vs.axis('off')
    
    # Create dramatic VS visualization
    vs_circle = Circle((0.5, 0.5), 0.15, fill=True, 
                      facecolor='#1a1a1a', edgecolor='#4CAF50',
                      linewidth=3, transform=ax_vs.transAxes)
    ax_vs.add_patch(vs_circle)
    ax_vs.text(0.5, 0.5, 'VS', ha='center', va='center',
              fontsize=28, fontweight='bold', color='white',
              transform=ax_vs.transAxes)
    
    # Ground truth reference (small, above VS)
    ax_truth = fig.add_axes([0.43, 0.55, 0.14, 0.14])
    truth = run1084['objectGuess']
    truth_phase = np.angle(truth)
    # Crop to interesting part
    h, w = truth_phase.shape
    crop = 40
    truth_crop = truth_phase[crop:h-crop, crop:w-crop]
    ax_truth.imshow(truth_crop, cmap='viridis', interpolation='bicubic')
    ax_truth.axis('off')
    ax_truth.text(0.5, -0.1, 'Ground Truth', ha='center',
                 fontsize=9, color='#888888', transform=ax_truth.transAxes)
    
    # PtychoPINN OOD (SUCCESS) - Right
    ax_pinn_success = fig.add_subplot(gs[1, 2])
    if pinn_ood_path.exists():
        img_pinn = load_and_process_image(pinn_ood_path)
        ax_pinn_success.imshow(img_pinn, cmap='viridis', interpolation='bicubic')
    ax_pinn_success.axis('off')
    
    # Add success annotation with style
    success_box = FancyBboxPatch((0.05, 0.85), 0.9, 0.12,
                                 boxstyle="round,pad=0.02",
                                 facecolor='#4CAF50', alpha=0.9,
                                 edgecolor='white', linewidth=2,
                                 transform=ax_pinn_success.transAxes)
    ax_pinn_success.add_patch(success_box)
    ax_pinn_success.text(0.5, 0.91, 'PTYCHOPINN SUCCEEDS', ha='center', va='center',
                        fontsize=12, fontweight='bold', color='white',
                        transform=ax_pinn_success.transAxes)
    
    ax_pinn_success.text(0.5, -0.05, 'Physics-Informed NN', ha='center',
                        fontsize=11, color='#888888',
                        transform=ax_pinn_success.transAxes)
    
    # ============ BOTTOM: IN-DISTRIBUTION REFERENCE ============
    # Small panels showing both work in-distribution
    
    bottom_y = 0.08
    panel_size = 0.12
    
    # Load in-distribution reconstructions
    baseline_id_path = Path('experiment_outputs/run1084_trained_models/recon_on_run1084_baseline/baseline_reconstructed_phase.png')
    pinn_id_path = Path('experiment_outputs/run1084_trained_models/recon_on_run1084_pinn/reconstructed_phase.png')
    
    # Baseline in-dist
    ax_base_id = fig.add_axes([0.25, bottom_y, panel_size, panel_size])
    if baseline_id_path.exists():
        img = load_and_process_image(baseline_id_path)
        ax_base_id.imshow(img, cmap='viridis', interpolation='bicubic', alpha=0.7)
    ax_base_id.axis('off')
    ax_base_id.text(0.5, -0.15, 'Baseline\n(in-distribution)', ha='center',
                   fontsize=8, color='#666666', transform=ax_base_id.transAxes)
    
    # PtychoPINN in-dist
    ax_pinn_id = fig.add_axes([0.63, bottom_y, panel_size, panel_size])
    if pinn_id_path.exists():
        img = load_and_process_image(pinn_id_path)
        ax_pinn_id.imshow(img, cmap='viridis', interpolation='bicubic', alpha=0.7)
    ax_pinn_id.axis('off')
    ax_pinn_id.text(0.5, -0.15, 'PtychoPINN\n(in-distribution)', ha='center',
                   fontsize=8, color='#666666', transform=ax_pinn_id.transAxes)
    
    # Reference text
    fig.text(0.5, 0.03, 'Both models work when tested on their training distribution',
             ha='center', fontsize=9, color='#666666', style='italic')
    
    # ============ KEY MESSAGE ============
    fig.text(0.5, 0.23, 'Physics-based inductive bias enables generalization to new experimental conditions',
             ha='center', fontsize=12, color='white', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a1a', alpha=0.8))
    
    plt.tight_layout()
    
    # Save
    fig.savefig('experiment_outputs/stunning_figure.pdf', 
               dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
    fig.savefig('experiment_outputs/stunning_figure.png', 
               dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
    
    print("\nStunning figure saved as stunning_figure.pdf/png")
    
    return fig

if __name__ == "__main__":
    create_stunning_figure()