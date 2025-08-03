#!/usr/bin/env python3
"""
generate_2x2_visualization.py - Generate visualizations for 2x2 probe study

This script creates:
1. Side-by-side reconstruction comparison (2x2 grid)
2. Probe comparison figure (default vs hybrid)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import sys


def setup_logging():
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate visualizations for 2x2 probe parameterization study"
    )
    parser.add_argument(
        "study_dir",
        type=Path,
        help="Path to study output directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for figures (default: study_dir)"
    )
    return parser.parse_args()


def load_reconstruction(study_dir: Path, experiment: str) -> tuple:
    """Load reconstruction data from an experiment."""
    recon_file = study_dir / experiment / "evaluation" / "reconstructions_aligned.npz"
    
    if not recon_file.exists():
        # Try unaligned version
        recon_file = study_dir / experiment / "evaluation" / "reconstructions.npz"
    
    if not recon_file.exists():
        return None, None
    
    data = np.load(recon_file)
    
    # Handle different possible key names
    recon = None
    if 'pinn_reconstruction' in data:
        recon = data['pinn_reconstruction']
    elif 'PtychoPINN_reconstruction' in data:
        recon = data['PtychoPINN_reconstruction']
    elif 'ptychopinn_complex' in data:
        recon = data['ptychopinn_complex']
    elif 'ptychopinn_amplitude' in data and 'ptychopinn_phase' in data:
        # Reconstruct complex from amplitude and phase
        amp = data['ptychopinn_amplitude']
        phase = data['ptychopinn_phase']
        recon = amp * np.exp(1j * phase)
    else:
        # Try to find any reconstruction key
        for key in data.keys():
            if 'reconstruction' in key.lower() and 'complex' in key.lower():
                recon = data[key]
                break
    
    if recon is None:
        return None, None
    
    # Get PSNR values from metadata if available
    psnr_amp = None
    psnr_phase = None
    
    # Try to load metrics
    metrics_file = study_dir / experiment / "evaluation" / "comparison_metrics.csv"
    if metrics_file.exists():
        import pandas as pd
        metrics = pd.read_csv(metrics_file)
        psnr_rows = metrics[metrics['metric'] == 'psnr']
        if len(psnr_rows) > 0:
            psnr_amp = psnr_rows['amplitude'].values[0]
            psnr_phase = psnr_rows['phase'].values[0]
    
    return recon, (psnr_amp, psnr_phase)


def create_reconstruction_comparison(study_dir: Path, output_dir: Path):
    """Create 2x2 grid of reconstruction comparisons."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('2x2 Probe Parameterization Study: Reconstruction Comparison', fontsize=16)
    
    experiments = [
        ("gs1_default", "Gridsize 1, Default Probe"),
        ("gs1_hybrid", "Gridsize 1, Hybrid Probe"),
        ("gs2_default", "Gridsize 2, Default Probe"),
        ("gs2_hybrid", "Gridsize 2, Hybrid Probe")
    ]
    
    for idx, (exp, title) in enumerate(experiments):
        row = idx // 2
        col_base = (idx % 2) * 2
        
        recon, psnr = load_reconstruction(study_dir, exp)
        
        if recon is not None:
            # Amplitude
            ax_amp = axes[row, col_base]
            im_amp = ax_amp.imshow(np.abs(recon), cmap='hot')
            ax_amp.set_title(f'{title}\nAmplitude', fontsize=12)
            ax_amp.axis('off')
            plt.colorbar(im_amp, ax=ax_amp, fraction=0.046, pad=0.04)
            
            # Phase
            ax_phase = axes[row, col_base + 1]
            im_phase = ax_phase.imshow(np.angle(recon), cmap='twilight', 
                                       vmin=-np.pi, vmax=np.pi)
            ax_phase.set_title(f'Phase', fontsize=12)
            ax_phase.axis('off')
            plt.colorbar(im_phase, ax=ax_phase, fraction=0.046, pad=0.04)
            
            # Add PSNR annotations if available
            if psnr[0] is not None:
                ax_amp.text(0.02, 0.98, f'PSNR: {psnr[0]:.1f} dB', 
                           transform=ax_amp.transAxes, color='white',
                           verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3',
                                                             facecolor='black', alpha=0.7))
            if psnr[1] is not None:
                ax_phase.text(0.02, 0.98, f'PSNR: {psnr[1]:.1f} dB',
                            transform=ax_phase.transAxes, color='white',
                            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3',
                                                              facecolor='black', alpha=0.7))
        else:
            # Clear axes if no data
            axes[row, col_base].text(0.5, 0.5, f'{title}\nNo Data', 
                                    ha='center', va='center', fontsize=14,
                                    transform=axes[row, col_base].transAxes)
            axes[row, col_base].axis('off')
            axes[row, col_base + 1].axis('off')
    
    plt.tight_layout()
    output_file = output_dir / '2x2_reconstruction_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_file


def create_probe_comparison(study_dir: Path, output_dir: Path):
    """Create probe comparison figure."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('Probe Comparison: Default vs Hybrid', fontsize=16)
    
    # Load probes
    default_probe_file = study_dir / 'default_probe.npy'
    hybrid_probe_file = study_dir / 'hybrid_probe.npy'
    
    if not default_probe_file.exists() or not hybrid_probe_file.exists():
        logging.error("Probe files not found!")
        return None
    
    default_probe = np.load(default_probe_file)
    hybrid_probe = np.load(hybrid_probe_file)
    
    # Default probe
    axes[0, 0].imshow(np.abs(default_probe), cmap='hot')
    axes[0, 0].set_title('Default Probe\nAmplitude')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(np.angle(default_probe), cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[0, 1].set_title('Default Probe\nPhase')
    axes[0, 1].axis('off')
    
    # Hybrid probe
    axes[1, 0].imshow(np.abs(hybrid_probe), cmap='hot')
    axes[1, 0].set_title('Hybrid Probe\nAmplitude')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(np.angle(hybrid_probe), cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[1, 1].set_title('Hybrid Probe\nPhase')
    axes[1, 1].axis('off')
    
    # Difference plots
    amp_diff = np.abs(hybrid_probe) - np.abs(default_probe)
    phase_diff = np.angle(hybrid_probe) - np.angle(default_probe)
    
    # Wrap phase difference to [-pi, pi]
    phase_diff = np.angle(np.exp(1j * phase_diff))
    
    im_amp_diff = axes[0, 2].imshow(amp_diff, cmap='RdBu_r')
    axes[0, 2].set_title('Amplitude\nDifference')
    axes[0, 2].axis('off')
    plt.colorbar(im_amp_diff, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    im_phase_diff = axes[1, 2].imshow(phase_diff, cmap='RdBu_r', vmin=-np.pi, vmax=np.pi)
    axes[1, 2].set_title('Phase\nDifference')
    axes[1, 2].axis('off')
    plt.colorbar(im_phase_diff, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    # Add statistics
    stats_text = (
        f"Default Probe:\n"
        f"  Amp mean: {np.abs(default_probe).mean():.4f}\n"
        f"  Phase std: {np.angle(default_probe).std():.4f}\n\n"
        f"Hybrid Probe:\n"
        f"  Amp mean: {np.abs(hybrid_probe).mean():.4f}\n"
        f"  Phase std: {np.angle(hybrid_probe).std():.4f}"
    )
    
    fig.text(0.98, 0.5, stats_text, transform=fig.transFigure,
             fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    output_file = output_dir / 'probe_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_file


def main():
    logger = setup_logging()
    args = parse_arguments()
    
    if not args.study_dir.exists():
        logger.error(f"Study directory not found: {args.study_dir}")
        sys.exit(1)
    
    output_dir = args.output_dir or args.study_dir
    output_dir.mkdir(exist_ok=True)
    
    # Create visualizations
    logger.info("Creating reconstruction comparison...")
    recon_file = create_reconstruction_comparison(args.study_dir, output_dir)
    if recon_file:
        logger.info(f"Saved reconstruction comparison to: {recon_file}")
    
    logger.info("Creating probe comparison...")
    probe_file = create_probe_comparison(args.study_dir, output_dir)
    if probe_file:
        logger.info(f"Saved probe comparison to: {probe_file}")
    
    logger.info("Visualization generation complete!")


if __name__ == "__main__":
    main()