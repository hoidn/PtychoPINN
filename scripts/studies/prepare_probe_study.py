#!/usr/bin/env python3
"""
prepare_probe_study.py - Prepare probe pair for parameterization study

This script creates the two probes needed for the study:
1. Default probe: Specified amplitude with flat (zero) phase
2. Hybrid probe: Same amplitude with aberrated phase from another source

The probes are created BEFORE simulation, as they will be used as inputs
to generate different datasets.
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
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare probe pair for parameterization study"
    )
    parser.add_argument(
        "--amplitude-source",
        type=Path,
        required=True,
        help="Source for probe amplitude (.npz with probeGuess or .npy probe file)"
    )
    parser.add_argument(
        "--phase-source",
        type=Path,
        required=True,
        help="Source for aberrated phase (.npz with probeGuess or .npy probe file)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for probe files"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization of probes"
    )
    parser.add_argument(
        "--amplitude-key",
        type=str,
        default="probeGuess",
        help="Key name for probe in NPZ file (default: probeGuess)"
    )
    parser.add_argument(
        "--phase-key",
        type=str,
        default="probeGuess",
        help="Key name for probe in NPZ file (default: probeGuess)"
    )
    return parser.parse_args()


def load_probe(file_path: Path, key: str = "probeGuess") -> np.ndarray:
    """Load probe from NPZ or NPY file."""
    if file_path.suffix == '.npz':
        data = np.load(file_path)
        if key not in data:
            available_keys = list(data.keys())
            raise KeyError(f"Key '{key}' not found in {file_path}. Available keys: {available_keys}")
        return data[key]
    elif file_path.suffix == '.npy':
        return np.load(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")


def create_probe_pair(amplitude_probe: np.ndarray, phase_probe: np.ndarray) -> tuple:
    """
    Create default and hybrid probes.
    
    Args:
        amplitude_probe: Probe to use for amplitude
        phase_probe: Probe to use for phase (in hybrid)
        
    Returns:
        default_probe, hybrid_probe
    """
    # Ensure same shape
    if amplitude_probe.shape != phase_probe.shape:
        raise ValueError(f"Probe shapes don't match: {amplitude_probe.shape} vs {phase_probe.shape}")
    
    # Default probe: specified amplitude with flat phase
    default_probe = np.abs(amplitude_probe).astype(np.complex64)
    
    # Hybrid probe: same amplitude with aberrated phase
    hybrid_probe = np.abs(amplitude_probe) * np.exp(1j * np.angle(phase_probe))
    hybrid_probe = hybrid_probe.astype(np.complex64)
    
    return default_probe, hybrid_probe


def save_probes(default_probe: np.ndarray, hybrid_probe: np.ndarray, output_dir: Path):
    """Save probe pair to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    default_path = output_dir / "default_probe.npy"
    hybrid_path = output_dir / "hybrid_probe.npy"
    
    np.save(default_path, default_probe)
    np.save(hybrid_path, hybrid_probe)
    
    return default_path, hybrid_path


def create_visualization(default_probe: np.ndarray, hybrid_probe: np.ndarray, output_dir: Path):
    """Create visualization comparing the probes."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('Probe Pair for Parameterization Study', fontsize=16)
    
    # Default probe
    im1 = axes[0, 0].imshow(np.abs(default_probe), cmap='hot')
    axes[0, 0].set_title('Default Probe\nAmplitude')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
    
    im2 = axes[0, 1].imshow(np.angle(default_probe), cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[0, 1].set_title('Default Probe\nPhase (Flat)')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
    
    # Hybrid probe
    im3 = axes[1, 0].imshow(np.abs(hybrid_probe), cmap='hot')
    axes[1, 0].set_title('Hybrid Probe\nAmplitude (Same)')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
    
    im4 = axes[1, 1].imshow(np.angle(hybrid_probe), cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[1, 1].set_title('Hybrid Probe\nPhase (Aberrated)')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)
    
    # Phase difference
    phase_diff = np.angle(hybrid_probe) - np.angle(default_probe)
    phase_diff = np.angle(np.exp(1j * phase_diff))  # Wrap to [-pi, pi]
    
    im5 = axes[0, 2].imshow(phase_diff, cmap='RdBu_r', vmin=-np.pi, vmax=np.pi)
    axes[0, 2].set_title('Phase Difference\n(Hybrid - Default)')
    axes[0, 2].axis('off')
    plt.colorbar(im5, ax=axes[0, 2], fraction=0.046)
    
    # Statistics text
    stats_text = (
        f"Probe Statistics:\n"
        f"Shape: {default_probe.shape}\n"
        f"Amplitude mean: {np.abs(default_probe).mean():.4f}\n"
        f"Default phase std: {np.angle(default_probe).std():.4f}\n"
        f"Hybrid phase std: {np.angle(hybrid_probe).std():.4f}\n"
        f"Phase difference std: {phase_diff.std():.4f}"
    )
    
    axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes,
                    fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    output_path = output_dir / 'probe_pair_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def main():
    logger = setup_logging()
    args = parse_arguments()
    
    # Load probe sources
    logger.info(f"Loading amplitude source from: {args.amplitude_source}")
    amplitude_probe = load_probe(args.amplitude_source, args.amplitude_key)
    logger.info(f"  Loaded probe with shape {amplitude_probe.shape}")
    
    logger.info(f"Loading phase source from: {args.phase_source}")
    phase_probe = load_probe(args.phase_source, args.phase_key)
    logger.info(f"  Loaded probe with shape {phase_probe.shape}")
    
    # Create probe pair
    logger.info("Creating probe pair...")
    default_probe, hybrid_probe = create_probe_pair(amplitude_probe, phase_probe)
    
    # Save probes
    default_path, hybrid_path = save_probes(default_probe, hybrid_probe, args.output_dir)
    logger.info(f"Saved default probe to: {default_path}")
    logger.info(f"Saved hybrid probe to: {hybrid_path}")
    
    # Print statistics
    logger.info("\nProbe statistics:")
    logger.info(f"Default probe:")
    logger.info(f"  Shape: {default_probe.shape}")
    logger.info(f"  Dtype: {default_probe.dtype}")
    logger.info(f"  Amplitude mean: {np.abs(default_probe).mean():.6f}")
    logger.info(f"  Phase std: {np.angle(default_probe).std():.6f}")
    
    logger.info(f"\nHybrid probe:")
    logger.info(f"  Shape: {hybrid_probe.shape}")
    logger.info(f"  Dtype: {hybrid_probe.dtype}")
    logger.info(f"  Amplitude mean: {np.abs(hybrid_probe).mean():.6f}")
    logger.info(f"  Phase std: {np.angle(hybrid_probe).std():.6f}")
    
    # Create visualization if requested
    if args.visualize:
        logger.info("\nCreating visualization...")
        viz_path = create_visualization(default_probe, hybrid_probe, args.output_dir)
        logger.info(f"Saved visualization to: {viz_path}")
    
    logger.info("\nProbe preparation complete!")
    logger.info("These probes should be used as inputs to simulate separate datasets.")


if __name__ == "__main__":
    main()