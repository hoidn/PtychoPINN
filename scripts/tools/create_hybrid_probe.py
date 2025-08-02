#!/usr/bin/env python
"""
Create a hybrid probe by combining amplitude from one source with phase from another.

This tool enables the creation of synthetic probes for controlled experiments
by mixing the amplitude characteristics of one probe with the phase
characteristics of another. Both input probes must have exactly the same
dimensions - no automatic resizing is performed.

Example:
    python scripts/tools/create_hybrid_probe.py \\
        datasets/default_probe.npz \\
        datasets/fly64/fly001_64_train_converted.npz \\
        --output hybrid_probe.npy \\
        --visualize
        
Note: If probes have different dimensions, you must resize them to match
before using this tool.
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import logging

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ptycho.workflows.simulation_utils import load_probe_from_source
from ptycho.log_config import setup_logging

# Set up logger
logger = logging.getLogger(__name__)


def create_hybrid_probe(
    amplitude_source: np.ndarray,
    phase_source: np.ndarray,
    normalize: bool = False
) -> np.ndarray:
    """
    Create a hybrid probe by combining amplitude and phase from different sources.
    
    Parameters
    ----------
    amplitude_source : np.ndarray
        Probe to extract amplitude from
    phase_source : np.ndarray
        Probe to extract phase from
    normalize : bool, optional
        If True, normalize the hybrid probe to preserve total power
        
    Returns
    -------
    np.ndarray
        Hybrid probe with amplitude from source 1 and phase from source 2
        
    Raises
    ------
    ValueError
        If the input probes have different shapes. Both probes must have
        exactly the same dimensions - no automatic resizing is performed.
    """
    logger.info(f"Creating hybrid probe from amplitude {amplitude_source.shape} "
                f"and phase {phase_source.shape}")
    
    # Validate dimension match
    if amplitude_source.shape != phase_source.shape:
        raise ValueError(
            f"Probe dimensions must match exactly. Got amplitude source: {amplitude_source.shape} "
            f"and phase source: {phase_source.shape}. No automatic resizing will be performed."
        )
    
    # Extract amplitude and phase
    amplitude = np.abs(amplitude_source)
    phase = np.angle(phase_source)
    
    # Combine
    hybrid_probe = amplitude * np.exp(1j * phase)
    
    # Normalize if requested
    if normalize:
        # Calculate power normalization factor
        original_power = np.sum(np.abs(amplitude_source)**2)
        hybrid_power = np.sum(np.abs(hybrid_probe)**2)
        scale_factor = np.sqrt(original_power / hybrid_power)
        hybrid_probe *= scale_factor
        logger.debug(f"Applied normalization factor: {scale_factor:.4f}")
    
    # Ensure complex64 dtype
    hybrid_probe = hybrid_probe.astype(np.complex64)
    
    # Validate output
    if not np.all(np.isfinite(hybrid_probe)):
        raise ValueError("Hybrid probe contains NaN or Inf values")
    
    logger.info(f"Created hybrid probe with shape {hybrid_probe.shape}")
    logger.debug(f"Amplitude range: [{np.min(amplitude):.3f}, {np.max(amplitude):.3f}]")
    logger.debug(f"Phase range: [{np.min(phase):.3f}, {np.max(phase):.3f}]")
    
    return hybrid_probe


def visualize_probes(amplitude_source, phase_source, hybrid_probe, output_path):
    """Create a visualization comparing the source and hybrid probes."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Amplitude source
    im = axes[0, 0].imshow(np.abs(amplitude_source), cmap='gray')
    axes[0, 0].set_title('Amplitude Source\n(Amplitude)', fontsize=12)
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046)
    
    im = axes[1, 0].imshow(np.angle(amplitude_source), cmap='hsv', 
                          vmin=-np.pi, vmax=np.pi)
    axes[1, 0].set_title('Amplitude Source\n(Phase)', fontsize=12)
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
    
    # Phase source
    im = axes[0, 1].imshow(np.abs(phase_source), cmap='gray')
    axes[0, 1].set_title('Phase Source\n(Amplitude)', fontsize=12)
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)
    
    im = axes[1, 1].imshow(np.angle(phase_source), cmap='hsv',
                          vmin=-np.pi, vmax=np.pi)
    axes[1, 1].set_title('Phase Source\n(Phase)', fontsize=12)
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
    
    # Hybrid probe
    im = axes[0, 2].imshow(np.abs(hybrid_probe), cmap='gray')
    axes[0, 2].set_title('Hybrid Probe\n(Amplitude from Source 1)', fontsize=12)
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)
    
    im = axes[1, 2].imshow(np.angle(hybrid_probe), cmap='hsv',
                          vmin=-np.pi, vmax=np.pi)
    axes[1, 2].set_title('Hybrid Probe\n(Phase from Source 2)', fontsize=12)
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)
    
    # Remove axis ticks
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle('Hybrid Probe Creation', fontsize=16)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Visualization saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Create a hybrid probe from amplitude and phase sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'amplitude_source',
        type=str,
        help='Path to probe file (.npy or .npz) to extract amplitude from'
    )
    
    parser.add_argument(
        'phase_source', 
        type=str,
        help='Path to probe file (.npy or .npz) to extract phase from'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='hybrid_probe.npy',
        help='Output path for hybrid probe (default: hybrid_probe.npy)'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Create visualization comparing source and hybrid probes'
    )
    
    parser.add_argument(
        '--normalize',
        action='store_true',
        help='Normalize hybrid probe to preserve total power from amplitude source'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(
        output_dir=Path('.'),
        console_level=getattr(logging, args.log_level)
    )
    
    try:
        # Load probes
        logger.info(f"Loading amplitude source from: {args.amplitude_source}")
        amplitude_probe = load_probe_from_source(args.amplitude_source)
        
        logger.info(f"Loading phase source from: {args.phase_source}")
        phase_probe = load_probe_from_source(args.phase_source)
        
        # Create hybrid probe
        hybrid_probe = create_hybrid_probe(
            amplitude_probe, 
            phase_probe,
            normalize=args.normalize
        )
        
        # Save output
        output_path = Path(args.output)
        np.save(output_path, hybrid_probe)
        logger.info(f"Hybrid probe saved to: {output_path}")
        
        # Log statistics
        logger.info(f"Output shape: {hybrid_probe.shape}")
        logger.info(f"Output dtype: {hybrid_probe.dtype}")
        logger.info(f"Mean amplitude: {np.mean(np.abs(hybrid_probe)):.6f}")
        logger.info(f"Phase std dev: {np.std(np.angle(hybrid_probe)):.6f}")
        
        # Create visualization if requested
        if args.visualize:
            viz_path = output_path.with_suffix('.png')
            visualize_probes(amplitude_probe, phase_probe, hybrid_probe, viz_path)
            
    except Exception as e:
        logger.error(f"Error creating hybrid probe: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()