#!/usr/bin/env python3
"""
create_idealized_probe.py - Create an idealized probe with flat phase

This script creates a probe with:
- Amplitude from an experimental probe (realistic beam profile)
- Flat phase (all zeros - no aberrations)
"""

import argparse
import numpy as np
from pathlib import Path
import logging


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
        description="Create an idealized probe with experimental amplitude and flat phase"
    )
    parser.add_argument(
        "input_probe",
        type=Path,
        help="Input probe file (.npy or .npz) to extract amplitude from"
    )
    parser.add_argument(
        "output_file",
        type=Path,
        help="Output file path for the idealized probe (.npy)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Display visualization of the probe"
    )
    return parser.parse_args()


def load_probe(probe_path: Path) -> np.ndarray:
    """Load probe from file."""
    if probe_path.suffix == '.npz':
        data = np.load(probe_path)
        # Try common keys
        for key in ['probeGuess', 'probe', 'probe_guess']:
            if key in data:
                return data[key]
        raise ValueError(f"No probe found in {probe_path}. Available keys: {list(data.keys())}")
    else:
        return np.load(probe_path)


def create_idealized_probe(experimental_probe: np.ndarray) -> np.ndarray:
    """Create idealized probe with experimental amplitude and zero phase."""
    # Extract amplitude from experimental probe
    amplitude = np.abs(experimental_probe)
    
    # Create probe with flat phase (all zeros)
    idealized_probe = amplitude.astype(np.complex64)
    
    return idealized_probe


def visualize_probe(probe: np.ndarray, title: str = "Probe"):
    """Visualize probe amplitude and phase."""
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Amplitude
    im1 = ax1.imshow(np.abs(probe), cmap='hot')
    ax1.set_title(f'{title} - Amplitude')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1)
    
    # Phase
    im2 = ax2.imshow(np.angle(probe), cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax2.set_title(f'{title} - Phase')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.show()


def main():
    logger = setup_logging()
    args = parse_arguments()
    
    # Load experimental probe
    logger.info(f"Loading experimental probe from: {args.input_probe}")
    experimental_probe = load_probe(args.input_probe)
    logger.info(f"Loaded probe with shape {experimental_probe.shape}, dtype {experimental_probe.dtype}")
    
    # Create idealized probe
    logger.info("Creating idealized probe with flat phase...")
    idealized_probe = create_idealized_probe(experimental_probe)
    
    # Save the idealized probe
    np.save(args.output_file, idealized_probe)
    logger.info(f"Idealized probe saved to: {args.output_file}")
    
    # Print statistics
    logger.info(f"Output shape: {idealized_probe.shape}")
    logger.info(f"Output dtype: {idealized_probe.dtype}")
    logger.info(f"Mean amplitude: {np.abs(idealized_probe).mean():.6f}")
    logger.info(f"Phase std dev: {np.angle(idealized_probe).std():.6f}")
    logger.info(f"Phase range: [{np.angle(idealized_probe).min():.6f}, {np.angle(idealized_probe).max():.6f}]")
    
    # Visualize if requested
    if args.visualize:
        visualize_probe(idealized_probe, "Idealized Probe")


if __name__ == "__main__":
    main()