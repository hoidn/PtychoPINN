#!/usr/bin/env python3
"""
Extract and save ground truth images from dataset NPZ files.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def extract_ground_truth(npz_path, output_dir=None):
    """Extract ground truth from NPZ and save as PNG."""
    npz_path = Path(npz_path)
    
    # Load data
    data = np.load(npz_path)
    
    if 'objectGuess' not in data:
        raise ValueError(f"No objectGuess found in {npz_path}")
    
    obj = data['objectGuess']
    amplitude = np.abs(obj)
    phase = np.angle(obj)
    
    # Determine output directory
    if output_dir is None:
        output_dir = npz_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine dataset name from path
    dataset_name = 'unknown'
    if 'fly64' in str(npz_path).lower():
        dataset_name = 'fly64'
    elif 'run1084' in str(npz_path).lower():
        dataset_name = 'run1084'
    
    # Save amplitude
    amp_path = output_dir / f"{dataset_name}_ground_truth_amplitude.png"
    fig = plt.figure(figsize=(4, 4), dpi=100)
    ax = plt.gca()
    im = ax.imshow(amplitude, cmap='viridis', interpolation='nearest', origin='lower')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(amp_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved amplitude to: {amp_path}")
    
    # Save phase
    phase_path = output_dir / f"{dataset_name}_ground_truth_phase.png"
    fig = plt.figure(figsize=(4, 4), dpi=100)
    ax = plt.gca()
    im = ax.imshow(phase, cmap='gray', interpolation='nearest', origin='lower')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(phase_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved phase to: {phase_path}")
    
    # Also save as NPZ for consistency
    npz_out_path = output_dir / f"{dataset_name}_ground_truth.npz"
    np.savez_compressed(
        npz_out_path,
        ground_truth_complex=obj,
        ground_truth_amplitude=amplitude,
        ground_truth_phase=phase,
        metadata={'source': str(npz_path), 'shape': obj.shape}
    )
    print(f"Saved NPZ to: {npz_out_path}")
    
    return amp_path, phase_path, npz_out_path

def main():
    parser = argparse.ArgumentParser(description="Extract ground truth from dataset NPZ")
    parser.add_argument("npz_path", help="Path to dataset NPZ file")
    parser.add_argument("--output-dir", help="Output directory (default: same as NPZ)")
    
    args = parser.parse_args()
    
    try:
        extract_ground_truth(args.npz_path, args.output_dir)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())