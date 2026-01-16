#!/usr/bin/env python3
"""
A comprehensive tool to visualize a ptychography NPZ dataset.

This script generates a multi-panel plot showing:
1. The probe's amplitude and phase.
2. The scan point positions on the object.
3. The full ground truth object's amplitude and phase.
4. A RANDOM SAMPLE of 4 diffraction patterns.

It intelligently checks for diffraction data under the keys 'diff3d' or 'diffraction'.
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def visualize_full_dataset(npz_path: str, output_path: str, seed: int = None):
    """Loads and creates a comprehensive visualization of an NPZ dataset."""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Input file not found: {npz_path}")

    print(f"Loading data from: {npz_path}")
    with np.load(npz_path) as data:
        # Load all required arrays, providing defaults for missing ones
        probe = data.get('probeGuess')
        obj = data.get('objectGuess')
        xcoords = data.get('xcoords')
        ycoords = data.get('ycoords')

        # --- KEY CHANGE: Handle flexible diffraction key names ---
        diff_patterns = None
        if 'diff3d' in data:
            diff_patterns = data['diff3d']
            print("Found diffraction data under key: 'diff3d'")
        elif 'diffraction' in data:
            diff_patterns = data['diffraction']
            print("Found diffraction data under key: 'diffraction'")
        # ---------------------------------------------------------

    # Set the random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # --- Create a 2x4 plot for a comprehensive overview ---
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle(f"Visualization of: {os.path.basename(npz_path)}", fontsize=16)

    # --- Top Row: Probe, Object, and Scan Positions ---
    
    # 1. Probe Amplitude
    ax = axes[0, 0]
    if probe is not None:
        im = ax.imshow(np.abs(probe), cmap='viridis')
        ax.set_title(f"Probe Amplitude (Shape: {probe.shape})")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.text(0.5, 0.5, 'No Probe Data', ha='center'); ax.set_title("Probe Amplitude")

    # 2. Probe Phase
    ax = axes[0, 1]
    if probe is not None:
        im = ax.imshow(np.angle(probe), cmap='twilight')
        ax.set_title("Probe Phase")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.text(0.5, 0.5, 'No Probe Data', ha='center'); ax.set_title("Probe Phase")

    # 3. Object Amplitude
    ax = axes[0, 2]
    if obj is not None:
        im = ax.imshow(np.abs(obj), cmap='gray')
        ax.set_title(f"Object Amplitude (Shape: {obj.shape})")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.text(0.5, 0.5, 'No Object Data', ha='center'); ax.set_title("Object Amplitude")

    # 4. Scan Positions on Object
    ax = axes[0, 3]
    if obj is not None and xcoords is not None and ycoords is not None:
        ax.imshow(np.abs(obj), cmap='gray', alpha=0.7)
        ax.scatter(xcoords, ycoords, s=5, c='red', alpha=0.5)
        ax.set_title(f"Scan Positions (n={len(xcoords)})")
        ax.set_aspect('equal'); ax.set_xlim(0, obj.shape[1]); ax.set_ylim(obj.shape[0], 0)
    else:
        ax.text(0.5, 0.5, 'No Object/Scan Data', ha='center'); ax.set_title("Scan Positions")

    # --- Bottom Row: Random Sample of Diffraction Patterns ---
    
    if diff_patterns is not None:
        num_patterns = diff_patterns.shape[0]
        # Select 4 random indices, or fewer if not enough patterns are available
        sample_size = min(4, num_patterns)
        random_indices = np.random.choice(num_patterns, size=sample_size, replace=False)
        
        for i, ax_idx in enumerate(range(4)):
            ax = axes[1, ax_idx]
            if i < len(random_indices):
                pattern_index = random_indices[i]
                im = ax.imshow(np.log1p(diff_patterns[pattern_index]), cmap='jet')
                ax.set_title(f"Random Diffraction [{pattern_index}]")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                # Hide unused axes if there are fewer than 4 patterns
                ax.axis('off')
    else:
        for ax_idx in range(4):
            ax = axes[1, ax_idx]
            axes[1, ax_idx].text(0.5, 0.5, 'No Diffraction Data', ha='center')
            axes[1, ax_idx].set_title("Diffraction Pattern")

    # --- Final Touches ---
    for ax in axes.flat:
        if not ax.get_title(): # Hide axes that were not used
             ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Visualization saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize a ptychography NPZ dataset with random sampling.")
    parser.add_argument("input_npz", help="Path to the source .npz file.")
    parser.add_argument("output_png", help="Path to save the output visualization PNG file.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for selecting patterns, for reproducibility.")
    args = parser.parse_args()

    try:
        visualize_full_dataset(args.input_npz, args.output_png, args.seed)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
