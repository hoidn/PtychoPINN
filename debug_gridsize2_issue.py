#!/usr/bin/env python
"""
Visual debugging script for the gridsize=2 synthetic data generation issue.

This script helps diagnose the coordinate grouping and patch extraction problems
that occur when using gridsize=2 in the simulation pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ptycho.config.config import TrainingConfig, ModelConfig
from ptycho.raw_data import RawData, get_relative_coords, get_image_patches
from scipy.spatial import cKDTree
import tensorflow as tf

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def visualize_coordinate_grouping(xcoords, ycoords, groups, gridsize):
    """Visualize how coordinates are grouped for gridsize > 1."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: All coordinates
    ax = axes[0]
    ax.scatter(xcoords, ycoords, s=10, alpha=0.5, c='blue')
    ax.set_title(f"All {len(xcoords)} Scan Positions")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Selected groups
    ax = axes[1]
    colors = plt.cm.rainbow(np.linspace(0, 1, min(20, len(groups))))
    for i, group in enumerate(groups[:20]):  # Show first 20 groups
        group_x = xcoords[group]
        group_y = ycoords[group]
        ax.scatter(group_x, group_y, s=30, c=[colors[i % len(colors)]], 
                  alpha=0.7, label=f"Group {i}" if i < 5 else "")
    ax.set_title(f"First {min(20, len(groups))} Groups (gridsize={gridsize})")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    if len(groups) <= 5:
        ax.legend()
    
    # Plot 3: Group connectivity
    ax = axes[2]
    ax.scatter(xcoords, ycoords, s=5, alpha=0.2, c='gray')
    for i, group in enumerate(groups[:10]):  # Show connectivity for first 10 groups
        center_idx = group[0]
        center_x = xcoords[center_idx]
        center_y = ycoords[center_idx]
        for neighbor_idx in group[1:]:
            neighbor_x = xcoords[neighbor_idx]
            neighbor_y = ycoords[neighbor_idx]
            ax.plot([center_x, neighbor_x], [center_y, neighbor_y], 
                   'r-', alpha=0.3, linewidth=0.5)
        ax.scatter(center_x, center_y, s=50, c='red', marker='*', zorder=5)
    ax.set_title("Group Connectivity (First 10 Groups)")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def test_coordinate_grouping(n_images=100, gridsize=2, object_size=224, probe_size=64):
    """Test the coordinate grouping logic with synthetic data."""
    
    logger.info(f"Testing with n_images={n_images}, gridsize={gridsize}")
    logger.info(f"Object size: {object_size}x{object_size}, Probe size: {probe_size}x{probe_size}")
    
    # Generate random coordinates
    buffer = probe_size // 2
    np.random.seed(42)
    xcoords = np.random.uniform(buffer, object_size - buffer, n_images)
    ycoords = np.random.uniform(buffer, object_size - buffer, n_images)
    
    logger.info(f"Generated {n_images} random scan positions")
    logger.info(f"X range: [{xcoords.min():.2f}, {xcoords.max():.2f}]")
    logger.info(f"Y range: [{ycoords.min():.2f}, {ycoords.max():.2f}]")
    
    # Method 1: Direct grouping (as done in simulate_and_save.py for gridsize>1)
    if gridsize > 1:
        C = gridsize ** 2
        dummy_diff = np.zeros((n_images, probe_size, probe_size))
        dummy_probe = np.ones((probe_size, probe_size), dtype=np.complex64)
        
        temp_raw = RawData(
            xcoords=xcoords,
            ycoords=ycoords,
            xcoords_start=xcoords,
            ycoords_start=ycoords,
            diff3d=dummy_diff,
            probeGuess=dummy_probe,
            scan_index=np.zeros(n_images, dtype=int)
        )
        
        # This is what simulate_and_save.py does
        config = TrainingConfig(
            model=ModelConfig(N=probe_size, gridsize=gridsize),
            n_images=n_images
        )
        
        grouped_data = temp_raw.generate_grouped_data(
            N=probe_size,
            K=C,
            nsamples=n_images,
            config=config
        )
        
        coords_nn = grouped_data['coords_nn']
        nn_indices = grouped_data['nn_indices']
        
        logger.info(f"Method 1 (generate_grouped_data) results:")
        logger.info(f"  coords_nn shape: {coords_nn.shape}")
        logger.info(f"  nn_indices shape: {nn_indices.shape}")
        logger.info(f"  Number of groups: {nn_indices.shape[0]}")
        
        # Visualize the grouping
        fig1 = visualize_coordinate_grouping(xcoords, ycoords, nn_indices, gridsize)
        fig1.suptitle("Method 1: generate_grouped_data (Data Preparation Function)", fontsize=14)
        
    # Method 2: Simple gridsize=1 approach
    scan_offsets_simple = np.stack([ycoords, xcoords], axis=1)
    logger.info(f"\nMethod 2 (simple gridsize=1) results:")
    logger.info(f"  scan_offsets shape: {scan_offsets_simple.shape}")
    
    # Compare coordinate extraction
    if gridsize > 1:
        global_offsets, local_offsets = get_relative_coords(coords_nn)
        scan_offsets = global_offsets if isinstance(global_offsets, np.ndarray) else global_offsets.numpy()
        
        logger.info(f"\nCoordinate extraction comparison:")
        logger.info(f"  Global offsets shape: {scan_offsets.shape}")
        logger.info(f"  Local offsets shape: {local_offsets.shape if local_offsets is not None else 'None'}")
        
        # Check for sample explosion
        expected_samples = n_images
        actual_samples = nn_indices.shape[0] * C
        logger.info(f"\nSample count analysis:")
        logger.info(f"  Expected samples (n_images): {expected_samples}")
        logger.info(f"  Actual samples (groups * C): {actual_samples}")
        logger.info(f"  Sample explosion factor: {actual_samples / expected_samples:.2f}x")
        
        if actual_samples > expected_samples * 2:
            logger.warning(f"⚠️ SAMPLE EXPLOSION DETECTED: {actual_samples} samples from {expected_samples} images!")
    
    plt.show()
    return grouped_data if gridsize > 1 else None

def test_patch_extraction(grouped_data, object_size=224, probe_size=64, gridsize=2):
    """Test patch extraction with grouped coordinates."""
    
    logger.info(f"\nTesting patch extraction...")
    
    # Create a synthetic object with visible pattern
    synthetic_object = np.zeros((object_size, object_size), dtype=np.complex64)
    # Add a checkerboard pattern for visibility
    for i in range(0, object_size, 20):
        for j in range(0, object_size, 20):
            if (i // 20 + j // 20) % 2 == 0:
                synthetic_object[i:i+20, j:j+20] = 1.0 + 0.5j
    
    # Extract coordinate information from grouped data
    coords_nn = grouped_data['coords_nn']
    global_offsets, local_offsets = get_relative_coords(coords_nn)
    
    logger.info(f"Attempting patch extraction:")
    logger.info(f"  Object shape: {synthetic_object.shape}")
    logger.info(f"  Global offsets shape: {global_offsets.shape}")
    logger.info(f"  Local offsets shape: {local_offsets.shape if local_offsets is not None else 'None'}")
    
    try:
        # This is what simulate_and_save.py tries to do
        Y_patches = get_image_patches(
            synthetic_object,
            global_offsets,
            local_offsets,
            N=probe_size,
            gridsize=gridsize
        )
        
        logger.info(f"✅ Patch extraction successful!")
        logger.info(f"  Y_patches shape: {Y_patches.shape}")
        
        # Visualize some patches
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        fig.suptitle(f"Extracted Patches (gridsize={gridsize})")
        
        Y_patches_np = Y_patches.numpy() if hasattr(Y_patches, 'numpy') else Y_patches
        for i, ax in enumerate(axes.flat):
            if i < min(8, Y_patches_np.shape[0]):
                ax.imshow(np.abs(Y_patches_np[i, :, :, 0] if Y_patches_np.ndim > 3 else Y_patches_np[i]))
                ax.set_title(f"Patch {i}")
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        logger.error(f"❌ Patch extraction failed: {e}")
        logger.error(f"   This is the error occurring in simulate_and_save.py!")
        import traceback
        traceback.print_exc()

def main():
    """Run the diagnostic tests."""
    
    print("=" * 70)
    print("GRIDSIZE=2 SYNTHETIC DATA GENERATION DIAGNOSTIC")
    print("=" * 70)
    
    # Test with different configurations
    configs = [
        {"n_images": 50, "gridsize": 1},  # This should work
        {"n_images": 50, "gridsize": 2},  # This has issues
    ]
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Testing configuration: {config}")
        print(f"{'='*60}")
        
        grouped_data = test_coordinate_grouping(**config)
        
        if grouped_data is not None and config["gridsize"] > 1:
            test_patch_extraction(grouped_data, gridsize=config["gridsize"])
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)
    print("\nKey findings:")
    print("1. generate_grouped_data() is a DATA PREPARATION function, not SIMULATION")
    print("2. It assumes existing diffraction patterns and groups them")
    print("3. For simulation, we need to generate coordinates THEN simulate diffraction")
    print("4. The coordinate format mismatch causes patch extraction failures")

if __name__ == "__main__":
    main()