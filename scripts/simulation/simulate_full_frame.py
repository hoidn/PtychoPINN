#!/usr/bin/env python3
"""
Full-frame ptychography simulation tool that ensures complete object coverage.

This script generates diffraction patterns with scan positions that cover the
entire object in a regular grid pattern, ensuring ground truth is fully visible
in comparison plots.
"""

import numpy as np
import argparse
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ptycho import params
from ptycho.raw_data import RawData
from ptycho.probe import get_default_probe
from ptycho.diffsim import sim_object_image


def generate_full_frame_positions(object_shape, probe_shape, n_positions, overlap=None):
    """
    Generate scan positions that fully cover the object in a regular grid.
    
    Args:
        object_shape: Tuple of (height, width) for the object
        probe_shape: Tuple of (height, width) for the probe
        n_positions: Number of scan positions desired
        overlap: Optional overlap fraction. If None, calculated to achieve n_positions
    
    Returns:
        xcoords, ycoords: Arrays of scan positions in pixels
    """
    obj_h, obj_w = object_shape
    probe_h, probe_w = probe_shape
    
    # Calculate the scan area (accounting for probe size)
    scan_w = obj_w - probe_w
    scan_h = obj_h - probe_h
    
    if scan_w <= 0 or scan_h <= 0:
        raise ValueError(f"Object ({obj_w}x{obj_h}) must be larger than probe ({probe_w}x{probe_h})")
    
    # If overlap not specified, calculate it to achieve desired n_positions
    if overlap is None:
        # Estimate grid dimensions needed for n_positions
        # n_positions ≈ n_x * n_y
        aspect_ratio = scan_w / scan_h
        n_y = int(np.sqrt(n_positions / aspect_ratio))
        n_x = int(n_positions / n_y)
        
        # Ensure minimum grid size
        n_x = max(2, n_x)
        n_y = max(2, n_y)
        
        # Calculate step sizes
        step_x = scan_w / (n_x - 1) if n_x > 1 else 0
        step_y = scan_h / (n_y - 1) if n_y > 1 else 0
        
        # Calculate effective overlap
        overlap_x = 1 - (step_x / probe_w) if probe_w > 0 else 0
        overlap_y = 1 - (step_y / probe_h) if probe_h > 0 else 0
        overlap = max(0, min(overlap_x, overlap_y))
        
        print(f"Calculated overlap: {overlap:.2f} to achieve ~{n_positions} positions")
    else:
        # Use specified overlap
        step_x = int(probe_w * (1 - overlap))
        step_y = int(probe_h * (1 - overlap))
        step_x = max(1, step_x)
        step_y = max(1, step_y)
        
        n_x = int(scan_w / step_x) + 1
        n_y = int(scan_h / step_y) + 1
    
    # Generate regular grid
    x_positions = []
    y_positions = []
    
    for iy in range(n_y):
        for ix in range(n_x):
            # Position in scan area
            x = ix * step_x if n_x > 1 else scan_w / 2
            y = iy * step_y if n_y > 1 else scan_h / 2
            
            # Offset to object coordinates (centered)
            x_centered = x + probe_w/2 - obj_w/2
            y_centered = y + probe_h/2 - obj_h/2
            
            x_positions.append(x_centered)
            y_positions.append(y_centered)
    
    # Convert to arrays
    xcoords = np.array(x_positions, dtype=np.float64)
    ycoords = np.array(y_positions, dtype=np.float64)
    
    actual_positions = len(xcoords)
    print(f"Generated {actual_positions} scan positions ({n_x}x{n_y} grid)")
    print(f"Actual overlap: {overlap:.2f}")
    print(f"Coverage: X=[{xcoords.min():.1f}, {xcoords.max():.1f}], Y=[{ycoords.min():.1f}, {ycoords.max():.1f}]")
    
    # If we have too few positions, duplicate some randomly
    if actual_positions < n_positions:
        n_extra = n_positions - actual_positions
        extra_indices = np.random.choice(actual_positions, n_extra, replace=True)
        xcoords = np.concatenate([xcoords, xcoords[extra_indices]])
        ycoords = np.concatenate([ycoords, ycoords[extra_indices]])
        print(f"Duplicated {n_extra} positions to reach {n_positions} total")
    
    # If we have too many, subsample
    elif actual_positions > n_positions:
        indices = np.random.choice(actual_positions, n_positions, replace=False)
        xcoords = xcoords[indices]
        ycoords = ycoords[indices]
        print(f"Subsampled to {n_positions} positions")
    
    return xcoords, ycoords


def create_synthetic_object(size, object_type='lines', seed=None):
    """Create a synthetic object for simulation."""
    if seed is not None:
        np.random.seed(seed)
    
    # Set params for object generation
    params.cfg['data_source'] = object_type
    params.cfg['size'] = size
    
    # Generate object
    obj = sim_object_image(size)
    
    # Remove channel dimension if present
    if obj.ndim == 3 and obj.shape[-1] == 1:
        obj = obj[..., 0]
    
    # Ensure complex type
    if not np.iscomplexobj(obj):
        obj = obj.astype(np.complex64)
    
    return obj


def main():
    parser = argparse.ArgumentParser(
        description="Simulate ptychography data with full-frame coverage",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to save the simulated data as .npz"
    )
    
    parser.add_argument(
        "--n-images",
        type=int,
        default=1000,
        help="Number of diffraction patterns to simulate"
    )
    
    parser.add_argument(
        "--probe-size",
        type=int,
        default=64,
        help="Size of the probe (NxN pixels)"
    )
    
    parser.add_argument(
        "--object-size",
        type=int,
        default=256,
        help="Size of the object (MxM pixels)"
    )
    
    parser.add_argument(
        "--object-type",
        type=str,
        default='lines',
        choices=['lines', 'grf', 'logo'],
        help="Type of synthetic object to generate"
    )
    
    parser.add_argument(
        "--overlap",
        type=float,
        default=None,
        help="Overlap fraction between adjacent probes (0-1). If not specified, calculated to achieve n-images"
    )
    
    parser.add_argument(
        "--probe-file",
        type=str,
        help="Optional path to external probe file (.npy or .npz)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible object generation"
    )
    
    parser.add_argument(
        "--nphotons",
        type=float,
        default=1e9,
        help="Number of photons for Poisson noise simulation"
    )
    
    parser.add_argument(
        "--gridsize",
        type=int,
        default=1,
        help="Grid size parameter for simulation"
    )
    
    args = parser.parse_args()
    
    # Set global parameters
    params.cfg['N'] = args.probe_size
    params.cfg['gridsize'] = args.gridsize
    params.cfg['nphotons'] = args.nphotons
    
    print(f"Generating full-frame simulation:")
    print(f"  Object: {args.object_size}x{args.object_size} ({args.object_type})")
    print(f"  Probe: {args.probe_size}x{args.probe_size}")
    print(f"  Target positions: {args.n_images}")
    
    # Generate or load probe
    if args.probe_file:
        print(f"Loading probe from: {args.probe_file}")
        if args.probe_file.endswith('.npy'):
            probe = np.load(args.probe_file)
        else:
            data = np.load(args.probe_file)
            probe = data['probeGuess'] if 'probeGuess' in data else data['probe']
    else:
        print("Generating default probe")
        params.cfg['default_probe_scale'] = 0.7
        probe = get_default_probe(args.probe_size, fmt='np')
        probe = probe.astype(np.complex64)
    
    # Generate synthetic object
    print(f"Generating {args.object_type} object with seed={args.seed}")
    synthetic_object = create_synthetic_object(args.object_size, args.object_type, args.seed)
    
    # Generate full-frame scan positions
    # Let the function calculate overlap to achieve the desired number of positions
    xcoords, ycoords = generate_full_frame_positions(
        object_shape=synthetic_object.shape,
        probe_shape=probe.shape,
        n_positions=args.n_images,
        overlap=args.overlap if args.overlap is not None else None
    )
    
    # Create coordinate arrays
    n_actual = len(xcoords)
    scan_index = np.zeros(n_actual, dtype=int)
    
    # Create RawData instance using simulation
    print(f"\nSimulating {n_actual} diffraction patterns...")
    
    # For gridsize > 1, we need to handle the data differently
    if args.gridsize > 1:
        # For gridsize > 1, create empty RawData and then simulate
        # This is a workaround for the channel dimension issue
        print(f"Note: Gridsize {args.gridsize} simulation needs special handling")
        
        # Create RawData with appropriate structure
        diff3d = np.zeros((n_actual, args.probe_size, args.probe_size), dtype=np.float32)
        xcoords_start = xcoords.copy()
        ycoords_start = ycoords.copy()
        
        raw_data = RawData(
            xcoords=xcoords,
            ycoords=ycoords,
            xcoords_start=xcoords_start,
            ycoords_start=ycoords_start,
            diff3d=diff3d,
            probeGuess=probe,
            scan_index=scan_index,
            objectGuess=synthetic_object
        )
        
        # Simulate using the object and probe
        # For now, we'll use gridsize=1 simulation and note this limitation
        print("WARNING: Using gridsize=1 simulation for gridsize>1 data generation")
        params.cfg['gridsize'] = 1  # Temporarily set to 1 for simulation
        simulated_data = RawData.from_simulation(
            xcoords=xcoords,
            ycoords=ycoords,
            probeGuess=probe,
            objectGuess=synthetic_object,
            scan_index=scan_index
        )
        raw_data.diff3d = simulated_data.diff3d
        params.cfg['gridsize'] = args.gridsize  # Restore gridsize
    else:
        # Use RawData.from_simulation to generate the data
        raw_data = RawData.from_simulation(
            xcoords=xcoords,
            ycoords=ycoords,
            probeGuess=probe,
            objectGuess=synthetic_object,
            scan_index=scan_index
        )
    
    # Extract simulated diffraction patterns (RawData stores as diff3d)
    simulated_diff = raw_data.diff3d
    
    # Save output with RawData-compatible format
    # Note: RawData expects 'diff3d' not 'diffraction'
    output_data = {
        'xcoords': raw_data.xcoords,
        'ycoords': raw_data.ycoords,
        'xcoords_start': raw_data.xcoords_start,
        'ycoords_start': raw_data.ycoords_start,
        'diff3d': simulated_diff,  # Use RawData-expected key name
        'probeGuess': raw_data.probeGuess,
        'objectGuess': raw_data.objectGuess,
        'scan_index': raw_data.scan_index,
    }
    
    # Save the data
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **output_data)
    
    print(f"\nSimulation complete!")
    print(f"Saved to: {output_path}")
    print(f"Data summary:")
    print(f"  - diff3d: {simulated_diff.shape}")
    print(f"  - objectGuess: {synthetic_object.shape}")
    print(f"  - probeGuess: {probe.shape}")
    print(f"  - scan positions: {n_actual}")
    
    # Verify coverage
    x_range = xcoords.max() - xcoords.min()
    y_range = ycoords.max() - ycoords.min()
    expected_x_range = args.object_size - args.probe_size
    expected_y_range = args.object_size - args.probe_size
    
    coverage_x = (x_range / expected_x_range) * 100
    coverage_y = (y_range / expected_y_range) * 100
    
    print(f"\nCoverage verification:")
    print(f"  X coverage: {coverage_x:.1f}%")
    print(f"  Y coverage: {coverage_y:.1f}%")
    
    if coverage_x < 90 or coverage_y < 90:
        print("WARNING: Object coverage is less than 90%! Consider increasing n_images.")


if __name__ == "__main__":
    main()