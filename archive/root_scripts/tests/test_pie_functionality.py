#!/usr/bin/env python3
"""
Test script to verify PIE algorithm functionality in pty-chi.
Tests all PIE variants (PIE, ePIE, rPIE, mPIE) to check for tensor dimension issues.
"""

import sys
import os
import numpy as np
import torch
from pathlib import Path
import traceback

# Add pty-chi to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pty-chi', 'src'))

import ptychi.api as api
from ptychi.api.task import PtychographyTask
from ptychi.utils import get_suggested_object_size, get_default_complex_dtype

def create_minimal_test_data():
    """Create minimal synthetic data for testing."""
    print("Creating minimal test data...")
    
    # Create simple test data
    N = 32  # Small probe size for quick testing
    n_images = 10  # Minimal number of patterns
    
    # Create probe (Gaussian)
    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, x)
    probe = np.exp(-(X**2 + Y**2) / 0.1).astype(np.complex128)
    
    # Create object (slightly larger)
    M = 64
    obj = np.ones((M, M), dtype=np.complex128)
    
    # Create scan positions (simple grid) - centered
    positions_x = np.array([-10, -5, 0, 5, 10, -10, -5, 0, 5, 10], dtype=np.float32)
    positions_y = np.array([-2.5, -2.5, -2.5, -2.5, -2.5, 2.5, 2.5, 2.5, 2.5, 2.5], dtype=np.float32)
    
    # Create random diffraction patterns (amplitude)
    diffraction = np.random.rand(n_images, N, N) * 10 + 1
    
    # Stack positions in Y,X format and ensure they're centered
    positions = np.stack([positions_y, positions_x], axis=-1)
    
    return {
        'probe': probe,
        'object': obj,
        'positions': positions,  # Already centered
        'diffraction': diffraction,
        'N': N,
        'n_images': n_images
    }

def test_algorithm(algorithm_name, data, n_epochs=10, batch_size=None):
    """Test a specific algorithm with the given data."""
    print(f"\n{'='*60}")
    print(f"Testing {algorithm_name}...")
    print(f"{'='*60}")
    
    try:
        # Determine object size BEFORE converting to torch
        # get_suggested_object_size expects numpy arrays
        probe_shape = data['probe'].shape  # (H, W)
        obj_size = get_suggested_object_size(
            data['positions'],  # numpy array
            probe_shape,  # tuple (H, W)
            extra=10  # Add some padding
        )
        print(f"Object size: {obj_size}")
        
        # NOW convert to torch tensors
        probe_torch = torch.from_numpy(data['probe']).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        positions_torch = torch.from_numpy(data['positions'])
        diffraction_torch = torch.from_numpy(data['diffraction']**2)  # Convert amplitude to intensity
        
        # Set up reconstruction task
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Configure algorithm-specific options
        if algorithm_name == 'PIE':
            options = api.PIEOptions()
            options.object_options.alpha = 0.1
            options.probe_options.alpha = 0.1
        elif algorithm_name == 'ePIE':
            options = api.EPIEOptions()
            options.object_options.alpha = 0.1
            options.probe_options.alpha = 0.1
        elif algorithm_name == 'rPIE':
            options = api.RPIEOptions()
        elif algorithm_name == 'mPIE':
            # mPIE might not be available in the API
            print("  Note: mPIE not available in API, skipping...")
            return False, "mPIE not available in API"
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
        # Set batch size
        if batch_size is None:
            batch_size = min(32, data['n_images'])
        
        print(f"Batch size: {batch_size}")
        
        # Configure common options
        options.data_options.data = diffraction_torch
        options.data_options.wavelength_m = 1e-9  # 1 nm
        options.data_options.detector_pixel_size_m = 1e-6  # 1 micron
        options.data_options.fft_shift = True
        
        # Configure object
        options.object_options.initial_guess = torch.ones(
            [1, *obj_size], 
            dtype=torch.complex64
        )
        options.object_options.pixel_size_m = 1e-8  # 10 nm
        options.object_options.optimizable = True
        
        # Configure probe  
        options.probe_options.initial_guess = probe_torch
        options.probe_options.optimizable = True
        
        # Configure positions
        options.probe_position_options.position_x_px = positions_torch[:, 1]  # X is column 1
        options.probe_position_options.position_y_px = positions_torch[:, 0]  # Y is column 0
        options.probe_position_options.optimizable = False
        
        # Reconstruction parameters
        options.reconstructor_options.num_epochs = n_epochs
        # PIE algorithms may use batch_size instead of chunk_length
        try:
            options.reconstructor_options.batch_size = batch_size
        except AttributeError:
            # Some algorithms might not support batch_size
            pass
        
        # Device selection
        if device == 'cuda':
            options.reconstructor_options.default_device = api.Devices.GPU
        else:
            options.reconstructor_options.default_device = api.Devices.CPU
        
        # Create task with options
        task = PtychographyTask(options)
        
        # Run reconstruction
        print(f"\nStarting reconstruction with {n_epochs} epochs...")
        task.run()
        
        print(f"\nReconstruction complete!")
        
        # Get reconstructed object
        obj_recon = task.get_data_to_cpu("object", as_numpy=True)[0]  # Remove batch dimension
        print(f"Reconstructed object shape: {obj_recon.shape}")
        print(f"Reconstructed object dtype: {obj_recon.dtype}")
        
        print(f"\n✓ {algorithm_name} completed successfully!")
        return True, None
        
    except Exception as e:
        error_msg = str(e)
        tb = traceback.format_exc()
        print(f"\n✗ {algorithm_name} FAILED with error:")
        print(f"  Error: {error_msg}")
        print(f"\nTraceback:")
        print(tb)
        
        # Check for the specific tensor dimension error
        if "dimension" in error_msg.lower() or "shape" in error_msg.lower():
            print(f"\n  → This appears to be a TENSOR DIMENSION ERROR")
        
        return False, error_msg

def main():
    """Run comprehensive PIE tests."""
    print("="*80)
    print("PIE FUNCTIONALITY TEST")
    print("="*80)
    
    # Create test data
    data = create_minimal_test_data()
    print(f"\nTest data created:")
    print(f"  Probe shape: {data['probe'].shape}")
    print(f"  Object shape: {data['object'].shape}")
    print(f"  Positions shape: {data['positions'].shape}")
    print(f"  Diffraction shape: {data['diffraction'].shape}")
    
    # Test each algorithm variant
    algorithms = ['PIE', 'ePIE', 'rPIE', 'mPIE']
    results = {}
    
    for algo in algorithms:
        success, error = test_algorithm(algo, data, n_epochs=20, batch_size=10)
        results[algo] = {'success': success, 'error': error}
    
    # Also test with different batch sizes for PIE
    print("\n" + "="*80)
    print("Testing PIE with different batch sizes...")
    print("="*80)
    
    for batch_size in [1, 5, 10]:
        print(f"\nTesting PIE with batch_size={batch_size}")
        success, error = test_algorithm('PIE', data, n_epochs=10, batch_size=batch_size)
        results[f'PIE_batch{batch_size}'] = {'success': success, 'error': error}
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for algo, result in results.items():
        status = "✓ PASSED" if result['success'] else "✗ FAILED"
        print(f"{algo:20s}: {status}")
        if not result['success'] and result['error']:
            # Extract key part of error
            error_lines = result['error'].split('\n')
            key_error = error_lines[0] if error_lines else result['error']
            print(f"  └─ Error: {key_error[:100]}...")
    
    # Check if the original bug is present
    failed_algos = [algo for algo, result in results.items() if not result['success']]
    if failed_algos:
        print(f"\n⚠️  {len(failed_algos)} algorithms failed")
        
        # Check if all PIE variants failed (indicates systemic bug)
        pie_variants = ['PIE', 'ePIE', 'rPIE', 'mPIE']
        all_pie_failed = all(algo in failed_algos for algo in pie_variants if algo in results)
        
        if all_pie_failed:
            print("  → ALL PIE variants failed - likely systemic bug in base PIE class")
        else:
            print(f"  → Some PIE variants failed: {', '.join(failed_algos)}")
    else:
        print("\n✅ All algorithms passed - PIE bug appears to be FIXED!")
    
    return len(failed_algos) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)