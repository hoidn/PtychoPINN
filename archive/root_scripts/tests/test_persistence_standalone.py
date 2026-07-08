#!/usr/bin/env python
"""Standalone model persistence test that properly handles TF initialization."""

import os
import sys
import tempfile
from pathlib import Path

# CRITICAL: Set environment before ANY imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

def main():
    """Run the model persistence tests."""
    print("=" * 60)
    print("Model Persistence Test Suite")
    print("=" * 60)
    
    # Import after environment setup
    from ptycho import params as p
    from ptycho.probe import get_default_probe
    import tensorflow as tf
    import numpy as np
    
    # Initialize required parameters
    print("\nInitializing parameters...")
    p.set('N', 64)
    p.set('gridsize', 1)
    p.set('probe.type', 'gaussian')
    p.set('probe.photons', 1e10)
    p.set('nphotons', 1e8)
    p.set('n_filters_scale', 2)
    p.set('offset', 0)
    p.set('gaussian_smoothing_sigma', 0.0)
    p.set('probe.trainable', False)
    p.set('probe.mask', False)
    p.set('intensity_scale', 1.0)
    
    # Set probe
    probe = get_default_probe(64)
    p.params()['probe'] = probe
    
    # Now import model-related modules (after params are set)
    from ptycho.model_manager import ModelManager
    from ptycho.model import create_model_with_gridsize
    
    success_count = 0
    total_tests = 3
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temp directory: {temp_dir}")
        
        # Test 1: Parameter Persistence
        print("\n" + "=" * 40)
        print("Test 1: Parameter Persistence")
        print("=" * 40)
        try:
            # Set specific parameters
            p.set('N', 128)
            p.set('gridsize', 2) 
            p.set('nphotons', 1e7)
            
            # Update probe for new N
            probe = get_default_probe(128)
            p.params()['probe'] = probe
            
            # Create and save model
            print("Creating model with N=128, gridsize=2...")
            model, _ = create_model_with_gridsize(gridsize=2, N=128)
            model_path = Path(temp_dir) / "param_test_model"
            
            print(f"Saving model to {model_path}...")
            ModelManager.save_model(model, str(model_path), {}, 1.0)
            
            # Change parameters
            p.set('N', 64)
            p.set('gridsize', 1)
            p.set('nphotons', 1e9)
            print("Changed params to N=64, gridsize=1")
            
            # Load model
            print("Loading model...")
            loaded_model = ModelManager.load_model(str(model_path))
            
            # Check if parameters were restored
            assert p.get('N') == 128, f"N not restored: got {p.get('N')}, expected 128"
            assert p.get('gridsize') == 2, f"gridsize not restored: got {p.get('gridsize')}, expected 2"
            assert p.get('nphotons') == 1e7, f"nphotons not restored: got {p.get('nphotons')}, expected 1e7"
            
            print("✅ PASSED: Parameters correctly restored!")
            success_count += 1
            
        except Exception as e:
            print(f"❌ FAILED: {str(e)}")
        
        # Test 2: Architecture Consistency
        print("\n" + "=" * 40)
        print("Test 2: Architecture Consistency")
        print("=" * 40)
        try:
            # Reset parameters
            p.set('gridsize', 2)
            p.set('N', 64)
            probe = get_default_probe(64)
            p.params()['probe'] = probe
            
            print("Creating gridsize=2 model...")
            model_gs2, _ = create_model_with_gridsize(gridsize=2, N=64)
            
            model_path2 = Path(temp_dir) / "arch_test_model"
            print(f"Saving model to {model_path2}...")
            ModelManager.save_model(model_gs2, str(model_path2), {}, 1.0)
            
            # Change gridsize
            p.set('gridsize', 1)
            print("Changed gridsize to 1")
            
            # Load model (should restore gridsize=2)
            print("Loading model...")
            loaded_model = ModelManager.load_model(str(model_path2))
            
            # Check architecture
            input_channels = loaded_model.input_shape[0][-1]
            expected_channels = 4  # gridsize=2 means 2x2=4 channels
            assert input_channels == expected_channels, \
                f"Wrong input channels: got {input_channels}, expected {expected_channels}"
            
            print("✅ PASSED: Architecture correctly restored!")
            success_count += 1
            
        except Exception as e:
            print(f"❌ FAILED: {str(e)}")
        
        # Test 3: Inference Consistency
        print("\n" + "=" * 40)
        print("Test 3: Inference Consistency")
        print("=" * 40)
        try:
            # Reset and create a gridsize=1 model
            p.set('gridsize', 1)
            p.set('N', 64)
            probe = get_default_probe(64)
            p.params()['probe'] = probe
            
            print("Creating inference model...")
            _, inference_model = create_model_with_gridsize(gridsize=1, N=64)
            
            # Generate test input
            print("Generating test inputs...")
            dummy_diffraction = tf.random.normal((2, 64, 64, 1))
            dummy_positions = tf.zeros((2, 1, 2, 1))
            
            # Get original output
            print("Getting original model output...")
            original_output = inference_model.predict([dummy_diffraction, dummy_positions], verbose=0)
            
            # Save model
            model_path3 = Path(temp_dir) / "inference_model"
            print(f"Saving model to {model_path3}...")
            ModelManager.save_model(inference_model, str(model_path3), {}, 1.0)
            
            # Load model
            print("Loading model...")
            loaded_model = ModelManager.load_model(str(model_path3))
            
            # Get loaded output
            print("Getting loaded model output...")
            loaded_output = loaded_model.predict([dummy_diffraction, dummy_positions], verbose=0)
            
            # Check outputs match
            np.testing.assert_allclose(original_output, loaded_output, rtol=1e-5)
            
            print("✅ PASSED: Inference outputs match!")
            success_count += 1
            
        except Exception as e:
            print(f"❌ FAILED: {str(e)}")
    
    # Final summary
    print("\n" + "=" * 60)
    print(f"Test Results: {success_count}/{total_tests} tests passed")
    if success_count == total_tests:
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"⚠️  {total_tests - success_count} test(s) failed")
    print("=" * 60)
    
    return 0 if success_count == total_tests else 1

if __name__ == "__main__":
    sys.exit(main())