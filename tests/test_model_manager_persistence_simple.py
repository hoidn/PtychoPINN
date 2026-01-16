#!/usr/bin/env python
"""Simplified model persistence test that can be run directly."""

import os
import sys
import tempfile
from pathlib import Path

# Set up environment before any TF imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

def test_model_persistence():
    """Test basic model save/load functionality."""
    print("Starting model persistence test...")
    
    # Import after environment setup
    from ptycho import params as p
    from ptycho.probe import get_default_probe
    
    # Initialize required parameters
    p.set('N', 64)
    p.set('gridsize', 1)
    p.set('probe.type', 'gaussian')
    p.set('probe.photons', 1e10)
    p.set('nphotons', 1e8)
    
    # Set probe
    probe = get_default_probe(64)
    p.params()['probe'] = probe
    
    # Now import model-related modules
    from ptycho.model_manager import ModelManager
    from ptycho.model import create_model_with_gridsize
    import tensorflow as tf
    import numpy as np
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temp directory: {temp_dir}")
        
        # Test 1: Parameter restoration
        print("\n--- Test 1: Parameter Restoration ---")
        # Set specific parameters
        p.set('N', 128)
        p.set('gridsize', 2) 
        p.set('nphotons', 1e8)
        
        # Create and save model
        model, _ = create_model_with_gridsize(gridsize=2, N=128)
        model_path = Path(temp_dir) / "test_model"
        ModelManager.save_model(model, str(model_path), {}, 1.0)
        print(f"Model saved with N=128, gridsize=2")
        
        # Change parameters
        p.set('N', 64)
        p.set('gridsize', 1)
        p.set('nphotons', 1e9)
        print(f"Changed params to N=64, gridsize=1")
        
        # Load model
        _ = ModelManager.load_model(str(model_path))
        
        # Check if parameters were restored
        assert p.get('N') == 128, f"N not restored: got {p.get('N')}, expected 128"
        assert p.get('gridsize') == 2, f"gridsize not restored: got {p.get('gridsize')}, expected 2"
        assert p.get('nphotons') == 1e8, f"nphotons not restored: got {p.get('nphotons')}, expected 1e8"
        print("✅ Parameter restoration test passed!")
        
        # Test 2: Architecture consistency
        print("\n--- Test 2: Architecture Consistency ---")
        # Reset and create a gridsize=1 model
        p.set('gridsize', 1)
        p.set('N', 64)
        _, inference_model = create_model_with_gridsize(gridsize=1, N=64)
        
        # Generate test input
        dummy_diffraction = tf.random.normal((2, 64, 64, 1))
        dummy_positions = tf.zeros((2, 1, 2, 1))
        
        # Get original output
        original_output = inference_model.predict([dummy_diffraction, dummy_positions])
        
        # Save model
        model_path2 = Path(temp_dir) / "inference_model"
        ModelManager.save_model(inference_model, str(model_path2), {}, 1.0)
        
        # Change gridsize
        p.set('gridsize', 2)
        
        # Load model (should restore gridsize=1)
        loaded_model = ModelManager.load_model(str(model_path2))
        
        # Get loaded output
        loaded_output = loaded_model.predict([dummy_diffraction, dummy_positions])
        
        # Check outputs match
        np.testing.assert_allclose(original_output, loaded_output, rtol=1e-6)
        print("✅ Architecture consistency test passed!")
        
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_model_persistence()