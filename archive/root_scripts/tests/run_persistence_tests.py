#!/usr/bin/env python
"""Run persistence tests in subprocess to avoid TF initialization conflicts."""

import subprocess
import sys
from pathlib import Path

def run_test_subprocess():
    """Run the test in a subprocess to ensure clean TF initialization."""
    
    test_script = """
import os
import sys
import tempfile
from pathlib import Path

# Set environment before ANY imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# NOW import everything
from ptycho import params as p
from ptycho.probe import get_default_probe
import tensorflow as tf
import numpy as np

print("\\n" + "=" * 60)
print("Model Persistence Test Suite")
print("=" * 60)

# Initialize parameters
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

# Import model modules
from ptycho.model_manager import ModelManager
from ptycho.model import create_model_with_gridsize

with tempfile.TemporaryDirectory() as temp_dir:
    print(f"\\nUsing temp directory: {temp_dir}")
    
    # Test 1: Basic save/load
    print("\\n--- Test 1: Basic Save/Load ---")
    try:
        model, _ = create_model_with_gridsize(gridsize=1, N=64)
        model_path = Path(temp_dir) / "test_model"
        ModelManager.save_model(model, str(model_path), {}, 1.0)
        loaded = ModelManager.load_model(str(model_path))
        print("✅ Basic save/load works")
    except Exception as e:
        print(f"❌ Failed: {e}")
        sys.exit(1)
    
    # Test 2: Parameter restoration
    print("\\n--- Test 2: Parameter Restoration ---")
    try:
        # Save with specific params
        p.set('nphotons', 1e7)
        model, _ = create_model_with_gridsize(gridsize=1, N=64)
        model_path = Path(temp_dir) / "param_model"
        ModelManager.save_model(model, str(model_path), {}, 2.5)
        
        # Change params
        p.set('nphotons', 1e9)
        
        # Load and check restoration
        loaded = ModelManager.load_model(str(model_path))
        assert p.get('nphotons') == 1e7, f"nphotons not restored"
        print("✅ Parameters restored correctly")
    except Exception as e:
        print(f"❌ Failed: {e}")
        sys.exit(1)

print("\\n" + "=" * 60)
print("✅ All persistence tests passed!")
print("=" * 60)
"""

    # Write test script to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_script)
        temp_script = f.name
    
    try:
        # Run in subprocess
        result = subprocess.run(
            [sys.executable, temp_script],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent)
        )
        
        print(result.stdout)
        if result.stderr:
            print("Warnings/Info:", result.stderr[:500])
        
        return result.returncode
        
    finally:
        # Clean up temp script
        Path(temp_script).unlink(missing_ok=True)

if __name__ == "__main__":
    sys.exit(run_test_subprocess())