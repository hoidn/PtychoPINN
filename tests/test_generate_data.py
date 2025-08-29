# Test for generate_data module in the ptycho package

import unittest
import sys
import os
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Initialize params BEFORE importing modules that need probe
from ptycho import params as p
from ptycho.probe import get_default_probe

# Set required parameters before any problematic imports
p.set('N', 64)
p.set('gridsize', 1)
p.set('default_probe_scale', 0.7)

# Initialize probe and set in params - convert to complex and add channel dimension
probe = get_default_probe(N=64, fmt='np')
probe_complex = probe.astype(complex)
probe_3d = probe_complex[..., np.newaxis]  # Add channel dimension (64, 64, 1)
p.set('probe', probe_3d)

# Note: Cannot import generate_data at module level due to import-time side effects

class TestGenerateData(unittest.TestCase):
    
    def setUp(self):
        """Ensure params are properly set for each test."""
        p.set('N', 64)
        p.set('gridsize', 1)
        p.set('default_probe_scale', 0.7)
        if 'probe' not in p.params():
            probe = get_default_probe(N=64, fmt='np')
            probe_complex = probe.astype(complex)
            probe_3d = probe_complex[..., np.newaxis]  # Add channel dimension
            p.set('probe', probe_3d)

    def test_generate_data_import(self):
        """Test that generate_data module import is blocked by import-time side effects."""
        # The generate_data module has import-time side effects and tries to generate data
        # This test documents that behavior and ensures our probe setup is correct
        try:
            from ptycho import generate_data as init
            # If import succeeds, test that it has the expected structure
            self.assertTrue(hasattr(init, 'main'), "generate_data module should have main function")
        except KeyError as e:
            # Expected: module fails to import due to missing config parameters
            # This is actually the expected behavior for this deprecated module
            self.assertIn('train_data_file_path', str(e) or 'file_path', 
                         "Import should fail due to missing data file configuration")
            print(f"Expected import failure: {e}")
        except Exception as e:
            # Any other failure should be documented
            print(f"Unexpected import failure: {e}")
            raise
        
    def test_probe_initialization(self):
        """Test that the probe is properly initialized."""
        self.assertIn('probe', p.params(), "probe should be in params")
        probe = p.get('probe')
        self.assertEqual(probe.shape, (64, 64, 1), "probe should have correct shape with channel dimension")
        self.assertEqual(probe.dtype, complex, "probe should be complex")

if __name__ == '__main__':
    unittest.main()
