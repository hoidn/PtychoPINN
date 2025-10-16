# file: tests/test_model_manager_persistence.py

import unittest
import tempfile
from pathlib import Path
import numpy as np
import tensorflow as tf
import sys
import os

# Add project root to path to allow for ptycho imports
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from ptycho import params as p
from ptycho.probe import get_default_probe

# Initialize probe before importing model to avoid KeyError
p.set('N', 64)
p.set('probe.type', 'gaussian')
p.set('probe.photons', 1e10)
probe = get_default_probe(64)
p.params()['probe'] = probe

from ptycho.model_manager import ModelManager
from ptycho.model import create_model_with_gridsize

class TestModelManagerPersistence(unittest.TestCase):

    def setUp(self):
        """Create a temporary directory and store original params."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.original_params = p.cfg.copy()
        print(f"\nCreated temp dir: {self.test_dir.name}")

    def tearDown(self):
        """Clean up the temporary directory and restore original params."""
        self.test_dir.cleanup()
        p.cfg = self.original_params
        print(f"Cleaned up temp dir and restored params.")

    def test_parameter_restoration_on_load(self):
        """
        CRITICAL TEST: Verify that loading a model restores its saved params.cfg,
        overwriting the current session's configuration.
        """
        print("--- Running Test: Parameter Restoration ---")
        # 1. Save a model with a specific, non-default configuration
        p.set('N', 128)
        p.set('gridsize', 2)
        p.set('nphotons', 1e8)
        model_to_save, _ = create_model_with_gridsize(gridsize=2, N=128)
        
        model_path = Path(self.test_dir.name) / "param_test_model"
        ModelManager.save_model(model_to_save, str(model_path), {}, 1.0)

        # 2. Change the current session's parameters to something different
        p.set('N', 64)
        p.set('gridsize', 1)
        p.set('nphotons', 1e9)
        
        # 3. Load the model
        _ = ModelManager.load_model(str(model_path))
        
        # 4. Assert that the global parameters have been updated to the saved values
        self.assertEqual(p.get('N'), 128, "Parameter 'N' was not restored correctly.")
        self.assertEqual(p.get('gridsize'), 2, "Parameter 'gridsize' was not restored correctly.")
        self.assertEqual(p.get('nphotons'), 1e8, "Parameter 'nphotons' was not restored correctly.")
        print("✅ Parameter restoration test passed.")

    def test_architecture_aware_loading(self):
        """
        CRITICAL TEST: Test that a model's architecture is correctly rebuilt based on
        the parameters restored from the saved artifact.
        """
        print("--- Running Test: Architecture-Aware Loading ---")
        # 1. Save a model with gridsize=2 (which has 4 input channels)
        model_gs2, _ = create_model_with_gridsize(gridsize=2, N=64)
        
        model_path = Path(self.test_dir.name) / "model_gs2"
        ModelManager.save_model(model_gs2, str(model_path), {}, 1.0)

        # 2. Set current session to a conflicting gridsize
        p.set('gridsize', 1)
        
        # 3. Load the gridsize=2 model. This should first restore gridsize=2,
        #    then build the model with the correct architecture.
        loaded_model = ModelManager.load_model(str(model_path))
        
        # 4. Assert that the loaded model has the correct architecture for gridsize=2
        # The input layer for a gridsize=2 model should have 4 channels.
        self.assertEqual(loaded_model.input_shape[0][-1], 4, "Loaded model has incorrect input shape for gridsize=2.")
        print("✅ Architecture-aware loading test passed.")

    def test_inference_consistency_after_load(self):
        """
        CRITICAL TEST: Ensure a loaded model produces identical output to the
        original model for the same input.
        """
        print("--- Running Test: Inference Consistency ---")
        p.set('N', 64)
        p.set('gridsize', 1)
        
        # 1. Create and save a model
        _, original_inference_model = create_model_with_gridsize(gridsize=1, N=64)
        
        # 2. Generate a dummy input
        dummy_diffraction = tf.random.normal((2, 64, 64, 1))
        dummy_positions = tf.zeros((2, 1, 2, 1))
        
        # 3. Get output from the original model
        original_output = original_inference_model.predict([dummy_diffraction, dummy_positions])
        
        model_path = Path(self.test_dir.name) / "inference_model"
        ModelManager.save_model(original_inference_model, str(model_path), {}, 1.0)
        
        # 4. Load the model
        loaded_model = ModelManager.load_model(str(model_path))
        
        # 5. Get output from the loaded model
        loaded_output = loaded_model.predict([dummy_diffraction, dummy_positions])
        
        # 6. Assert that the outputs are numerically identical
        np.testing.assert_allclose(original_output, loaded_output, rtol=1e-6,
                                   err_msg="Model output changed after save/load cycle.")
        print("✅ Inference consistency test passed.")

if __name__ == '__main__':
    unittest.main()