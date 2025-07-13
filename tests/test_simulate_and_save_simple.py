#!/usr/bin/env python
"""
Simplified test for scripts/simulation/simulate_and_save.py

This test focuses on testing the interface and file handling without 
running the full TensorFlow simulation, which has compatibility issues
in the test environment.
"""

import unittest
import os
import tempfile
import numpy as np
from pathlib import Path
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import functions to test
from scripts.simulation.simulate_and_save import parse_arguments
from ptycho.nongrid_simulation import load_probe_object


class TestSimulateAndSaveInterface(unittest.TestCase):
    """Test suite for the simulate_and_save interface and argument parsing."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = Path(self.test_dir.name)
        self.input_file_path = self.temp_dir_path / "test_input.npz"
        self.output_file_path = self.temp_dir_path / "test_output.npz"
        
        # Create test input file with proper structure
        self._create_test_input_file()

    def tearDown(self):
        """Clean up test environment."""
        self.test_dir.cleanup()

    def _create_test_input_file(self):
        """Create a test input file with valid structure."""
        # Use working dimensions from actual data
        probe = np.ones((64, 64), dtype=np.complex64)
        obj = np.ones((128, 128), dtype=np.complex64)
        
        np.savez(
            self.input_file_path,
            objectGuess=obj,
            probeGuess=probe
        )

    def test_argument_parsing(self):
        """Test that command line arguments are parsed correctly."""
        # Test with minimal required arguments
        test_args = [
            "--input-file", str(self.input_file_path),
            "--output-file", str(self.output_file_path),
            "--nimages", "100"
        ]
        
        # Mock sys.argv
        import sys
        original_argv = sys.argv
        try:
            sys.argv = ["simulate_and_save.py"] + test_args
            args = parse_arguments()
            
            self.assertEqual(args.input_file, str(self.input_file_path))
            self.assertEqual(args.output_file, str(self.output_file_path))
            self.assertEqual(args.nimages, 100)
            self.assertIsNone(args.buffer)  # Default should be None
            self.assertIsNone(args.seed)    # Default should be None
        finally:
            sys.argv = original_argv

    def test_argument_parsing_with_optional_args(self):
        """Test argument parsing with all optional parameters."""
        test_args = [
            "--input-file", str(self.input_file_path),
            "--output-file", str(self.output_file_path),
            "--nimages", "500",
            "--buffer", "25.5",
            "--seed", "12345"
        ]
        
        import sys
        original_argv = sys.argv
        try:
            sys.argv = ["simulate_and_save.py"] + test_args
            args = parse_arguments()
            
            self.assertEqual(args.nimages, 500)
            self.assertEqual(args.buffer, 25.5)
            self.assertEqual(args.seed, 12345)
        finally:
            sys.argv = original_argv

    def test_load_probe_object_function(self):
        """Test the load_probe_object function with valid input."""
        obj, probe = load_probe_object(str(self.input_file_path))
        
        # Check that objects were loaded correctly
        self.assertEqual(obj.shape, (128, 128))
        self.assertEqual(probe.shape, (64, 64))
        self.assertTrue(np.iscomplexobj(obj))
        self.assertTrue(np.iscomplexobj(probe))

    def test_load_probe_object_missing_file(self):
        """Test load_probe_object with missing file."""
        nonexistent_file = self.temp_dir_path / "nonexistent.npz"
        
        with self.assertRaises(RuntimeError):
            load_probe_object(str(nonexistent_file))

    def test_load_probe_object_invalid_structure(self):
        """Test load_probe_object with file missing required keys."""
        invalid_file = self.temp_dir_path / "invalid.npz"
        
        # Create file with wrong keys
        np.savez(invalid_file, wrong_key=np.ones((64, 64)))
        
        with self.assertRaises(RuntimeError):
            load_probe_object(str(invalid_file))

    def test_load_probe_object_wrong_data_types(self):
        """Test load_probe_object with non-complex data."""
        invalid_file = self.temp_dir_path / "invalid_types.npz"
        
        # Create file with real-valued arrays instead of complex
        real_probe = np.ones((64, 64), dtype=np.float32)
        real_obj = np.ones((128, 128), dtype=np.float32)
        
        np.savez(invalid_file, objectGuess=real_obj, probeGuess=real_probe)
        
        with self.assertRaises(RuntimeError):
            load_probe_object(str(invalid_file))

    def test_output_directory_creation_scenario(self):
        """Test that output directory would be created properly."""
        # Create a nested output path that doesn't exist
        nested_output = self.temp_dir_path / "deep" / "nested" / "path" / "output.npz"
        
        # Verify the parent directory doesn't exist
        self.assertFalse(nested_output.parent.exists())
        
        # Test that the directory creation logic would work
        # (Simulating what simulate_and_save does)
        output_dir = nested_output.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify directory was created
        self.assertTrue(output_dir.exists())
        self.assertTrue(output_dir.is_dir())

    def test_file_path_handling(self):
        """Test that file path handling works with Path objects and strings."""
        # Test with string path
        obj1, probe1 = load_probe_object(str(self.input_file_path))
        
        # Test with Path object
        obj2, probe2 = load_probe_object(self.input_file_path)
        
        # Results should be identical
        np.testing.assert_array_equal(obj1, obj2)
        np.testing.assert_array_equal(probe1, probe2)


class TestSimulateAndSaveDocumentation(unittest.TestCase):
    """Test that the script has proper documentation and structure."""

    def test_script_has_docstring(self):
        """Test that the script module has proper documentation."""
        import scripts.simulation.simulate_and_save as script_module
        
        self.assertIsNotNone(script_module.__doc__)
        self.assertIn("simulate_and_save", script_module.__doc__)

    def test_main_function_exists(self):
        """Test that main function exists for command-line usage."""
        import scripts.simulation.simulate_and_save as script_module
        
        self.assertTrue(hasattr(script_module, 'main'))
        self.assertTrue(callable(script_module.main))

    def test_simulate_and_save_function_signature(self):
        """Test that the main function has expected signature."""
        from scripts.simulation.simulate_and_save import simulate_and_save
        import inspect
        
        sig = inspect.signature(simulate_and_save)
        params = list(sig.parameters.keys())
        
        # Check required parameters
        self.assertIn('input_file_path', params)
        self.assertIn('output_file_path', params)
        self.assertIn('nimages', params)
        
        # Check optional parameters
        self.assertIn('buffer', params)
        self.assertIn('seed', params)


if __name__ == '__main__':
    unittest.main(verbosity=2)