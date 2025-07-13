#!/usr/bin/env python
"""
Unit tests for scripts/simulation/simulate_and_save.py

This test module verifies that the simulate_and_save script correctly:
1. Creates output NPZ files with the expected structure
2. Handles edge cases and error conditions appropriately
3. Produces data with the correct number of scan positions
"""

import unittest
import os
import tempfile
import numpy as np
from pathlib import Path
import sys

# Add the project root to the Python path to allow imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the function we want to test
from scripts.simulation.simulate_and_save import simulate_and_save


class TestSimulateAndSave(unittest.TestCase):
    """Test suite for the simulate_and_save function."""

    def setUp(self):
        """
        Set up test environment before each test.
        Creates a temporary directory and dummy input data.
        """
        # Create a temporary directory for test files
        self.test_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = Path(self.test_dir.name)
        
        # Define file paths
        self.input_file_path = self.temp_dir_path / "test_input.npz"
        self.output_file_path = self.temp_dir_path / "test_output.npz"
        
        # Create dummy input data
        self._create_dummy_input_file()

    def tearDown(self):
        """
        Clean up test environment after each test.
        Removes the temporary directory and its contents.
        """
        self.test_dir.cleanup()

    def _create_dummy_input_file(self):
        """
        Create a dummy input NPZ file using actual working probe/object data.
        This ensures compatibility with the simulation code.
        """
        # Load actual working data to use as test template
        working_data_path = "/home/ollie/Documents/PtychoPINN/datasets/fly/fly001.npz"
        
        try:
            with np.load(working_data_path) as data:
                # Extract the probe and object from working data
                dummy_probe = data['probeGuess'].copy()
                dummy_object = data['objectGuess'].copy()
        except FileNotFoundError:
            # Fallback: create simple dummy data with known working dimensions
            # Based on actual working data shapes: probe (64, 64), object (232, 232)
            dummy_probe = np.ones((64, 64), dtype=np.complex64)
            dummy_object = np.ones((232, 232), dtype=np.complex64)
        
        # Save to NPZ file
        np.savez(
            self.input_file_path,
            objectGuess=dummy_object,
            probeGuess=dummy_probe
        )

    def test_creates_output_file(self):
        """Test that the script successfully creates an output NPZ file."""
        # Run the function
        simulate_and_save(
            input_file_path=self.input_file_path,
            output_file_path=self.output_file_path,
            nimages=50,  # Use enough images to avoid coordinate grouping issues
            seed=42      # For reproducibility
        )
        
        # Check that output file was created
        self.assertTrue(
            self.output_file_path.exists(),
            "Output file was not created"
        )

    def test_output_file_structure(self):
        """Test that the output file contains all expected keys."""
        # Run the function
        simulate_and_save(
            input_file_path=self.input_file_path,
            output_file_path=self.output_file_path,
            nimages=50,
            seed=42
        )
        
        # Load and verify the output file structure
        with np.load(self.output_file_path) as data:
            expected_keys = [
                'xcoords',
                'ycoords', 
                'xcoords_start',
                'ycoords_start',
                'diff3d',
                'probeGuess',
                'objectGuess',
                'scan_index'
            ]
            
            for key in expected_keys:
                self.assertIn(
                    key, data.keys(),
                    f"Expected key '{key}' not found in output file"
                )

    def test_correct_number_of_diffraction_patterns(self):
        """Test that the number of diffraction patterns matches nimages."""
        nimages = 50
        
        simulate_and_save(
            input_file_path=self.input_file_path,
            output_file_path=self.output_file_path,
            nimages=nimages,
            seed=42
        )
        
        with np.load(self.output_file_path) as data:
            # Check array shapes
            self.assertEqual(
                data['diff3d'].shape[0], nimages,
                f"Expected {nimages} diffraction patterns, got {data['diff3d'].shape[0]}"
            )
            self.assertEqual(
                data['xcoords'].shape[0], nimages,
                f"Expected {nimages} x-coordinates, got {data['xcoords'].shape[0]}"
            )
            self.assertEqual(
                data['ycoords'].shape[0], nimages,
                f"Expected {nimages} y-coordinates, got {data['ycoords'].shape[0]}"
            )
            self.assertEqual(
                data['scan_index'].shape[0], nimages,
                f"Expected {nimages} scan indices, got {data['scan_index'].shape[0]}"
            )

    def test_data_types_and_shapes(self):
        """Test that output data has correct types and dimensional structure."""
        simulate_and_save(
            input_file_path=self.input_file_path,
            output_file_path=self.output_file_path,
            nimages=20,
            seed=42
        )
        
        with np.load(self.output_file_path) as data:
            # Check coordinate arrays are 1D
            self.assertEqual(len(data['xcoords'].shape), 1, "xcoords should be 1D")
            self.assertEqual(len(data['ycoords'].shape), 1, "ycoords should be 1D")
            self.assertEqual(len(data['xcoords_start'].shape), 1, "xcoords_start should be 1D")
            self.assertEqual(len(data['ycoords_start'].shape), 1, "ycoords_start should be 1D")
            
            # Check diffraction patterns are 3D (nimages, height, width)
            self.assertEqual(len(data['diff3d'].shape), 3, "diff3d should be 3D")
            
            # Check probe and object are 2D
            self.assertEqual(len(data['probeGuess'].shape), 2, "probeGuess should be 2D")
            self.assertEqual(len(data['objectGuess'].shape), 2, "objectGuess should be 2D")
            
            # Check scan_index is 1D
            self.assertEqual(len(data['scan_index'].shape), 1, "scan_index should be 1D")
            
            # Check that probe and object are complex
            self.assertTrue(
                np.iscomplexobj(data['probeGuess']),
                "probeGuess should be complex-valued"
            )
            self.assertTrue(
                np.iscomplexobj(data['objectGuess']),
                "objectGuess should be complex-valued"
            )

    def test_reproducibility_with_seed(self):
        """Test that using the same seed produces identical results."""
        seed = 123
        nimages = 50
        
        # Run simulation twice with the same seed
        output_file_1 = self.temp_dir_path / "output1.npz"
        output_file_2 = self.temp_dir_path / "output2.npz"
        
        simulate_and_save(
            input_file_path=self.input_file_path,
            output_file_path=output_file_1,
            nimages=nimages,
            seed=seed
        )
        
        simulate_and_save(
            input_file_path=self.input_file_path,
            output_file_path=output_file_2,
            nimages=nimages,
            seed=seed
        )
        
        # Compare the results
        with np.load(output_file_1) as data1, np.load(output_file_2) as data2:
            np.testing.assert_array_equal(
                data1['xcoords'], data2['xcoords'],
                "xcoords should be identical with same seed"
            )
            np.testing.assert_array_equal(
                data1['ycoords'], data2['ycoords'],
                "ycoords should be identical with same seed"
            )
            np.testing.assert_array_equal(
                data1['diff3d'], data2['diff3d'],
                "diff3d should be identical with same seed"
            )

    def test_missing_input_file_raises_error(self):
        """Test that missing input file raises FileNotFoundError."""
        nonexistent_file = self.temp_dir_path / "nonexistent.npz"
        
        with self.assertRaises(FileNotFoundError):
            simulate_and_save(
                input_file_path=nonexistent_file,
                output_file_path=self.output_file_path,
                nimages=10,
                seed=42
            )

    def test_output_directory_creation(self):
        """Test that output directories are created if they don't exist."""
        # Create a nested output path
        nested_output = self.temp_dir_path / "subdir" / "nested" / "output.npz"
        
        # Ensure the directory doesn't exist initially
        self.assertFalse(nested_output.parent.exists())
        
        # Run the function
        simulate_and_save(
            input_file_path=self.input_file_path,
            output_file_path=nested_output,
            nimages=5,
            seed=42
        )
        
        # Check that the directory was created and file exists
        self.assertTrue(nested_output.exists(), "Output file should exist")
        self.assertTrue(nested_output.parent.exists(), "Output directory should be created")

    def test_custom_buffer_parameter(self):
        """Test that custom buffer parameter is handled correctly."""
        simulate_and_save(
            input_file_path=self.input_file_path,
            output_file_path=self.output_file_path,
            nimages=10,
            buffer=20.0,  # Custom buffer value
            seed=42
        )
        
        # Verify the file was created successfully
        self.assertTrue(self.output_file_path.exists())
        
        # Load and check basic structure
        with np.load(self.output_file_path) as data:
            self.assertEqual(data['diff3d'].shape[0], 10)


if __name__ == '__main__':
    # Enable verbose output for test results
    unittest.main(verbosity=2)