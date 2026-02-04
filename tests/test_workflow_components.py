#!/usr/bin/env python3
"""Unit tests for ptycho.workflows.components module."""

import unittest
import tempfile
import shutil
import os
import zipfile
from pathlib import Path
import numpy as np
import tensorflow as tf
from unittest.mock import patch, MagicMock

# Add project root to path
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ptycho.workflows.components import load_inference_bundle, DiffractionToObjectAdapter


class TestLoadInferenceBundle(unittest.TestCase):
    """Test the centralized load_inference_bundle function."""
    
    def setUp(self):
        """Set up temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()
        self.model_dir = Path(self.test_dir) / "test_model"
        self.model_dir.mkdir()
        
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.test_dir)
        
    def create_mock_model_archive(self, include_diffraction_model=True):
        """Create a mock wts.h5.zip file with model data."""
        # Create a simple Keras model for testing
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(64, 64, 1)),
            tf.keras.layers.Conv2D(16, 3, padding='same'),
            tf.keras.layers.Conv2D(1, 3, padding='same')
        ])
        
        # Save the model to a temporary h5 file
        temp_model_path = self.model_dir / "temp_model.h5"
        model.save(temp_model_path)
        
        # Create the wts.h5.zip archive
        zip_path = self.model_dir / "wts.h5.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            if include_diffraction_model:
                # Add the diffraction_to_obj model
                zf.write(temp_model_path, "diffraction_to_obj.h5")
            else:
                # Add a different model to test error handling
                zf.write(temp_model_path, "some_other_model.h5")
            
            # Add params file (mock configuration)
            params_content = b"gridsize: 2\nN: 64\nnepochs: 100"
            zf.writestr("params.yaml", params_content)
        
        # Clean up temp file
        temp_model_path.unlink()
        
    def test_load_valid_model_directory(self):
        """Test loading from a valid model directory."""
        # Create mock model archive
        self.create_mock_model_archive(include_diffraction_model=True)
        
        # Mock ModelManager.load_multiple_models
        with patch('ptycho.workflows.components.ModelManager.load_multiple_models') as mock_load:
            # Create a mock model
            mock_model = MagicMock(spec=tf.keras.Model)
            mock_load.return_value = {'diffraction_to_obj': mock_model}
            
            # Test loading
            model, config = load_inference_bundle(self.model_dir)
            
            # Verify results
            self.assertIsNotNone(model)
            self.assertIsInstance(model, DiffractionToObjectAdapter)
            self.assertEqual(model._model, mock_model)
            self.assertIsInstance(config, dict)
            
            # Verify ModelManager was called correctly
            expected_path = str(self.model_dir / "wts.h5")
            mock_load.assert_called_once_with(expected_path)
            
    def test_nonexistent_directory(self):
        """Test error handling for non-existent directory."""
        fake_dir = Path(self.test_dir) / "nonexistent"
        
        with self.assertRaises(ValueError) as context:
            load_inference_bundle(fake_dir)
            
        self.assertIn("does not exist", str(context.exception))
        
    def test_not_a_directory(self):
        """Test error handling when path is not a directory."""
        # Create a file instead of directory
        fake_file = Path(self.test_dir) / "not_a_dir.txt"
        fake_file.write_text("test")
        
        with self.assertRaises(ValueError) as context:
            load_inference_bundle(fake_file)
            
        self.assertIn("not a directory", str(context.exception))
        
    def test_missing_model_archive(self):
        """Test error handling when wts.h5.zip is missing."""
        # Create directory without model archive
        empty_dir = Path(self.test_dir) / "empty_model"
        empty_dir.mkdir()
        
        with self.assertRaises(FileNotFoundError) as context:
            load_inference_bundle(empty_dir)
            
        self.assertIn("wts.h5.zip", str(context.exception))
        
    def test_missing_diffraction_model(self):
        """Test error handling when diffraction_to_obj model is missing."""
        # Create mock archive without diffraction_to_obj model
        self.create_mock_model_archive(include_diffraction_model=False)
        
        with patch('ptycho.workflows.components.ModelManager.load_multiple_models') as mock_load:
            # Return dict without diffraction_to_obj
            mock_load.return_value = {'some_other_model': MagicMock()}
            
            with self.assertRaises(KeyError) as context:
                load_inference_bundle(self.model_dir)
                
            self.assertIn("diffraction_to_obj", str(context.exception))
            
    def test_path_conversion(self):
        """Test that string paths are converted to Path objects."""
        self.create_mock_model_archive(include_diffraction_model=True)
        
        with patch('ptycho.workflows.components.ModelManager.load_multiple_models') as mock_load:
            mock_model = MagicMock(spec=tf.keras.Model)
            mock_load.return_value = {'diffraction_to_obj': mock_model}
            
            # Pass string path instead of Path object
            model, config = load_inference_bundle(str(self.model_dir))
            
            self.assertIsNotNone(model)
            
    def test_exception_propagation(self):
        """Test that exceptions from ModelManager are properly propagated."""
        self.create_mock_model_archive(include_diffraction_model=True)
        
        with patch('ptycho.workflows.components.ModelManager.load_multiple_models') as mock_load:
            # Simulate an error in ModelManager
            mock_load.side_effect = RuntimeError("Mock loading error")
            
            with self.assertRaises(RuntimeError) as context:
                load_inference_bundle(self.model_dir)
                
            self.assertIn("Mock loading error", str(context.exception))


if __name__ == '__main__':
    unittest.main()
