import numpy as np
import pytest
from ptycho.xpp import load

def test_load_function_with_synthetic_data():
    # Generate synthetic data similar to the expected experimental format
    diffraction_patterns = np.random.rand(2, 1000, 64, 64)
    scan_points = np.random.rand(2, 1000, 2)
    np.savez('synthetic_data.npz', diffraction=diffraction_patterns, scan_points=scan_points)

    # Load the synthetic data using the load function with the correct argument
    data = load('test')  # Assuming you want to test with 'test' data

    # Assert that the data is loaded and normalized correctly
    assert data['X'].shape == (2, 1000, 64, 64)
    assert data['scan_points'].shape == (2, 1000, 2)
    # Add more assertions as needed to verify the normalization and other processing steps

# Additional tests for other functionalities
