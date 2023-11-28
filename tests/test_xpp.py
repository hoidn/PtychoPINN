import numpy as np
import pytest
import tensorflow as tf
from ptycho.xpp import load

def test_load_function_with_synthetic_data():
    # Load the sample experimental dataset using the load function
    data = load('test')  # Load 'test' data from the sample experimental dataset

    # Get the gridsize from params.py
    from ptycho.params import get
    gridsize = get('gridsize')

    # Assert that the data is loaded correctly and has the expected shape
    # Note: The shape of the tensors is (b, 64, 64, gridsize**2)
    assert data['X'].shape == (data['X'].shape[0], 64, 64, gridsize**2), "Loaded tensor X has incorrect shape"
    # Add more assertions as needed to verify the normalization and other processing steps

def test_load_function_normalization():
    # Load the sample experimental dataset using the load function
    data = load('test')  # Load 'test' data from the sample experimental dataset

    # Calculate the expected normalization factor
    N = 64  # Assuming N is the size of the diffraction patterns
    X_full = data['X']
    X_full_norm = ((N / 2)**2) / np.mean(tf.reduce_sum(X_full**2, axis=[1, 2]))

    # Assert that the normalization factor in the loaded data matches the expected value
    assert np.isclose(data['norm_Y_I'], X_full_norm), "Normalization factor does not match expected value"

# Additional tests for other functionalities
