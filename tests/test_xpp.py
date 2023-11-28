import numpy as np
import pytest
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

# Additional tests for other functionalities
