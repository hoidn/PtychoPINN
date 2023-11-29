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

#def test_load_function_normalization():
#    # Load the sample experimental dataset using the load function
#    data = load('test')  # Load 'test' data from the sample experimental dataset
#
#    # Calculate the expected normalization factor
#    N = 64  # Assuming N is the size of the diffraction patterns
#    X_full = data['X']
#    X_full_norm = ((N / 2)**2) / np.mean(tf.reduce_sum(X_full**2, axis=[1, 2]))
#
#    # Assert that the normalization factor in the loaded data matches the expected value
#    assert np.isclose(data['norm_Y_I'], X_full_norm), "Normalization factor does not match expected value"

# Additional tests for other functionalities
import numpy as np
import pytest
from ptycho import xpp

@pytest.mark.parametrize("dataset", ['train', 'test'])
def test_loading_of_data(dataset):
    # Load data
    data = xpp.load(dataset)

    # Assertions to ensure data is loaded correctly
    assert isinstance(data, dict), "Loaded data should be a dictionary."
    assert 'X' in data, "Loaded data should have 'X' key."
    assert 'Y_I' in data, "Loaded data should have 'Y_I' key."
    assert 'Y_phi' in data, "Loaded data should have 'Y_phi' key."

def test_normalization():
    # Load test data
    test_data = xpp.load('test')

    # Using the provided test case to ensure normalization is correct
    mean_value = np.mean(test_data['X'])
    expected_mean = 0.3497164627096171  # This value is from the provided test case
    assert np.isclose(mean_value, expected_mean), f"Mean of 'X' array should be close to {expected_mean}."

@pytest.mark.parametrize("dataset", ['train', 'test'])
def test_data_splitting(dataset):
    # Load data
    data = xpp.load(dataset)

    # Assuming the dataset is split 50-50 for train and test
    assert len(data['X']) > 0, f"{dataset} 'X' array should not be empty."

