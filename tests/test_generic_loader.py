import pkg_resources
import numpy as np
import os
from ptycho.loader import RawData
from ptycho.xpp import load_xpp_data
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses most TensorFlow warnings

def create_sample_data_file(file_path, xcoords, ycoords, xcoords_start, ycoords_start, diff3d, probeGuess, scan_index):
    np.savez(file_path, xcoords=xcoords, ycoords=ycoords, xcoords_start=xcoords_start, ycoords_start=ycoords_start, diff3d=diff3d, probeGuess=probeGuess, scan_index=scan_index)

def get_test_data_path():
    path = 'datasets/Run1084_recon3_postPC_shrunk_3.npz'
    return pkg_resources.resource_filename(__name__, 'datasets/Run1084_recon3_postPC_shrunk_3.npz')

def test_generic_loader(remove = True, path = None):
    if path is None:
        path = get_test_data_path()
    # Load RawData instances using the 'xpp' method
    train_data, test_data = load_xpp_data(path)

    # Define file paths for output
    train_data_file_path = 'train_data.npz'
    test_data_file_path = 'test_data.npz'

    # Use RawData.to_file() to write them to file
    train_data.to_file(train_data_file_path)
    test_data.to_file(test_data_file_path)

    print(f"Train data written to {train_data_file_path}")
    print(f"Train data written to {test_data_file_path}")

    # Load data using the 'generic' method
    train_raw_data = RawData.from_file(train_data_file_path)
    test_raw_data = RawData.from_file(test_data_file_path)

    # Perform assertions to verify the data is loaded correctly
    assert np.array_equal(train_raw_data.xcoords, train_data.xcoords)
    assert np.array_equal(train_raw_data.ycoords, train_data.ycoords)
    assert np.array_equal(train_raw_data.diff3d, train_data.diff3d)
    assert np.array_equal(train_raw_data.probeGuess, train_data.probeGuess)
    assert np.array_equal(train_raw_data.scan_index, train_data.scan_index)

    if remove:
        # Clean up the created files
        os.remove(train_data_file_path)
        os.remove(test_data_file_path)
    return train_data, test_data

if __name__ == '__main__':
    test_generic_loader()
