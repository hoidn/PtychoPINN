import numpy as np
import os
from ptycho.loader import RawData
from ptycho.xpp import load_ptycho_data
import tensorflow as tf
import pkg_resources
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses most TensorFlow warnings

def create_sample_data_file(file_path, xcoords, ycoords, xcoords_start, ycoords_start, diff3d, probeGuess, scan_index):
    np.savez(file_path, xcoords=xcoords, ycoords=ycoords, xcoords_start=xcoords_start, ycoords_start=ycoords_start, diff3d=diff3d, probeGuess=probeGuess, scan_index=scan_index)

def test_generic_loader(remove=True, data_file_path = None, train_size = 512):
    if data_file_path is None:
        data_file_path = pkg_resources.resource_filename('ptycho', 'datasets/Run1084_recon3_postPC_shrunk_3.npz')

    # Load RawData instances using the 'xpp' method
    test_data, train_data, _ = load_ptycho_data(data_file_path, train_size = train_size)

#    test_data.xcoords, test_data.ycoords = -test_data.xcoords, -test_data.ycoords
#    train_data.xcoords, train_data.ycoords = -train_data.xcoords, -train_data.ycoords
#    test_data.xcoords_start, test_data.ycoords_start = -test_data.xcoords_start, -test_data.ycoords_start
#    train_data.xcoords_start, train_data.ycoords_start = -train_data.xcoords_start, -train_data.ycoords_start

    # Define file paths for output
    train_data_file_path = 'train_data.npz'
    test_data_file_path = 'test_data.npz'

    # Use RawData.to_file() to write them to file
    train_data.to_file(train_data_file_path)
    test_data.to_file(test_data_file_path)

    print(f"Train data written to {train_data_file_path}")
    print(f"Test data written to {test_data_file_path}")

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
