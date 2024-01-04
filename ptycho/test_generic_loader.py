import numpy as np
import os
from ptycho.loader import RawData

def create_sample_data_file(file_path, xcoords, ycoords, xcoords_start, ycoords_start, diff3d, probeGuess, scan_index):
    np.savez(file_path, xcoords=xcoords, ycoords=ycoords, xcoords_start=xcoords_start, ycoords_start=ycoords_start, diff3d=diff3d, probeGuess=probeGuess, scan_index=scan_index)

def test_generic_loader():
    # Create sample data
    xcoords = np.array([0, 1, 2])
    ycoords = np.array([0, 1, 2])
    xcoords_start = np.array([0, 0, 0])
    ycoords_start = np.array([0, 0, 0])
    diff3d = np.random.rand(3, 10, 10)
    probeGuess = np.random.rand(10, 10)
    scan_index = np.array([0, 0, 0])

    # Define file paths
    train_data_file_path = 'train_data.npz'
    test_data_file_path = 'test_data.npz'

    # Create data files
    create_sample_data_file(train_data_file_path, xcoords, ycoords, xcoords_start, ycoords_start, diff3d, probeGuess, scan_index)
    create_sample_data_file(test_data_file_path, xcoords, ycoords, xcoords_start, ycoords_start, diff3d, probeGuess, scan_index)

    # Load data using the 'generic' method
    train_raw_data, test_raw_data = RawData.from_files(train_data_file_path, test_data_file_path)

    # Perform assertions to verify the data is loaded correctly
    assert np.array_equal(train_raw_data.xcoords, xcoords)
    assert np.array_equal(train_raw_data.ycoords, ycoords)
    assert np.array_equal(train_raw_data.diff3d, diff3d)
    assert np.array_equal(train_raw_data.probeGuess, probeGuess)
    assert np.array_equal(train_raw_data.scan_index, scan_index)

    # Clean up the created files
    os.remove(train_data_file_path)
    os.remove(test_data_file_path)

if __name__ == '__main__':
    test_generic_loader()
