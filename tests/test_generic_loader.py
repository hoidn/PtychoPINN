import numpy as np
import os
from ptycho.loader import RawData
from ptycho.xpp import ptycho_data

def create_sample_data_file(file_path, xcoords, ycoords, xcoords_start, ycoords_start, diff3d, probeGuess, scan_index):
    np.savez(file_path, xcoords=xcoords, ycoords=ycoords, xcoords_start=xcoords_start, ycoords_start=ycoords_start, diff3d=diff3d, probeGuess=probeGuess, scan_index=scan_index)

def test_generic_loader(remove = True):
    # Load RawData instances using the 'xpp' method
    train_data = ptycho_data

    # Define file paths for output
    train_data_file_path = 'train_data.npz'

    # Use RawData.to_file() to write them to file
    train_data.to_file(train_data_file_path)

    print(f"Train data written to {train_data_file_path}")

    # Load data using the 'generic' method
    train_raw_data = RawData.from_file(train_data_file_path)

    # Perform assertions to verify the data is loaded correctly
    assert np.array_equal(train_raw_data.xcoords, train_data.xcoords)
    assert np.array_equal(train_raw_data.ycoords, train_data.ycoords)
    assert np.array_equal(train_raw_data.diff3d, train_data.diff3d)
    assert np.array_equal(train_raw_data.probeGuess, train_data.probeGuess)
    assert np.array_equal(train_raw_data.scan_index, train_data.scan_index)

    if remove:
        # Clean up the created files
        os.remove(train_data_file_path)

if __name__ == '__main__':
    test_generic_loader()
