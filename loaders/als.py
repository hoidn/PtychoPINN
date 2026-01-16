import numpy as np
from ptycho.raw_data import RawData

import pkg_resources

def load_single_object(file_path, train_size=512):
    """
    Load ptychography data from a file and return RawData objects. We ASSUME we're processing
    a single object. The first train_size samples will be used for training and the entire dataset 
    will be used for evaluation.

    Args:
        file_path: Path to the data file.
        train_size (int, optional): Number of data points to include in the training set. Defaults to 512.

    Returns:
        tuple: A tuple containing two RawData objects:
            - ptycho_data: RawData object containing the full dataset.
            - ptycho_data_train: RawData object containing a subset of the data for training.
    """
    # Load data from file
    data = np.load(file_path)

    # Extract required arrays from loaded data
    xcoords = data['xcoords']
    ycoords = data['ycoords']
    xcoords_start = data['xcoords_start']
    ycoords_start = data['ycoords_start']
    diff3d = data['diffraction']
    probeGuess = data['probeGuess']
    objectGuess = data['objectGuess']

    # Create scan_index array
    scan_index = np.zeros(diff3d.shape[0], dtype=int)

    # Create RawData object for the full dataset
    ptycho_data = RawData(xcoords, ycoords, xcoords_start, ycoords_start,
                          diff3d, probeGuess, scan_index, objectGuess=objectGuess)

    # Create RawData object for the training subset
    ptycho_data_train = RawData(xcoords[:train_size], ycoords[:train_size],
                                xcoords_start[:train_size], ycoords_start[:train_size],
                                diff3d[:train_size], probeGuess,
                                scan_index[:train_size], objectGuess=objectGuess)

    return ptycho_data, ptycho_data_train, data

