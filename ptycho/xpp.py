import numpy as np
import os
import pkg_resources
from ptycho import loader
from ptycho.loader import RawData, PtychoDataContainer, load

def load_xpp_data(file_path, gridh=32, gridw=32, train_size=512):
    """
    Load and prepare XPCS data for ptychography reconstruction.
    Args:
        file_path (str): Path to the .npz file containing the XPCS data.
        gridh (int, optional): Height of the grid. Defaults to 32.
        gridw (int, optional): Width of the grid. Defaults to 32.
        train_size (int, optional): Size of the training data. Defaults to 512.
    Returns:
        tuple: A tuple containing train_data_container and test_data_container.
    """
    if not isinstance(gridh, int) or gridh <= 0:
        raise ValueError("gridh must be a positive integer.")
    if not isinstance(gridw, int) or gridw <= 0:
        raise ValueError("gridw must be a positive integer.")
    if not isinstance(train_size, int) or train_size <= 0:
        raise ValueError("train_size must be a positive integer.")

    # Load the .npz file
    obj = np.load(file_path)
    # Prepare the data
    xcoords = obj['xcoords'][:gridh * gridw]
    ycoords = obj['ycoords'][:gridh * gridw]
    xcoords_start = obj['xcoords_start'][:gridh * gridw]
    ycoords_start = obj['ycoords_start'][:gridh * gridw]
    diff3d = np.transpose(obj['diffraction'][:, :, :gridh * gridw], [2, 0, 1])
    probeGuess = obj['probeGuess']
    objectGuess = obj['objectGuess']
    # Initialize RawData objects
    scan_index = np.zeros(diff3d.shape[0], dtype=int)
    ptycho_data = RawData(xcoords, ycoords, xcoords_start, ycoords_start, diff3d, probeGuess, scan_index, objectGuess=objectGuess)
    ptycho_data_train = RawData(xcoords[:train_size], ycoords[:train_size], xcoords_start[:train_size], ycoords_start[:train_size], diff3d[:train_size], probeGuess, scan_index[:train_size], objectGuess=objectGuess)

    return ptycho_data_train, ptycho_data

def get_data_containers(data_file_path=None, N=64, train_frac=0.5, **kwargs):
    """
    Get the train and test data containers.
    Args:
        data_file_path (str, optional): Path to the .npz file containing the XPCS data.
                                        If None, the default file path will be used. Defaults to None.
        N (int, optional): Size of the image. Defaults to 64.
        train_frac (float, optional): Fraction of the data to be used for training. Defaults to 0.5.
    Returns:
        tuple: A tuple containing train_data_container and test_data_container.
    """
    if data_file_path is None:
        data_file_path = pkg_resources.resource_filename(__name__, 'datasets/Run1084_recon3_postPC_shrunk_3.npz')
    elif not isinstance(data_file_path, str):
        raise TypeError("data_file_path must be a string.")
    elif not os.path.isfile(data_file_path):
        raise FileNotFoundError(f"File not found: {data_file_path}")

    ptycho_data_train, ptycho_data = load_xpp_data(data_file_path)
    train_data_container = load(lambda: ptycho_data_train.generate_grouped_data(64, K=7, nsamples=1), which='train')
    test_data_container = load(lambda: ptycho_data.generate_grouped_data(64, K=7, nsamples=1), which='test')
    return train_data_container, test_data_container

def get_data(data_file_path=None, N=64, train_frac=0.5, **kwargs):
    """
    Get the ptychography data and split it into training and test sets.
    Args:
        data_file_path (str, optional): Path to the .npz file containing the XPCS data.
                                        If None, the default file path will be used. Defaults to None.
        N (int, optional): Size of the image. Defaults to 64.
        train_frac (float, optional): Fraction of the data to be used for training. Defaults to 0.5.
    Returns:
        tuple: A tuple containing the grouped data and train_frac.
    """
    train_data_container, _ = get_data_containers(data_file_path, N, train_frac, **kwargs)
    return train_data_container.generate_grouped_data(N, K=7, nsamples=1), train_frac
