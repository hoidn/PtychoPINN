from .params import params
from ptycho import diffsim as datasets

class RawData:
    def __init__(self, xcoords, ycoords, xcoords_start, ycoords_start, diff3d, probeGuess, scan_index):
        # Sanity checks
        self._check_data_validity(xcoords, ycoords, xcoords_start, ycoords_start, diff3d, probeGuess, scan_index)

        # Assigning values if checks pass
        self.xcoords = xcoords
        self.ycoords = ycoords
        self.xcoords_start = xcoords_start
        self.ycoords_start = ycoords_start
        self.diff3d = diff3d
        self.probeGuess = probeGuess
        self.scan_index = scan_index

    @staticmethod
    def from_files(train_data_file_path, test_data_file_path):
        """
        Static method to instantiate RawData objects from training and test data files.

        The data files should be NumPy .npz files with the following keys:
        - 'xcoords': x coordinates of the scan points
        - 'ycoords': y coordinates of the scan points
        - 'xcoords_start': starting x coordinates for the scan
        - 'ycoords_start': starting y coordinates for the scan
        - 'diff3d': diffraction patterns
        - 'probeGuess': initial guess of the probe function
        - 'scan_index': array indicating the scan index for each diffraction pattern

        Args:
            train_data_file_path (str): Path to the training data file.
            test_data_file_path (str): Path to the test data file.

        Returns:
            tuple: A tuple containing the instantiated RawData objects for training and test data.
        """
        # Load training data
        train_data = np.load(train_data_file_path)
        train_raw_data = RawData(
            xcoords=train_data['xcoords'],
            ycoords=train_data['ycoords'],
            xcoords_start=train_data['xcoords_start'],
            ycoords_start=train_data['ycoords_start'],
            diff3d=train_data['diff3d'],
            probeGuess=train_data['probeGuess'],
            scan_index=train_data['scan_index']
        )

        # Load test data
        test_data = np.load(test_data_file_path)
        test_raw_data = RawData(
            xcoords=test_data['xcoords'],
            ycoords=test_data['ycoords'],
            xcoords_start=test_data['xcoords_start'],
            ycoords_start=test_data['ycoords_start'],
            diff3d=test_data['diff3d'],
            probeGuess=test_data['probeGuess'],
            scan_index=test_data['scan_index']
        )

        return train_raw_data, test_raw_data

    def _check_data_validity(self, xcoords, ycoords, xcoords_start, ycoords_start, diff3d, probeGuess, scan_index):
        # Check if all inputs are numpy arrays
        if not all(isinstance(arr, np.ndarray) for arr in [xcoords, ycoords, xcoords_start, ycoords_start, diff3d, probeGuess, scan_index]):
            raise ValueError("All inputs must be numpy arrays.")

        # Check if coordinate arrays have matching shapes
        if not (xcoords.shape == ycoords.shape == xcoords_start.shape == ycoords_start.shape):
            raise ValueError("Coordinate arrays must have matching shapes.")

        # Add more checks as necessary, for example:
        # - Check if 'diff3d' has the expected number of dimensions
        # - Check if 'probeGuess' has a specific shape or type criteria
        # Check if 'scan_index' has the correct length
        if len(scan_index) != diff3d.shape[0]:
            raise ValueError("Length of scan_index array must match the number of diffraction patterns.")

class PtychoDataset:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

class PtychoData:
    def __init__(self, X, Y_I, Y_phi, YY_full, coords_nominal, coords_true, probe, scan_index):
        from .tf_helper import combine_complex
        self.X = X
        self.Y = combine_complex(Y_I, Y_phi)
        self.YY_full = YY_full
        self.coords_nominal = coords_nominal
        self.coords_true = coords_true
        self.probe = probe
        self.scan_index = scan_index
