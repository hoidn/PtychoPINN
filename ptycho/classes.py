from .params import params
from ptycho import diffsim as datasets

class RawData:
    def __init__(self, xcoords, ycoords, xcoords_start, ycoords_start, diff3d, probeGuess):
        # Sanity checks
        self._check_data_validity(xcoords, ycoords, xcoords_start, ycoords_start, diff3d, probeGuess)

        # Assigning values if checks pass
        self.xcoords = xcoords
        self.ycoords = ycoords
        self.xcoords_start = xcoords_start
        self.ycoords_start = ycoords_start
        self.diff3d = diff3d
        self.probeGuess = probeGuess

    def _check_data_validity(self, xcoords, ycoords, xcoords_start, ycoords_start, diff3d, probeGuess):
        # Check if all inputs are numpy arrays
        if not all(isinstance(arr, np.ndarray) for arr in [xcoords, ycoords, xcoords_start, ycoords_start, diff3d, probeGuess]):
            raise ValueError("All inputs must be numpy arrays.")

        # Check if coordinate arrays have matching shapes
        if not (xcoords.shape == ycoords.shape == xcoords_start.shape == ycoords_start.shape):
            raise ValueError("Coordinate arrays must have matching shapes.")

        # Add more checks as necessary, for example:
        # - Check if 'diff3d' has the expected number of dimensions
        # - Check if 'probeGuess' has a specific shape or type criteria

class PtychoDataset:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

class PtychoData:
    def __init__(self, X, Y_I, Y_phi, YY_full, coords_nominal, coords_true, probe):
        from .tf_helper import combine_complex
        self.X = X
        self.Y = combine_complex(Y_I, Y_phi)
        self.YY_full = YY_full
        self.coords_nominal = coords_nominal
        self.coords_true = coords_true
        self.probe = probe
