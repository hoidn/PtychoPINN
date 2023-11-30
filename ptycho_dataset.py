class PtychoDataset:
    def __init__(self, train_X, train_coords, train_probe, test_X, test_coords, test_probe, train_Y=None, test_Y=None):
        """
        Initializes the PtychoDataset object with training and evaluation data.

        Args:
            train_X (np.ndarray): 4D array of training diffraction patterns.
            train_coords (np.ndarray): 3D array of training scan point coordinates.
            train_probe (np.ndarray): 2D complex-valued array representing the probe function for training.
            test_X (np.ndarray): 4D array of evaluation diffraction patterns.
            test_coords (np.ndarray): 3D array of evaluation scan point coordinates.
            test_probe (np.ndarray): 2D complex-valued array representing the probe function for evaluation.
            train_Y (np.ndarray): 2D complex-valued array representing the ground truth for training. Optional.
            test_Y (np.ndarray): 2D complex-valued array representing the ground truth for evaluation. Optional.
        """
        self.train_X = train_X
        self.train_coords = train_coords
        self.train_probe = train_probe
        self.test_X = test_X
        self.test_coords = test_coords
        self.test_probe = test_probe
        self.train_Y = train_Y
        self.test_Y = test_Y
