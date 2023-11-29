class PtychoDataset:
    def __init__(self, train_diffraction, train_coords, train_probe, test_diffraction, test_coords, test_probe, train_ground_truth=None, test_ground_truth=None):
        """
        Initializes the PtychoDataset object with training and evaluation data.

        Args:
            train_diffraction (np.ndarray): 4D array of training diffraction patterns.
            train_coords (np.ndarray): 3D array of training scan point coordinates.
            train_probe (np.ndarray): 2D complex-valued array representing the probe function for training.
            test_diffraction (np.ndarray): 4D array of evaluation diffraction patterns.
            test_coords (np.ndarray): 3D array of evaluation scan point coordinates.
            test_probe (np.ndarray): 2D complex-valued array representing the probe function for evaluation.
            train_ground_truth (np.ndarray): 2D complex-valued array representing the ground truth for training. Optional.
            test_ground_truth (np.ndarray): 2D complex-valued array representing the ground truth for evaluation. Optional.
        """
        self.train_diffraction = train_diffraction
        self.train_coords = train_coords
        self.train_probe = train_probe
        self.test_diffraction = test_diffraction
        self.test_coords = test_coords
        self.test_probe = test_probe
        self.train_ground_truth = train_ground_truth
        self.test_ground_truth = test_ground_truth
