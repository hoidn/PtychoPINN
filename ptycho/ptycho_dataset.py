class PtychoDataset:
    def __init__(self, train_X, train_coords, train_probe, test_X, test_coords, test_probe,
                 train_Y=None, test_Y=None):
        """
        Initializes the PtychoDataset object with training and evaluation data.

        Args:
            train_X (np.ndarray): 4D array of training diffraction patterns.
            train_coords (np.ndarray): 3D array of training scan point coordinates.
            train_probe (np.ndarray): 2D complex-valued array representing the probe function for trai
ning.
            test_X (np.ndarray): 4D array of evaluation diffraction patterns.
            test_coords (np.ndarray): 3D array of evaluation scan point coordinates.
            test_probe (np.ndarray): 2D complex-valued array representing the probe function for evalu
ation.
            train_Y (np.ndarray): 2D complex-valued array representing the ground truth for training. 
Optional.
            test_Y (np.ndarray): 2D complex-valued array representing the ground truth for evaluation.
 Optional.
        """
        self.data = {
            'train': {
                'X': train_X,
                'coords': train_coords,
                'probe': train_probe,
                'Y': train_Y
            },
            'test': {
                'X': test_X,
                'coords': test_coords,
                'probe': test_probe,
                'Y': test_Y
            }
        }
