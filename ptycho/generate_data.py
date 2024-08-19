import numpy as np
from .data_preprocessing import generate_data
from . import params as p

def main(probeGuess = None):
    X_train, Y_I_train, Y_phi_train, X_test, Y_I_test, Y_phi_test, YY_ground_truth, ptycho_dataset, YY_test_full, norm_Y_I_test = generate_data(probeGuess)
    print(np.linalg.norm(ptycho_dataset.train_data.X[0]) / np.linalg.norm(np.abs(ptycho_dataset.train_data.Y[0])))
    return X_train, Y_I_train, Y_phi_train, X_test, Y_I_test, Y_phi_test, YY_ground_truth, ptycho_dataset, YY_test_full, norm_Y_I_test

X_train, Y_I_train, Y_phi_train, X_test, Y_I_test, Y_phi_test, YY_ground_truth, ptycho_dataset, YY_test_full, norm_Y_I_test = main(probeGuess = p.get('probe'))
