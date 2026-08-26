"""DEPRECATED: Legacy data wrapper. Use ptycho.data_preprocessing directly.

The former import-time simulation side effect (W3.2) is gone: callers invoke
:func:`run` explicitly and read fields off the returned namespace.
"""
import numpy as np
from types import SimpleNamespace

from .data_preprocessing import generate_data
from . import params as p

_FIELDS = (
    "X_train", "Y_I_train", "Y_phi_train", "X_test", "Y_I_test", "Y_phi_test",
    "YY_ground_truth", "ptycho_dataset", "YY_test_full", "norm_Y_I_test",
)

_result = None


# TODO passing the probe should be mandatory, to enforce side-effect free behavior.
def main(probeGuess = None):
    X_train, Y_I_train, Y_phi_train, X_test, Y_I_test, Y_phi_test, YY_ground_truth, ptycho_dataset, YY_test_full, norm_Y_I_test = generate_data(probeGuess)
    print(np.linalg.norm(ptycho_dataset.train_data.X[0]) / np.linalg.norm(np.abs(ptycho_dataset.train_data.Y[0])))
    return X_train, Y_I_train, Y_phi_train, X_test, Y_I_test, Y_phi_test, YY_ground_truth, ptycho_dataset, YY_test_full, norm_Y_I_test


def run(probeGuess=None, force=False):
    """Run the legacy simulation once and memoize the result namespace."""
    global _result
    if _result is None or force:
        if probeGuess is None:
            probeGuess = p.get('probe')
        _result = SimpleNamespace(**dict(zip(_FIELDS, main(probeGuess))))
    return _result
