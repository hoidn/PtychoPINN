from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf
from ptycho import params

def shuffle_data(X, Y_I, Y_phi, random_state=0):
    indices = np.arange(len(Y_I))
    indices_shuffled = shuffle(indices, random_state=random_state)

    X_shuffled = X[indices_shuffled]
    Y_I_shuffled = Y_I[indices_shuffled]
    Y_phi_shuffled = Y_phi[indices_shuffled]

    return X_shuffled, Y_I_shuffled, Y_phi_shuffled, indices_shuffled

def get_clipped_object(YY_full, outer_offset):
    borderleft, borderright, clipleft, clipright = get_clip_sizes(outer_offset)

    extra_size = (YY_full.shape[1] - (params.cfg["N"] + (params.cfg["gridsize"] - 1) * params.cfg["offset"])) % (outer_offset // 2)
    if extra_size > 0:
        YY_ground_truth = YY_full[:, :-extra_size, :-extra_size]
    else:
        print("discarding length {} from test image".format(extra_size))
        YY_ground_truth = YY_full
    YY_ground_truth = YY_ground_truth[:, clipleft:-clipright, clipleft:-clipright]
    return YY_ground_truth

def get_clip_sizes(outer_offset):
    N = params.cfg["N"]
    gridsize = params.cfg["gridsize"]
    offset = params.cfg["offset"]
    bordersize = (N - outer_offset / 2) / 2
    borderleft = int(np.ceil(bordersize))
    borderright = int(np.floor(bordersize))
    clipsize = bordersize + ((gridsize - 1) * offset) // 2
    clipleft = int(np.ceil(clipsize))
    clipright = int(np.floor(clipsize))
    return borderleft, borderright, clipleft, clipright

def stitch_data(b, norm_Y_I_test=1, norm=True, part="amp", outer_offset=None, nimgs=None):
    if nimgs is None:
        nimgs = params.get("nimgs_test")
    if outer_offset is None:
        outer_offset = params.get("outer_offset_test")
    nsegments = int(np.sqrt((int(tf.size(b)) / nimgs) / (params.cfg["N"]**2)))
    if part == "amp":
        getpart = np.absolute
    elif part == "phase":
        getpart = np.angle
    elif part == "complex":
        def getpart(x):
            return x
    else:
        raise ValueError
    if norm:
        img_recon = np.reshape(
            (norm_Y_I_test * getpart(b)),
            (-1, nsegments, nsegments, params.cfg["N"], params.cfg["N"], 1))
    else:
        img_recon = np.reshape(
            (getpart(b)),
            (-1, nsegments, nsegments, params.cfg["N"], params.cfg["N"], 1))
    borderleft, borderright, clipleft, clipright = get_clip_sizes(outer_offset)
    img_recon = img_recon[:, :, :, borderleft:-borderright, borderleft:-borderright, :]
    tmp = img_recon.transpose(0, 1, 3, 2, 4, 5)
    stitched = tmp.reshape(-1, np.prod(tmp.shape[1:3]), np.prod(tmp.shape[1:3]), 1)
    return stitched

def reassemble(b, norm_Y_I=1.0, part="amp", **kwargs):
    stitched = stitch_data(b, norm_Y_I, norm=False, part=part, **kwargs)
    return stitched

def process_simulated_data(X_train, Y_I_train, Y_phi_train, X_test, Y_I_test, Y_phi_test, YY_test_full, outer_offset_test):
    X_train, Y_I_train, Y_phi_train, indices_shuffled = shuffle_data(np.array(X_train), np.array(Y_I_train), np.array(Y_phi_train))
    if params.get("outer_offset_train") is not None:
        YY_ground_truth_all = get_clipped_object(YY_test_full, outer_offset_test)
        YY_ground_truth = YY_ground_truth_all[0, ...]
        print("DEBUG: generating grid-mode ground truth image")
    else:
        YY_ground_truth = None
        print("DEBUG: No ground truth image in non-grid mode")
    return X_train, Y_I_train, Y_phi_train, YY_ground_truth
