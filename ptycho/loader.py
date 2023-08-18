import numpy as np
import tensorflow as tf
from scipy.spatial import cKDTree

from .params import params

def get_neighbor_indices(xcoords, ycoords, K = 3):
    # Combine x and y coordinates into a single array
    points = np.column_stack((xcoords, ycoords))

    # Create a KDTree
    tree = cKDTree(points)

    # Query for K nearest neighbors for each point
    distances, nn_indices = tree.query(points, k=K+1)  # +1 because the point itself is included in the results
    return nn_indices

def sample_rows(indices, n, m):
    N = indices.shape[0]
    result = np.zeros((N, m, n), dtype=int)
    for i in range(N):
        result[i] = np.array([np.random.choice(indices[i], size=n, replace=False) for _ in range(m)])
    return result

def get_relative_coords(coords_nn):
    assert len(coords_nn.shape) == 4
    coords_offsets = np.mean(coords_nn, axis = 3)[..., None]
    coords_relative = coords_nn - np.mean(coords_nn, axis = 3)[..., None]
    return coords_offsets, coords_relative

def crop12(arr, size):
    N, M = arr.shape[1:3]
    return arr[:, N // 2 - (size) // 2: N // 2+ (size) // 2, N // 2 - (size) // 2: N // 2 + (size) // 2, ...]

def get_image_patches(gt_image, global_offsets, local_offsets):
    from . import tf_helper as hh
    gridsize = params()['gridsize']
    N = params()['N']
    B = global_offsets.shape[0] #* gridsize**2

    gt_repeat = tf.repeat(
        tf.repeat(gt_image[None, ...], B, axis = 0)[..., None],
        gridsize**2, axis = 3)

    gt_repeat_f = hh._channel_to_flat(gt_repeat)

    offsets_c = tf.cast(tf.squeeze(global_offsets + local_offsets), tf.float32)[:, None, :, :]

    offsets_f = hh._channel_to_flat(offsets_c)

    gt_translated = hh._flat_to_channel(
        crop12(hh.translate(tf.squeeze(gt_repeat_f)[..., None], tf.squeeze(offsets_f)), N)
    )
    return gt_translated

def get_neighbor_diffraction_and_positions(diff3d, xcoords, ycoords,
        xcoords_start = None, ycoords_start = None, K = 6, C = None,
        nsamples = 10):
    """
    xcoords and ycoords: 1d coordinate arrays
    diff3d: np array of shape (B, N, N)
    """
    gridsize = params()['gridsize']
    if C is None:
        C = gridsize**2
    nn_indices = get_neighbor_indices(xcoords, ycoords, K = K)
    nn_indices = sample_rows(nn_indices, C, nsamples).reshape(-1, C)

    diff4d_nn = np.transpose(diff3d[nn_indices], [0, 2, 3, 1])
    coords_nn = np.transpose(np.array([xcoords[nn_indices], ycoords[nn_indices]]),
                [1, 0, 2])[:, None, :, :]
    # Do this instead if you don't want random sampling
#    diff4d_nn = diff3d[..., None]
#    coords_nn = np.transpose(np.array([xcoords, ycoords]))[:, None, :, None]
    coords_offsets, coords_relative = get_relative_coords(coords_nn)

    if xcoords_start is not None:
        coords_start_nn = np.transpose(np.array([xcoords_start[nn_indices], ycoords_start[nn_indices]]),
                    [1, 0, 2])[:, None, :, :]
        coords_start_offsets, coords_start_relative = get_relative_coords(coords_start_nn)
    else:
        coords_start_offsets = coords_start_relative = None
    return {'diffraction': diff4d_nn,
        'coords_offsets': coords_offsets,
        'coords_relative': coords_relative,
        'coords_start_offsets': coords_start_offsets,
        'coords_start_relative': coords_start_relative,
        'coords_nn': coords_nn,
        'coords_start_nn': coords_start_nn}
