import numpy as np
from scipy.spatial import cKDTree

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

def get_neighbor_diffraction_and_positions(diff3d, xcoords, ycoords,
        xcoords_start = None, ycoords_start = None, K = 6, C = 4,
        nsamples = 10):
    """
    xcoords and ycoords: 1d coordinate arrays
    diff3d: np array of shape (B, N, N)
    """
    nn_indices = get_neighbor_indices(xcoords, ycoords, K = K)
    nn_indices = sample_rows(nn_indices, C, nsamples).reshape(-1, C)

    diff4d_nn = np.transpose(diff3d[nn_indices], [0, 2, 3, 1])
    diff4d_nn.shape

    coords_nn = np.transpose(np.array([xcoords[nn_indices], ycoords[nn_indices]]),
                [1, 0, 2])[:, None, :, :]
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
