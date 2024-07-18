
import numpy as np
from scipy.spatial import cKDTree
#All methods for patch generation that used to be in loader will go here
#Will be imported into dset_loader for generating patches of grid_size ** 2

def get_neighbor_diffraction_and_positions(PtychoDataset, index, N, K=6, C=None, nsamples=10):
    """
    Returns, for a single experimental dataset, the relative offsets and coordinates for nearest neighboring groups of diffraction
    images of stack size grid_size ** 2. Can be iteratively run to calculate for all experiments

    Difference from PtychoPINN V1 (tf_helper):
    diff4d_nn is not calculated in this function. We will be calculating diff4d_nn when writing
    the memmap. 
    ---
    Inputs:
    PtychoDataset - Dataset object from dset_loader. Contains all relevant dataset information
        - image_map: shape (N x M* x H x W). M* is the number of diffraction patterns per
          experiment. N is # of diffraction experiments
    Index - Index of diffraction experiment to be used. First dimension of any tensor in dataset



    """
    
    nn_indices, coords_nn = group_coords(PtychoDataset, index,
                                         K = K, C = C, nsamples = nsamples)


    #diff4d_nn = PtychoDataset.data_dict['diffraction'][index][nn_indices]

    coords_offsets, coords_relative = get_relative_coords(coords_nn)

    if PtychoDataset.data_dict['xstart'][index] is not None:
        #Dimensions: M* x C x 1 x 2
        coords_start_nn = np.stack(PtychoDataset[index].xcoords_start[nn_indices],
                                   PtychoDataset[index].ycoords_start[nn_indices],
                                   axis = 2)[:, :, None, :]
        #coords_start_nn = coords_start_nn[:, :, ::-1, :]
        coords_start_offsets, coords_start_relative = get_relative_coords(coords_start_nn)
    else:
        coords_start_offsets = coords_start_relative = None

    #Adding other relevant parameters to data dict
    PtychoDataset.data_dict['coords_offsets'] = coords_offsets
    PtychoDataset.data_dict['coords_relative'] = coords_relative
    PtychoDataset.data_dict['coords_start_offsets'] = coords_start_offsets
    PtychoDataset.data_dict['coords_start_relative'] = coords_start_relative
    PtychoDataset.data_dict['coords_nn'] = coords_nn
    PtychoDataset.data_dict['coords_start_nn'] = coords_start_nn
    PtychoDataset.data_dict['nn_indices'] = nn_indices

    dset = {
        'diffraction': diff4d_nn,
        'coords_offsets': coords_offsets,
        'coords_relative': coords_relative,
        'coords_start_offsets': coords_start_offsets,
        'coords_start_relative': coords_start_relative,
        'coords_nn': coords_nn,
        'coords_start_nn': coords_start_nn,
        'nn_indices': nn_indices,
        'objectGuess': ptycho_data.objectGuess
    }
    X_full = normalize_data(dset, N)
    dset['X_full'] = X_full
    print('neighbor-sampled diffraction shape', X_full.shape)
    return dset

def group_coords(xcoords, ycoords, params, C):
    """
    Assemble a flat dataset into solution regions using nearest-neighbor grouping.
    ---
    Inputs:
    PtychoDataset - Dataset object from dset_loader. Contains all relevant dataset information
    Index - Index of diffraction experiment to be used. First dimension of any tensor in dataset
    K - Number of nearest neighbors to select from
    C - Number of total images in a single solution region (i.e. K choose C for a single solution region)
    nsamples - Number of distinct solution regions to sample from the K nearest neighbors
               (e.g.) nsamples = 10, K = 6, C = 4. 6C4 = 15 but only 10 will be selected

    Returns:
        nn_indices: shape (M, C), M = total number of solution regions
        coords_nn: shape (M, C, 1, 2)
    """
    if C is None:
        C = params['n_images']
    #No overlaps enforced
    if C == 1:
        nn_indices = get_neighbor_self_indices(xcoords,
                                               ycoords)
    #Yes overlaps enforced
    else:
        nn_indices = get_neighbor_indices(xcoords,
                                          ycoords, K=params['K'])
        nn_indices = sample_rows(nn_indices, C, params['n_subsample']).reshape(-1, C)

    #Get final array of coordinates (M* x C x 1 x 2)
    coords_nn = np.stack([xcoords[nn_indices],
                          ycoords[nn_indices]],axis=2)[:, :, None, :]
    
    return nn_indices, coords_nn

def get_neighbor_self_indices(xcoords, ycoords):
    """
    Assign each pattern index to itself
    """
    N = len(xcoords)
    nn_indices = np.arange(N).reshape(N, 1) 
    return nn_indices

def get_neighbor_indices(xcoords, ycoords, K = 6):
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

def get_relative_coords(coords_nn, local_offset_sign=1):
    """
    Calculate the relative coordinates and offsets from the nearest neighbor coordinates.

    Args:
        coords_nn (np.ndarray): Array of nearest neighbor coordinates with shape (M, C, 1, 2).

    Returns:
        tuple: A tuple containing coords_offsets and coords_relative.
        coords_relative: Array of relative coordinates with shape (M, C, 1, 2).
        coords_offsets: Array of offsets with shape (M, 1, 1, 2).
    """
    assert len(coords_nn.shape) == 4
    #Center of mass coordinate for every combination of coordinates
    coords_offsets = np.mean(coords_nn, axis=1)[:, None, :, :]
    #Subtract center of mass coordinate from every coordinate to get relative coordinate
    #coords_offsets is broadcast to match second dimension of coords_nn
    coords_relative = local_offset_sign * (coords_nn - coords_offsets)

    return coords_offsets, coords_relative


