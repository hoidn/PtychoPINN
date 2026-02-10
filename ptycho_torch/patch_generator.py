
import numpy as np
from scipy.spatial import cKDTree, KDTree
from scipy.sparse import coo_matrix
from ptycho_torch.config_params import DataConfig, ModelConfig # Removed TrainingConfig
from typing import Tuple, Optional
#All methods for patch generation that used to be in loader will go here
#Will be imported into dset_loader for generating patches of grid_size ** 2

def group_coords(xcoords_full, ycoords_full,
                 xcoords_bounded, ycoords_bounded,
                 neighbor_function,
                 valid_mask,
                 data_config: DataConfig, C: int = None): # Added data_config
    """
    Assemble a flat dataset into solution regions using nearest-neighbor grouping.
    ---
    Inputs:
    xcoords_full - full unbounded x coordinate set from raw data
    y_coords_full - same as x, see above
    xcoords_bounded - bounded x coordinate set to trim edges
    ycoords_bounded - same as y
    valid_mask - mask on xcoords_full to get us our indices of interest
    neighbor_function - nearest neighbor calculation function, need to consolidate these all into one call
    data_config - DataConfig class

    Returns:
        nn_indices: shape (M, C), M = total number of solution regions
        coords_nn: shape (M, C, 1, 2)
    """

    if C is None:
        C = data_config.C # Use config C
    #No overlaps enforced
    if C == 1:
        # For C==1, use bounded (valid) coordinates, not full coordinates
        nn_indices = get_neighbor_self_indices(xcoords_bounded,
                                               ycoords_bounded)
        # Map bounded indices to global indices using valid_mask
        nn_indices_global = valid_mask[nn_indices.flatten()].reshape(-1, 1)

        # Apply n_subsample: repeat each index n_subsample times
        # This creates multiple samples per coordinate, matching the calculate_length expectation
        n_subsample = data_config.n_subsample
        nn_indices_global = np.repeat(nn_indices_global, n_subsample, axis=0)

        # Create coords_nn with shape (N*n_subsample, 1, 1, 2) for C==1 case using global indices
        coords_nn = np.stack([xcoords_full[nn_indices_global],
                            ycoords_full[nn_indices_global]], axis=2)[:, :, None, :]
        nn_indices = nn_indices_global
    #Yes overlaps enforced
    else:
        #Various neighbor sampling procedures
        if data_config.neighbor_function == '4_quadrant':
            _, nn_indices, coords_nn = get_fixed_quadrant_neighbors_c4(xcoords_full, ycoords_full,
                                                                       xcoords_bounded, ycoords_bounded,
                                                                       valid_mask,
                                                                       data_config)

        else:
            print('begin nn_indices')
            # Get neighbor indices in bounded coordinate space
            nn_indices_bounded = neighbor_function(xcoords_bounded, ycoords_bounded,
                                                  data_config, K=data_config.K)
            print('nn_indices_bounded_shape', nn_indices_bounded.shape)
            
            # CRITICAL FIX: Map bounded indices back to global indices
            nn_indices_global = map_bounded_to_global_indices(nn_indices_bounded, valid_mask)
            
            # Sample and reshape
            nn_indices = sample_rows(nn_indices_global, C, data_config.n_subsample).reshape(-1, C)

            # Get final array of coordinates using GLOBAL indices
            coords_nn = np.stack([xcoords_full[nn_indices],
                                ycoords_full[nn_indices]], axis=2)[:, :, None, :]

    
    return nn_indices, coords_nn

def get_neighbor_self_indices(xcoords, ycoords):
    """
    Assign each pattern index to itself
    """
    N = len(xcoords)
    nn_indices = np.arange(N).reshape(N, 1) 
    return nn_indices

def get_neighbor_indices(xcoords, ycoords, data_config, K = 6):
    # Combine x and y coordinates into a single array
    points = np.column_stack((xcoords, ycoords))

    # Create a KDTree
    tree = cKDTree(points)

    # Query for K nearest neighbors for each point
    distances, nn_indices = tree.query(points, k=K+1)  # +1 because the point itself is included in the results
    return nn_indices

def get_neighbors_indices_within_bounds(xcoords, ycoords,
                                        data_config, K):
    """
    Finds neighbors using KDTree.sparse_distance_matrix. Can apply min and max distance metrics

    Args:
        points (np.ndarray): Array of shape [N, 2] with coordinates.
        min_dist (float): Minimum distance for neighbors.
        max_dist (float): Maximum distance for neighbors.
        K (int): Maximum number of neighbors per point.

    Returns:
        dict: A dictionary where keys are point indices (0 to N-1) and
              values are tuples (neighbor_indices, neighbor_distances).
              neighbor_indices and neighbor_distances are sorted NumPy arrays.
    """

    #Defining vars
    points = np.column_stack((xcoords, ycoords))
    N = points.shape[0]
    if N == 0:
        return {}
    min_dist = data_config.min_neighbor_distance
    max_dist = data_config.max_neighbor_distance

    tree = KDTree(points)

    # Compute a sparse matrix (dictionary of keys format) containing pairs (i, j)
    # where distance(points[i], points[j]) <= max_dist AND i < j.
    # The value associated with the key (i, j) is the distance.
    sparse_dist_matrix_dok = tree.sparse_distance_matrix(tree, max_distance=max_dist)

    # Use lists to collect potential neighbors for each point before sorting and capping
    # Store tuples of (distance, neighbor_index)
    neighbors_per_point = [[] for _ in range(N)]

    # Iterate through the items (key-value pairs) of the DOK matrix
    # Key is a tuple (i, j), value is the distance dist
    for (i, j), dist in sparse_dist_matrix_dok.items():
        # Apply minimum distance filter.
        # DOK from sparse_distance_matrix usually stores i < j pairs.
        if dist >= min_dist:
            # Add j as a neighbor for i, and i as a neighbor for j
            neighbors_per_point[i].append((dist, j))
            neighbors_per_point[j].append((dist, i)) # Add the symmetric pair
    
    # Initialize the result array with the placeholder
    result_indices = np.full((N, K), -1, dtype=int)

    # Process each point's collected neighbors
    for i in range(N):
        candidates = neighbors_per_point[i]
        if not candidates:
            continue # Row already filled with placeholders

        # Sort candidates by distance (ascending)
        candidates.sort(key=lambda x: x[0])

        # Extract indices of sorted neighbors
        sorted_neighbor_indices = [idx for dist, idx in candidates]

        # Determine how many neighbors to fill (up to K)
        num_neighbors_found = len(sorted_neighbor_indices)
        num_to_fill = min(K, num_neighbors_found)

        if num_to_fill > 0:
            result_indices[i, :num_to_fill] = sorted_neighbor_indices[:num_to_fill]

    return result_indices

def map_bounded_to_global_indices(bounded_indices, valid_mask):
    """
    Maps indices from bounded coordinate space back to global coordinate space.
    
    Args:
        bounded_indices: Array of shape (N, K) with indices relative to bounded arrays
        valid_mask: Boolean mask or index array that maps bounded positions to global positions
    
    Returns:
        global_indices: Array of shape (N, K) with indices relative to global arrays
    """
    # Get the global indices corresponding to valid (bounded) positions
    if valid_mask.dtype == bool:
        # If valid_mask is boolean, get the indices where it's True
        global_position_map = np.where(valid_mask)[0]
    else:
        # If valid_mask is already an index array
        global_position_map = valid_mask
    
    # Map bounded indices to global indices
    global_indices = np.full_like(bounded_indices, -1)
    
    for i in range(bounded_indices.shape[0]):
        for j in range(bounded_indices.shape[1]):
            bounded_idx = bounded_indices[i, j]
            if bounded_idx >= 0 and bounded_idx < len(global_position_map):
                global_indices[i, j] = global_position_map[bounded_idx]
            else:
                global_indices[i, j] = -1  # Invalid index marker
    
    return global_indices


def get_fixed_quadrant_neighbors_c4(
    xcoords: np.ndarray,
    ycoords: np.ndarray,
    xcoords_bounded: np.ndarray,
    ycoords_bounded: np.ndarray,
    valid_mask: np.ndarray,
    data_config: DataConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    Finds groups of 4 neighbors for central points, enforcing a fixed spatial
    quadrant layout (TL, TR, BL, BR) relative to the center.

    Selects one random candidate if multiple neighbors exist in a quadrant.
    Discards central points if neighbors in all four quadrants are not found
    within the specified distance bounds and K-neighbor search.

    Searches for neighbors within all of xcoords, ycoords.
    However, only iterates through points within the bounded arrays which are valid.
    Therefore, we can pick 

    Assumes a standard Cartesian coordinate system where:
        +y is UP
        -y is DOWN
        -x is LEFT
        +x is RIGHT
    Therefore:
        TL: dx < 0, dy > 0
        TR: dx > 0, dy > 0
        BL: dx < 0, dy < 0
        BR: dx > 0, dy < 0

    Args:
        xcoords (np.ndarray): 1D array of x-coordinates for all points. Full xcoords from npz file
        ycoords (np.ndarray): 1D array of y-coordinates for all points. Full ycoords from npz file
        xcoords_bounded (np.ndarray): 1D array of bounded x coordinates, based on specified bounds
        ycoords_bounded (np.darray): 1D array of bounded y coordinates, see above
        data_config (DataConfig): Configuration object containing attributes like
                                   min_neighbor_distance, max_neighbor_distance.
        k_neighbors_query (int): How many nearest neighbors to initially query
                                 using KDTree for each central point. Should be
                                 sufficiently large to find candidates in all
                                 quadrants (e.g., 10-20+).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - final_center_indices: Indices of central points (shape M,) for which
                                    a valid group was found.
            - final_neighbor_indices: Indices of the 4 neighbors ordered consistently
                                     [TL, TR, BL, BR] (shape M, 4).
            - final_coords_nn: Coordinates of the neighbors (shape M, 4, 1, 2).
    """
    k_neighbors_query = getattr(data_config, 'K_quadrant', 60)

    if k_neighbors_query <= 4:
        raise ValueError("k_neighbors_query must be greater than 4 to potentially find neighbors in all quadrants.")

    points = np.column_stack((xcoords, ycoords))
    points_bounded = np.column_stack((xcoords_bounded, ycoords_bounded))
    n_points_bounded = points_bounded.shape[0]

    min_dist = getattr(data_config, 'min_neighbor_distance', 0.0)
    max_dist = getattr(data_config, 'max_neighbor_distance', np.inf)

    print(f"Building KDTree for {points.shape[0]} points...")
    tree = cKDTree(points)

    valid_center_indices = [] # Stores index of the center element which you used to look
    valid_neighbor_groups = [] # Stores the [TL, TR, BL, BR] indices

    #Quadrant order
    quad_order = ["TL", "TR", "BL", "BR"]

    print(f"Processing {n_points_bounded} potential central points...")

    for i in range(n_points_bounded):
        center_coord = points_bounded[i]

        # Query K nearest neighbors (include self initially)
        # Increase k slightly just in case some neighbours are filtered out by distance
        query_k = k_neighbors_query + 1
        distances, indices = tree.query(center_coord, k=query_k)

        # --- Filter Neighbors ---
        valid_neighbor_indices = []
        for k_idx, dist in zip(indices, distances):
            if k_idx == i:
                continue
            
            if min_dist <= dist < max_dist:
                valid_neighbor_indices.append(k_idx)

        if not valid_neighbor_indices:
            print(f"Point {i}: No neighbors passed distance filters.")
            continue # Skip quadrant check if no neighbors passed distance

        # --- Categorize by Quadrant ---
        quadrant_candidates = {q: [] for q in quad_order}
        neighbor_coords = points[valid_neighbor_indices]
        dx = neighbor_coords[:, 0] - center_coord[0]
        dy = neighbor_coords[:, 1] - center_coord[1]
        
        # Reshaping bound definitions based on data_config
        # Need to make this even more robust in future
        if data_config.scan_pattern == 'Isotropic':
            x_lower_bound, x_upper_bound = 0, 10 #Arbitrary large value
            y_lower_bound, y_upper_bound = 0, 10
            y_bound = y_lower_bound
        elif data_config.scan_pattern == 'Rectangular':
            x_lower_bound, x_upper_bound = 0, 12
            y_lower_bound, y_upper_bound = 0.7, 2
            y_bound = y_upper_bound

        # bound = 0.7
        # x_bound = 12
        # y_bound = 2

        #Deciding quadrants

        for neighbor_idx, d_x, d_y in zip(valid_neighbor_indices, dx, dy):
            if d_x < x_lower_bound and d_x > -x_upper_bound and d_y > -y_lower_bound and d_y < y_upper_bound:
                quadrant_candidates["TL"].append(neighbor_idx)
            elif d_x > x_lower_bound and d_x < x_upper_bound and d_y > -y_lower_bound and d_y < y_upper_bound:
                quadrant_candidates["TR"].append(neighbor_idx)
            elif d_x < x_lower_bound and d_x > -x_upper_bound and d_y < -y_bound:
                quadrant_candidates["BL"].append(neighbor_idx)
            elif d_x > x_lower_bound and d_x < x_upper_bound and d_y < -y_bound:
                quadrant_candidates["BR"].append(neighbor_idx)
            # Ignore points exactly on axes relative to the center

        #Add center point in there also
        #Need to append the true index of the point in "points", not in "points_bounded"
        for quad in quad_order:
            quadrant_candidates[quad].append(valid_mask[i])

        # # #Check failure conditiont
        # # #Check if center is the only value in at least 2 quadrants
        num_quads_with_only_center = sum(1 for quad in quad_order if len(quadrant_candidates[quad]) == 1)
        if num_quads_with_only_center >= 2:
            print(f"quadrant candidates TL: {quadrant_candidates['TL']}")
            print(f"quadrant candidates TR: {quadrant_candidates['TR']}")
            print(f"quadrant candidates BL: {quadrant_candidates['BL']}")
            print(f"quadrant candidates BR: {quadrant_candidates['BR']}")
            continue
        if not all(quadrant_candidates.values()):
            continue

        # --- Check Availability and Select ---
        for _ in range(data_config.n_subsample):
            center_count = 2
            while center_count > 1:
                idx_tl = np.random.choice(quadrant_candidates["TL"])
                idx_tr = np.random.choice(quadrant_candidates["TR"])
                idx_bl = np.random.choice(quadrant_candidates["BL"])
                idx_br = np.random.choice(quadrant_candidates["BR"])

                candidate_set = [idx_tl, idx_tr, idx_bl, idx_br]
                center_count = candidate_set.count(i)

            valid_center_indices.append(i)
            valid_neighbor_groups.append(candidate_set)

    print(f"Finished quadrant neighbor processing. Found {len(valid_center_indices)} valid C=4 groups.")

    # Convert results to NumPy arrays
    final_center_indices = np.array(valid_center_indices, dtype=int)
    final_neighbor_indices = np.array(valid_neighbor_groups, dtype=int) # Shape (M, 4)

    # Fetch coordinates for the final neighbor indices
    # Need careful indexing here
    # final_neighbor_indices is (M, 4). points[final_neighbor_indices] will be (M, 4, 2)
    coords_for_groups = points[final_neighbor_indices] # Shape (M, 4, 2)
    # Reshape to match expected output (M, 4, 1, 2)
    final_coords_nn = coords_for_groups[:, :, np.newaxis, :]

    return final_center_indices, final_neighbor_indices, final_coords_nn


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
