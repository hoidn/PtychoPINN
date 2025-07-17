import numpy as np
import tensorflow as tf
from typing import Tuple, Optional
from scipy.spatial import cKDTree
from ptycho import params
from ptycho.autotest.debug import debug
from ptycho import diffsim as datasets
from ptycho import tf_helper as hh

# Constants, # TODO cleanup / refactor
local_offset_sign = 1
key_coords_offsets = 'coords_start_offsets'
key_coords_relative = 'coords_start_relative'

class RawData:
    #@debug
    def __init__(self, xcoords, ycoords, xcoords_start, ycoords_start, diff3d, probeGuess,
             scan_index, objectGuess = None, Y = None, norm_Y_I = None):
        # Sanity checks
        self._check_data_validity(xcoords, ycoords, xcoords_start, ycoords_start, diff3d,
                    probeGuess, scan_index)

        # TODO these should go in the data validation method
        assert len(xcoords.shape) == 1, f"Expected xcoords to be 1D, got shape {xcoords.shape}"
        assert len(ycoords.shape) == 1, f"Expected ycoords to be 1D, got shape {ycoords.shape}"
        assert len(xcoords_start.shape) == 1, f"Expected xcoords_start to be 1D, got shape {xcoords_start.shape}"
        assert len(ycoords_start.shape) == 1, f"Expected ycoords_start to be 1D, got shape {ycoords_start.shape}"
        if diff3d is not None:
            assert len(diff3d.shape) == 3, f"Expected diff3d to be 3D, got shape {diff3d.shape}"
            print(f"diff3d shape: {diff3d.shape}")
            assert diff3d.shape[1] == diff3d.shape[2]
        if probeGuess is not None:
            assert len(probeGuess.shape) == 2, f"Expected probeGuess to be 2D, got shape {probeGuess.shape}"
            print(f"probeGuess shape: {probeGuess.shape}")
        if scan_index is not None:
            assert len(scan_index.shape) == 1, f"Expected scan_index to be 1D, got shape {scan_index.shape}"
            print(f"scan_index shape: {scan_index.shape}")
        if objectGuess is not None:
            print(f"objectGuess shape: {objectGuess.shape}")
            assert len(objectGuess.shape) == 2

        print(f"xcoords shape: {xcoords.shape}")
        print(f"ycoords shape: {ycoords.shape}")
        print(f"xcoords_start shape: {xcoords_start.shape}")
        print(f"ycoords_start shape: {ycoords_start.shape}")

        # Assigning values if checks pass
        self.xcoords = xcoords
        self.ycoords = ycoords
        self.xcoords_start = xcoords_start
        self.ycoords_start = ycoords_start
        self.diff3d = diff3d
        self.probeGuess = probeGuess
        self.scan_index = scan_index
        self.objectGuess = objectGuess
        # TODO validity checks
        self.Y = Y
        self.norm_Y_I = norm_Y_I

    @staticmethod
    #@debug
    def from_coords_without_pc(xcoords, ycoords, diff3d, probeGuess, scan_index,
                               objectGuess=None):
        """
        Static method to create a RawData instance without separate start coordinates.
        The start coordinates are set to be the same as the xcoords and ycoords.

        Args:
            xcoords (np.ndarray): x coordinates of the scan points.
            ycoords (np.ndarray): y coordinates of the scan points.
            diff3d (np.ndarray): diffraction patterns.
            probeGuess (np.ndarray): initial guess of the probe function.
            scan_index (np.ndarray): array indicating the scan index for each diffraction pattern.
            objectGuess (np.ndarray, optional): initial guess of the object. Defaults to None.

        Returns:
            RawData: An instance of the RawData class.
        """
        return RawData(xcoords, ycoords, xcoords, ycoords, diff3d, probeGuess, scan_index, objectGuess)

    @staticmethod
    def from_simulation(xcoords, ycoords, probeGuess,
                 objectGuess, scan_index = None):
        """
        Create a RawData instance from simulation data.

        Args:
            xcoords (np.ndarray): x coordinates of the scan points.
            ycoords (np.ndarray): y coordinates of the scan points.
            probeGuess (np.ndarray): initial guess of the probe function.
            objectGuess (np.ndarray): initial guess of the object.
            scan_index (np.ndarray, optional): array indicating the scan index for each diffraction pattern.

        Returns:
            RawData: An instance of the RawData class with simulated data.
        """
        from ptycho.diffsim import illuminate_and_diffract
        xcoords_start = xcoords
        ycoords_start = ycoords
        global_offsets, local_offsets, nn_indices = calculate_relative_coords(
                    xcoords, ycoords)

        Y_obj = get_image_patches(objectGuess, global_offsets, local_offsets) 
        Y_I = tf.math.abs(Y_obj)
        Y_phi = tf.math.angle(Y_obj)
        X, Y_I_xprobe, Y_phi_xprobe, intensity_scale = illuminate_and_diffract(Y_I, Y_phi, probeGuess)
        norm_Y_I = datasets.scale_nphotons(X)
        assert X.shape[-1] == 1, "gridsize must be set to one when simulating in this mode"
        # TODO RawData should have a method for generating the illuminated ground truth object
        return RawData(xcoords, ycoords, xcoords_start, ycoords_start, tf.squeeze(X).numpy(),
                       probeGuess, scan_index, objectGuess,
                       Y = Y_obj.numpy(),
                       norm_Y_I = norm_Y_I)

    #@debug
    def __str__(self):
        parts = [
            "RawData:",
            f"  xcoords: {self.xcoords.shape if self.xcoords is not None else 'None'}",
            f"  ycoords: {self.ycoords.shape if self.ycoords is not None else 'None'}",
            f"  xcoords_start: {self.xcoords_start.shape if self.xcoords_start is not None else 'None'}",
            f"  ycoords_start: {self.ycoords_start.shape if self.ycoords_start is not None else 'None'}",
            f"  diff3d: {self.diff3d.shape if self.diff3d is not None else 'None'}",
            f"  probeGuess: {self.probeGuess.shape if self.probeGuess is not None else 'None'}",
            f"  scan_index: {self.scan_index.shape if self.scan_index is not None else 'None'}",
            f"  objectGuess: {self.objectGuess.shape if self.objectGuess is not None else 'None'}"
        ]
        return "\n".join(parts)

    #@debug
    def to_file(self, file_path: str) -> None:
        """
        Method to write the RawData object to a file using numpy.savez.

        Args:
            file_path (str): Path to the file where the data will be saved.
        """
        np.savez(file_path,
                 xcoords=self.xcoords,
                 ycoords=self.ycoords,
                 xcoords_start=self.xcoords_start,
                 ycoords_start=self.ycoords_start,
                 diff3d=self.diff3d,
                 probeGuess=self.probeGuess,
                 objectGuess=self.objectGuess,
                 scan_index=self.scan_index)

    @staticmethod
    #@debug
    def from_file(train_data_file_path: str) -> 'RawData':
        """
        Static method to create a RawData instance from a file.

        Args:
            train_data_file_path (str): Path to the file containing the data.

        Returns:
            RawData: An instance of the RawData class.
        """
        # Load training data
        train_data = np.load(train_data_file_path)
        train_raw_data = RawData(
            xcoords=train_data['xcoords'],
            ycoords=train_data['ycoords'],
            xcoords_start=train_data['xcoords_start'],
            ycoords_start=train_data['ycoords_start'],
            diff3d=train_data['diff3d'],
            probeGuess=train_data['probeGuess'],
            objectGuess=train_data['objectGuess'],
            scan_index=train_data['scan_index']
        )
        return train_raw_data

    @staticmethod
    #@debug
    def from_files(train_data_file_path, test_data_file_path):
        """
        Static method to instantiate RawData objects from training and test data files.

        The data files should be NumPy .npz files with the following keys:
        - 'xcoords': x coordinates of the scan points
        - 'ycoords': y coordinates of the scan points
        - 'xcoords_start': starting x coordinates for the scan
        - 'ycoords_start': starting y coordinates for the scan
        - 'diff3d': diffraction patterns
        - 'probeGuess': initial guess of the probe function
        - 'scan_index': array indicating the scan index for each diffraction pattern

        Args:
            train_data_file_path (str): Path to the training data file.
            test_data_file_path (str): Path to the test data file.

        Returns:
            tuple: A tuple containing the instantiated RawData objects for training and test data.
        """
        # Load training data
        train_raw_data = RawData.from_file(train_data_file_path)

        # Load test data
        test_raw_data = RawData.from_file(test_data_file_path)

        return train_raw_data, test_raw_data

    #@debug
    def generate_grouped_data(self, N, K = 7, nsamples = 1):
        """
        Generate nearest-neighbor solution region grouping.

        Args:
            N (int): Size of the solution region.
            K (int, optional): Number of nearest neighbors. Defaults to 7.
            nsamples (int, optional): Number of samples. Defaults to 1.

        Returns:
            dict: Dictionary containing grouped data.
        """
        print('DEBUG:', 'nsamples:', nsamples)
        return get_neighbor_diffraction_and_positions(self, N, K=K, nsamples=nsamples)

    #@debug
    def _check_data_validity(self, xcoords, ycoords, xcoords_start, ycoords_start, diff3d, probeGuess, scan_index):
        """
        Check if the input data is valid.

        Args:
            xcoords (np.ndarray): x coordinates of the scan points.
            ycoords (np.ndarray): y coordinates of the scan points.
            xcoords_start (np.ndarray): starting x coordinates for the scan.
            ycoords_start (np.ndarray): starting y coordinates for the scan.
            diff3d (np.ndarray): diffraction patterns.
            probeGuess (np.ndarray): initial guess of the probe function.
            scan_index (np.ndarray): array indicating the scan index for each diffraction pattern.

        Raises:
            ValueError: If coordinate arrays don't have matching shapes.
        """
        # Check if coordinate arrays have matching shapes
        if not (xcoords.shape == ycoords.shape == xcoords_start.shape == ycoords_start.shape):
            raise ValueError("Coordinate arrays must have matching shapes.")

#@debug
def calculate_relative_coords(xcoords, ycoords, K = 6, C = None, nsamples = 10):
    """
    Group scan indices and coordinates into solution regions, then
    calculate coords_offsets (global solution region coordinates) and
    coords_relative (local solution patch coords) from ptycho_data using
    the provided index_grouping_cb callback function.

    Args:
        xcoords (np.ndarray): x coordinates of the scan points.
        ycoords (np.ndarray): y coordinates of the scan points.
        K (int, optional): Number of nearest neighbors. Defaults to 6.
        C (int, optional): Number of coordinates per solution region. Defaults to None.
        nsamples (int, optional): Number of samples. Defaults to 10.

    Returns:
        tuple: A tuple containing coords_offsets, coords_relative, and nn_indices.
    """
    nn_indices, coords_nn = group_coords(xcoords, ycoords, K = K, C = C, nsamples = nsamples)
    coords_offsets, coords_relative = get_relative_coords(coords_nn)
    return coords_offsets, coords_relative, nn_indices

#@debug
def get_image_patches(gt_image, global_offsets, local_offsets, N=None, gridsize=None):
    """
    Generate and return image patches in channel format using a single canvas.

    Args:
        gt_image (tensor): Ground truth image tensor.
        global_offsets (tensor): Global offset tensor.
        local_offsets (tensor): Local offset tensor.
        N (int, optional): Patch size. If None, uses params.get('N').
        gridsize (int, optional): Grid size. If None, uses params.get('gridsize').

    Returns:
        tensor: Image patches in channel format.
    """
    # Use explicit parameters if provided, otherwise fall back to global params
    # This follows the project's hybrid modernization pattern
    N = N if N is not None else params.get('N')
    gridsize = gridsize if gridsize is not None else params.get('gridsize')
    B = global_offsets.shape[0]
    c = gridsize**2

    # Pad the ground truth image once
    gt_padded = hh.pad(gt_image[None, ..., None], N // 2)

    # Calculate the combined offsets by adding global and local offsets
    offsets_c = tf.cast((global_offsets + local_offsets), tf.float32)
    offsets_f = hh._channel_to_flat(offsets_c)

    # Create a canvas to store the extracted patches
    canvas = np.zeros((B, N, N, c), dtype=np.complex64)

    # Iterate over the combined offsets and extract patches one by one
    for i in range(B * c):
        offset = -offsets_f[i, :, :, 0]
        translated_patch = hh.translate(gt_padded, offset)
        canvas[i // c, :, :, i % c] = np.array(translated_patch)[0, :N, :N, 0]

    # Convert the canvas to a TensorFlow tensor and return it
    return tf.convert_to_tensor(canvas)

#@debug
def group_coords(xcoords: np.ndarray, ycoords: np.ndarray, K: int, C: Optional[int], nsamples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assemble a flat dataset into solution regions using nearest-neighbor grouping.

    Args:
        xcoords (np.ndarray): x coordinates of the scan points.
        ycoords (np.ndarray): y coordinates of the scan points.
        K (int): Number of nearest neighbors to consider.
        C (Optional[int]): Number of coordinates per solution region. If None, uses gridsize^2.
        nsamples (int): Number of samples to generate.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - nn_indices: shape (M, C)
            - coords_nn: shape (M, 1, 2, C)
    """
    gridsize = params.get('gridsize')
    if C is None:
        C = gridsize**2
    if C == 1:
        nn_indices = get_neighbor_self_indices(xcoords, ycoords)
    else:
        nn_indices = get_neighbor_indices(xcoords, ycoords, K=K)
        nn_indices = sample_rows(nn_indices, C, nsamples).reshape(-1, C)

    coords_nn = np.transpose(np.array([xcoords[nn_indices],
                            ycoords[nn_indices]]),
                            [1, 0, 2])[:, None, :, :]
    return nn_indices, coords_nn[:, :, :, :]

#@debug
def get_relative_coords(coords_nn):
    """
    Calculate the relative coordinates and offsets from the nearest neighbor coordinates.

    Args:
        coords_nn (np.ndarray): Array of nearest neighbor coordinates with shape (M, 1, 2, C).

    Returns:
        tuple: A tuple containing coords_offsets and coords_relative.
    """
    assert len(coords_nn.shape) == 4
    coords_offsets = np.mean(coords_nn, axis=3)[..., None]
    coords_relative = local_offset_sign * (coords_nn - coords_offsets)
    return coords_offsets, coords_relative

#@debug
def get_neighbor_self_indices(xcoords, ycoords):
    """
    Assign each pattern index to itself.

    Args:
        xcoords (np.ndarray): x coordinates of the scan points.
        ycoords (np.ndarray): y coordinates of the scan points.

    Returns:
        np.ndarray: Array of self-indices.
    """
    N = len(xcoords)
    nn_indices = np.arange(N).reshape(N, 1) 
    return nn_indices

#@debug
def get_neighbor_indices(xcoords, ycoords, K = 3):
    """
    Get K nearest neighbor indices for each point.

    Args:
        xcoords (np.ndarray): x coordinates of the scan points.
        ycoords (np.ndarray): y coordinates of the scan points.
        K (int, optional): Number of nearest neighbors to find. Defaults to 3.

    Returns:
        np.ndarray: Array of nearest neighbor indices.
    """
    # Combine x and y coordinates into a single array
    points = np.column_stack((xcoords, ycoords))

    # Create a KDTree
    tree = cKDTree(points)

    # Query for K nearest neighbors for each point
    distances, nn_indices = tree.query(points, k=K+1)  # +1 because the point itself is included in the results
    return nn_indices

#@debug
def sample_rows(indices, n, m):
    """
    Sample rows from the given indices.

    Args:
        indices (np.ndarray): Array of indices to sample from.
        n (int): Number of samples per row.
        m (int): Number of rows to generate.

    Returns:
        np.ndarray: Sampled indices array.
    """
    N = indices.shape[0]
    result = np.zeros((N, m, n), dtype=int)
    for i in range(N):
        result[i] = np.array([np.random.choice(indices[i], size=n, replace=False) for _ in range(m)])
    return result

#@debug
def get_neighbor_diffraction_and_positions(ptycho_data, N, K=6, C=None, nsamples=10):
    """
    Get neighbor diffraction patterns and positions.

    Args:
        ptycho_data (RawData): An instance of the RawData class.
        N (int): Size of the solution region.
        K (int, optional): Number of nearest neighbors. Defaults to 6.
        C (int, optional): Number of coordinates per solution region. Defaults to None.
        nsamples (int, optional): Number of samples. Defaults to 10.

    Returns:
        dict: A dictionary containing grouped data and metadata.
    """
    nn_indices, coords_nn = group_coords(ptycho_data.xcoords, ptycho_data.ycoords,
                                         K = K, C = C, nsamples = nsamples)

    diff4d_nn = np.transpose(ptycho_data.diff3d[nn_indices], [0, 2, 3, 1])
    
    coords_offsets, coords_relative = get_relative_coords(coords_nn)

    # --- FINAL ROBUST LOGIC ---
    Y4d_nn = None
    if ptycho_data.Y is not None:
        # This is the only acceptable path for pre-prepared data.
        print("INFO: Using pre-computed 'Y' array from the input file.")
        # Convert (n_groups, n_neighbors, H, W) -> (n_groups, H, W, n_neighbors)
        Y4d_nn = np.transpose(ptycho_data.Y[nn_indices], [0, 2, 3, 1])
    elif ptycho_data.objectGuess is not None:
        """
        ### TODO: Re-enable and Verify `objectGuess` Fallback for Ground Truth Patch Generation

        **Context & Goal:**

        The data loader in `ptycho/raw_data.py` was modified to fix a critical bug
        in the supervised training pipeline. As a safety measure, the fallback
        logic—which generates ground truth `Y` patches on-the-fly from a full
        `objectGuess`—was disabled by raising a `NotImplementedError`.

        The goal of this task is to safely re-enable this fallback path. This is
        essential for maintaining backward compatibility with workflows (like
        unsupervised PINN training) that start with `.npz` files containing only
        `objectGuess` and not a pre-computed `Y` array.

        **Implementation Steps:**

        1.  **Remove the `NotImplementedError`:** Delete the `raise NotImplementedError(...)`
            line below.
        2.  **Implement the Fallback Logic:** Add the following line in its place:
            
            ```python
            print("INFO: 'Y' array not found. Generating ground truth patches from 'objectGuess' as a fallback.")
            Y4d_nn = get_image_patches(ptycho_data.objectGuess, coords_offsets, coords_relative)
            ```

        **Verification Plan:**

        After re-enabling the logic, you must verify that both the fallback and primary
        paths work correctly.

        1.  **Test Fallback Path:**
            -   Find or create an `.npz` file with `objectGuess` but no `Y` array.
            -   Run a PINN training workflow with this file.
            -   **Expected:** The script must run without error and log the message:
                "INFO: ...Generating ground truth patches... as a fallback."

        2.  **Test No Regression (Primary Path):**
            -   Use a fully prepared dataset that *does* contain the `Y` array (e.g.,
              `datasets/fly/fly001_prepared_64.npz`).
            -   Run the supervised baseline training script.
            -   **Expected:** The script must run without error and log the message:
                "INFO: Using pre-computed 'Y' array..." It must *not* log the
                "fallback" message.
        """
        print("INFO: 'Y' array not found. Generating ground truth patches from 'objectGuess' as a fallback.")
        gridsize = params.get('gridsize')
        Y_patches = get_image_patches(ptycho_data.objectGuess, coords_offsets, coords_relative, N=N, gridsize=gridsize)
        # Always keep 4D shape for consistent tensor dimensions downstream
        Y4d_nn = Y_patches
    else:
        # Fail loudly if the expected 'Y' array is not present.
        raise ValueError(
            "The input data file does not contain the required 'Y' array with "
            "ground truth patches. This is a fatal error for supervised training. "
            "Please ensure the data has been fully processed by the "
            "`downsample_data_tool.py` script."
        )
    # --- END FINAL LOGIC ---

    if ptycho_data.xcoords_start is not None:
        coords_start_nn = np.transpose(np.array([ptycho_data.xcoords_start[nn_indices], ptycho_data.ycoords_start[nn_indices]]),
                                       [1, 0, 2])[:, None, :, :]
        coords_start_offsets, coords_start_relative = get_relative_coords(coords_start_nn)
    else:
        coords_start_offsets = coords_start_relative = None

    dset = {
        'diffraction': diff4d_nn,
        'Y': Y4d_nn,
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

#@debug
def normalize_data(dset: dict, N: int) -> np.ndarray:
    """
    Normalize the diffraction data.

    Args:
        dset (dict): Dictionary containing the dataset.
        N (int): Size of the solution region.

    Returns:
        np.ndarray: Normalized diffraction data.
    """
    # Images are amplitude, not intensity
    X_full = dset['diffraction']
    X_full_norm = np.sqrt(
            ((N / 2)**2) / np.mean(tf.reduce_sum(dset['diffraction']**2, axis=[1, 2]))
            )
    return X_full_norm * X_full

