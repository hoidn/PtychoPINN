#This code is for generating "fake" data from a known ptychography dataset

#Imports
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import time
from functools import partial

#Other functions
import ptycho_torch.patch_generator as pg
from ptycho_torch.datagen.objects import create_complex_layered_procedural_object, downscale_complex_image, create_complex_polyhedra
from ptycho_torch.datagen.objects import create_dead_leaves, create_white_noise_object, create_simplex_noise_object
from ptycho_torch.datagen.probe import generate_zernike_probe, generate_random_fzp, generate_random_zernike
import ptycho_torch.helper as hh
from ptycho_torch.config_params import TrainingConfig, DataConfig, ModelConfig
from skimage.draw import line_aa, disk, rectangle, ellipse, circle_perimeter_aa

#Random utils
from glob import glob


def load_probe_object(file_path):
    """
    Load object and probe guesses from a .npz file.

    Args:
        file_path (str): Path to the .npz file containing objectGuess and probeGuess.

    Returns:
        tuple: A tuple containing (objectGuess, probeGuess)

    Raises:
        ValueError: If required data is missing from the .npz file or if data is invalid.
        RuntimeError: If an error occurs during file loading.
    """
    try:
        with np.load(file_path) as data:
            if 'objectGuess' not in data or 'probeGuess' not in data:
                raise ValueError("The .npz file must contain 'objectGuess' and 'probeGuess'")
            
            objectGuess = data['objectGuess']
            probeGuess = data['probeGuess']

        if probeGuess.shape == 2:
            scale_factor = np.sqrt(np.sum(np.abs(probeGuess)**2))
            probeGuess /= scale_factor

        # Validate extracted data
        if objectGuess.ndim != 2 or probeGuess.ndim != 2:
            raise ValueError("objectGuess and probeGuess must be 2D arrays")
        if not np.iscomplexobj(objectGuess) or not np.iscomplexobj(probeGuess):
            raise ValueError("objectGuess and probeGuess must be complex-valued")

        return objectGuess, probeGuess

    except Exception as e:
        raise RuntimeError(f"Error loading data from {file_path}: {str(e)}")
    
def simulate_from_npz(obj_path, probe_path,
                      nimages, data_config,
                      probe_arg):
    """
    Load object and probe guesses from .npz files and generate simulated ptychography data.

    Args:
        file_path (str): Path to the .npz file containing objectGuess and probeGuess.
        nimages (int): Number of scan positions to generate.
        buffer (float, optional): Border size to avoid when generating coordinates. 
                                  If None, defaults to 5% of the smaller dimension of objectGuess.
        random_seed (int, optional): Seed for random number generation. If None, uses system time.

    Returns:
        RawData: A RawData instance containing the simulated ptychography data.

    Raises:
        ValueError: If input parameters are invalid.
        RuntimeError: If an error occurs during simulation or file loading.
    """
    # TODO there should be an option to use the same scan point coords as in the dataset for the simulation. This
    # would be useful for consistency checks
    # Load guesses from file
    objectGuess = np.load(obj_path)['objectGuess']
    probeGuess = np.load(probe_path)['probeGuess']

    # Set default buffer if not provided
    if buffer is None:
        buffer = min(objectGuess.shape) * 0.05  # 5% of the smaller dimension

    # Generate simulated data
    return generate_simulated_data(objectGuess, probeGuess, nimages, data_config, probe_arg)

def simulate_multiple_experiments(obj_list,
                                  probe_list,
                                  images_per_experiment,
                                  img_shape,
                                  data_config,
                                  probe_arg,
                                  save_dir,
                                  save_bool = True):
    """
    Generates multiple simulated ptychography experiments using randomized probes

    Args:
        obj_list (List): List of generated objects from another step
        probe_list (List): List of generated probes from another step.
             Must be equal to or shorter in length than the obj_list length.
        images_per_experiment (int): Number of experiments
        img_shape (int, int): Tuple of object image dimensions
        data_config (DataConfig): DataConfig object, only used for N
        probe_arg (Dict): Additional probe arguments to create a range of usable probes
        save_dir (Path): Path to save all generated experiments 
    """

    if len(probe_list) > len(obj_list):
        raise ValueError("Probe list must be less than or equal to object list length!")
    
    n_experiments = len(obj_list)
    n_probes = len(probe_list)

    for i in range(n_experiments):
        start = time.time()
        print(f"----Beginning simulation for experiment {i}----")

        #Object and probe
        probe_idx = i % n_probes
        probe_i = probe_list[probe_idx]
        probe_name_i = probe_arg['probe_name_idx'][i]
        obj_i = obj_list[i]

        raw_data = generate_simulated_data(obj_i, probe_i,
                                           images_per_experiment,
                                           data_config, probe_arg)
        

        if save_bool:
            save_name = "synthetic_" + str(i)
            save_path = save_dir + '/' + save_name
            
            np.savez(save_path,
                    diff3d = raw_data['diff3d'],
                    label = raw_data['label'],
                    objectGuess = raw_data['objectGuess'],
                    probeGuess = raw_data['probeGuess'].squeeze(),
                    xcoords = raw_data['xcoords'],
                    ycoords = raw_data['ycoords'],
                    probeName = probe_name_i,
                    )
            end = time.time()
            print(f"----Finished saving for experiment {i} in {end-start} seconds----")

def assemble_precomputed_images(dir_list, mode, single_dir = False):
    """
    Just appends all probes or objects into single list from directories of interest.
    """

    precomputed_images = []
    
    # Turn single directory string into a list
    if single_dir:
        dir_list = [dir_list]
    
    for dir in dir_list:
        file_list = glob(dir + '/*.npz')
        for file in file_list:
            if mode == 'probe':
                file_im = np.load(file)['probeGuess']
                #Removed normalization because we want the raw probe
            elif mode == 'object':
                file_im = np.load(file)['objectGuess']

            precomputed_images.append(file_im)
            

    return precomputed_images

def simulate_synthetic_objects(img_shape, data_config, nimages, obj_method, obj_arg):
    """
    Generates list of synthetic objects from pre-built methods. Wrapper function.

    Args:
        img_shape (int, int): Tuple of image dimensions, generally square.
        nimages (int): Number of images in list
        obj_method (str): String descriptor that must match with obj methods listed internally
        obj_arg (Dict): Object arguments (hyperparameters)
    """
    #Internal object methods list
    obj_methods = ['procedural', 'polyhedra','dead_leaves','white_noise','simplex_noise','blurred_white_noise']
    
    #Check validity
    if obj_method not in obj_methods:
        raise ValueError("Method not in supported methods")
    
    #Select procedural generation method
    if obj_method == 'dead_leaves':
        obj_func = partial(create_dead_leaves, obj_arg = obj_arg)
    elif obj_method == 'procedural':
        obj_func = create_complex_layered_procedural_object
    elif obj_method == 'polyhedra':
        obj_func = create_complex_polyhedra
    elif obj_method == 'white_noise':
        obj_func = partial(create_white_noise_object, obj_arg = obj_arg)
    elif obj_method == 'simplex_noise':
        obj_func = create_simplex_noise_object
    elif obj_method == 'blurred_white_noise':
        obj_arg['blur'] = True
        obj_func = partial(create_white_noise_object, obj_arg = obj_arg)

    #Create list
    obj_list = []

    #Populate list
    for i in range(nimages):
        obj = obj_func(img_shape)
        obj_list.append(obj)

    return obj_list

def simulate_synthetic_probes(data_config, nimages, probe_method, probe_arg):
    """
    Generates list of synthetic probes from pre-built methods. Wrapper function.

    Args:
        img_shape (int, int): Tuple of image dimensions, generally square.
        nimages (int): Number of images in list
        obj_method (str): String descriptor that must match with probe methods listed internally
        probe_arg (dict): Specific generation settings for the probe
    """
    #Internal methods
    probe_methods = ['zernike', 'fzp']

    #Check internal list
    if probe_method not in probe_methods:
        raise ValueError("Method not in supported probe methods")

    #Assign method
    if probe_method == 'zernike':
        probe_func = generate_random_zernike
    elif probe_method == 'fzp':
        probe_func = generate_random_fzp

    #List
    probe_list = []

    for i in range(nimages):
        probe = probe_func(shape = (data_config.N,data_config.N),
                       probe_arg = probe_arg)
        probe_list.append(probe)

    return probe_list    


def generate_simulated_data(objectGuess, probeGuess, nimages,
                            data_config, probe_arg, random_seed=None):
    """
    Generate simulated ptychography data using random scan positions.

    Args:
        objectGuess (np.ndarray): Complex-valued 2D array representing the object.
        probeGuess (np.ndarray): Complex-valued 2D array representing the probe.
        nimages (int): Number of scan positions to generate.
        buffer (float): Border size to avoid when generating coordinates.
        random_seed (int, optional): Seed for random number generation. If None, uses system time.

    Returns:
        RawData: A RawData instance containing the simulated ptychography data.

    Raises:
        ValueError: If input parameters are invalid.
        RuntimeError: If an error occurs during simulation.
    """
    # Input validation
    if objectGuess.ndim != 2 or probeGuess.ndim < 2:
        raise ValueError("objectGuess must be 2d, probe must be greater than 2d")
    if not np.iscomplexobj(objectGuess) or not np.iscomplexobj(probeGuess):
        raise ValueError("objectGuess and probeGuess must be complex-valued")
    if nimages <= 0:
        raise ValueError("nimages must be positive and buffer must be non-negative")
    
    N = data_config.N

    # Get object dimensions
    height, width = objectGuess.shape

    # Ensure buffer doesn't exceed image dimensions
    buffer = N//2

    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    # Calculate grid parameters
    grid_size = int(np.ceil(np.sqrt(nimages)))
    x_step = (width - 2*buffer) / (grid_size - 1) if grid_size > 1 else 0
    y_step = (height - 2*buffer) / (grid_size - 1) if grid_size > 1 else 0
    
    # Generate grid coordinates with jitter
    jitter_amount = min(x_step, y_step) * 0.3  # 20% jitter
    
    x_grid = np.linspace(buffer, width - buffer, grid_size)
    y_grid = np.linspace(buffer, height - buffer, grid_size)
    xx, yy = np.meshgrid(x_grid, y_grid)
    
    # Add jitter and flatten
    xcoords = (xx.flatten() + np.random.uniform(-jitter_amount, jitter_amount, grid_size**2))[:nimages].astype(np.float32)
    xcoords = np.round(xcoords, 2)
    ycoords = (yy.flatten() + np.random.uniform(-jitter_amount, jitter_amount, grid_size**2))[:nimages].astype(np.float32)
    ycoords = np.round(ycoords, 2)
    #Convert object to proper data format
    objectGuess = objectGuess.astype(np.complex64)
    probeGuess = probeGuess.astype(np.complex64)

    # Create scan_index
    scan_index = np.zeros(nimages, dtype = int)

    # Generate simulated data
    print("Beginning simulation...")
    raw_data = from_simulation(xcoords, ycoords, probeGuess, objectGuess,
                               data_config, probe_arg, scan_index, batch_size = 3000)

    return raw_data

def generate_data_from_experiment(diff, objectGuess, probeGuess, xcoords, ycoords,
                            data_config, probe_arg, batch_size = 3000, random_seed=None):
    """
    Extract object patches matching scan positions from actual experimental data

    Args:
        objectGuess (np.ndarray): Complex-valued 2D array representing the object.
        probeGuess (np.ndarray): Complex-valued 2D array representing the probe.
        nimages (int): Number of scan positions to generate.
        buffer (float): Border size to avoid when generating coordinates.
        random_seed (int, optional): Seed for random number generation. If None, uses system time.

    Returns:
        RawData: A RawData instance containing the simulated ptychography data.

    Raises:
        ValueError: If input parameters are invalid.
        RuntimeError: If an error occurs during simulation.
    """
    # Input validation
    if objectGuess.ndim != 2 or probeGuess.ndim < 2:
        raise ValueError("objectGuess must be 2d, probe must be greater than 2d")
    if not np.iscomplexobj(objectGuess) or not np.iscomplexobj(probeGuess):
        raise ValueError("objectGuess and probeGuess must be complex-valued")
    
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    N = data_config.N

    # Get object dimensions
    height, width = objectGuess.shape

    # Ensure buffer doesn't exceed image dimensions
    buffer = N//2

    # Limit xcoords and ycoords to within the buffered area
    xcoords = xcoords[(xcoords >= buffer) & (xcoords <= width - buffer)]
    ycoords = ycoords[(ycoords >= buffer) & (ycoords <= height - buffer)]
 
    #Convert object to proper data format
    objectGuess = objectGuess.astype(np.complex64)
    #Normalize objectGuess amp to 1
    obj_amp, obj_phase = np.abs(objectGuess), np.angle(objectGuess)
    obj_amp /= np.max(obj_amp)
    #Reassembling obj_amp
    objectGuess = obj_amp * np.exp(1j * obj_phase)

    probeGuess = probeGuess.astype(np.complex64)

    # Create scan_index
    n_images = len(xcoords)
    scan_index = np.zeros(n_images, dtype = int)
    

    # Generate simulated data
    print("Beginning extraction...")

    # Convert coordinates and static data to torch (keep on CPU initially)
    xcoords_cpu = torch.from_numpy(xcoords)
    ycoords_cpu = torch.from_numpy(ycoords)
    objectGuess_torch = torch.from_numpy(objectGuess).to(device)

    # Initialize output arrays on CPU to store results
    obj_patches_list = []

    print(f"Processing {n_images} images in batches of {batch_size}...")
    
    # Process in batches
    for batch_start in range(0, n_images, batch_size):
        batch_end = min(batch_start + batch_size, n_images)

        # Get batch coordinates and move to GPU
        batch_xcoords = xcoords_cpu[batch_start:batch_end].to(device)
        batch_ycoords = ycoords_cpu[batch_start:batch_end].to(device)

        # Get image patches for this batch
        print("  Getting image patches...")
        batch_obj_patches = get_image_patches(objectGuess_torch,
                                            batch_xcoords, batch_ycoords,
                                            data_config)
    
        obj_patches_list.append(batch_obj_patches.squeeze().detach().cpu())
        
        # Clear GPU memory for this batch
        del batch_obj_patches, batch_xcoords, batch_ycoords
        if device == 'cuda':
            torch.cuda.empty_cache()

    # Concatenate all batches
    print("Concatenating results...")
    sampled_obj_patches = torch.cat(obj_patches_list, dim=0).numpy()

    output_dict = {
        'diff3d': diff,
        'label': sampled_obj_patches,
        'objectGuess': objectGuess,
        'probeGuess': probeGuess,
        'xcoords': xcoords,
        'ycoords': ycoords,
        'scan_index': scan_index
    }

    return output_dict

def from_simulation(xcoords, ycoords,
                    probeGuess, objectGuess,
                    data_config: DataConfig,
                    probe_arg, 
                    scan_index = None,
                    device = 'cuda',
                    batch_size = 32):  # Add batch_size parameter
    """
    Generate simulated ptychography data using random scan positions with batching.

    Args:
        xcoords, ycoords: (n_images,) 1-D numpy arrays - Coordinates for randomly sampled positions 
        probeGuess: (h,w) 2-D numpy array - Complex guess for the probe
        objectGuess: (h,w) 2-D numpy array - Complex guess for object
        data_Config: DataConfig - DataConfig object, only need this for the N parameter (size of diff pattern)
        batch_size: int - Number of images to process at once (default: 32)
    """
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    n_images = len(xcoords)

    # Re-normalize probe if it hasn't been already
    probeGuess, _ = hh.normalize_probe(probeGuess)
    
    # Convert coordinates and static data to torch (keep on CPU initially)
    xcoords_cpu = torch.from_numpy(xcoords)
    ycoords_cpu = torch.from_numpy(ycoords)
    probeGuess_torch = torch.from_numpy(probeGuess).to(device)
    objectGuess_torch = torch.from_numpy(objectGuess).to(device)

    # Initialize output arrays on CPU to store results
    diff_patterns_list = []
    obj_patches_list = []

    print(f"Processing {n_images} images in batches of {batch_size}...")
    
    # Process in batches
    for batch_start in range(0, n_images, batch_size):
        batch_end = min(batch_start + batch_size, n_images)
        
        # print(f"Processing batch {batch_start//batch_size + 1}/{(n_images + batch_size - 1)//batch_size} "
        #       f"(images {batch_start}-{batch_end-1})")
        
        # Get batch coordinates and move to GPU
        batch_xcoords = xcoords_cpu[batch_start:batch_end].to(device)
        batch_ycoords = ycoords_cpu[batch_start:batch_end].to(device)

        # Get image patches for this batch
        print("  Getting image patches...")
        batch_obj_patches = get_image_patches(objectGuess_torch,
                                            batch_xcoords, batch_ycoords,
                                            data_config)

        # Multiply probe and object, get scaled diffraction pattern
        print("  Diffracting...")
        batch_diff_patches, _, scaled_probe = hh.illuminate_and_diffract(
            batch_obj_patches.squeeze(), 
            probeGuess_torch,
            nphotons = np.random.randint(5e5, 1e6)
        )
        
        # Apply Poisson scaling to simulate real data
        print("  Poisson scaling...")
        batch_diff_patches = hh.poisson_scale(batch_diff_patches)

        # Apply detector beamstop
        print("  Applying detector beamstop...")
        batch_diff_patches = apply_detector_beamstop_torch(
            batch_diff_patches,
            probe_arg['beamstop_diameter']
        )

        # Move batch results to CPU and store
        diff_patterns_list.append(batch_diff_patches.detach().cpu())
        obj_patches_list.append(batch_obj_patches.squeeze().detach().cpu())
        
        # Clear GPU memory for this batch
        del batch_obj_patches, batch_diff_patches, batch_xcoords, batch_ycoords
        if device == 'cuda':
            torch.cuda.empty_cache()

    # Concatenate all batches
    print("Concatenating results...")
    diff_obj_patches = torch.cat(diff_patterns_list, dim=0).numpy()
    sampled_obj_patches = torch.cat(obj_patches_list, dim=0).numpy()
    
    # Move remaining data back to CPU
    objectGuess = objectGuess_torch.detach().cpu().numpy()
    probeGuess = scaled_probe.detach().cpu().numpy()

    output_dict = {
        'diff3d': diff_obj_patches,
        'label': sampled_obj_patches,
        'objectGuess': objectGuess,
        'probeGuess': probeGuess,
        'xcoords': xcoords,
        'ycoords': ycoords,
        'scan_index': scan_index
    }

    return output_dict

def get_image_patches(objectGuess,
                      xcoords, ycoords,
                      data_config):
    """
    Get and return image patches from single canvas

    Input
    -----
    objectGuess: torch.Tensor (H, W)
    xcoords: torch.Tensor (n_images)
    ycoords: torch.Tensor (n_images)

    Output:
        (n_images, H, W)

    
    """

    # --- Parameters ---
    N = data_config.N                 # Patch size (e.g., 64)
    M_y, M_x = objectGuess.shape      # Full canvas size (e.g., 512, 512)
    n_images = len(xcoords)           # Number of patches

    if N > M_y or N > M_x:
        raise ValueError(f"Patch size N ({N}) cannot be larger than object dimensions ({M_y}, {M_x})")

    # --- Calculate Required Translation Offset ---
    # Center of the large canvas (pixel coordinates)
    center_x = (M_x - 1) / 2.0
    center_y = (M_y - 1) / 2.0

    # Desired shift for each patch: (center_x - patch_center_x, center_y - patch_center_y)
    # This is the offset needed to move the image content at (x,y) to the center (cx,cy)
    offset_x = center_x - xcoords
    offset_y = center_y - ycoords

    # Reshape offset for the Translation function: (n_images, 1, 2)
    offsets = torch.stack([offset_x, offset_y], dim=-1).reshape(n_images, 1, 2)

    # --- Apply Translation ---
    # The Translation function needs a batch dimension for the object. Repeat it.
    object_batch = objectGuess.unsqueeze(0).repeat(n_images, 1, 1) # Shape (n_images, M_y, M_x)

    # Apply the translation using the revised function (assumed available)
    # No jitter applied here (jitter_amt=0)
    translated_full_patches = hh.Translation(object_batch, offsets, jitter_amt=0)
    # Output shape: (n_images, M_y, M_x)

    # --- Extract Center N x N Patch ---
    # Calculate start/end indices for the central crop
    start_y = (M_y - N) // 2
    end_y = start_y + N # Slice goes up to, but not including, end_y
    start_x = (M_x - N) // 2
    end_x = start_x + N # Slice goes up to, but not including, end_x

    # Perform the crop
    trimmed_patches = translated_full_patches.squeeze()[:, start_y:end_y, start_x:end_x]

    # Output shape: (n_images, N, N)

    return trimmed_patches

# Miscellanous functions (not suitable for helper)

def apply_detector_beamstop_torch(
    diff_pattern_intensity: torch.Tensor,
    detector_beamstop_diameter_pix: float
    ) -> torch.Tensor:
    """
    Applies a circular detector beamstop mask to a batch of diffraction patterns.

    Args:
        diff_pattern_intensity: 3D PyTorch tensor of diffraction intensities
                                with shape (B, H, W), where B is the batch size.
        detector_beamstop_diameter_pix: Diameter of the central stop in pixels.

    Returns:
        torch.Tensor: Diffraction pattern intensities with the center masked
                      to zero, same shape as input.
    """
    # --- Input Validation ---
    if diff_pattern_intensity.ndim != 3:
        raise ValueError(f"Input tensor must be 3D (B, H, W), but got shape {diff_pattern_intensity.shape}")
    if detector_beamstop_diameter_pix <= 0:
        return diff_pattern_intensity # No beamstop to apply

    # --- Get Shape and Parameters ---
    B, H, W = diff_pattern_intensity.shape
    radius_pix = detector_beamstop_diameter_pix / 2.0
    device = diff_pattern_intensity.device
    dtype = diff_pattern_intensity.dtype # Match input dtype for coords

    # --- Create Coordinate Grids (centered on the detector grid) ---
    # Assumes fftshift has been applied, so DC component is at the center.
    center_y_det = (H - 1) / 2.0
    center_x_det = (W - 1) / 2.0

    # Create 1D coordinate tensors relative to the center
    y_coords = torch.arange(H, dtype=dtype, device=device) - center_y_det
    x_coords = torch.arange(W, dtype=dtype, device=device) - center_x_det

    # Create 2D coordinate grids using meshgrid
    # Use 'xy' indexing: xx shape (H, W), yy shape (H, W)
    xx_det, yy_det = torch.meshgrid(x_coords, y_coords, indexing='xy')

    # --- Calculate Radial Distance from Center ---
    rr_det_pix = torch.sqrt(xx_det**2 + yy_det**2) # Shape (H, W)

    # --- Create Mask: True (1) *outside* the beamstop radius ---
    # Mask has shape (H, W)
    detector_bs_mask = (rr_det_pix > radius_pix)

    # --- Apply Mask ---
    # Multiply the batch tensor (B, H, W) by the mask (H, W).
    # PyTorch broadcasting handles this correctly.
    # Boolean mask multiplication usually works directly, or convert: .to(dtype)
    masked_diff_pattern = diff_pattern_intensity * detector_bs_mask

    return masked_diff_pattern

