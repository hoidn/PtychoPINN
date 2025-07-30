#This code is for generating "fake" data from a known ptychography dataset

#Imports
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

#Other functions
import ptycho_torch.patch_generator as pg
import ptycho_torch.helper as hh
from ptycho_torch.config_params import TrainingConfig, DataConfig, ModelConfig



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

        # Validate extracted data
        if objectGuess.ndim != 2 or probeGuess.ndim != 2:
            raise ValueError("objectGuess and probeGuess must be 2D arrays")
        if not np.iscomplexobj(objectGuess) or not np.iscomplexobj(probeGuess):
            raise ValueError("objectGuess and probeGuess must be complex-valued")

        return objectGuess, probeGuess

    except Exception as e:
        raise RuntimeError(f"Error loading data from {file_path}: {str(e)}")
    
def simulate_from_npz(file_path, nimages, buffer=None, random_seed=None):
    """
    Load object and probe guesses from a .npz file and generate simulated ptychography data.

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
    objectGuess, probeGuess = load_probe_object(file_path)

    # Set default buffer if not provided
    if buffer is None:
        buffer = min(objectGuess.shape) * 0.05  # 5% of the smaller dimension

    # Generate simulated data
    return generate_simulated_data(objectGuess, probeGuess, nimages, random_seed)
    

def generate_simulated_data(objectGuess, probeGuess, nimages, random_seed=None):
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
    if objectGuess.ndim != 2 or probeGuess.ndim != 2:
        raise ValueError("objectGuess and probeGuess must be 2D arrays")
    if not np.iscomplexobj(objectGuess) or not np.iscomplexobj(probeGuess):
        raise ValueError("objectGuess and probeGuess must be complex-valued")
    if nimages <= 0:
        raise ValueError("nimages must be positive and buffer must be non-negative")
    
    N = DataConfig().get('N')

    # Get object dimensions
    height, width = objectGuess.shape

    # Ensure buffer doesn't exceed image dimensions
    buffer = N//2

    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate random coordinates (floats)
    xcoords = np.random.uniform(buffer, width - buffer, nimages)
    ycoords = np.random.uniform(buffer, height - buffer, nimages)

    # Create scan_index
    scan_index = np.zeros(nimages, dtype = int)

    # Generate simulated data
    raw_data = from_simulation(xcoords, ycoords, probeGuess, objectGuess, scan_index)

    return raw_data

def from_simulation(xcoords, ycoords, probeGuess, objectGuess, scan_index = None):
    """
    Generate simulated ptychography data using random scan positions.

    """
    #Convert all input vars to torch
    xcoords = torch.from_numpy(xcoords)
    ycoords = torch.from_numpy(ycoords)
    probeGuess = torch.from_numpy(probeGuess)
    objectGuess = torch.from_numpy(objectGuess)

    sampled_obj_patches = get_image_patches(objectGuess, probeGuess, xcoords, ycoords)

    diff_obj_patches, obj, scaling = hh.illuminate_and_diffract(sampled_obj_patches.squeeze(), probeGuess)
    output_scaling_factor = hh.scale_nphotons(diff_obj_patches)

    output_dict = {
        'diff3d':  diff_obj_patches,
        'original_object': sampled_obj_patches.squeeze(),
        'object': obj,
        'obj_scale_factor': output_scaling_factor,
        'diff_scale_factor': scaling,
        'xcoords': xcoords,
        'ycoords': ycoords,
        'xcoords_start': xcoords,
        'ycoords_start': ycoords,
        'scan_index': scan_index
    }

    return output_dict
def get_image_patches(objectGuess, probeGuess, xcoords, ycoords):
    """
    Get and return image patches from single canvas

    Input
    -----
    objectGuess: np.ndarray (H, W)
    xcoords: np.darray (N)
    ycoords: np.darray (N)

    
    """

    #No need to pad canvas. We will never sample anywhere from outside the canvas
    N = DataConfig().get('N')
    n_images = len(xcoords)
    #Need to add batch dimension to objectGuess
    canvas = objectGuess[None].expand(n_images,-1,-1)

    #All coordinates are in pixel units
    #Remember that F.grid_sample needs coords in [-1, 1], so need to shift
    h_obj, w_obj = objectGuess.shape
    h_prob, w_prob = probeGuess.shape
    x, y = torch.arange(h_prob), torch.arange(w_prob)
    #Create "grid coordinates" at every xcoord location
    x_shifted, y_shifted = ((xcoords - N/2)[:,None] + x)/ (h_obj-1), \
                           ((ycoords - N/2)[:,None] + y)/ (w_obj-1) #map to [0,1]
    
    #Creating a meshgrid of size (H x W x 2)
    grid = torch.stack([x_shifted.unsqueeze(-1).expand(n_images, -1, y_shifted.shape[1]),
                        y_shifted.unsqueeze(1).expand(n_images, x_shifted.shape[1], -1)],
                        dim = -1) * 2 - 1 #map to [-1,1]
    #F.grid_sample behavior needs tranpose
    grid = torch.transpose(grid, 1, 2)
    grid = grid.to(torch.float32)

    #F.grid_sample needs (N,C,H,W), so we put in 1 dummy C dimension
    sampled_real = F.grid_sample(canvas.unsqueeze(1).real, grid,
                                 mode = 'bilinear', align_corners = True)

    sampled_imag = F.grid_sample(canvas.unsqueeze(1).imag, grid,
                                 mode = 'bilinear', align_corners = True)
    
    sampled_complex = torch.view_as_complex(torch.stack([sampled_real, sampled_imag],
                                                        dim = -1))
    
    return sampled_complex
    #Note to albert:
    #Continue developing get_image_patches adapated to pytorch.. shouldn't be too hard







