#This code is for generating "fake" data from a known ptychography dataset

#Imports
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

#Other functions
import ptycho_torch.patch_generator as pg
import ptycho_torch.helper as hh
from ptycho_torch.config_params import Params



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
    return generate_simulated_data(objectGuess, probeGuess, nimages, buffer, random_seed)
    

def generate_simulated_data(objectGuess, probeGuess, nimages, buffer, random_seed=None):
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
    if nimages <= 0 or buffer < 0:
        raise ValueError("nimages must be positive and buffer must be non-negative")
    
    N = Params().get('N')

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
    scan_index = np.zeros(nimages, dtype=int)

    # Generate simulated data
    raw_data = from_simulation(xcoords, ycoords, probeGuess, objectGuess, scan_index)
    return raw_data

def from_simulation(xcoords, ycoords, probe, objectGuess, scan_index):
    """
    Generate simulated ptychography data using random scan positions.

    """

    xcoords_start = xcoords
    ycoords_start = ycoords

    sampled_diff3d = get_image_patches(objectGuess, xcoords, ycoords)



    norm_factor = hh.scale_nphotons(objectGuess)



    X, Y_I_xprobe, Y_phi_xprobe, intensity_scale = illuminate_and_diffract(Y_I, Y_phi, probeGuess)
    norm_Y_I = datasets.scale_nphotons(X)
    assert X.shape[-1] == 1, "gridsize must be set to one when simulating in this mode"
    # TODO RawData should have a method for generating the illuminated ground truth object
    return RawData(xcoords, ycoords, xcoords_start, ycoords_start, tf.squeeze(X).numpy(),
                    probeGuess, scan_index, objectGuess,
                    Y = tf.squeeze(hh.combine_complex( Y_I_xprobe, Y_phi_xprobe)).numpy(),
                    norm_Y_I = norm_Y_I)

def get_image_patches(objectGuess, xcoords, ycoords):
    """
    Get and return image patches from single canvas

    Input
    -----
    objectGuess: np.ndarray (H, W)
    xcoords: np.darray (N)
    ycoords: np.darray (N)

    
    """

    N = Params().get('N')
    gridsize = Params().get('gridsize')

    B = len(xcoords)
    c = gridsize ** 2

    #No need to pad canvas. We will never sample anywhere from outside the canvas
    canvas = torch.from_numpy(objectGuess)

    #Use grid_sample to sample all coordinates to get smaller patches
    #All coordinates are in pixel units
    #Remember that F.grid_sample needs coords in [-1, 1], so need to shift
    h, w = canvas.shape
    x, y = torch.arange(h), torch.arange(w)
    x_shifted, y_shifted = x / (h-1), y / (w-1) #map to [0,1]
    #Creating a meshgrid of size (H x W x 2)
    grid = torch.stack([x_shifted.unsqueeze(-1).expand(-1, len(y_shifted)),
                        y_shifted.unsqueeze(0).expand(len(x_shifted.shape), -1)],
                        dim = -1) * 2 - 1 #map to [-1,1]
    #F.grid_sample behavior needs tranpose
    grid = torch.transpose(grid, 1, 2)

    #F.grid_sample needs (N,C,H,W), so we put in two dummy N,C dimensions
    sampled_real = F.grid_sample(canvas[None,None,...].real, grid,
                                 mode = 'bilinear', align_corners = True)

    sampled_imag = F.grid_sample(canvas[None,None,...].imag, grid,
                                 mode = 'bilinear', align_corners = True)
    
    sampled_complex = torch.view_as_complex(torch.stack([sampled_real, sampled_imag],
                                                        dim = -1))
    
    return sampled_complex
    #Note to albert:
    #Continue developing get_image_patches adapated to pytorch.. shouldn't be too hard







